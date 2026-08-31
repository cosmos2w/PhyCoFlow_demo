from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

import audit_training_cost_v4 as audit
import benchmark_training_replay_v4 as replay


class TrainingCostV4AuditTest(unittest.TestCase):
    def test_checkpoint_optimizer_steps_are_explicit_and_consistent(self) -> None:
        metadata = audit.extract_checkpoint_metadata(
            {
                "epoch": 7,
                "optimizer": {
                    "state": {
                        "0": {"step": 42},
                        "1": {"step": 42},
                    }
                },
            }
        )
        self.assertEqual(metadata["update_count"], 42)
        self.assertEqual(metadata["update_count_status"], "explicit_consistent")
        self.assertEqual(metadata["update_count_sources"], ["checkpoint.optimizer.state[*].step"])

    def test_conflicting_global_and_optimizer_steps_are_not_promoted(self) -> None:
        metadata = audit.extract_checkpoint_metadata(
            {
                "global_step": 42,
                "optimizer": {"state": {"0": {"step": 41}, "1": {"step": 41}}},
            }
        )
        self.assertEqual(metadata["update_count_status"], "explicit_conflict")
        self.assertIsNone(metadata["update_count"])

    def test_config_device_ids_and_run_name_do_not_become_timing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run_20260101_010203"
            run_dir.mkdir()
            config_path = run_dir / "run_config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "device_ids": [2],
                        "run_name": "Baseline_demo_20260101_010203",
                        "epochs": 1000,
                    }
                ),
                encoding="utf-8",
            )
            evidence = audit.discover_declared_evidence(run_dir, config_path)
            self.assertEqual(evidence["wall_clock"], [])
            self.assertEqual(evidence["gpu"], [])
            summary = audit._gpu_evidence_summary(evidence["gpu"])
            self.assertEqual(summary["status"], "incomparable")

    def test_stage_total_requires_duration_gpu_identity_and_pinned_identity(self) -> None:
        stage = {
            "name": "demo stage 1",
            "identity_status": "pass",
            "metadata": {"update_count_status": "explicit_consistent"},
            "evidence": {
                "wall_clock": {"status": "pass", "seconds": 7200},
                "gpu": {"status": "pass", "active_gpu_count": 2},
            },
        }
        status, gpu_hours, reasons = audit._stage_total_status([stage])
        self.assertEqual(status, "defensible")
        self.assertEqual(gpu_hours, 4.0)
        self.assertEqual(reasons, [])

        stage["evidence"]["wall_clock"] = {"status": "missing", "seconds": None}
        status, gpu_hours, reasons = audit._stage_total_status([stage])
        self.assertEqual(status, "unavailable")
        self.assertIsNone(gpu_hours)
        self.assertTrue(any("wall-clock" in reason for reason in reasons))

    def test_real_contract_declares_exact_eight_and_strict_replay_gate(self) -> None:
        config_path = PACKAGE_ROOT / "configs" / "training_cost_audit_v4.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], "figure5-training-cost-audit-v4")
        self.assertEqual(config["method_order"], audit.METHOD_ORDER)
        self.assertEqual(len(config["checkpoints"]), 8)
        self.assertTrue(config["evidence_policy"]["forbid_filesystem_mtime"])
        self.assertEqual(config["promotion"]["replay_requires"]["measured_updates"], 100)
        self.assertEqual(config["promotion"]["replay_requires"]["measured_blocks"], 10)
        self.assertEqual(config["promotion"]["replay_requires"]["updates_per_block"], 10)
        self.assertEqual(
            config["promotion"]["replay_requires"]["timing_boundary"],
            "synchronized_update_core_preloaded_batch",
        )
        report = audit.audit_config(
            config,
            repo_root=REPO_ROOT,
            no_hash=True,
            no_checkpoint_metadata=True,
        )
        self.assertFalse(report["promotable"])
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["checks"]["historical_gpu_hours"])
        self.assertTrue(all(row["classification"] == "replay_required" for row in report["records"]))

    def test_replay_plan_is_read_only_and_has_all_required_stages(self) -> None:
        config = yaml.safe_load((PACKAGE_ROOT / "configs" / "training_cost_audit_v4.yaml").read_text(encoding="utf-8"))
        synthetic_audit = {
            "records": [
                {
                    "method": entry["method"],
                    "stages": [
                        {
                            "name": f"{entry['method']} stage {entry.get('stage', 1)}",
                            "role": "adopted_checkpoint_training_stage",
                            "path": entry["path"],
                            "config_path": entry["config_path"],
                            "training_config": {},
                            "metadata": {"update_count": 10, "update_count_sources": ["fixture"]},
                        }
                    ],
                }
                for entry in config["checkpoints"]
            ]
        }
        synthetic_audit["records"][3]["stages"].append(
            {
                "name": "Latent FM shared autoencoder",
                "role": "required_stage_1",
                "path": config["checkpoints"][3]["dependencies"][0]["path"],
                "config_path": config["checkpoints"][3]["dependencies"][0]["config_path"],
                "training_config": {},
                "metadata": {"update_count": 20, "update_count_sources": ["fixture"]},
            }
        )
        plan = replay.build_replay_plan(config, synthetic_audit)
        self.assertEqual(plan["status"], "planned")
        self.assertEqual(len(plan["rows"]), 9)
        self.assertEqual(plan["protocol"]["warmup_updates"], 20)
        self.assertEqual(plan["protocol"]["measured_updates"], 100)
        self.assertEqual(plan["protocol"]["measured_blocks"], 10)
        self.assertEqual(plan["protocol"]["updates_per_block"], 10)
        self.assertTrue(all(row["write_checkpoint"] is False for row in plan["rows"]))
        self.assertTrue(all(row["mutate_archive"] is False for row in plan["rows"]))
        self.assertFalse(plan["safety"]["execution_enabled"])

    def test_formal_source_uses_adopted_canonical_batches_and_explicit_unavailability(self) -> None:
        run_dir = (
            PACKAGE_ROOT
            / "results"
            / "ValidationV4"
            / "TrainingCost"
            / "training_replay_formal_v4r2"
        )
        if not run_dir.is_dir():
            self.skipTest("formal V4 replay products are local generated evidence")
        manifest = yaml.safe_load((run_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = yaml.safe_load((run_dir / "qa.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["protocol"]["batch_policy"],
            "adopted canonical batch size for every stage",
        )
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(
            summary["promoted_methods"],
            ["DMF-Gen", "FFM-FNO", "FFM-Perceiver", "SiT", "MLP-RBF", "Senseiver"],
        )

    def test_replay_execute_requires_explicit_non_destructive_confirmation(self) -> None:
        self.assertEqual(replay.main(["--execute"]), 2)


if __name__ == "__main__":
    unittest.main()
