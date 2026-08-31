from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


class Figure5V4PipelineTest(unittest.TestCase):
    def _config(self) -> dict:
        return yaml.safe_load((PACKAGE_ROOT / "configs" / "figure5_v4.yaml").read_text(encoding="utf-8"))

    def test_v4_contract_is_additive_and_replaces_lower_row_with_training_cost_and_stress(self) -> None:
        config = self._config()
        self.assertEqual(config["schema_version"], "figure5-validation-v4")
        self.assertEqual(list(config["figure"]["panel_map"]), list("abcde"))
        self.assertNotIn("row_headers", config["figure"])
        self.assertEqual(config["figure"]["panel_map"]["d"], "cost_training_compute")
        self.assertEqual(config["figure"]["panel_map"]["e"], "cost_scalability_envelope")
        self.assertEqual(config["formal_inputs"]["training_cost_run_id"], "training_replay_formal_v4r2")
        self.assertEqual(config["formal_protocol"]["scale_stress"]["native_query_counts"], [1024, 4096, 16384, 40300])
        scale = config["formal_protocol"]["scale_stress"]
        self.assertEqual(scale["throughput_query_counts"], [100000, 250000, 500000, 1000000, 2000000, 4000000])
        self.assertEqual(scale["adaptive_query_cap"], 8000000)
        self.assertEqual(scale["query_spec"]["generator"], "torch.quasirandom.SobolEngine")
        self.assertEqual(scale["query_spec"]["sequence_policy"], "exact_sensor_prefix_then_sobol_suffix")
        self.assertTrue(config["formal_protocol"]["scale_stress"]["no_accuracy_claim_above_native"])
        self.assertEqual(config["style"]["method_colors"]["DMF-Gen"], "#E63946")
        self.assertEqual(config["style"]["method_markers"]["DMF-Gen"], "o")

    def test_non_strict_build_keeps_v3_abc_and_marks_v4_de_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            timestamp = "20990102_0000"
            config = self._config()
            config["formal_inputs"]["training_cost_root"] = str(output_root / "missing_training")
            config["formal_inputs"]["scale_root"] = str(output_root / "missing_scale")
            config_path = output_root / "figure5_v4_missing.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "build_figure5_v4.py"),
                    "--config",
                    str(config_path),
                    "--timestamp",
                    timestamp,
                    "--output-root",
                    str(output_root),
                ],
                check=True,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertIn('"d": "pending"', result.stdout)
            self.assertIn('"e": "pending"', result.stdout)
            bundle = output_root / "figures" / "generated" / timestamp
            self.assertEqual(len(list(bundle.glob("*.svg"))), 6)
            self.assertFalse(list(bundle.glob("*.pdf")))
            for path in bundle.glob("*.svg"):
                root = ET.parse(path).getroot()
                self.assertTrue(any(node.tag.endswith("text") for node in root.iter()))
            manifest = json.loads((output_root / "results" / "derived" / timestamp / "build_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "figure5-validation-v4")
            self.assertTrue(manifest["no_v4_fallback"])
            modes = {row["panel"]: row["mode"] for row in manifest["sources"]}
            self.assertEqual(modes, {"a": "formal", "b": "formal", "c": "formal", "d": "pending", "e": "pending"})

    def test_strict_formal_rejects_absent_v4_d_and_e_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config()
            config["formal_inputs"]["training_cost_root"] = str(root / "missing_training")
            config["formal_inputs"]["scale_root"] = str(root / "missing_scale")
            config_path = root / "figure5_v4_missing.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "build_figure5_v4.py"),
                    "--config",
                    str(config_path),
                    "--timestamp",
                    "20990102_0001",
                    "--output-root",
                    str(root / "out"),
                    "--strict-formal",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-formal panels", result.stderr)
            self.assertIn("d", result.stderr)
            self.assertIn("e", result.stderr)
            self.assertFalse((root / "out" / "figures").exists())

    def test_v4_scale_loader_never_accepts_v3_query_tables(self) -> None:
        from utils.figure5_v4_data import load_figure5_v4_data

        config = self._config()
        config["formal_inputs"]["scale_root"] = str(REPO_ROOT / "Dis_SI_Process" / "results" / "ValidationV3" / "CostClean")
        config["formal_inputs"]["scale_run_id"] = "formal_cost_clean_v3_20260830_v3"
        data, records = load_figure5_v4_data(config, REPO_ROOT)
        self.assertEqual(data["modes"]["e"], "pending")
        record = next(item for item in records if item.panel == "e")
        self.assertIn("schema_version", record.note)
        self.assertIn("figure5-validation-v4-scale-stress-1", record.note)

    def test_v3_native_scaling_prefix_is_canonicalized_without_v4_rows(self) -> None:
        from utils.figure5_v4_data import _load_v3_scaling_native

        native, errors = _load_v3_scaling_native(self._config(), REPO_ROOT)
        self.assertEqual(errors, [])
        self.assertIsNotNone(native)
        assert native is not None
        latency = native["latency"]
        dmf_counts = latency.loc[latency["method"].eq("DMF-Gen"), "N"].astype(int).tolist()
        fno_counts = latency.loc[latency["method"].eq("FFM-FNO"), "N"].astype(int).tolist()
        self.assertEqual(dmf_counts, [1024, 4096, 16384, 40300])
        self.assertEqual(fno_counts, [40300])
        self.assertTrue((latency["query_region"] == "native_validated").all())
        self.assertTrue((latency["source_schema"] == "figure5-validation-v3-cost-1").all())

    def test_scale_attempts_accept_a_prefix_ending_at_a_boundary(self) -> None:
        from utils.figure5_v4_data import _prefix_counts

        declared = [100000, 250000, 500000, 1000000, 2000000, 4000000, 8000000]
        self.assertTrue(_prefix_counts([100000, 250000], declared))
        self.assertTrue(_prefix_counts(declared, declared))
        self.assertFalse(_prefix_counts([100000, 500000], declared))
        self.assertFalse(_prefix_counts([100000, 250000, 500000, 1000000, 4000000], declared))

    def test_scale_consumer_merges_v3_native_prefix_and_v4_stress_prefix(self) -> None:
        from utils.figure5_v4_data import _load_scale_stress

        config = self._config()
        stress_counts = [100000, 250000, 500000, 1000000, 2000000, 4000000, 8000000]
        variable_methods = ["DMF-Gen", "FFM-Perceiver", "MLP-RBF", "Senseiver"]
        methods = config["paper_contract"]["method_order"]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "scale_stress_formal_v4"
            run_dir.mkdir()
            spec_hash = "spec-hash"
            manifest = {
                "schema_version": "figure5-validation-v4-scale-stress-1",
                "status": "complete",
                "formal": True,
                "run_id": run_dir.name,
                "dummy_query_spec_sha256": spec_hash,
                "dummy_query_spec": {
                    "generator": "torch.quasirandom.SobolEngine",
                    "include_sensor_prefix": True,
                    "sequence_policy": "exact_sensor_prefix_then_sobol_suffix",
                },
                "protocol": {
                    "native_query_count": 40300,
                    "sensor_count": 256,
                    "predeclared_query_counts": stress_counts[:-1],
                    "candidate_query_counts": stress_counts,
                    "global_query_cap": 8000000,
                },
            }
            qa = {
                "status": "pass",
                "support_methods_exact": True,
                "all_eligible_attempted": True,
                "candidate_counts_predeclared": True,
                "shared_query_hash_per_count": True,
                "largest_success_first_failure_recorded": True,
                "latency_iqr_valid": True,
                "throughput_only_no_accuracy_claim": True,
                "no_unsupported_scaling_curve": True,
                "geometry_preparation_separate": True,
                "repeat_rows_present": True,
                "identity_pass": True,
                "gpu_clean_before": True,
                "gpu_clean_after": True,
                "fixed_grid_methods_have_no_scaling_curve": True,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run_dir / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
            support_rows = []
            for method in methods:
                eligible = method in variable_methods
                support_rows.append(
                    {
                        "method": method,
                        "status": "ok",
                        "native_query_supported": eligible,
                        "query_scaling_eligible": eligible,
                        "native_only": not eligible,
                        "decision_basis": "canonical arbitrary query coordinates" if eligible else "fixed native grid",
                    }
                )
            pd.DataFrame(support_rows).to_csv(run_dir / "native_query_support_audit.csv", index=False)
            query_rows = [
                {
                    "N": count,
                    "query_sha256": f"query-{count}",
                    "spec_sha256": spec_hash,
                    "throughput_only": True,
                    "accuracy_claim": False,
                    "generator": "torch.quasirandom.SobolEngine",
                    "sensor_count": 256,
                }
                for count in stress_counts
            ]
            pd.DataFrame(query_rows).to_csv(run_dir / "query_coordinates_manifest.csv", index=False)
            summary_rows = []
            boundary_rows = []
            for method in variable_methods:
                counts = stress_counts[:2] if method == "DMF-Gen" else stress_counts
                for count in counts:
                    failed = method == "DMF-Gen" and count == 250000
                    summary_rows.append(
                        {
                            "method": method,
                            "N": count,
                            "status": "boundary_failure" if failed else "ok",
                            "median_latency_ms": "" if failed else 1.0,
                            "latency_q25_ms": "" if failed else 0.9,
                            "latency_q75_ms": "" if failed else 1.1,
                            "peak_allocated_mib": "" if failed else 10.0,
                            "query_sha256": f"query-{count}",
                            "spec_sha256": spec_hash,
                            "throughput_only": True,
                            "accuracy_claim": False,
                        }
                    )
                boundary_rows.append(
                    {
                        "method": method,
                        "largest_success_N": "100000" if method == "DMF-Gen" else "8000000",
                        "first_failure_N": "250000" if method == "DMF-Gen" else "",
                        "termination_reason": "first_failure" if method == "DMF-Gen" else "global_cap_reached",
                    }
                )
            pd.DataFrame(summary_rows).to_csv(run_dir / "scale_stress_summary.csv", index=False)
            pd.DataFrame(boundary_rows).to_csv(run_dir / "boundary_summary.csv", index=False)
            config["formal_inputs"]["scale_root"] = temporary
            config["formal_inputs"]["scale_run_id"] = run_dir.name
            loaded, errors = _load_scale_stress(config, REPO_ROOT)
            self.assertEqual(errors, [])
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertIn("variable_query_supported", loaded["support"].columns)
            self.assertEqual(
                set(
                    loaded["support"].loc[
                        loaded["support"]["variable_query_supported"], "method"
                    ]
                ),
                set(variable_methods),
            )
            latency = loaded["latency"]
            self.assertEqual(
                latency.loc[latency["method"].eq("DMF-Gen"), "N"].astype(int).tolist(),
                [1024, 4096, 16384, 40300, 100000, 250000],
            )
            self.assertEqual(latency.loc[latency["method"].eq("FFM-FNO"), "N"].astype(int).tolist(), [40300])
            self.assertEqual(
                int(loaded["boundary"].loc[loaded["boundary"]["method"].eq("DMF-Gen"), "first_failed_N"].iloc[0]),
                250000,
            )


if __name__ == "__main__":
    unittest.main()
