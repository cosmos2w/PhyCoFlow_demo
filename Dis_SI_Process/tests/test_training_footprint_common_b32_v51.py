from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

import benchmark_training_footprint_common_b32_v51 as benchmark  # noqa: E402


CONFIG_PATH = PACKAGE_ROOT / "configs" / "training_footprint_common_b32_v51.yaml"


class CommonBatchContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_contract_is_common_b32_fixed_m256_and_exact_protocol(self) -> None:
        self.assertEqual(self.config["schema_version"], "figure5-training-footprint-common-b32-v51")
        self.assertEqual(self.config["split"], "train")
        self.assertEqual(self.config["batch_size"], 32)
        self.assertEqual(self.config["sensor_count"], 256)
        self.assertEqual(self.config["condition"], "Cond_T")
        self.assertEqual(self.config["dtype"], "float32")
        protocol = self.config["protocol"]
        self.assertEqual(protocol["warmup_updates"], 20)
        self.assertEqual(protocol["measured_blocks"], 10)
        self.assertEqual(protocol["updates_per_block"], 10)
        self.assertEqual(protocol["measured_updates"], 100)
        self.assertIsNone(self.config["sensor_plan"]["path"])
        self.assertIn("deterministic", self.config["sensor_plan"]["rule"])

    def test_plan_contains_nine_attempts_with_explicit_query_and_native_modes(self) -> None:
        benchmark._validate_contract(self.config)
        self.assertEqual(len(self.config["stages"]), 9)
        self.assertEqual(
            [row["method"] for row in self.config["stages"]],
            list(benchmark.METHOD_ORDER[:3]) + ["Latent FM", "Latent FM", "SiT", "MLP-RBF", "Geo-FNO", "Senseiver"],
        )
        self.assertTrue(
            all(
                row["n_training_targets"] == 4096
                for row in self.config["stages"]
                if row["training_target_mode"] == "query_4096"
            )
        )
        self.assertTrue(
            all(
                row["n_training_targets"] == 40300
                for row in self.config["stages"]
                if row["training_target_mode"] == "native_full_grid"
            )
        )

    def test_execute_requires_explicit_clean_gpu_confirmation(self) -> None:
        self.assertEqual(benchmark.main(["--execute"]), 2)


class CommonBatchMeasurementBoundaryTest(unittest.TestCase):
    def test_fixed_sparse_is_exactly_m256_and_uses_temperature_field(self) -> None:
        batch = {
            "coords": torch.arange(32 * 64 * 3, dtype=torch.float32).reshape(32, 64, 3),
            "fields": torch.arange(32 * 64 * 5, dtype=torch.float32).reshape(32, 64, 5),
            "sensor_indices": torch.tensor([list(range(16))] * 32, dtype=torch.long),
            "sensor_field_ids": torch.full((32, 16), 2, dtype=torch.long),
        }
        sparse = benchmark._fixed_sparse(batch)
        self.assertEqual(tuple(sparse["obs_coords"].shape), (32, 16, 3))
        self.assertEqual(tuple(sparse["obs_values"].shape), (32, 16, 1))
        self.assertTrue(torch.equal(sparse["obs_field_ids"], torch.full((32, 16), 2, dtype=torch.long)))
        self.assertTrue(torch.all(sparse["obs_mask"] == 1))

    def test_timing_quantiles_and_drift_gate_are_explicit(self) -> None:
        values = [10.0 + 0.01 * (index % 10) for index in range(100)]
        blocks = [values[offset : offset + 10] for offset in range(0, 100, 10)]
        summary = benchmark._timing_summary(values, blocks, tolerance=0.25)
        self.assertLess(summary["update_time_p10_ms"], summary["update_time_q25_ms"])
        self.assertLess(summary["update_time_q25_ms"], summary["update_time_median_ms"])
        self.assertLess(summary["update_time_median_ms"], summary["update_time_q75_ms"])
        self.assertLess(summary["update_time_q75_ms"], summary["update_time_p90_ms"])
        self.assertEqual(summary["stability_status"], "pass")

    def test_optimizer_and_ema_byte_boundaries_are_tensor_only(self) -> None:
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters())
        loss = model(torch.ones(2, 4)).sum()
        loss.backward()
        optimizer.step()
        self.assertGreater(benchmark._module_bytes(model)[0], 0)
        self.assertGreater(benchmark._gradient_bytes(model), 0)
        self.assertGreater(benchmark._optimizer_bytes(optimizer), 0)
        self.assertEqual(benchmark._ema_bytes(None), 0)


class CommonBatchOutcomeGateTest(unittest.TestCase):
    def test_geo_fno_success_at_common_batch_is_valid(self) -> None:
        self.assertEqual(
            benchmark._geo_fno_outcome(
                {"status": "ok", "batch_size": 32, "measured_updates": 100}
            ),
            "success_at_common_batch",
        )

    def test_geo_fno_oom_at_common_batch_is_also_valid(self) -> None:
        self.assertEqual(
            benchmark._geo_fno_outcome(
                {"status": benchmark.OOM_STATUS, "batch_size": 32, "measured_updates": 0}
            ),
            "oom_at_common_batch",
        )

    def test_after_gate_filters_current_pid_but_keeps_foreign_rows(self) -> None:
        rows = [
            {"pid": 1234, "used_memory": "120 MiB"},
            {"pid": 9876, "used_memory": "240 MiB"},
        ]
        filtered = benchmark._foreign_process_rows(rows, current_pid=1234)
        self.assertEqual(filtered, [rows[1]])


class CommonBatchTrainingStepInstrumentationTest(unittest.TestCase):
    def test_sit_spike_state_reset_changes_only_transient_runtime_dict(self) -> None:
        state = {"ema_loss": 0.0058015, "ema_grad": 0.12, "skipped": 37, "other": "preserved"}
        runtime = SimpleNamespace(method="SiT", bundle=SimpleNamespace(components={"spike_state": state}))
        info = benchmark._reset_sit_spike_state(runtime)
        self.assertIsNotNone(info)
        self.assertEqual(state["ema_loss"], None)
        self.assertEqual(state["ema_grad"], None)
        self.assertEqual(state["skipped"], 0)
        self.assertEqual(state["other"], "preserved")
        self.assertFalse(info["checkpoint_mutation"])

    def test_counter_delta_and_success_contract_require_all_steps_and_native_ema(self) -> None:
        start = benchmark.UpdateCounters().snapshot()
        end = benchmark.UpdateCounters(
            optimizer_step_attempts=20 + 100,
            optimizer_step_successes=20 + 100,
            ema_update_attempts=20 + 100,
            ema_update_successes=20 + 100,
        ).snapshot()
        row = benchmark._counter_delta(start, end)
        self.assertEqual(row["optimizer_step_attempts"], 120)
        stage = {
            "optimizer_step_attempts_warmup": 20,
            "optimizer_step_successes_warmup": 20,
            "optimizer_step_skips_warmup": row["optimizer_step_skips"],
            "optimizer_step_attempts_measured": 100,
            "optimizer_step_successes_measured": 100,
            "optimizer_step_skips_measured": row["optimizer_step_skips"],
            "ema_update_attempts_warmup": 20,
            "ema_update_successes_warmup": 20,
            "ema_update_skips_warmup": 0,
            "ema_update_attempts_measured": 100,
            "ema_update_successes_measured": 100,
            "ema_update_skips_measured": 0,
            "ema_expected": True,
        }
        self.assertTrue(benchmark._stage_counter_contract_pass(stage))
        stage["ema_update_successes_measured"] = 99
        self.assertFalse(benchmark._stage_counter_contract_pass(stage))


if __name__ == "__main__":
    unittest.main()
