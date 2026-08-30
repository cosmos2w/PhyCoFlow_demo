from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_ValidationPlans/validation_v1.yaml"


class ValidationPlanTest(unittest.TestCase):
    def test_frozen_state_map_matches_dataset_split_algorithm(self) -> None:
        plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
        indices = np.arange(10000, dtype=np.int64)
        np.random.default_rng(42).shuffle(indices)
        expected = sorted(int(value) for value in indices[9000:])
        self.assertEqual(plan["test_states"]["evaluation_indices"], list(range(1000)))
        self.assertEqual(plan["test_states"]["original_hdf5_time_indices"], expected)
        self.assertEqual(len(plan["cohorts"]["calibration_200"]["evaluation_indices"]), 200)
        self.assertEqual(
            plan["cohorts"]["pilot"]["original_hdf5_time_indices"],
            [5, 840, 1764, 2634, 3525, 4462, 5542, 6528, 7342, 8155, 9170, 9977],
        )

    def test_checkpoint_and_field_contract_is_exact(self) -> None:
        plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
        self.assertEqual(plan["dataset"]["field_order"], ["Y_CH4", "Y_CO", "T", "U1", "p"])
        self.assertEqual(
            [item["method"] for item in plan["checkpoints"]],
            ["DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT", "MLP-RBF", "Geo-FNO", "Senseiver"],
        )
        self.assertTrue(all(item["path"].endswith("/Cond_T/last.pt") for item in plan["checkpoints"]))
        self.assertEqual(plan["inference"]["observation_consistency"], "default_hard")
        self.assertEqual(plan["inference"]["measured_nfe"], 2)


if __name__ == "__main__":
    unittest.main()
