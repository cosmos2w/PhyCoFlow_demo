from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from Dis_SI_Process.utils.figure5_v5_data import derive_lifecycle_v5, error_capture_curve


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO_ROOT / "Dis_SI_Process" / "configs" / "figure5_v5.yaml").read_text())
class TestFigure5V5(unittest.TestCase):
    def test_error_capture_known_ranking(self) -> None:
        uncertainty = np.asarray([4.0, 3.0, 2.0, 1.0])
        error = np.asarray([4.0, 3.0, 2.0, 1.0])
        observed = error_capture_curve(uncertainty, error, (0.25, 0.50, 1.0))
        np.testing.assert_allclose(observed, [0.4, 0.7, 1.0])

    def test_error_capture_is_monotone_and_ends_at_one(self) -> None:
        observed = error_capture_curve(
            np.asarray([0.2, 0.8, 0.1, 0.4]),
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            (0.25, 0.50, 0.75, 1.0),
        )
        self.assertTrue(np.all(np.diff(observed) >= 0))
        self.assertEqual(observed[-1], 1.0)

    def test_lifecycle_uses_all_stages_and_two_gpu_geofno(self) -> None:
        lifecycle, stages, manifest, qa = derive_lifecycle_v5(CONFIG, REPO_ROOT)
        self.assertEqual(qa["status"], "pass")
        self.assertEqual(manifest["metric_label"], "Replay-equivalent model-core training GPU-hours")
        self.assertEqual(len(lifecycle), 8)
        self.assertEqual(len(stages), 9)
        self.assertEqual(int(lifecycle.set_index("method").loc["Latent FM", "stage_count"]), 2)
        self.assertEqual(int(stages.set_index("method").loc["Geo-FNO", "gpu_count"]), 2)
        latent_stages = stages.loc[stages["method"].eq("Latent FM"), "replay_equivalent_gpu_hours"]
        self.assertAlmostEqual(
            float(lifecycle.set_index("method").loc["Latent FM", "replay_equivalent_gpu_hours"]),
            float(latent_stages.sum()),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
