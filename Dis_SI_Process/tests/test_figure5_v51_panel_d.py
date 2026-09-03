from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "explore_figure5_v51_panel_d.py"
SPEC = importlib.util.spec_from_file_location("figure5_v51_panel_d", SCRIPT_PATH)
assert SPEC and SPEC.loader
PANEL_D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PANEL_D)


class Figure5V51PanelDTest(unittest.TestCase):
    def test_existing_formal_source_is_complete_and_stage_aware(self) -> None:
        config = PANEL_D.load_config(PACKAGE_ROOT / "configs" / "figure5_v5.yaml")
        data = PANEL_D.load_existing_formal(config, REPO_ROOT)
        self.assertEqual(data["mode"], "existing_formal")
        self.assertEqual(len(data["summary"]), 8)
        self.assertEqual(len(data["stages"]), 9)
        self.assertEqual(int(data["summary"].set_index("method").loc["Latent FM", "stage_count"]), 2)
        self.assertEqual(data["manifest"]["metric_label"], "Replay-equivalent model-core training GPU-hours")

    def test_shared_v51_contract_is_translated_without_losing_lifecycle_methods(self) -> None:
        config = PANEL_D.load_config(PACKAGE_ROOT / "configs" / "figure5_v51_exploration.yaml")
        self.assertEqual(len(config["paper_contract"]["method_order"]), 8)
        self.assertIn("Senseiver", config["style"]["method_colors"])
        self.assertIn("Geo-FNO", config["style"]["method_markers"])

    def test_common_batch_loader_strictly_waits_when_path_is_absent(self) -> None:
        config = PANEL_D.load_config(PACKAGE_ROOT / "configs" / "figure5_v5.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            result = PANEL_D.load_common_b32(config, REPO_ROOT, Path(temporary) / "not_available")
        self.assertEqual(result["status"], "strict_wait")
        self.assertIn("absent", result["reason"])
        self.assertNotIn("summary", result)

    def test_d1_accuracy_column_is_at_least_forty_percent_of_width(self) -> None:
        config = PANEL_D.load_config(PACKAGE_ROOT / "configs" / "figure5_v5.yaml")
        data = PANEL_D.load_existing_formal(config, REPO_ROOT)
        fig = PANEL_D.make_d1(data, config, "existing_formal")
        try:
            fig.canvas.draw()
            widths = [axis.get_position().width for axis in fig.axes[:4]]
            self.assertGreater(widths[0] / sum(widths), 0.40)
            self.assertLess(widths[0] / sum(widths), 0.50)
        finally:
            plt.close(fig)

    def test_existing_plot_stage_source_excludes_memory_columns(self) -> None:
        config = PANEL_D.load_config(PACKAGE_ROOT / "configs" / "figure5_v5.yaml")
        data = PANEL_D.load_existing_formal(config, REPO_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            paths = PANEL_D._write_source_tables(data, Path(temporary), "existing_formal")
            header = Path(paths["stage_source"]).read_text(encoding="utf-8").splitlines()[0]
        self.assertNotIn("peak_allocated", header)
        self.assertNotIn("peak_reserved", header)


if __name__ == "__main__":
    unittest.main()
