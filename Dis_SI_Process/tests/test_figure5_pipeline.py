from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class Figure5PipelineTest(unittest.TestCase):
    def test_v3_contract_has_five_panels_and_no_main_nfe_or_v2_fallback(self) -> None:
        config = yaml.safe_load((PACKAGE_ROOT / "configs" / "figure5_draft.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], "figure5-validation-v3")
        self.assertEqual(list(config["figure"]["panel_map"]), list("abcde"))
        self.assertNotIn("nfe", " ".join(config["figure"]["panel_map"].values()).lower())
        self.assertEqual(config["formal_protocol"]["uq"]["field_weights"], [0.25, 0.25, 0.25, 0.25])
        self.assertTrue(config["build_policy"]["strict_formal"]["reject_validation_v2_cost"])

    def test_v3_build_emits_six_named_svg_only_outputs_with_editable_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            timestamp = "20990101_0000"
            subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "build_figure5_draft.py"),
                    "--timestamp",
                    timestamp,
                    "--output-root",
                    str(output_root),
                ],
                check=True,
                cwd=PACKAGE_ROOT.parent,
            )
            bundle = output_root / "figures" / "generated" / timestamp
            svgs = sorted(bundle.glob("*.svg"))
            self.assertEqual(len(svgs), 6)
            self.assertFalse(list(bundle.glob("*.pdf")))
            expected = {
                f"fig5a_normalized_crps_{timestamp}.svg",
                f"fig5b_spread_error_methods_{timestamp}.svg",
                f"fig5c_accuracy_latency_clean_{timestamp}.svg",
                f"fig5d_query_latency_{timestamp}.svg",
                f"fig5e_query_memory_{timestamp}.svg",
                f"fig5_composed_v3_{timestamp}.svg",
            }
            self.assertEqual({path.name for path in svgs}, expected)
            for path in svgs:
                root = ET.parse(path).getroot()
                self.assertTrue(any(node.tag.endswith("text") for node in root.iter()))

    def test_strict_formal_rejects_incomplete_validation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            config_path = output_root / "missing_formal_sources.yaml"
            config_text = (PACKAGE_ROOT / "configs" / "figure5_draft.yaml").read_text(encoding="utf-8")
            config_text = config_text.replace(
                "uq_root: Dis_SI_Process/results/ValidationV3/UQCompare",
                f"uq_root: {output_root / 'missing_uncertainty'}",
            ).replace(
                "cost_root: Dis_SI_Process/results/ValidationV3/CostClean",
                f"cost_root: {output_root / 'missing_cost'}",
            )
            config_path.write_text(config_text, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "build_figure5_draft.py"),
                    "--config",
                    str(config_path),
                    "--timestamp",
                    "20990101_0001",
                    "--output-root",
                    temporary,
                    "--strict-formal",
                ],
                cwd=PACKAGE_ROOT.parent,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-formal panels", result.stderr)


if __name__ == "__main__":
    unittest.main()
