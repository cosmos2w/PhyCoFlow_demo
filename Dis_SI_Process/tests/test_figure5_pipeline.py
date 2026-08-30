from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class Figure5PipelineTest(unittest.TestCase):
    def test_draft_build_emits_svg_only_with_editable_text(self) -> None:
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
            self.assertEqual(len(svgs), 9)
            self.assertFalse(list(bundle.glob("*.pdf")))
            for path in svgs:
                root = ET.parse(path).getroot()
                self.assertTrue(any(node.tag.endswith("text") for node in root.iter()))

    def test_strict_formal_rejects_current_missing_validation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE_ROOT / "scripts" / "build_figure5_draft.py"),
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
