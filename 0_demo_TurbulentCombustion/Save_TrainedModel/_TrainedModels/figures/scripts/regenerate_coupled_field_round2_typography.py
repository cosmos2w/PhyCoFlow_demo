#!/usr/bin/env python
"""Regenerate the round-2 coupled-field typography revision as PDF only."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-id",
        default=f"round2_typography_{datetime.now():%Y%m%d_%H%M}",
        help="Unique output suffix; an existing PDF is never overwritten.",
    )
    args = parser.parse_args()
    output = (
        ROOT / "_Process_Figures/Assembled/Composite"
        / f"CoupledFieldReconstruction_{args.output_id}.pdf"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {output}")
    command = [
        sys.executable,
        str(ROOT / "_Scripts/91_assemble_coupled_field_publication.py"),
        "--run-id", "paper_full_20260711",
        "--layout", str(ROOT / "_Scripts/publication_layout_coupled_field_round2_typography.yaml"),
        "--output-id", args.output_id,
        "--formats", "pdf",
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
