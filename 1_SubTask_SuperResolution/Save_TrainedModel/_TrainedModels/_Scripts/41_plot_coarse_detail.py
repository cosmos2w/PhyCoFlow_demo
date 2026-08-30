#!/usr/bin/env python
"""Export publication panel d and its relative-detail-L2 diagnostic."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.config import add_common_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", default=str(Path(__file__).with_name("publication_layout_unified_v2.yaml")))
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id")
    parser.add_argument("--base-data-run-id", default="2026-07-15_11-32")
    args = parser.parse_args()
    cmd = [
        sys.executable, str(Path(__file__).with_name("96_export_unified_v2_panels.py")),
        "--config", str(args.config), "--layout", str(args.layout),
        "--panels", "d", "--base-data-run-id", str(args.base_data_run_id),
    ]
    if args.run_id: cmd.extend(["--run-id", args.run_id])
    if args.data_run_id: cmd.extend(["--data-run-id", args.data_run_id])
    if args.cache_manifest: cmd.extend(["--cache-manifest", str(args.cache_manifest)])
    if args.representatives: cmd.extend(["--representatives", str(args.representatives)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
