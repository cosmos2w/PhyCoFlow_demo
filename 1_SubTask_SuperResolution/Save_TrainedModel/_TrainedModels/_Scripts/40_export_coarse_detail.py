#!/usr/bin/env python
"""Export scale-separated fidelity metrics from audited caches only."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.io_utils import read_csv
from common.panels_de_data import export_coarse_detail_fidelity
from common.workflow import cache_manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--canonical-run-id", default="formal_20260712")
    parser.add_argument("--sensor-count", type=int)
    parser.add_argument("--cutoff", choices=["L", "M"])
    parser.add_argument(
        "--layout", default=str(Path(__file__).with_name("publication_layout_unified_v2.yaml")),
    )
    args = parser.parse_args()
    cfg = load_config(args.config); ensure_output_dirs(); rid = run_id(args.run_id)
    with Path(args.layout).open("r", encoding="utf-8") as handle:
        panel_cfg = (yaml.safe_load(handle) or {}).get("panel_d", {})
    manifest_path = cache_manifest_path(args.cache_manifest)
    canonical_path = RESULTS_DIR / "CanonicalTestIndex" / f"CanonicalTestIndex_{args.canonical_run_id}.csv"
    output_dir = RESULTS_DIR / "UnifiedPublicationV2"
    outputs = export_coarse_detail_fidelity(
        cfg, read_csv(manifest_path), canonical_path, output_dir, rid,
        projector_resolution=args.cutoff or panel_cfg.get("projector_resolution", "M"),
        sensor_count=args.sensor_count or int(cfg["sensor_plan"]["default_count"]),
        bootstrap_samples=int(cfg["coarse_detail"].get("bootstrap_samples", 2000)),
    )[:3]
    for path in outputs:
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
