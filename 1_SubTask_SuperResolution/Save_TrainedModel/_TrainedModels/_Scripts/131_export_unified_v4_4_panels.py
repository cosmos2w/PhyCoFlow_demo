#!/usr/bin/env python
"""Export standalone V4_4 panels and preserve the validated V4_3 SI bundle."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from common.config import FIGURES_DIR, add_common_args, ensure_output_dirs, load_config
from common.io_utils import write_json
from common.publication_panels_unified_v4_4 import (
    PANEL_OUTPUT_NAMES,
    center_panel_e_tick_labels,
    draw_panel,
    panel_label,
)

HERE = Path(__file__).resolve().parent
V4_3_RELEASE = (
    HERE.parents[2] / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v4_3_20260914_2330"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assembler = _load("unified_v4_4_assembler_for_export", HERE / "130_assemble_mixed_resolution_unified_v4_4.py")
exporter37 = _load("unified_v3_7_exporter_for_v4_4", HERE / "116_export_unified_v3_7_panels.py")
base = exporter37.base


def export_standalone_panels(ctx, cfg, layout, root: Path, rid: str):
    outputs, metadata = [], {}
    panel_root = root / "panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    for label in "abcde":
        fig, ax, _container = assembler._create_standalone_canvas_v4_4(layout, label)
        panel_label(ax, label)
        metadata[label] = draw_panel(label, ax, ctx)
        base.finish_figure(fig)
        if label == "e":
            metadata[label].update(center_panel_e_tick_labels(ax, strict=True))
        panel_base = panel_root / f"{PANEL_OUTPUT_NAMES[label]}_{rid}"
        outputs.extend(base.save_triplet(fig, panel_base, cfg))
    return outputs, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v4_4.yaml")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = args.run_id
    if not rid:
        raise ValueError("--run-id is required for additive V4_4 exports")
    cfg = load_config(args.config)
    assembler.v4.apply_v4_style_contract(cfg)
    ensure_output_dirs()
    layout = base.load_layout(Path(args.layout).resolve())
    ctx = base.make_context(args, cfg, layout, rid)
    root = FIGURES_DIR / "UnifiedV4_4" / rid
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4_4 export directory: {root}")
    root.mkdir(parents=True)
    panel_outputs, panel_meta = export_standalone_panels(ctx, cfg, layout, root, rid)

    for name in ("si", "tables"):
        source = V4_3_RELEASE / name
        if not source.is_dir():
            raise FileNotFoundError(f"Validated V4_3 {name} directory is missing: {source}")
        shutil.copytree(source, root / name)
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v4_4_{rid}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["standalone_outputs"] = [base.record(path) for path in panel_outputs]
    manifest["standalone_panel_metadata"] = panel_meta
    manifest["si_outputs"] = {
        "preserved_v4_3": [base.record(path) for path in sorted((root / "si").rglob("*")) if path.is_file()]
    }
    manifest["si_metadata"] = {
        "source_revision": "V4_3", "restyled": False,
        "reason": "SI is outside the active V4_4 main-figure scope",
        "source_directory": str((V4_3_RELEASE / "si").resolve()),
    }
    manifest["table_outputs"] = {
        "preserved_v4_3": [base.record(path) for path in sorted((root / "tables").rglob("*")) if path.is_file()]
    }
    v4_manifest = json.loads((V4_3_RELEASE / "source_manifest_v4_3.json").read_text(encoding="utf-8"))
    for key in (
        "all_render_cache_sources", "all_render_source_records", "referenced_shared_sources",
        "table_sources", "extended_multiscale_distribution_source",
    ):
        if key in v4_manifest:
            manifest[key] = v4_manifest[key]
    manifest["si_root"] = str(root.resolve())
    manifest["release_status"] = "art reviewed, scientific release pending"
    write_json(manifest_path, manifest)
    print(f"[OK] {root}")


if __name__ == "__main__":
    main()
