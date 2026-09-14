#!/usr/bin/env python
"""Export V3-7 standalone panels, validated SI figures, and LaTeX tables."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import global_style as manuscript
import numpy as np
from common.config import FIGURES_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import apply_style
from common.publication_panels_unified_v3_7 import (
    PANEL_OUTPUT_NAMES,
    draw_panel,
    panel_label,
)

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


assembler = _load("unified_v3_7_assembler", HERE / "115_assemble_mixed_resolution_unified_v3_7.py")
exporter36 = _load("unified_v3_6_exporter", HERE / "113_export_unified_v3_6_panels.py")
base = exporter36.base


def _collect_cache_paths(payload):
    return exporter36._collect_cache_paths(payload)


def _collect_source_paths(payload):
    found = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "sources" and isinstance(value, list):
                found.update(Path(path) for path in value)
            else:
                found.update(_collect_source_paths(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_collect_source_paths(value))
    return found


def export_standalone_panels(ctx, cfg, layout, root, rid):
    outputs, metadata = [], {}; panel_root = root / "panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    for label in "abcde":
        fig, ax, _container = assembler.create_standalone_canvas(layout, label)
        panel_label(ax, label); metadata[label] = draw_panel(label, ax, ctx)
        base.finish_figure(fig)
        panel_base = panel_root / f"{PANEL_OUTPUT_NAMES[label]}_{rid}"
        outputs.extend(base.save_triplet(fig, panel_base, cfg))
    return outputs, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v3_7.yaml")
    parser.add_argument("--cache-manifest", type=Path); parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args(); rid = args.run_id
    if not rid:
        raise ValueError("--run-id is required for additive V3-7 exports")
    cfg = load_config(args.config); manuscript.register_local_arial(); apply_style(cfg); ensure_output_dirs()
    layout = base.load_layout(args.layout); ctx = base.make_context(args, cfg, layout, rid)
    root = FIGURES_DIR / "UnifiedV3_7" / rid
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing V3-7 export directory: {root}")
    root.mkdir(parents=True)
    panel_outputs, panel_meta = export_standalone_panels(ctx, cfg, layout, root, rid)
    si_layout = dict(layout); si_layout["si_hybrid"] = dict(layout["si_v3_7"])
    # V3-6 delegates the validated SI export implementation to its earlier
    # exporter chain; preserve that exact source path for V3-7 as well.
    si_outputs, si_meta = exporter36.exporter35.exporter34.exporter33.exporter32.export_si(
        ctx, cfg, si_layout, root, rid,
    )
    table_outputs, table_sources = base.export_tables(layout, root, rid)
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v3_7_{rid}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["standalone_outputs"] = [base.record(path) for path in panel_outputs]
    manifest["standalone_panel_metadata"] = panel_meta
    manifest["si_outputs"] = {key: [base.record(path) for path in paths] for key, paths in si_outputs.items()}
    manifest["si_metadata"] = si_meta
    manifest["table_outputs"] = {
        key: {**{k: v for k, v in item.items() if k != "path"}, **base.record(item["path"])}
        for key, item in table_outputs.items()
    }
    manifest["table_sources"] = [base.record(path) for path in table_sources]
    all_cache_paths = sorted(_collect_cache_paths({"main": manifest.get("panels", {}),
                                                   "standalone": panel_meta, "si": si_meta}))
    manifest["all_render_cache_sources"] = [base.record(path) for path in all_cache_paths]
    all_source_paths = sorted(_collect_source_paths({"main": manifest.get("panels", {}),
                                                     "standalone": panel_meta, "si": si_meta}))
    manifest["all_render_source_records"] = [base.record(path) for path in all_source_paths]
    shared_paths = set()
    for cache_path in all_cache_paths:
        with np.load(cache_path, allow_pickle=False) as cache:
            if "metadata_json" not in cache:
                continue
            metadata = json.loads(str(cache["metadata_json"]))
        for key in ("truth_ref", "grid_ref", "sensor_plan_path"):
            referenced = metadata.get(key)
            if referenced:
                shared_paths.add(Path(referenced))
    manifest["referenced_shared_sources"] = [base.record(path) for path in sorted(shared_paths)]
    manifest["si_root"] = str(root.resolve())
    base.write_json(manifest_path, manifest)
    print(f"[OK] {root}")


if __name__ == "__main__":
    main()
