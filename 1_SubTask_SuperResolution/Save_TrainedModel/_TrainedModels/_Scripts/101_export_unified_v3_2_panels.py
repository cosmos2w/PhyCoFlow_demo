#!/usr/bin/env python
"""Export V3-2 standalone panels, four SI figures, and LaTeX tables."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

import global_style as manuscript
import common.panel_c_tuning as panel_c_tuning
from common.config import FIGURES_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import apply_style
from common.physical_figure_layout import create_standalone_canvas, resolve_physical_layout
from common.publication_panels_unified_v3_2 import PANEL_OUTPUT_NAMES, draw_panel, panel_label


HERE = Path(__file__).resolve().parent
BASE_EXPORTER = HERE / "96_export_unified_v3_panels.py"
_spec = importlib.util.spec_from_file_location("unified_v3_base_exporter", BASE_EXPORTER)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load V3 base exporter: {BASE_EXPORTER}")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def _collect_cache_paths(payload):
    found = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "cache_sources" and isinstance(value, list):
                found.update(Path(path) for path in value)
            else:
                found.update(_collect_cache_paths(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_collect_cache_paths(value))
    return found


def _fix_expanded_gallery_colorbars(fig, metadata):
    """Give the legacy expanded gallery finite, explicit publication ticks."""
    targets = {
        str(metadata["field"]): ("Field value", metadata["field_limits"]),
        "Absolute error": ("Absolute error", metadata["error_limits"]),
    }
    axes, seen = [], set()
    pending = list(fig.axes)
    while pending:
        axis = pending.pop()
        if id(axis) in seen:
            continue
        seen.add(id(axis))
        axes.append(axis)
        pending.extend(getattr(axis, "child_axes", []))
    for axis in axes:
        original_title = axis.get_title()
        if original_title not in targets:
            continue
        title, limits = targets[original_title]
        low, high = map(float, limits)
        ticks = np.linspace(low, high, 3)
        labels = [f"{value:.3f}" if abs(value) >= 1e-3 else f"{value:.1e}" for value in ticks]
        axis.set_xlim(low, high)
        axis.set_xticks(ticks, labels)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.offsetText.set_visible(False)
        axis.set_title(title, pad=1.5)
        manuscript.tag_font_role(axis.title, "axis_label")
        for label in axis.get_xticklabels():
            manuscript.tag_font_role(label, "tick_label")


def export_standalone_panels(ctx, cfg, physical, root, rid):
    outputs, metadata = [], {}
    panel_root = root / "panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    for label in "abcd":
        fig, ax, _container = create_standalone_canvas(physical, label)
        panel_label(ax, label)
        metadata[label] = draw_panel(label, ax, ctx)
        base.finish_figure(fig)
        panel_base = panel_root / f"{PANEL_OUTPUT_NAMES[label]}_{rid}"
        outputs.extend(base.save_triplet(fig, panel_base, cfg))
    return outputs, metadata


def export_si(ctx, cfg, layout, root, rid):
    si_root = root / "si"
    si_root.mkdir(parents=True, exist_ok=True)
    records, metadata = {}, {}
    sizes = layout["si_hybrid"]

    width, height = map(float, sizes["sensor_sweep_size_mm"])
    fig, parent = base._blank_parent(width, height)
    metadata["Sx1"] = base.draw_sensor_efficiency(parent, ctx, standalone=True, show_legend=True)
    base.finish_figure(fig)
    records["Sx1_complete_sensor_sweeps"] = base.save_triplet(
        fig, si_root / f"SI_Sx1_complete_sensor_sweeps_{rid}", cfg,
    )

    width, height = map(float, sizes["qualitative_gallery_size_mm"])
    fig, parent = base._blank_parent(width, height)
    previous_sensor_count = panel_c_tuning.SENSOR_COUNT
    panel_c_tuning.SENSOR_COUNT = 256
    try:
        metadata["Sx2"] = base.draw_v2_panel_c(
            parent, ctx, standalone=False, show_legend=False, version=1,
        )
    finally:
        panel_c_tuning.SENSOR_COUNT = previous_sensor_count
    metadata["Sx2"]["expansion_contract"] = (
        "all four validated models across Mixed-HML, Zero-H-balanced, and Zero-H-M-rich at 256 sensors"
    )
    sx2_models = ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"]
    sx2_recipes = ["3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"]
    sx2_cache_paths = []
    for model in sx2_models:
        for recipe in sx2_recipes:
            payload = ctx.cache(model, recipe, int(metadata["Sx2"]["snapshot"]), 256)
            if payload is None:
                raise RuntimeError(f"Missing expanded SI cache for {model}/{recipe}/n=256")
            sx2_cache_paths.append(payload[2]["cache_path"])
    metadata["Sx2"]["cache_sources"] = sorted(sx2_cache_paths)
    metadata["Sx2"]["cache_cell_count"] = len(sx2_cache_paths)
    metadata["Sx2"]["display_model_order"] = sx2_models
    metadata["Sx2"]["display_recipe_order"] = sx2_recipes
    _fix_expanded_gallery_colorbars(fig, metadata["Sx2"])
    base.finish_figure(fig)
    records["Sx2_expanded_recipe_gallery"] = base.save_triplet(
        fig, si_root / f"SI_Sx2_expanded_recipe_gallery_{rid}", cfg,
    )

    width, height = map(float, sizes["multiscale_qualitative_size_mm"])
    fig, parent = base._blank_parent(width, height)
    metadata["Sx3"] = base.draw_multiscale_components(parent, ctx, standalone=True, show_legend=False)
    metadata["Sx3"]["content_type"] = "multiscale_qualitative"
    base.finish_figure(fig)
    records["Sx3_complete_three_scale_qualitative"] = base.save_triplet(
        fig, si_root / f"SI_Sx3_complete_three_scale_qualitative_{rid}", cfg,
    )

    width, height = map(float, sizes["multiscale_quantitative_size_mm"])
    fig, parent = base._blank_parent(width, height)
    metadata["Sx4"] = base.draw_multiscale_fidelity(parent, ctx, standalone=True, show_legend=False)
    metadata["Sx4"]["content_type"] = "multiscale_quantitative"
    base.finish_figure(fig)
    records["Sx4_complete_three_scale_quantitative"] = base.save_triplet(
        fig, si_root / f"SI_Sx4_complete_three_scale_quantitative_{rid}", cfg,
    )
    return records, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v3_2.yaml")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = args.run_id
    if not rid:
        raise ValueError("--run-id is required for additive V3-2 exports")

    cfg = load_config(args.config)
    manuscript.register_local_arial()
    apply_style(cfg)
    ensure_output_dirs()
    layout = base.load_layout(args.layout)
    physical = resolve_physical_layout(layout)
    ctx = base.make_context(args, cfg, layout, rid)
    root = FIGURES_DIR / "UnifiedV3_2" / rid
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing V3-2 export directory: {root}")
    root.mkdir(parents=True)

    panel_outputs, panel_meta = export_standalone_panels(ctx, cfg, physical, root, rid)
    si_outputs, si_meta = export_si(ctx, cfg, layout, root, rid)
    table_outputs, table_sources = base.export_tables(layout, root, rid)
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v3_2_{rid}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["standalone_outputs"] = [base.record(path) for path in panel_outputs]
    manifest["standalone_panel_metadata"] = panel_meta
    manifest["si_outputs"] = {
        key: [base.record(path) for path in paths] for key, paths in si_outputs.items()
    }
    manifest["si_metadata"] = si_meta
    manifest["table_outputs"] = {
        key: {**{k: v for k, v in item.items() if k != "path"}, **base.record(item["path"])}
        for key, item in table_outputs.items()
    }
    manifest["table_sources"] = [base.record(path) for path in table_sources]
    all_cache_paths = sorted(_collect_cache_paths({
        "main": manifest.get("panels", {}),
        "standalone": panel_meta,
        "si": si_meta,
    }))
    manifest["all_render_cache_sources"] = [base.record(path) for path in all_cache_paths]
    shared_paths = set()
    for cache_path in all_cache_paths:
        with np.load(cache_path, allow_pickle=False) as cache:
            if "metadata_json" not in cache:
                continue
            cache_metadata = json.loads(str(cache["metadata_json"]))
        for key in ("truth_ref", "grid_ref", "sensor_plan_path"):
            referenced = cache_metadata.get(key)
            if referenced:
                shared_paths.add(Path(referenced))
    manifest["referenced_shared_sources"] = [base.record(path) for path in sorted(shared_paths)]
    manifest["si_root"] = str(root.resolve())
    base.write_json(manifest_path, manifest)
    print(f"[OK] {root}")


if __name__ == "__main__":
    main()
