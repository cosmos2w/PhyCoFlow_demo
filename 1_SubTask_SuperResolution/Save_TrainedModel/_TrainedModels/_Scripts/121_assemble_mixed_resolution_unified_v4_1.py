#!/usr/bin/env python
"""Assemble the author-authorized mixed-resolution Figure V4_1 revision."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import global_style as manuscript
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import save_figure
from common.io_utils import write_json
from common.physical_figure_layout import validate_panel_text_boundaries
from common.publication_panels_unified_v4_1 import draw_panel, panel_label


HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4 = _load("mixed_resolution_v4_base_for_v4_1", HERE / "118_assemble_mixed_resolution_unified_v4.py")
base = v4.base
V4_BASELINE = (
    HERE.parents[2] / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v4_20260914_1620"
)


def _v4_baseline_anchor() -> dict:
    paths = {
        "release_pdf": V4_BASELINE / "MixedResolution_unified_v4_20260914_1620.pdf",
        "release_svg": V4_BASELINE / "MixedResolution_unified_v4_20260914_1620.svg",
        "release_png": V4_BASELINE / "MixedResolution_unified_v4_20260914_1620.png",
        "release_manifest": V4_BASELINE / "source_manifest_v4.json",
        "release_qa": V4_BASELINE / "qa_v4.json",
        "renderer": HERE / "118_assemble_mixed_resolution_unified_v4.py",
        "exporter": HERE / "119_export_unified_v4_panels.py",
        "audit": HERE / "120_audit_mixed_resolution_v4.py",
        "panel_renderer": HERE / "common" / "publication_panels_unified_v4.py",
        "layout": HERE / "publication_layout_unified_v4.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V4 baseline anchor is incomplete: {missing}")
    return {key: v4._record(path) for key, path in paths.items()}


def _author_delta_contract(panel_meta: dict) -> dict:
    return {
        "authorization": "explicit user instruction in V4_1 revision task",
        "underlying_source_arrays_changed": False,
        "panel_a": {
            "change": "same-size shared ROI translated right; boxes, connectors, and insets remain consistent",
            "before_shift_parent_fraction": [-0.150, 0.180],
            "after_shift_parent_fraction": [-0.050, 0.180],
            "same_native_field": True,
        },
        "panel_b": {
            "change": "five redundant DMF-Gen labels hidden and saved Mixed-HML sweep restored",
            "before_sweep_recipes": ["4_ZeroH_Balanced", "5_ZeroH_MRich"],
            "after_sweep_recipes": ["3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"],
            "saved_rows_only": True,
            "bar_values_and_confidence_intervals_unchanged": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v4_1.yaml")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = base.publication_timestamp(args.run_id)
    layout_path = Path(args.layout).resolve()
    cfg = load_config(args.config)
    style_record = v4.apply_v4_style_contract(cfg)
    style_record["scope"] = "V4_1 process-local override; V4 and V3-7 sources untouched"
    ensure_output_dirs()
    layout = base.load_layout(layout_path)
    ctx = base.make_context(args, cfg, layout, rid)
    v3_anchor = v4._v3_7_baseline_anchor()
    v4_anchor = _v4_baseline_anchor()
    results_before = v4._tree_state(RESULTS_DIR)
    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v4_1_{rid}"
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v4_1_{rid}.json"
    targets = [out.with_suffix(ext) for ext in (".svg", ".pdf", ".png")] + [manifest_path]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to overwrite existing V4_1 artifacts for {rid}")

    fig, axes, containers, rects, width, height, shared_axes, cbar_parent = v4.create_canvas(layout)
    panel_meta = {}
    for label in ("a", "b"):
        panel_label(axes[label], label)
        panel_meta[label] = draw_panel(label, axes[label], ctx)
    panel_label(axes["c"], "c")
    panel_meta["c"] = draw_panel(
        "c", axes["c"], ctx, shared_axes=shared_axes["c"], colorbar_parent=cbar_parent,
    )
    panel_label(axes["d"], "d")
    panel_meta["d"] = draw_panel("d", axes["d"], ctx, shared_axes=shared_axes["d"])
    panel_label(axes["e"], "e")
    panel_meta["e"] = draw_panel("e", axes["e"], ctx)

    for axis in fig.axes:
        for item in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            if item.get_visible() and item.get_text():
                item.set_gid("font-role:tick_label")
    geometric_axis_count = base.enforce_geometric_aspects(fig)
    typography_qa = manuscript.enforce_figure_typography(fig, font_family=manuscript.FONT_FAMILY)
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_artist_qa = v4._model_artist_qa(fig, cfg)
    geometry_qa = v4._geometry_qa(fig, containers, rects)
    alignment_qa = v4._panel_cd_alignment_qa(fig, shared_axes, cfg["figure_style"]["paper_dpi"])
    if not model_artist_qa["passed"]:
        raise ValueError(f"V4_1 model artist contract failed: {model_artist_qa}")
    if not geometry_qa["passed"]:
        raise ValueError(f"V4_1 geometry failed: {geometry_qa}")
    if not alignment_qa["passed"]:
        raise ValueError(f"V4_1 shared c+d alignment failed: {alignment_qa}")
    qa = {
        "geometric_axis_count": geometric_axis_count,
        "typography_qa": typography_qa,
        "frame_lineweight_qa": frame_qa,
        "model_artist_qa": model_artist_qa,
        "geometry_qa": geometry_qa,
        "text_overflow_in": dict(manuscript.validate_text_within_canvas(fig)),
        "panel_text_clearance_qa": validate_panel_text_boundaries(fig, containers),
    }
    outputs = save_figure(
        fig, out, cfg, formats=("svg", "pdf", "png"),
        dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
    )
    plt.close(fig)
    results_after = v4._tree_state(RESULTS_DIR)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_1 rendering")

    manifest = v4.build_manifest(
        ctx, cfg, layout, layout_path, outputs, panel_meta, rects, width, height,
        qa, alignment_qa, style_record, args, rid, v3_anchor, results_before, results_after,
    )
    manifest.update({
        "workflow_label": "mixed_resolution_unified_v4_1",
        "schema_version": "4.1",
        "revision": "V4_1",
        "source_visual_revision": "V4",
        "v4_baseline_anchor": v4_anchor,
        "author_authorized_deltas": _author_delta_contract(panel_meta),
        "renderer": v4._record(Path(__file__).resolve()),
        "panel_renderer": v4._record(HERE / "common" / "publication_panels_unified_v4_1.py"),
        "exporter": v4._record(HERE / "122_export_unified_v4_1_panels.py"),
    })
    manifest["figure_contract"].update({
        "core_conclusion": (
            "The saved results continue to show resolution-transfer behavior and DMF-Gen's "
            "relative fidelity; V4_1 adds the saved Mixed-HML sweep and a richer shared viewport."
        ),
        "revision_scope": (
            "Author-authorized V4_1 presentation revision using unchanged saved fields and metrics; "
            "explicit ROI, visible-annotation, and sweep-inventory deltas are recorded."
        ),
        "underlying_source_arrays_unchanged": True,
    })
    manifest["selection_contract"]["panel_b"]["sweep_recipes"] = list(layout["panel_b_v4"]["sweep_recipes"])
    manifest["selection_contract"]["panel_b"]["sweep_count"] = 3
    manifest["selection_contract"]["panel_d"]["semantic_colormaps"] = {
        "truth_component": layout["panel_d_v4"]["truth_component_cmap"],
        "residuals": layout["panel_d_v4"]["residual_cmap"],
    }
    manifest["style_contract"]["semantic_colormaps"] = {
        "panel_c_physical_field": cfg["rendering"]["cmap"],
        "panel_c_absolute_error": cfg["rendering"]["error_cmap"],
        "panel_d_truth_component": layout["panel_d_v4"]["truth_component_cmap"],
        "panel_d_residuals": layout["panel_d_v4"]["residual_cmap"],
    }
    write_json(manifest_path, manifest)
    print(f"[OK] {out}.pdf")
    print(f"[OK] {manifest_path}")


if __name__ == "__main__":
    main()
