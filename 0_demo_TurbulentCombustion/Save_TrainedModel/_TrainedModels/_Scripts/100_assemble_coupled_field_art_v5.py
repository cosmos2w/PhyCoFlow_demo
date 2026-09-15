#!/usr/bin/env python
"""Render Figure 4 art V5 from the validated V4 display and frozen data."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PolyCollection
import numpy as np


HERE = Path(__file__).resolve().parent
V4_PATH = HERE / "98_assemble_coupled_field_art_v4.py"
FACE_ALPHA = 0.75


def _load_v4():
    spec = importlib.util.spec_from_file_location("coupled_field_art_v4_for_v5", V4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V4 renderer: {V4_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v4 = _load_v4()
v3 = v4.v3
v2 = v4.v2
base = v4.base


def _rgba_with_alpha(colors, alpha: float):
    rgba = np.asarray(colors, dtype=float).copy()
    if rgba.size == 0:
        return rgba
    rgba = np.atleast_2d(rgba)
    rgba[:, 3] = alpha
    return rgba


def _opaque_existing_edges(colors):
    rgba = np.asarray(colors, dtype=float).copy()
    if rgba.size == 0:
        return rgba
    rgba = np.atleast_2d(rgba)
    rgba[rgba[:, 3] > 0, 3] = 1.0
    return rgba


def _apply_v5_art(fig) -> dict:
    v4_qa = v4._final_v4(fig)
    b_axes, bar_axes, spectra, _, violin_axes = v4._panel_axes(fig)

    title_before = [ax.get_title() for ax in bar_axes]
    title_after = [re.sub(r"\s*\(unobs\.\)\s*$", "", title) for title in title_before]
    if any(before == after for before, after in zip(title_before, title_after)):
        raise RuntimeError(f"Panel-c title did not contain the expected '(unobs.)' suffix: {title_before}")
    for ax, title in zip(bar_axes, title_after):
        ax.set_title(title, fontsize=base.SIZE_SUBPLOT_TITLE, pad=2)

    b_image_records = []
    for axis_index, ax in enumerate(b_axes):
        if len(ax.images) != 1:
            raise RuntimeError(f"Expected one image in Panel-b matrix axis {axis_index}; found {len(ax.images)}")
        image = ax.images[0]
        image.set_alpha(FACE_ALPHA)
        b_image_records.append({"axis_index": axis_index, "alpha": float(image.get_alpha())})

    bar_records = []
    for axis_index, ax in enumerate(bar_axes):
        for patch_index, patch in enumerate(ax.patches):
            face = np.asarray(patch.get_facecolor(), dtype=float)
            edge = np.asarray(patch.get_edgecolor(), dtype=float)
            patch.set_alpha(None)
            patch.set_facecolor((*face[:3], FACE_ALPHA))
            if edge.size >= 4 and edge[3] > 0:
                patch.set_edgecolor((*edge[:3], 1.0))
            bar_records.append({
                "axis_index": axis_index,
                "patch_index": patch_index,
                "face_alpha": float(patch.get_facecolor()[3]),
                "edge_alpha": float(patch.get_edgecolor()[3]),
            })

    violin_records = []
    for axis_index, ax in enumerate(violin_axes):
        bodies = [collection for collection in ax.collections if isinstance(collection, PolyCollection) and collection.get_paths()]
        if len(bodies) != 8:
            raise RuntimeError(f"Expected eight violin bodies in Panel-d axis {axis_index}; found {len(bodies)}")
        for body_index, body in enumerate(bodies):
            faces = body.get_facecolors()
            edges = body.get_edgecolors()
            body.set_alpha(None)
            body.set_facecolors(_rgba_with_alpha(faces, FACE_ALPHA))
            if np.asarray(edges).size:
                body.set_edgecolors(_opaque_existing_edges(edges))
            violin_records.append({
                "axis_index": axis_index,
                "body_index": body_index,
                "face_alphas": [float(value) for value in body.get_facecolors()[:, 3]],
                "edge_alphas": [float(value) for value in body.get_edgecolors()[:, 3]],
            })

    c_xlabels = sorted({ax.get_xlabel() for ax in spectra if ax.get_xlabel()})
    d_xlabels = sorted({ax.get_xlabel() for ax in violin_axes if ax.get_xlabel()})
    fig.canvas.draw()
    collision = v4._collision_gate(fig)
    return {
        **v4_qa,
        **collision,
        "revision": "art_v5",
        "panel_c_title_before": title_before,
        "panel_c_title_after": title_after,
        "panel_c_spectrum_xlabels": c_xlabels,
        "panel_d_violin_xlabels": d_xlabels,
        "panel_b_heatmap_alpha_records": b_image_records,
        "panel_c_bar_alpha_records": bar_records,
        "panel_d_violin_alpha_records": violin_records,
        "panel_b_heatmap_count": len(b_image_records),
        "panel_c_bar_patch_count": len(bar_records),
        "panel_d_violin_body_count": len(violin_records),
        "face_alpha": FACE_ALPHA,
    }


def _apply_v5_contract() -> None:
    v4._apply_v4_contract()


def _install_v5_save_hook() -> None:
    original_save = base._save_figure

    def save_v5(fig, output_base: Path, formats: list[str], dpi: int):
        qa = _apply_v5_art(fig)
        outputs = original_save(fig, output_base, formats, dpi)
        (output_base.parent / f"{output_base.name}_art_qa.json").write_text(
            json.dumps(
                qa,
                indent=2,
                sort_keys=True,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
            ) + "\n",
            encoding="utf-8",
        )
        return outputs

    base._save_figure = save_v5


def main() -> int:
    _apply_v5_contract()
    _install_v5_save_hook()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
