#!/usr/bin/env python
"""Render art V5.1 with six shared Panel-a color bars beside the V5 maps."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
V5_PATH = HERE / "100_assemble_coupled_field_art_v5.py"
V5_MANIFEST = (HERE.parent / "_Process_Figures" / "Assembled" / "Composite"
               / "CoupledFieldReconstruction_art_v5_20260915_1035_source_manifest.json")
LAYOUT_PATH = HERE / "publication_layout_coupled_field_art_v5_1.yaml"


def _load_v5():
    spec = importlib.util.spec_from_file_location("coupled_field_art_v5_for_v5_1", V5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V5 renderer: {V5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v5 = _load_v5()
v4 = v5.v4
base = v5.base


def _scaled_tick(value: float, exponent: int) -> str:
    scaled = value / (10.0 ** exponent)
    if abs(scaled) < 0.005:
        return "0"
    return rf"${scaled:.3g}\times 10^{{{exponent}}}$"


def _panel_a_v5_1(fig, geometry: dict, scientific: dict) -> dict:
    maps = list(getattr(fig, "_v5_1_panel_a_maps", []))
    if len(maps) != 42:
        raise RuntimeError(f"Expected 42 Panel-a maps, found {len(maps)}")
    old_centers = sorted({round((ax.get_position().x0 + ax.get_position().x1) / 2, 7)
                          for ax in maps})
    if len(old_centers) != 7:
        raise RuntimeError(f"Expected seven Panel-a map columns, found {len(old_centers)}")
    old_centers = np.asarray(old_centers)
    columns = [[] for _ in range(7)]
    for ax in maps:
        midpoint = (ax.get_position().x0 + ax.get_position().x1) / 2
        columns[int(np.argmin(abs(old_centers - midpoint)))].append(ax)
    if any(len(column) != 6 for column in columns):
        raise RuntimeError("Each Panel-a column must contain six map rows")

    canvas_width_mm = float(fig.get_size_inches()[0]) * 25.4
    left_mm = min(ax.get_position().x0 for ax in maps) * canvas_width_mm
    right_mm = float(geometry["map_right_mm"])
    gap_mm = float(geometry["map_column_gap_mm"])
    bar_x_mm = float(geometry["colorbar_x_mm"])
    bar_width_mm = float(geometry["colorbar_width_mm"])
    map_width_mm = (right_mm - left_mm - 6 * gap_mm) / 7
    if map_width_mm <= 0 or not right_mm + 2 < bar_x_mm:
        raise RuntimeError("Panel-a map and colorbar columns have insufficient clearance")
    centers_mm = np.asarray([left_mm + map_width_mm / 2 + i * (map_width_mm + gap_mm)
                             for i in range(7)])
    for index, column in enumerate(columns):
        for ax in column:
            box = ax.get_position()
            ax.set_position([(centers_mm[index] - map_width_mm / 2) / canvas_width_mm,
                             box.y0, map_width_mm / canvas_width_mm, box.height], which="both")
            ax.set_aspect("auto")

    headers = list(v5.v4.v3.PANEL_A_COLUMNS)
    found_headers = set()
    for artist in fig.texts:
        label = artist.get_text()
        normalized = " ".join(label.replace("$U_1$", "U1").replace("$p$", "p").split())
        if normalized in headers:
            index = headers.index(normalized)
            artist.set_x(float(centers_mm[index] / canvas_width_mm))
            found_headers.add(normalized)
        elif label == "Conditioning progression (DMF-Gen)":
            artist.set_x(float(np.mean(centers_mm[:3]) / canvas_width_mm))
        elif label == "Baseline comparisons: conditioned on T only":
            artist.set_x(float(np.mean(centers_mm[3:]) / canvas_width_mm))
    if len(found_headers) != 7:
        raise RuntimeError(f"Panel-a header inventory changed: {found_headers}")
    divider_mm = float((centers_mm[2] + centers_mm[3]) / 2)
    dividers = [artist for artist in fig.artists if isinstance(artist, Line2D)]
    if len(dividers) != 1:
        raise RuntimeError(f"Expected one Panel-a divider, found {len(dividers)}")
    dividers[0].set_xdata([divider_mm / canvas_width_mm] * 2)

    row_boxes = sorted((ax.get_position() for ax in columns[0]), key=lambda box: box.y0, reverse=True)
    height_fraction = float(geometry["colorbar_height_fraction"])
    fields = list(scientific["fields"])
    bar_records = []
    for row_index, box in enumerate(row_boxes):
        field = fields[row_index // 2]
        is_error = row_index % 2 == 1
        limits = (scientific["robust_error_limits"] if is_error
                  else scientific["field_value_limits"])[field]
        lo, hi = map(float, limits)
        cmap = base._panel_a_colormap(error=True) if is_error else base._panel_a_colormap(field)
        bar_height = box.height * height_fraction
        bar_bottom = box.y0 + (box.height - bar_height) / 2
        cax = fig.add_axes([bar_x_mm / canvas_width_mm, bar_bottom,
                            bar_width_mm / canvas_width_mm, bar_height],
                           label=f"panel-a-{field}-{'error' if is_error else 'value'}-bar")
        cbar = fig.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap=cmap),
                            cax=cax, ticks=[lo, hi], orientation="vertical")
        exponent = int(np.floor(np.log10(max(abs(lo), abs(hi)))))
        cbar.ax.set_yticklabels([_scaled_tick(lo, exponent), _scaled_tick(hi, exponent)])
        cbar.ax.tick_params(axis="y", which="both", left=False, right=True,
                            labelleft=False, labelright=True, length=1.5, pad=1.0,
                            labelsize=float(geometry["colorbar_tick_size_pt"]))
        cbar.outline.set_linewidth(0.55)
        bar_records.append({"field": field, "kind": "error" if is_error else "value",
                            "limits": [lo, hi], "cmap": cmap.name,
                            "bbox_mm": [bar_x_mm, bar_bottom * fig.get_size_inches()[1] * 25.4,
                                        bar_width_mm, bar_height * fig.get_size_inches()[1] * 25.4]})

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 25.4 / fig.dpi
    text_overflow = []
    for ax in fig.axes[-6:]:
        for tick in ax.get_yticklabels():
            if tick.get_visible() and tick.get_text():
                box = tick.get_window_extent(renderer)
                if box.x1 * scale > canvas_width_mm - 0.4:
                    text_overflow.append((tick.get_text(), box.x1 * scale))
    collision = v4._collision_gate(fig)
    result = {
        "revision": "art_v5_1", "map_count": len(maps), "colorbar_count": len(bar_records),
        "map_left_mm": left_mm, "map_right_mm": right_mm,
        "map_width_mm": map_width_mm, "map_gap_mm": gap_mm,
        "bar_to_map_gap_mm": bar_x_mm - right_mm,
        "divider_x_mm": divider_mm, "colorbars": bar_records,
        "tick_canvas_overflow": text_overflow,
        **collision,
    }
    result["passed"] = (not text_overflow and collision["text_text_overlap_count"] == 0
                        and collision["text_nonowned_axes_overlap_count"] == 0)
    if not result["passed"]:
        raise RuntimeError(f"Panel-a V5.1 layout failed: {result}")
    return result


def main() -> int:
    v5._apply_v5_contract()
    original_qualitative = base.draw_qualitative_panel

    def capture_panel_a_maps(fig, slot, *args, **kwargs):
        before = set(fig.axes)
        result = original_qualitative(fig, slot, *args, **kwargs)
        fig._v5_1_panel_a_maps = [axis for axis in fig.axes if axis not in before]
        return result

    base.draw_qualitative_panel = capture_panel_a_maps
    original_save = base._save_figure
    scientific = json.loads(V5_MANIFEST.read_text(encoding="utf-8"))
    import yaml
    geometry = yaml.safe_load(LAYOUT_PATH.read_text(encoding="utf-8"))["panel_a_v5_1"]
    written_base = None

    def save_v5_1(fig, output_base: Path, formats: list[str], dpi: int):
        nonlocal written_base
        v5_qa = v5._apply_v5_art(fig)
        panel_a_qa = _panel_a_v5_1(fig, geometry, scientific)
        outputs = original_save(fig, output_base, formats, dpi)
        (output_base.parent / f"{output_base.name}_art_qa.json").write_text(
            json.dumps({**v5_qa, "panel_a_v5_1": panel_a_qa}, indent=2, sort_keys=True,
                       default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n",
            encoding="utf-8",
        )
        written_base = output_base
        return outputs

    base._save_figure = save_v5_1
    status = base.main()
    if written_base is None:
        raise RuntimeError("V5.1 renderer did not export the figure")
    manifest_path = written_base.parent / f"{written_base.name}_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qualitative_colourbar_policy"] = (
        "six right-side bars; exact physical lower and upper limits; "
        "reconstruction and absolute error use their V5 palettes and normalizations"
    )
    manifest["qualitative_geometry_lock"]["colorbar_multiplier"] = (
        "none; each bar shows exact scientific endpoint labels"
    )
    manifest["panel_a_v5_1_layout"] = geometry
    manifest["source_visual_revision"] = "art_v5"
    manifest["revision"] = "art_v5_1"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
