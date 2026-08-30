#!/usr/bin/env python
"""Assemble existing panels, including standalone per-field contour exports.

The ``contour_grid`` YAML panel type resolves the current contour contract:
``Fig_<GT|Rec|Err>_<field>_s<snapshot>_<model>_<condition>_<run-id>.png``.
It never loads a model or rebuilds a reconstruction.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import string
import subprocess
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.config import FIGURES_DIR, SCRIPT_DIR, add_common_args, load_config, run_id
from common.figure_style import add_panel_label, apply_style, mark_missing, save_figure
from global_style import (
    COMPOSITE_MARGIN_BOTTOM_IN,
    COMPOSITE_MARGIN_LEFT_IN,
    COMPOSITE_MARGIN_RIGHT_IN,
    COMPOSITE_MARGIN_TOP_IN,
    COMPOSITE_WIDTH_IN,
    GENERIC_PANEL_ROW_HEIGHT_IN,
    PANEL_HSPACE_IN,
    PANEL_WSPACE_IN,
    SIZE_SUBPLOT_TITLE,
    adaptive_composite_height,
    gridspec_space_from_inches,
)


def _model_directory(cfg: dict[str, Any], requested: str) -> str:
    """Resolve a display name or alias to the model directory from YAML."""
    needle = str(requested).lower()
    for method in cfg["methods"]:
        names = [method["name"], method["directory"], *method.get("aliases", [])]
        if any(needle == str(name).lower() for name in names):
            return method["directory"]
    return requested


def _read_panel(ax, path: Path, missing_text: str, cfg: dict) -> None:
    """Draw a preview panel, preferring a same-name PNG for PDF sources.

    The final ``--vector-pdf`` path embeds the requested PDF directly.  The
    ordinary Matplotlib preview cannot decode PDFs consistently, so this
    branch uses the matching exporter-produced PNG where it is available.
    """
    preview_path = path.with_suffix(".png") if path.suffix.lower() == ".pdf" and path.with_suffix(".png").exists() else path
    if not preview_path.exists():
        mark_missing(ax, missing_text, cfg)
        return
    try:
        ax.imshow(plt.imread(preview_path), aspect="equal")
        ax.axis("off")
    except Exception:
        mark_missing(ax, "Unreadable", cfg)


def _path_panel(panel: dict, layout_path: Path, rid: str) -> Path:
    raw = str(panel.get("path", "")).replace("{run_id}", rid)
    path = Path(raw)
    return path if path.is_absolute() else (layout_path.parent / path).resolve()


def _contour_path(panel: dict, cfg: dict, field: str, kind: str, rid: str) -> Path:
    model_dir = _model_directory(cfg, str(panel["model"]))
    condition = str(panel["condition"])
    snapshot = int(panel.get("snapshot", 0))
    contour_rid = str(panel.get("contour_run_id", rid)).replace("{run_id}", rid)
    extension = str(panel.get("extension", "png")).lstrip(".")
    filename = f"Fig_{kind}_{field}_s{snapshot:04d}_{model_dir}_{condition}_{contour_rid}.{extension}"
    return FIGURES_DIR / "_Contours" / model_dir / condition / filename


def _contour_grid(fig, slot, panel: dict, cfg: dict, rid: str) -> None:
    """Populate a nested field-by-kind grid from individual contour panels."""
    fields = [str(value) for value in panel.get("fields", [item["key"] for item in cfg["fields"]])]
    kinds = [str(value) for value in panel.get("kinds", ["GT", "Rec", "Err"])]
    hspace = float(panel.get("hspace", 0.05))
    wspace = float(panel.get("wspace", 0.03))
    grid = slot.subgridspec(len(fields), len(kinds), hspace=hspace, wspace=wspace)

    for row, field in enumerate(fields):
        for col, kind in enumerate(kinds):
            ax = fig.add_subplot(grid[row, col])
            path = _contour_path(panel, cfg, field, kind, rid)
            _read_panel(ax, path, panel.get("missing_text", "Missing"), cfg)


def _layout_box(panel: dict, layout: dict) -> tuple[float, float, float, float]:
    """Return an arbitrary YAML grid panel as x, top-y, width, height in mm."""
    width, height = float(layout.get("width_mm", 183)), float(layout.get("height_mm", 120))
    rows, cols = int(layout.get("rows", 1)), int(layout.get("cols", 1))
    gap_x = float(layout.get("wspace", .08)) * width / cols
    gap_y = float(layout.get("hspace", .08)) * height / rows
    width_weights = np.asarray(layout.get("width_ratios", [1.0] * cols), dtype=float)
    height_weights = np.asarray(layout.get("height_ratios", [1.0] * rows), dtype=float)
    if width_weights.size != cols or height_weights.size != rows or np.any(width_weights <= 0) or np.any(height_weights <= 0):
        raise ValueError("width_ratios/height_ratios must contain one positive value per grid column/row.")
    widths = (width - gap_x * (cols - 1)) * width_weights / width_weights.sum()
    heights = (height - gap_y * (rows - 1)) * height_weights / height_weights.sum()
    row, col = int(panel.get("row", 0)), int(panel.get("col", 0))
    rowspan, colspan = int(panel.get("rowspan", 1)), int(panel.get("colspan", 1))
    x = float(np.sum(widths[:col]) + col * gap_x)
    top = float(height - np.sum(heights[:row]) - row * gap_y)
    panel_w = float(np.sum(widths[col : col + colspan]) + (colspan - 1) * gap_x)
    panel_h = float(np.sum(heights[row : row + rowspan]) + (rowspan - 1) * gap_y)
    return x, top, panel_w, panel_h


def _tex_path(path: Path) -> str:
    """Use detokenize so underscores/spaces in local panel paths remain safe."""
    return path.resolve().as_posix().replace("\\", "/")


def _tex_text(value: object) -> str:
    """Minimal escaping for layout titles/labels rendered by LaTeX."""
    text = str(value)
    for plain, escaped in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(plain, escaped)
    return text


def _tex_image(path: Path, x: float, top: float, width: float, height: float, missing: str) -> list[str]:
    if path.exists():
        return [
            rf"\node[anchor=north west,inner sep=0pt] at ({x:.4f}mm,{top:.4f}mm) {{\includegraphics[width={width:.4f}mm,height={height:.4f}mm,keepaspectratio]{{\detokenize{{{_tex_path(path)}}}}}}};"
        ]
    bottom = top - height
    return [
        rf"\fill[gray!20] ({x:.4f}mm,{bottom:.4f}mm) rectangle ({x + width:.4f}mm,{top:.4f}mm);",
        rf"\node at ({x + width / 2:.4f}mm,{bottom + height / 2:.4f}mm) {{\scriptsize\textbf{{{missing}}}}};",
    ]


def _vector_pdf(layout: dict, layout_path: Path, cfg: dict, rid: str, base: Path) -> Path:
    """Compose PDFs with LaTeX/TikZ so PDF source panels stay vector graphics."""
    width, height = float(layout.get("width_mm", 183)), float(layout.get("height_mm", 120))
    lines = [
        r"\documentclass[tikz,border=0pt]{standalone}",
        r"\usepackage{graphicx}",
        r"\begin{document}",
        rf"\begin{{tikzpicture}}[x=1mm,y=1mm]",
        rf"\useasboundingbox (0,0) rectangle ({width:.4f},{height:.4f});",
    ]
    for index, panel in enumerate(layout.get("panels", [])):
        x, top, panel_w, panel_h = _layout_box(panel, layout)
        label = _tex_text(panel.get("label", string.ascii_lowercase[index]))
        # Keep manuscript panel letters inside the page bounding box.  Add the
        # header *after* source artwork so it is not hidden by an image node.
        header = [rf"\node[anchor=north west,font=\bfseries,fill=white,fill opacity=.88,text opacity=1,inner sep=1pt] at ({x + 1.0:.4f}mm,{top - 1.0:.4f}mm) {{{label}}};"]
        if panel.get("title"):
            header.append(rf"\node[anchor=north,font=\scriptsize,fill=white,fill opacity=.88,text opacity=1,inner sep=1pt] at ({x + panel_w / 2:.4f}mm,{top - 1.0:.4f}mm) {{{_tex_text(panel['title'])}}};")
        if panel.get("type", "path") != "contour_grid":
            lines.extend(_tex_image(_path_panel(panel, layout_path, rid), x, top, panel_w, panel_h, panel.get("missing_text", "Missing")))
            lines.extend(header)
            continue
        fields = [str(value) for value in panel.get("fields", [item["key"] for item in cfg["fields"]])]
        kinds = [str(value) for value in panel.get("kinds", ["GT", "Rec", "Err"])]
        gap_x = float(panel.get("wspace", .03)) * panel_w / max(len(kinds), 1)
        gap_y = float(panel.get("hspace", .05)) * panel_h / max(len(fields), 1)
        cell_w = (panel_w - gap_x * (len(kinds) - 1)) / len(kinds)
        cell_h = (panel_h - gap_y * (len(fields) - 1)) / len(fields)
        for row, field in enumerate(fields):
            for col, kind in enumerate(kinds):
                cell_x = x + col * (cell_w + gap_x)
                cell_top = top - row * (cell_h + gap_y)
                lines.extend(_tex_image(_contour_path(panel, cfg, field, kind, rid), cell_x, cell_top, cell_w, cell_h, panel.get("missing_text", "Missing")))
        lines.extend(header)
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    with tempfile.TemporaryDirectory(prefix="phycoflow_vector_assembly_") as tmp:
        tex_path = Path(tmp) / "assembly.tex"
        tex_path.write_text("\n".join(lines), encoding="utf-8")
        result = subprocess.run(["pdflatex", "-interaction=batchmode", "-halt-on-error", "assembly.tex"], cwd=tmp, capture_output=True, text=True)
        pdf_path = Path(tmp) / "assembly.pdf"
        if result.returncode != 0 or not pdf_path.exists():
            raise RuntimeError(f"Vector PDF assembly failed: {result.stdout[-1000:]} {result.stderr[-1000:]}")
        base.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, base.with_suffix(".pdf"))
    return base.with_suffix(".pdf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=SCRIPT_DIR / "example_layout.yaml")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--vector-pdf", action="store_true", help="Compose PDF source panels through LaTeX/TikZ without rasterizing them.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rid = run_id(args.run_id)
    apply_style(cfg)
    layout = yaml.safe_load(args.layout.read_text(encoding="utf-8")) or {}
    rows, cols = int(layout.get("rows", 1)), int(layout.get("cols", 1))
    # Generic assembly is also width-locked and height-adaptive.  A layout may
    # provide exact row_heights_in; otherwise every major row receives the
    # portable global default.  Adding/removing rows therefore changes height
    # without changing the journal-standard 7.2-in width.
    row_heights = [float(value) for value in layout.get(
        "row_heights_in", [GENERIC_PANEL_ROW_HEIGHT_IN] * rows,
    )]
    if len(row_heights) != rows:
        raise ValueError("row_heights_in must contain one physical height per layout row.")
    width_in = COMPOSITE_WIDTH_IN
    height_in = adaptive_composite_height(row_heights)
    fig = plt.figure(
        figsize=(width_in, height_in), facecolor="white",
    )
    usable_width = width_in - COMPOSITE_MARGIN_LEFT_IN - COMPOSITE_MARGIN_RIGHT_IN
    usable_height = height_in - COMPOSITE_MARGIN_TOP_IN - COMPOSITE_MARGIN_BOTTOM_IN
    main_grid = fig.add_gridspec(
        rows, cols,
        left=COMPOSITE_MARGIN_LEFT_IN / width_in,
        right=1.0 - COMPOSITE_MARGIN_RIGHT_IN / width_in,
        top=1.0 - COMPOSITE_MARGIN_TOP_IN / height_in,
        bottom=COMPOSITE_MARGIN_BOTTOM_IN / height_in,
        wspace=gridspec_space_from_inches(usable_width, PANEL_WSPACE_IN, cols),
        hspace=gridspec_space_from_inches(usable_height, PANEL_HSPACE_IN, rows),
        width_ratios=layout.get("width_ratios"), height_ratios=layout.get("height_ratios"),
    )

    for index, panel in enumerate(layout.get("panels", [])):
        row, col = int(panel.get("row", 0)), int(panel.get("col", 0))
        rowspan, colspan = int(panel.get("rowspan", 1)), int(panel.get("colspan", 1))
        slot = main_grid[row : row + rowspan, col : col + colspan]
        panel_type = panel.get("type", "path")

        # A transparent parent axes owns the Nature-style letter for a nested
        # contour grid; ordinary path panels use the image axes directly.
        if panel_type == "contour_grid":
            parent = fig.add_subplot(slot)
            parent.set_axis_off()
            _contour_grid(fig, slot, panel, cfg, rid)
            if panel.get("title"):
                parent.text(.5, .99, panel["title"], transform=parent.transAxes, ha="center", va="top", fontsize=SIZE_SUBPLOT_TITLE)
            add_panel_label(
                parent, panel.get("label", string.ascii_lowercase[index]), cfg,
                x=float(panel.get("label_x", .01)), y=float(panel.get("label_y", .99)),
            )
            continue

        ax = fig.add_subplot(slot)
        _read_panel(ax, _path_panel(panel, args.layout, rid), panel.get("missing_text", "Missing"), cfg)
        if panel.get("title"):
            ax.set_title(panel["title"])
        add_panel_label(
            ax, panel.get("label", string.ascii_lowercase[index]), cfg,
            x=float(panel.get("label_x", .01)), y=float(panel.get("label_y", .99)),
        )

    name = layout.get("name", "AssembledFigure")
    base = FIGURES_DIR / "Assembled" / f"{name}_{rid}"
    normal_formats = args.formats
    if args.vector_pdf and args.formats:
        normal_formats = [fmt for fmt in args.formats if fmt != "pdf"]
    save_figure(fig, base, cfg, normal_formats, args.dpi, fixed_canvas=True)
    plt.close(fig)
    if args.vector_pdf:
        vector_path = _vector_pdf(layout, args.layout, cfg, rid, base)
        print(f"[OK] vector PDF: {vector_path}")
    print(f"[OK] {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
