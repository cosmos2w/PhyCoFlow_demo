# Unified publication-figure workflow

The a–f standalone figures and the composed figure use one native Matplotlib
render path. No standalone raster or vector panel is placed into the composite.

## Tuning locations

- `global_style.py`: manuscript font hierarchy, line weights, semantic model
  colors, field/heatmap colormaps, fixed 183 mm width, panel-label position, and
  physical safety tolerances.
- `publication_layout_unified_v2.yaml` → `physical_layout`: canvas width, one
  height per row, normalized width fractions, horizontal/vertical gaps, and
  panel-specific internal safe bounds. Panel widths and composite height are
  derived; they are never entered a second time.
- `common/panel_c_tuning.py`: Panel C sensor count, optional snapshot override,
  ROI, headers, image gaps, colorbars, and dividers.
- `common/publication_panels_unified_v2.py`: Panel B internal legend/text
  positions. Panel A reads its internal field/bar bounds from the main project
  configuration.
- `common/multiscale_wavelet_panels.py`: Panel D–F internal axes, heatmap,
  field-map, and colorbar defaults. The active values are exposed under
  `panel_d.layout`, `panel_e.layout`, and `panel_f.layout` in the layout YAML.

All panel-specific bounds are fractions of an exact physical parent rectangle.
The same bounds and `standalone=False` publication profile are used for both
destinations.

## Geometry contract

`common/physical_figure_layout.py` validates and creates the canvases.

1. For each row, horizontal gaps are subtracted from the canvas width and the
   remaining width is divided by normalized `width_fractions`.
2. One `height_mm` controls every panel in a row.
3. Composite height is the sum of row heights and vertical gaps.
4. Standalone page width and height must equal its composite panel rectangle.
5. Fixed exports use no tight crop and no constrained-layout engine.
6. Text outside the declared page raises an error instead of changing geometry.

Vertical spacing is controlled globally by `row_gap_mm` or locally by adding
`gap_after_mm` to a row. `minimum_row_gap_mm` is a configurable floor and may be
zero. The assembler measures visible text against every panel container and
checks cross-panel text intersections, so an unsafe compact setting fails with
the offending panel/text labels instead of silently overlapping.

## Typography and visual-style contract

`global_style.py` defines six semantic text roles: panel label, subplot title,
axis label, tick label, legend, and annotation. Each role has a manuscript-wide
default, while explicitly tagged dense-panel text may declare a local resolved
point size through `tag_font_role(..., size_pt=...)`. Every entry point calls
`enforce_figure_typography` after all nested axes and colorbars are drawn. Axis
objects are classified structurally; free-standing headers/row labels are
explicitly tagged. The export manifest records role defaults, local overrides,
and counts, rejects untagged ad-hoc sizes, and the fixed-canvas bbox audit still
rejects any local size that crosses the declared page boundary.

Model curves are tagged with their model identity and audited against the
global color, alpha, and primary line-width values. Standalone and composite
exports therefore cannot silently diverge in typography or model styling.

## Representative snapshots

`representative_snapshots` in the layout YAML defines one shared default and
explicit nullable overrides for qualitative panels. Panel C and Panel E inherit
the configured shared value by default. `common/representative_snapshots.py`
records the shared value, override, and resolved value. Cache lookup is read-only
and never invokes model inference.

## Entry points

- `96_export_unified_v2_panels.py`: exact standalone a–f pages.
- `97_assemble_mixed_resolution_unified_v2.py`: exact native composite.
- `81_plot_multiscale_components.py`: named Panel E PDF/SVG/PNG export.
- `82_plot_multiscale_fidelity.py`: named Panel F and SI interval exports.
- `98_audit_unified_v2.py`: numerical provenance, output, physical-size,
  geometry, and visual-contract audit.

Run all scripts in the `fig` conda environment. Plotting consumes validated CSV
files and reconstruction caches without modifying either source.
