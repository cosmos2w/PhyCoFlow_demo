# Mixed-resolution Figure V3-3 revision brief for Codex

## Goal
Build **V3-3** from the current V3-2 hybrid figure while preserving the successful macro-narrative and restoring the quantitative/physical depth expected for a Nature-family main figure.

The current V3-2 has the correct scientific sequence, but several reductions went too far. V3-3 should **add depth without returning to the visual wall of the original six-panel figure**.

The intended reading sequence is:

1. **Panel a** — establish the three physical discretizations and the five training-data recipes.
2. **Panel b** — show how architectures respond to removing H-resolution training fields and how the zero-H models respond to sensor density.
3. **Panel c** — provide direct physical evidence: full field, magnified local structure, and aligned local reconstruction error for the decisive zero-H setting.
4. **Panel d** — restore complete large/intermediate/fine qualitative scale evidence and keep the two fine-scale trend plots as the compact quantitative summary.
5. **Panel e** — restore the full multi-scale quantitative matrices so the main figure visibly carries the complete scale-resolved workload.

No model training, new inference, or metric recomputation is requested. Use the exact validated V3-2 sources/caches and the earlier validated unified-v2/V3 sources.

---

# 1. Comparison of the current V3-2 and the earlier full figure

## What V3-2 improved

V3-2 successfully:
- made the training-data logic in panel a immediately understandable;
- integrated the duplicated 512-sensor comparison and sensor sweeps into one quantitative panel b;
- focused panel c on one zero-H recipe instead of repeating three recipes;
- simplified the multiscale argument into one compact panel d.

These structural improvements must be preserved.

## What V3-2 removed too aggressively

V3-2 currently loses or weakens:
- the **magnified local field evidence** that made the resolution difference visually convincing;
- the full **Large / Intermediate / Fine** scale progression;
- the **complete numerical matrices** for spatial-pattern correlation and variance-allocation bias;
- one useful qualitative baseline, **MLP-RBF**, which helps establish contrast with a simpler regression model.

V3-3 should restore exactly these high-value elements and no more.

---

# 2. Required V3-3 global layout

Use **five lettered panels**: `a, b, c, d, e`.

The recommended overall topology is:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ a  Resolution context + training recipes                              │
├─────────────────────────────────────────────────────────────────────────┤
│ b  Recipe transfer (grouped bars) + two zero-H sensor sweeps          │
├────────────────────────────────────────────┬────────────────────────────┤
│ c  5-column physical reconstruction       │ d  multi-scale evidence  │
│    Full / Zoom / Local error               │    + stacked line plots  │
├────────────────────────────────────────────┤                            │
│ e  full quantitative scale matrices        │                            │
└────────────────────────────────────────────┴────────────────────────────┘
```

### Preferred outer GridSpec

Use an outer `GridSpec` with:
- `nrows = 4`
- `ncols = 20`
- target `height_ratios ≈ [1.05, 2.30, 2.55, 1.80]`

Suggested assignments:
- `a = gs[0, 0:20]`
- `b = gs[1, 0:20]`
- `c = gs[2, 0:13]`  → approximately 65% of page width
- `d = gs[2:4, 13:20]` → vertical channel, approximately 35% width
- `e = gs[3, 0:13]` → bottom-left quantitative matrices

Starting outer spacing:
- `hspace ≈ 0.30`
- `wspace ≈ 0.30`

Tune at final print width, not on a large monitor.

### Figure dimensions

Start from:
- width: `183 mm`
- height: `225–240 mm`

Do not exceed approximately `240 mm` unless the project template demonstrates that the larger figure still fits the target full-page Nature layout.

---

# 3. Panel a — restore resolution context and improve the L/M/H legend

## 3.1 Restore the localized zoom effect

The current L/M/H thumbnails are too passive. Reintroduce a common physical ROI to make the discretization differences immediately visible.

For each of the three resolution thumbnails:

- use the **same physical region** in L, M and H;
- draw a thin black ROI rectangle on the parent field;
- add a small magnified inset using the exact same crop;
- connect the ROI to the inset with two subtle connector lines;
- use `interpolation="nearest"` in the inset so that the L/M/H pixel structure remains visible;
- keep the same field-value normalization across all three resolutions;
- no inset axes/ticks.

Recommended style:
- ROI border: `0.55–0.7 pt`, black or dark gray;
- connectors: `0.4–0.55 pt`, alpha `0.45–0.60`;
- inset size: about `35–42%` of the parent thumbnail width;
- do not let connectors cross neighboring thumbnails.

If an inset placed inside the thumbnail hides important field structure, place the inset immediately below or to the lower-right of the thumbnail while retaining the connectors.

## 3.2 Replace the sparse horizontal L/M/H key

Remove the current tiny floating horizontal swatches.

Create a **three-row vertical resolution block**:

```text
■  L   Low resolution      32 × 32
■  M   Medium resolution   64 × 64
■  H   High resolution    128 × 128
```

Use the same L/M/H colors as the stacked training bars.

The block should:
- occupy the intentional negative space between the thumbnails and the bar chart, or directly beneath/alongside the thumbnail group;
- use three aligned rows;
- use the manuscript’s final-size font, not miniature legend text;
- align the grid dimensions vertically so the block reads like a compact table.

To avoid redundant text, the thumbnail titles may be reduced to only `L`, `M`, `H`, with the dimensions carried in the vertical block. If the current `L / 32×32` titles remain visually cleaner, retain them but still use the vertical block as the formal resolution definition.

## 3.3 Keep the successful training-recipe bars

Preserve:
- H-only;
- H-limited;
- Mixed-HML;
- Zero-H-balanced;
- Zero-H-M-rich;
- `1.00×, 0.34×, 0.44×, 0.16×, 0.19×`;
- the grouping `contains H-resolution training fields` versus `zero-H training`.

Do not reintroduce extra setup panels.

## 3.4 Suggested inner panel-a topology

Use approximately:

```text
[ L/M/H thumbnails ] [ 3-row resolution block ] [ training-recipe bars ]
      28–30%                    12–14%                    56–60%
```

A practical inner width ratio is close to:
`3.2 : 1.35 : 6.8`.

---

# 4. Panel b — grouped categorical transfer + continuous zero-H sweeps

The current three stacked line plots are scientifically clear but aesthetically repetitive. V3-3 must distinguish the categorical recipe-transfer experiment from the continuous sensor-density sweeps.

## 4.1 Top axis: replace recipe-transfer lines with a grouped bar chart

### Categories

Use the five recipes in this exact order:
1. H-only
2. H-limited
3. Mixed-HML
4. Zero-H-balanced
5. Zero-H-M-rich

### Bars

For each recipe, show four adjacent bars:
- DMF-Gen
- FFM-Perceiver
- Senseiver
- MLP-RBF

Use the existing method colors consistently.

### Statistical information

Use the validated 512-sensor mean physical relative-$L_2$ and its accepted 95% interval if available in the source summary.

Add thin error bars above each bar.

### Y-axis

Preferred: **linear error axis** for the grouped bars.

Reason:
- the bar chart is categorical;
- the numerical range is small enough for a linear scale;
- using a linear top chart and logarithmic lower sweeps intentionally breaks aesthetic repetition;
- the lower sweep plots retain logarithmic scaling for sensor-count behavior.

Set the upper limit from the validated data with approximately 8–12% headroom.

If the renderer/audit contract strongly requires the log scale, a log-bar implementation is acceptable only if the bar baseline is represented without visual distortion and documented explicitly. Linear is preferred.

### Zero-H region

Retain the subtle background shading over:
- Zero-H-balanced
- Zero-H-M-rich

with a small label:
`no H-resolution training fields`.

### Precision annotations

Do not print 20 large numeric labels.

Preferred policy:
- annotate the DMF-Gen value above each recipe bar with three significant decimals;
- optionally annotate the best non-DMF baseline only in the two zero-H groups if this remains collision-free;
- retain all exact values in the figure report/SI table.

### Optional trend cue

A thin red line connecting the centers/tops of the five DMF-Gen bars is allowed and encouraged if it improves transfer readability.

Do not overlay four full trend lines unless final-size QA confirms the result remains clean.

## 4.2 Lower axes: retain the two current zero-H line sweeps

Keep:
- Zero-H-balanced
- Zero-H-M-rich

with:
- sensor counts 64, 128, 256, 384, 512;
- H-grid density labels;
- log-y scale;
- all four methods;
- existing uncertainty treatment.

## 4.3 Expand and separate the method legend

The current legend is too compressed.

Create a dedicated legend area above the panel-b plotting axes.

Requirements:
- `ncol=4`;
- increased font size relative to V3-2;
- increased `columnspacing`;
- increased `handletextpad`;
- no overlap with panel a or the top-axis title;
- no tiny compressed bounding box.

Suggested starting parameters:
- legend font: `7.0–7.6 pt` at final 183-mm width;
- marker/handle length: visually consistent with the main lines;
- `columnspacing ≈ 1.5–2.0`;
- `handletextpad ≈ 0.45–0.65`.

Use a frame only if it materially improves separation; otherwise no frame.

## 4.4 Increase vertical pacing

Increase separation between:
- grouped bar plot;
- lower two sensor-sweep plots.

Use a dedicated spacer or increase inner `hspace`.

Suggested inner panel-b design:
- top legend row;
- grouped-bar row;
- lower two-line-plot row.

Starting `hspace` between top and bottom plot rows: approximately `0.48–0.62`.

Ensure y-axis labels, zero-H titles, and sensor-density labels do not visually collide.

---

# 5. Panel c — five-column physical proof with explicit magnification

Panel c should become the principal physical-evidence panel.

## 5.1 Fixed setting

Use one validated setting only:
- **Zero-H-M-rich**
- same validated representative snapshot used in V3-2
- same validated sensor count and sensor plan

Do not add multiple recipes.

## 5.2 Five columns

Use this exact column order:

1. Ground truth
2. DMF-Gen
3. FFM-Perceiver
4. Senseiver
5. MLP-RBF

### Rationale

MLP-RBF is restored because it gives the reader a direct visual contrast against a simpler regression/interpolation-style baseline. It should not be added elsewhere as an additional contour family.

The MLP-RBF reconstruction must come from the existing validated cache for the same representative state/recipe/sensor plan. Do not run new inference. If the exact matching cached result is absent, stop and document that deficiency.

## 5.3 Three aligned rows

### Row 1 — Full field

For all five columns:
- same full H-resolution field extent;
- same field-value color limits;
- same ROI box;
- same axis box dimensions;
- no axes/ticks.

For model columns:
- annotate `Rel. $L_2$ = ...` using the validated full-field error.

### Row 2 — Zoomed field

For all five columns:
- exact same physical crop;
- same field-value scale as Row 1;
- same axis dimensions;
- no independent colorbar;
- no axes/ticks.

### Row 3 — Local absolute error

For model columns:
- show the local absolute error over the same zoomed physical crop.

For Ground truth:
- use the **sensor-layout tile**.

The sensor-layout tile must:
- have the **same axes bounding-box width and height** as the adjacent error heatmaps;
- use the same visual aspect ratio;
- preserve the physical coordinate aspect inside that box through padding/letterboxing if necessary;
- not shrink relative to neighboring tiles.

## 5.4 Frustum / magnification connectors

Restore a Nature-style magnification cue.

For each column:
- connect two corners of the ROI rectangle in Row 1 to the corresponding two corners of the Row-2 zoomed tile;
- use `ConnectionPatch` or an equivalent figure-coordinate connector;
- lines should be subtle and sit behind text/data annotations.

Recommended style:
- color `#444444` or black;
- linewidth `0.45–0.60 pt`;
- alpha `0.35–0.50`;
- no arrows.

The connectors must not cross into neighboring columns.

## 5.5 Standardize local-error annotation

Replace:
- `Local rel. $L_2$ = ...`

with:
- `Rel. $L_2$ = ...`

throughout the local-error row.

The row label itself already establishes that these are local errors.

## 5.6 Colorbars and scientific notation

Use exactly two shared horizontal colorbars:
- Field value
- Zoom-region absolute error

The current scientific-notation multipliers collide with the colorbar bounds/ticks.

Fix this by:
- suppressing automatic offset text on the colorbar axes;
- manually placing `$\times10^{-2}$` / `$\times10^{-3}$` with sufficient padding to the right of the bar;
- expanding the right margin of each colorbar axis;
- ensuring the exponent does not touch the last tick or colorbar outline.

Suggested implementation:
- `cbar.ax.xaxis.get_offset_text().set_visible(False)`
- add exponent with `cbar.ax.text(...)` outside the final tick using axes coordinates;
- provide at least `0.04–0.06` axes-width equivalent whitespace after the last tick.

Audit the exported PDF/SVG, not only the PNG.

---

# 6. Panel d — right-hand vertical channel with complete multi-scale physical evidence

Panel d should occupy the dedicated approximately 35% right-hand channel and span the two lower rows.

It must contain:
1. complete Large / Intermediate / Fine qualitative scale evidence;
2. the two stacked fine-scale line plots.

## 6.1 Reintroduce Large scale

Restore the complete three-scale qualitative block:

Rows:
- Large
- Intermediate
- Fine

Columns:
1. Ground-truth component
2. DMF-Gen residual
3. Senseiver residual

Use the same validated representative state and recipe as before.

Do not add FFM-Perceiver/MLP-RBF residual maps here; their quantitative scale behavior is represented below and in panel e.

## 6.2 Qualitative map formatting

Use compact consistent labels:
- column headings only once;
- scale labels at left;
- no axes/ticks;
- residual `Rel. $L_2$` value in each residual tile.

Expected validated representative values include:
- Large: DMF-Gen ~0.02, Senseiver ~0.05
- Intermediate: DMF-Gen ~0.21, Senseiver ~0.38
- Fine: DMF-Gen ~0.68, Senseiver ~4.81

Audit exact values from the validated source rather than hard-coding rounded numbers.

Use the original validated per-scale color limits. If separate scale colorbars are required, use very compact row-specific colorbars or report the limits in the caption/report; do not let colorbars dominate the right channel.

## 6.3 Stack the two fine-scale line plots vertically

Below the three-scale map block, stack:

### Plot 1
Fine-scale spatial pattern correlation

### Plot 2
Fine-scale variance-allocation bias

Use:
- Mixed-HML
- Zero-H-balanced
- Zero-H-M-rich

for:
- DMF-Gen
- FFM-Perceiver
- Senseiver
- MLP-RBF

Keep the existing uncertainty/error-bar definition.

Use:
- horizontal reference at correlation `= 1`;
- horizontal reference at bias `= 0`.

The two plots should share x alignment. Only the bottom plot needs full recipe labels if the upper plot becomes cramped.

## 6.4 Suggested panel-d internal GridSpec

Use an inner grid with approximately:

- top qualitative block: `45–50%` of panel-d height;
- correlation plot: `24–27%`;
- bias plot: `24–27%`.

For the top block:
- `3 rows × 3 columns`;
- optional very narrow fourth column for per-scale colorbar(s).

Starting height ratio:
`[1.0, 1.0, 1.0, 0.20, 1.05, 1.05]`
if explicit spacer rows are helpful.

---

# 7. Panel e — restore the complete quantitative matrices

Panel e occupies the **bottom-left**, directly beneath panel c.

This panel restores the full scale-resolved numerical workload from the older figure.

## 7.1 Two matrices, stacked vertically

Restore:

1. Spatial pattern correlation
2. Variance allocation bias

Each matrix must include:
- 4 methods:
  - DMF-Gen
  - FFM-Perceiver
  - Senseiver
  - MLP-RBF
- 3 training recipes:
  - Mixed-HML
  - Zero-H-balanced
  - Zero-H-M-rich
- 3 scale groups per recipe:
  - Large
  - Intermediate
  - Fine

This gives a `4 × 9` matrix per metric.

## 7.2 Stack rather than place side-by-side

Because panel e has approximately 65% page width, stack the two matrices vertically.

This allows:
- readable cell annotations;
- readable scale labels;
- clear recipe group headers.

Do not squeeze the two `4 × 9` matrices side-by-side.

## 7.3 Restore numerical annotations

Print the validated cell values in the matrices.

Use concise formatting:
- pattern correlation: 2 decimals;
- variance allocation bias: signed 2 decimals, with `+` for positive values if space permits.

Keep method labels only on the left.

## 7.4 Recipe and scale headers

For each matrix:
- group columns by recipe with a thin separator;
- label recipe once above each 3-column group;
- label scale columns `Large`, `Interm.`, `Fine`.

If vertical space is tight, show recipe/scale headers fully on the upper matrix and use aligned columns for the lower matrix, but the lower matrix must remain interpretable.

## 7.5 Link panel e to panel d

Panel d extracts the fine-scale trend from panel e.

Make this relationship visually explicit with a subtle device:
- draw a thin outline around the `Fine` column of each recipe group in both matrices, **or**
- add a small bracket/marker beneath the Fine columns with a note in the figure report that these are summarized in panel d.

Do not use a heavy red box.

## 7.6 Validated matrix values

Do not recompute. Audit against the existing multiscale summary.

Key expected pattern-correlation values include:

### Mixed-HML
- DMF-Gen: 1.00 / 0.99 / 0.93
- FFM-Perceiver: 1.00 / 0.75 / 0.37
- Senseiver: 1.00 / 0.97 / 0.84
- MLP-RBF: 0.99 / 0.61 / 0.25

### Zero-H-balanced
- DMF-Gen: 1.00 / 0.95 / 0.46
- FFM-Perceiver: 1.00 / 0.74 / 0.32
- Senseiver: 1.00 / 0.94 / 0.07
- MLP-RBF: 0.99 / 0.52 / 0.14

### Zero-H-M-rich
- DMF-Gen: 1.00 / 0.97 / 0.81
- FFM-Perceiver: 1.00 / 0.76 / 0.37
- Senseiver: 1.00 / 0.93 / -0.09
- MLP-RBF: 0.99 / 0.53 / 0.17

Key expected variance-allocation-bias values include the previous validated values, for example:
- DMF-Gen remains near zero across scales;
- MLP-RBF exhibits the large positive fine-scale biases in the zero-H recipes.

Use the actual source table as authority.

---

# 8. What remains in SI

The main figure becomes denser again, so avoid restoring any additional evidence.

Keep in SI:
- complete five-recipe sensor-count sweeps;
- multi-recipe qualitative reconstruction gallery;
- any extra representative snapshots;
- per-snapshot wavelet distributions;
- full tables of sensor-count metrics and confidence intervals.

The main figure should show the breadth of the workload without becoming a catalogue.

---

# 9. Panel lettering and manuscript reference map

New main-figure panel roles:

- `a`: training resolutions and recipes
- `b`: recipe transfer + zero-H sensor sweeps
- `c`: full / zoom / local error physical reconstruction
- `d`: large/intermediate/fine qualitative components + fine-scale line plots
- `e`: full correlation/bias matrices

Update manuscript references accordingly.

Create `figure_reference_update_v3_3.md` mapping old V3-2 references to V3-3.

---

# 10. Data and provenance requirements

## No new model evaluation

Use only:
- existing validated reconstruction caches;
- validated sensor-sweep summary;
- validated all-recipe 512-sensor summary;
- validated multiscale-wavelet summary;
- validated representative-state data.

## MLP-RBF qualitative column

Before rendering panel c:
- verify exact same snapshot, recipe, sensor count, truth field and coordinate identity;
- verify the cached MLP-RBF reconstruction exists;
- record its cache path/hash in the manifest.

If not available, do not silently substitute a different snapshot or rerun inference. Mark the build incomplete and report the missing source.

## Versioning

Start from the exact local V3-2 renderer/assembler that produced:

`MixedResolution_unified_v3_2_hybrid_20260913_1739.pdf`

Create additive V3-3 scripts/output directories. Do not overwrite V3-2.

---

# 11. Required outputs

## Main
- V3-3 composed SVG
- V3-3 composed PDF
- high-resolution PNG
- standalone a–e SVG/PDF/PNG

## SI
- existing complete sweep
- multi-recipe qualitative gallery
- complete tables as LaTeX
- any extended multiscale distributions already available

## Documentation
- `source_manifest_v3_3.json`
- `qa_v3_3.json`
- `quantitative_figure_report_v3_3.md`
- `figure_reference_update_v3_3.md`
- `completion_report_v3_3.md`

---

# 12. V3-3 QA contract

Fail the build if any of the following occurs:

## Panel a
1. L/M/H thumbnails do not use the same ROI.
2. zoom connectors are missing or visibly cross into neighboring thumbnails.
3. the vertical L/M/H resolution block does not explicitly include 32×32 / 64×64 / 128×128.
4. the old sparse horizontal legend remains.

## Panel b
5. top transfer plot is still a wide line chart rather than the requested grouped categorical display.
6. the method legend remains cramped.
7. vertical spacing causes titles, y labels or x labels to collide.
8. zero-H groups are not visually identifiable.

## Panel c
9. MLP-RBF is absent despite a valid cached result.
10. columns are not exactly aligned.
11. the sensor-layout tile has a different axes box/aspect footprint than the local-error tiles.
12. magnification connectors are missing.
13. local-error labels still say `Local rel. L2`.
14. colorbar exponent text overlaps the bar outline/ticks.
15. any model uses a different crop, color scale or snapshot.

## Panel d
16. the Large scale is not restored.
17. the two fine-scale line plots are not stacked in the right-hand channel.
18. the qualitative scale block uses different representative evidence than panel c without documentation.

## Panel e
19. either full matrix is missing.
20. matrices are side-by-side and too compressed to read.
21. validated cell values do not match the source table.
22. Fine columns do not align with the fine-scale curves in panel d.

## Global
23. source CSVs/caches are modified.
24. the final figure exceeds the accepted typography/collision thresholds at manuscript width.
25. visual depth is restored by shrinking text rather than by intelligent spatial reallocation.

---

# 13. Success criterion

V3-3 should make this five-step argument visible without reading the caption:

> L/M/H correspond to genuinely different spatial discretizations. The training recipes progressively remove H-resolution information. DMF-Gen retains the strongest transfer to H resolution and remains robust across zero-H sensor budgets. In the decisive zero-H reconstruction, this advantage is visible in the magnified local flow structure and local error, including against both generative, attention-based and simple regression baselines. The complete large/intermediate/fine analysis and quantitative matrices then show that the local visual improvement is part of a systematic scale-resolved fidelity advantage rather than an isolated example.

If the figure conveys that argument but still appears like a dense catalogue, continue simplifying alignment, labels, legends and colorbars before promoting the release.
