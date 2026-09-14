# Mixed-resolution Figure V3-5: geometric harmonization and matrix-layout revision

## Purpose
Build **V3-5** from the current validated V3-4 figure. This round is a targeted layout correction, not a scientific redesign. Preserve all validated values, representative states, recipes, color limits, and method identities unless explicitly changed below.

Primary source figure:
- `MixedResolution_unified_v3_4_20260914_1040.pdf`

The current V3-4 already has the correct scientific evidence. The remaining defects are structural:
- panel a has an insufficient separation between the L/M/H field block and the training-recipe chart;
- panel b wastes vertical space above the top chart while compressing the legend/sweeps below;
- panels c and d use incompatible internal row geometries;
- panel d duplicates the fine-scale quantitative evidence already contained in panel e;
- panel e is too small and vertically stacked, despite containing the complete quantitative multiscale evidence.

V3-5 must therefore improve **spatial topology and cross-panel geometry**, not add new results.

---

# 1. Final panel roles
Keep five lettered panels:

- **a** resolution context + training recipes
- **b** high-resolution reconstruction performance + two zero-H sensor sweeps
- **c** full field / zoomed field / local-error reconstruction evidence
- **d** Large / Intermediate / Fine qualitative scale decomposition
- **e** complete quantitative multiscale matrices

Delete the two fine-scale line charts from the main figure. Their information is already encoded in panel e.

---

# 2. Recommended global layout

Use four main vertical bands:

```text
Row 1:  a  full width
Row 2:  b  full width
Row 3:  c (about 65%) | d (about 35%)
Row 4:  e  full width, two matrices side-by-side
```

Recommended outer GridSpec:

```python
outer = GridSpec(
    4, 20,
    height_ratios=[0.92, 1.62, 1.95, 1.18],
    hspace=0.22,
    wspace=0.24,
)

a = outer[0, :]
b = outer[1, :]
c = outer[2, :13]
d = outer[2, 13:]
e = outer[3, :]
```

Starting figure size:
- width: 183 mm
- height: approximately 205--220 mm

Do not enlarge height unless final-size QA shows that panel e still cannot be read cleanly.

---

# 3. Panel a — create a true breathing channel and stronger multiscale zooms

## 3.1 Fix the collision between the H field and `Training cases`
The current V3-4 places the right edge of the H heatmap too close to the bar-chart y-axis and its `Training cases` label.

### Required geometry
Use a nested horizontal GridSpec with an explicit **blank spacer column** between the two content types.

Recommended width ratios:

```python
panel_a = GridSpecFromSubplotSpec(
    1, 5,
    subplot_spec=a,
    width_ratios=[1.0, 1.0, 1.0, 0.42, 4.75],
    wspace=0.12,
)
```

- columns 0--2: L/M/H heatmaps
- column 3: intentionally blank breathing channel
- column 4: training-recipe bar chart

Do not place any legend/text in the spacer column.

### Target separation
At final 183-mm width, maintain a visible blank channel equivalent to roughly **5--7 mm** between:
- the outer edge of the H heatmap/inset system,
- and the bar-chart y-axis label/ticks.

Slightly compress the bar-chart width rather than shrinking the heatmaps.

## 3.2 Move the ROI to a more discriminative region
The current ROI does not reveal the strongest resolution difference.

Move the common ROI:
- approximately **10--15% of the parent width leftward** from the current position;
- approximately **8--12% upward**;
- center it on the steep red/blue transition where the field contains the strongest local gradient.

The exact crop must be identical in physical coordinates for L, M and H.

Do not independently optimize the ROI for each resolution.

## 3.3 Enlarge the magnified insets
Increase inset size substantially.

Target:
- inset width = approximately **45--52%** of parent heatmap width;
- inset height determined by the same physical ROI aspect ratio;
- allow the inset to extend **8--15% outside the parent tile** toward bottom-right if needed.

This controlled boundary breaking is allowed and preferred over covering the main gradient region.

### Alignment
All three parent heatmaps must:
- have identical axes dimensions;
- have identical vertical alignment;
- use identically positioned insets relative to their parent axes;
- use identical ROI and connector geometry.

### Pixel visibility
Use nearest-neighbor rendering in the inset so the 32x32 / 64x64 / 128x128 discretization contrast is unmistakable.

## 3.4 Preserve current resolution titles
Keep centered two-line titles:
- `Low resolution` / `32 × 32`
- `Medium resolution` / `64 × 64`
- `High resolution` / `128 × 128`

Do not reintroduce L/M/H side tags.

---

# 4. Panel b — pull the top chart upward and create a dedicated legend strip

## 4.1 Current problem
V3-4 leaves too much vertical negative space between panel a and the grouped-bar chart. The `b` tag is correctly high, but the top plot starts too low. This forces the legend and lower sweeps into a cramped area.

## 4.2 Use a 3-row internal GridSpec
Use:

```python
panel_b = GridSpecFromSubplotSpec(
    3, 2,
    subplot_spec=b,
    height_ratios=[1.00, 0.16, 0.88],
    hspace=0.34,
    wspace=0.20,
)
```

- row 0, columns 0:2: grouped-bar chart
- row 1, columns 0:2: dedicated legend-only axis
- row 2, col 0: Zero-H-balanced sweep
- row 2, col 1: Zero-H-M-rich sweep

Turn the legend-only axis off and place the legend in its center.

## 4.3 Pull the grouped-bar chart upward
Reduce the outer gap between panels a and b.

The top edge of the grouped-bar plotting box should sit visually close to the `b` tag baseline; there should not be a large empty strip above it.

Target:
- top-axis bounding box begins within approximately **2--4 mm** below the `b` tag at final print size.

Do not move the `b` tag downward to solve this.

## 4.4 Preserve title placement
Keep:
- `High resolution reconstruction (512 sensors)`
inside the top-left negative space of the grouped-bar axes.

The title must not overlap the 0.20 tick or y-axis label.

## 4.5 Legend placement
Move the legend entirely into the dedicated middle strip.

Requirements:
- single row, 4 methods;
- centered horizontally;
- font slightly smaller than V3-4 if necessary;
- preserve comfortable `handletextpad` and `columnspacing`;
- no overlap with either the grouped bars or lower sweep titles.

## 4.6 Zero-H annotation
Keep `no H-resolution training fields`, but place it inside the shaded zero-H region at the upper-right portion of the grouped-bar axes where it cannot collide with bars or error bars.

---

# 5. Panels c and d — force exact 3-row geometric alignment

## 5.1 Core requirement
The three visual rows in panels c and d must have **the same physical height and the same vertical boundaries**.

Map rows as:

| Shared row | Panel c | Panel d |
|---|---|---|
| 1 | Full field | Large |
| 2 | Zoomed field | Intermediate |
| 3 | Local error | Fine |

This must be enforced geometrically, not approximated by independent `hspace` tuning.

## 5.2 Use one shared parent GridSpec for c+d image rows
Create one shared lower-image GridSpec across both panels, e.g.:

```python
visual_cd = GridSpecFromSubplotSpec(
    3, 9,
    subplot_spec=outer[2, :],
    height_ratios=[1, 1, 1],
    width_ratios=[1,1,1,1,1, 0.28, 1,1,1],
    hspace=0.08,
    wspace=0.12,
)
```

Suggested allocation:
- panel c: columns 0--4
- spacer between c/d: column 5
- panel d: columns 6--8

This shared grid should replace separate independent c and d row layouts.

The outer panel letters `c` and `d` can still be positioned independently at the upper-left of their respective groups.

## 5.3 Panel c: tighten interstitial whitespace
The current V3-4 has excessive horizontal and vertical padding, making the five heatmaps too small.

Requirements:
- use the shared equal-height rows above;
- reduce c-column `wspace` until the five columns read as one matrix while retaining clear separation;
- target intra-column gap approximately **1.5--2.5 mm** at print width;
- target row gap approximately **1.5--2.5 mm**;
- enlarge the heatmap tiles to consume the recovered space.

### Colorbars
Move panel-c colorbars into a compact dedicated strip **below the shared c/d visual row**, not by increasing the c row heights.

Reduce the blank gap above the colorbars.

The colorbar strip should not affect row alignment with panel d.

## 5.4 Panel d: slightly shrink image tiles to match panel c
Panel d tiles are currently too large relative to panel c.

Use the shared grid so the rows align exactly.

Within each d tile:
- preserve aspect ratio;
- center the image within the common cell;
- do not exceed the panel-c row height.

Panel d should no longer feel vertically compressed once the line plots are removed.

## 5.5 Remove the arrow/funnel from panel d
Delete the floating arrow/connector pointing from Fine images toward quantitative charts.

No replacement connector is needed.

The grid structure itself should provide the visual logic.

---

# 6. Delete the fine-scale line plots from panel d

Remove entirely from the main figure:
- `Fine-scale pattern correlation` line plot
- `Fine-scale variance-allocation bias` line plot

Reason:
- these values are already fully encoded in panel e;
- retaining both representations creates visual redundancy;
- their removal allows panel e to become the primary quantitative multiscale evidence.

Do not delete the underlying data or SI versions.

Panel d after this change contains only the 3x3 qualitative Large/Intermediate/Fine block.

---

# 7. Panel e — full-width horizontal 1x2 matrix layout

## 7.1 Promote panel e to full width
Panel e should now span the full bottom row of the figure.

Use:

```python
e = outer[3, :]
```

Reclaim the former line-plot space for larger matrices.

## 7.2 Place the two metric matrices side-by-side
Use a 1x2 outer matrix layout:

```text
[ Spatial pattern correlation ]   [ Variance allocation bias ]
```

Recommended width ratio:
- `1 : 1`

Use a moderate central gap between the two metric blocks.

## 7.3 Segment each metric matrix by recipe using three sub-axes
Do not render each metric as one continuous 4x9 image.

For each metric, create an inner 1x3 GridSpec:
- Mixed-HML
- Zero-H-balanced
- Zero-H-M-rich

Each sub-axis is a 4x3 heatmap:
- Large
- Interm.
- Fine

Recommended:
```python
metric_grid = GridSpecFromSubplotSpec(
    1, 3,
    subplot_spec=metric_slot,
    wspace=0.10,
)
```

This produces a real visual gap between recipe blocks instead of fake separator lines.

## 7.4 Y-axis labels
### Left metric: Spatial pattern correlation
Show method labels only on the leftmost recipe sub-axis:
- DMF-Gen
- FFM-Perceiver
- Senseiver
- MLP-RBF

Hide y labels on the other two recipe sub-axes.

### Right metric: Variance allocation bias
Hide method labels on **all three** recipe sub-axes because they duplicate the left metric.

Keep row alignment exact across both metric blocks.

## 7.5 Recipe titles and scale labels
For each metric:
- title each sub-axis with its recipe name;
- x labels `Large`, `Interm.`, `Fine`;
- keep typography identical between left and right metric blocks.

## 7.6 Numeric annotations
Retain the validated cell values.

Ensure annotation font size increases relative to V3-4 due to the larger cells.

Formatting:
- correlation: 2 decimals
- bias: signed 2 decimals where appropriate

## 7.7 Colorbar design — important scientific correction
Do **not** use one literal numeric colorbar for both matrices.

The two metrics have different meanings and numeric ranges:
- correlation: approximately `[-0.1, 1.0]`
- variance-allocation bias: approximately `[-1.4, 1.4] pp`

A single shared numeric color scale would be scientifically incorrect.

### Required compromise: one unified colorbar band
Create a single visual colorbar **band** spanning the full bottom of panel e, containing two aligned horizontal colorbars:

```text
[ correlation colorbar ---------------- ]   [ bias colorbar ---------------- ]
```

Requirements:
- both colorbars share the same vertical position and height;
- their total width visually anchors panel e across the page;
- left bar labeled `Correlation`;
- right bar labeled `Bias (pp)`;
- no redundant vertical colorbars beside the matrices.

This satisfies the desired clean horizontal anchoring without conflating two different metrics.

---

# 8. Cross-panel final alignment targets

## a/b boundary
- reduce the a-to-b vertical gap relative to V3-4;
- panel b top chart should visually start near its panel tag.

## c/d rows
- exact shared y boundaries for all three image rows;
- no independent row heights.

## c colorbars vs e
- keep c colorbars compact so they do not push panel e downward unnecessarily.

## panel e
- use the entire full-width bottom band and enlarge matrix numbers until comfortably readable at final journal width.

---

# 9. Preserve scientific sources

No new inference or metric computation.

Keep:
- same representative snapshot;
- same Zero-H-M-rich qualitative recipe;
- same MLP-RBF cache;
- same 512-sensor bar values and confidence intervals;
- same zero-H sensor sweeps;
- same Large/Intermediate/Fine wavelet components and residuals;
- same correlation/bias matrix values.

Do not modify validated source CSVs or reconstruction caches.

---

# 10. Deliverables
Create additive V3-5 outputs:

## Main
- composed SVG
- composed PDF
- high-resolution PNG
- standalone a--e SVG/PDF/PNG

## Documentation
- `source_manifest_v3_5.json`
- `qa_v3_5.json`
- `quantitative_figure_report_v3_5.md`
- `figure_reference_update_v3_5.md`
- `completion_report_v3_5.md`

The report should explicitly state that V3-5 removes redundant fine-scale line plots and promotes the complete matrices to the primary quantitative multiscale summary.

---

# 11. V3-5 QA failure conditions

Fail the build if any of the following occur.

## Panel a
1. `Training cases` still visually collides with the H-resolution heatmap/inset.
2. no explicit blank spacer exists between field thumbnails and bar chart.
3. ROI is not identical across L/M/H.
4. ROI is not moved toward the high-gradient transition.
5. inset size remains too small to reveal resolution differences.
6. inset positions differ between L/M/H or destroy horizontal alignment.

## Panel b
7. the top grouped-bar plot still sits substantially below the `b` tag.
8. legend is not in a dedicated middle strip.
9. legend crowds either the top chart or lower sweep titles.
10. `no H-resolution training fields` overlaps bars/error bars.

## Panels c/d
11. c and d use separate independent row geometries.
12. Full/Large, Zoomed/Intermediate, Local-error/Fine rows do not share exact y boundaries.
13. panel c retains large unused gaps around tiles or above colorbars.
14. panel d tiles remain visibly larger than panel c tiles.
15. the fine-scale arrow/funnel remains.
16. the two fine-scale line plots remain in the main figure.

## Panel e
17. matrices remain vertically stacked.
18. panel e does not span full width.
19. recipe groups are not separated by real horizontal gaps.
20. right-hand metric repeats method labels.
21. matrix numbers remain difficult to read at final width.
22. one numeric colorbar is incorrectly used for both correlation and bias.
23. vertical colorbars remain beside matrices instead of the unified bottom colorbar band.

## Global
24. source data are modified or recomputed.
25. the final figure gains space by shrinking text below the accepted journal-size typography.

---

# 12. Success criterion
At manuscript width, V3-5 should read as a single engineered composition:

- panel a has a clear physical separation between discretization examples and training-budget design, and the enlarged high-gradient insets visibly distinguish L/M/H;
- panel b starts immediately under its tag, with categorical performance above and sensor-density behavior below;
- panels c and d form one mathematically aligned 3-row visual band;
- panel d supplies qualitative scale evidence only, without duplicated fine-scale charts;
- panel e becomes the full-width quantitative conclusion of the scale analysis, with two large side-by-side matrices and a clean dual-colorbar bottom band.

If the visual row alignment between c and d is not immediately obvious, do not promote the release.
