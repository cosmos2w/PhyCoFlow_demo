# Mixed-resolution Figure V3-4 typographic and spatial refinement brief for Codex

## Objective
Refine the current **V3-3** mixed-resolution figure into a **V3-4 publication-polish version**. This round is **not** a scientific redesign. The data integration and panel logic are already in place. V3-4 must therefore focus on **typographic de-collision, panel spacing, visual hierarchy, inset quality, and matrix readability**, while preserving the validated quantitative content and the established five-panel narrative.

The current V3-3 has the right information, but several local layout and styling choices still keep it below publication standard. This revision should make the figure look intentional, calm, and editorially polished without changing the scientific claims.

Use the current V3-3 figure as the primary baseline:
- `/mnt/data/MixedResolution_unified_v3_3_20260913_2352.pdf`

For the panel-a inset treatment, use the attached reference image as the style target:
- `/mnt/data/ghostwriter_images/context/b6f81bed-6b96-5470-8555-f947734f6ea7.png`

No retraining, new inference, or metric recomputation is requested.

---

# 1. Preserve the established scientific structure
Maintain the five main panels and their scientific roles:

- **a** resolution context + training recipes
- **b** high-resolution reconstruction performance at 512 sensors + zero-H sweeps
- **c** full / zoom / local-error physical reconstruction
- **d** large/intermediate/fine qualitative evidence + fine-scale trend plots
- **e** full correlation and bias matrices

Do not add or remove panels. Do not alter the underlying data selection unless necessary for exact alignment with the existing validated V3-3 sources.

---

# 2. Panel a — inset restoration and margin control

## 2.1 Revert strictly to the old inset style
The current magnified insets are poorly resolved and too detached. Rebuild them to match the attached reference style as closely as possible.

### Required inset geometry
For each of the three heatmaps:
- place the magnified inset **inside the bottom-right corner** of the parent heatmap;
- keep the inset visually nested within the parent tile rather than floating outside it;
- use a black border on the ROI box and on the inset box;
- connect ROI to inset with two clean diagonal connector lines;
- keep the inset large enough to show discretization differences clearly, but small enough not to dominate the parent tile.

### Required visual style
- connectors and boxes must be solid black or near-black;
- connector line width should match the ROI border or be slightly lighter;
- inset fill should preserve the same data colormap as the parent heatmap;
- the pixel/coarsening character must be visible, especially for 32×32 and 64×64.

### Quality requirement
The final inset must look visually close to the reference image, especially in:
- nesting position,
- crispness,
- balance between parent field and magnified crop.

## 2.2 Remove the side L / M / H tags
Remove any left-side or free-floating `L`, `M`, `H` tags associated with the heatmaps.

## 2.3 Evenly distribute the three heatmaps horizontally
The three heatmaps should form a clean, evenly spaced row.

### Required labeling
Each heatmap must have **two centered text lines**, either above or below the image:
- line 1: descriptive label
  - `Low resolution`
  - `Medium resolution`
  - `High resolution`
- line 2: grid size
  - `$32 \times 32$`
  - `$64 \times 64$`
  - `$128 \times 128$`

Use identical alignment and spacing for all three.

### Preferred placement
Place the two lines **above** each heatmap, matching the uploaded reference image’s general feel, unless this causes crowding with panel-a letter placement.

## 2.4 Remove the redundant exposure explanatory label
Delete the text:
- `Values above bars: relative spatial-field exposure`

This text is unnecessary once the exposure values remain printed directly above the bars.

## 2.5 Collapse excess vertical whitespace inside panel a
After removing that text, reduce the vertical space between:
- the heatmap/inset row,
- and the training-recipe bar-chart row.

The objective is to remove the current orphaned gap and make panel a read as one compact setup panel.

### Suggested implementation
- reduce inner `hspace` within panel a;
- if needed, slightly shift the bar chart upward;
- do not compress so much that labels collide.

## 2.6 Keep the recipe bars and grouping
Retain:
- H-only
- H-limited
- Mixed-HML
- Zero-H-balanced
- Zero-H-M-rich
- relative exposure values above bars
- grouping labels for `contains H-resolution training fields` and `zero-H training`

Do not otherwise redesign the recipe bar section.

---

# 3. Panel b — typographic de-collision and legend anchoring

## 3.1 Reduce legend dominance
The method legend is currently too visually dominant.

### Requirements
- decrease legend font size modestly;
- preserve or slightly increase internal spacing between marker and text;
- ensure the legend no longer competes with the plots themselves.

The goal is **lighter**, not cramped.

## 3.2 Move the legend to the horizontal gap between the top and bottom plot blocks
The legend should not sit in the top-right corner.

### Required placement
Anchor the legend in the horizontal negative space **between**:
- the top grouped-bar chart,
- and the two zero-H line charts below.

This should make the panel feel more integrated and free the upper margin.

### Formatting
- one row,
- 4 entries,
- centered or slightly left-of-center depending on the available gap,
- no overlap with any axes.

## 3.3 Rename and reposition the top-axis title
Rename the top grouped-bar section title to:
- `High resolution reconstruction (512 sensors)`

Move this title **inside the top grouped-bar chart bounding box**, in the empty top-left negative space.

### Requirements
- do not place this title above the axis;
- keep it visually separated from the y-axis tick labels and bars;
- use moderate font weight.

This change should also allow the whole panel-b block to shift upward slightly.

## 3.4 Resolve the `no H-resolution training fields` overlap
The annotation on the right side of the grouped-bar chart is colliding with the bars.

### Required fix
Reposition the annotation so that:
- it still clearly labels the shaded zero-H region,
- it does not overlap any bar, error bar, or bar annotation,
- it remains visually associated with the right-side shaded region.

Acceptable options include:
- moving it higher,
- moving it into the upper-right corner of the shaded area,
- placing it as a small centered label near the top of the shaded region.

## 3.5 Increase vertical pacing between the three subplots
Increase `hspace` inside panel b so that:
- y-axis labels breathe,
- sweep titles are not cramped,
- the legend can sit comfortably between the top and bottom plot blocks.

This should improve readability without making the panel unnecessarily tall.

---

# 4. Panel c — visual consistency and frustum-line refinement

## 4.1 Remove redundant error labels from the first row
The `Rel. $L_2 = ...$` labels currently appear in both the first and third rows.

### Required change
Keep these labels **only** on the third row (`Local error`).
Remove them from the first row (`Full field`).

This is mandatory.

## 4.2 Convert the frustum/magnification connectors to dashed black lines
The current solid grey lines are too heavy and inconsistent with the ROI boxes.

### Required style
- dashed lines,
- black or near-black,
- line weight matched to the black bounding boxes,
- visually subtle but clearly present.

The dashed connectors should run from the Row-1 ROI box to the Row-2 zoomed field, column by column.

## 4.3 Add black borders around every zoomed-field tile
Every heatmap in the `Zoomed field` row must have an explicit black border.

### Requirements
- border width consistent across all columns,
- same visual weight as the ROI box,
- border should visually define Row 2 as the magnified counterpart of Row 1.

## 4.4 Keep the sensor-layout tile matched to the adjacent heatmaps
Preserve the current requirement that the `Sensor layout` tile has the same axes footprint as the local-error heatmaps.

Verify again after panel-c compression.

## 4.5 Standardize row-3 metric text
Use only:
- `Rel. $L_2 = ...$`

Do not use `Local rel. $L_2$`.

## 4.6 Fix colorbar exponent clipping
The scientific-notation exponent text is currently colliding with ticks / colorbar edges.

### Required fixes
- give the colorbar axes more right-side padding;
- hide automatic offset text if necessary and place the exponent manually;
- ensure `×10^{-2}` / `×10^{-3}` or equivalent notation does not touch the final tick or border.

### QA target
This must be verified in the final PDF/SVG, not only the screen PNG.

---

# 5. Panel d — distinct visual encoding and clearer flow to fine-scale charts

## 5.1 Differentiate the Ground-truth component from panel-c Ground truth
The `Truth component` column in panel d currently looks too similar to panel c’s `Ground truth`, causing visual fatigue.

### Required change
Make the `Truth component` column visually distinct by one of the following approved methods:

#### Preferred option A
Use a clearly different diverging or spectral colormap from panel c.

#### Acceptable option B
Retain a similar colormap but overlay contour lines, so the component reads as an analytical decomposition rather than another reconstruction field.

#### Acceptable option C
Use a grayscale or desaturated map if and only if the component remains legible.

The chosen treatment must distinguish panel d from panel c while preserving interpretability.

## 5.2 Anchor the fine-scale narrative visually
The connection between the `Fine` row and the fine-scale line plots should be more explicit.

### Required structure
- draw a subtle bounding box around the entire `Fine` row of qualitative tiles;
- create a minimalist connector from that row to the two fine-scale line charts below.

### Connector options
Acceptable options include:
- a light arrow,
- a funnel-like shaded connector,
- a bracket + arrow combination.

Requirements:
- must be subtle;
- must not dominate the panel;
- must clearly suggest `these charts quantify the Fine row`.

## 5.3 Slightly expand the vertical spacing inside panel d
Panel d currently feels cramped compared with panel c.

Increase vertical `hspace` modestly so that:
- scale rows breathe,
- the line charts are not pressed into the qualitative block,
- the fine-row connector remains legible.

Do not over-expand; the right-hand channel is already narrow.

---

# 6. Cross-panel alignment between c and d

## 6.1 Compress panel c vertically
Currently panel c is slightly too loose.

Reduce vertical `hspace` between:
- Full field,
- Zoomed field,
- Local error.

Do not compress so far that row labels or colorbars collide.

## 6.2 Expand panel d slightly
As noted above, panel d should breathe slightly more.

## 6.3 Alignment objective
After these adjustments, the image rows in panels c and d should align **as closely as possible horizontally**, taking into account that panel c includes colorbars below.

This alignment is important because it makes the lower half of the figure feel architected rather than assembled.

## 6.4 Use saved vertical space to strengthen panel e
Any vertical space recovered from panel-c compression should be transferred primarily to panel e.

---

# 7. Panel e — matrix segmentation and expansion

## 7.1 Introduce visible separation between the three recipe blocks
The three conditions
- Mixed-HML
- Zero-H-balanced
- Zero-H-M-rich

should read as distinct blocks, not one continuous matrix.

### Required change
Add a slight horizontal gap (`wspace`-like visual separation) between these three recipe groups in each matrix.

Possible implementation strategies:
- insert thin blank spacer columns,
- increase inter-group column spacing,
- draw slightly larger separators than the within-group scale separators.

The separation should be visible but not heavy.

## 7.2 Expand panel e
Use the vertical space saved from the upper panels to increase the size of panel e.

### Priority
Make the numerical annotations significantly easier to read.

### Requirements
- enlarge cell area enough that values no longer feel cramped;
- retain stacked matrices, not side-by-side matrices;
- preserve existing validated values and group labeling.

## 7.3 Preserve matrix readability hierarchy
Ensure that after expansion:
- method labels remain clear;
- recipe labels remain clearly grouped;
- Large / Interm. / Fine headers are legible;
- cell annotations remain centered and uncluttered.

---

# 8. Global typographic and spatial requirements

## 8.1 Do not change the scientific content
No new evidence should be introduced. V3-4 is a publication-polish revision.

## 8.2 Maintain editorial calm
Wherever possible, improve readability by:
- moving labels into negative space,
- increasing local padding,
- reducing clutter,
- differentiating repeated visual motifs,
not by shrinking fonts.

## 8.3 Final figure should look balanced at manuscript width
Perform final QA at the intended manuscript print width.

---

# 9. Deliverables
Create an additive **V3-4** release with:

## Main outputs
- composed V3-4 figure in SVG
- composed V3-4 figure in PDF
- high-resolution PNG
- standalone panels a–e in SVG/PDF/PNG

## Documentation
- `source_manifest_v3_4.json`
- `qa_v3_4.json`
- `quantitative_figure_report_v3_4.md`
- `figure_reference_update_v3_4.md`
- `completion_report_v3_4.md`

The quantitative report for V3-4 should emphasize that this round changed layout/typography/panel polish rather than scientific content.

---

# 10. QA contract for V3-4
The build fails if any of the following remain true:

## Panel a
1. Insets are still external/floating rather than nested in the bottom-right corner.
2. The inset style does not visually approximate the provided reference image.
3. Side `L/M/H` tags remain.
4. The three heatmaps are not evenly spaced.
5. The two-line descriptive labels are missing or misaligned.
6. The `Values above bars: relative spatial-field exposure` label is still present.
7. A visible orphaned vertical gap remains between the heatmap row and bar-chart row.

## Panel b
8. The legend still sits in the top-right corner.
9. The legend is still too visually dominant.
10. The top grouped-bar title has not been renamed to `High resolution reconstruction (512 sensors)`.
11. The title is not placed inside the top-axis negative space.
12. The `no H-resolution training fields` annotation still overlaps bars.
13. Vertical spacing between the top and lower plots remains cramped.

## Panel c
14. `Rel. $L_2$` labels still appear on the first row.
15. Frustum connectors are still solid grey rather than dashed black.
16. Zoomed-field tiles do not all have black borders.
17. Sensor-layout tile footprint differs from adjacent local-error tiles.
18. Colorbar exponent text still collides with ticks or borders.

## Panel d
19. `Truth component` is still visually too similar to panel-c `Ground truth`.
20. No explicit visual connector exists between the Fine row and the fine-scale charts.
21. Panel d remains more cramped than panel c without justification.

## Cross-panel / panel e
22. Panel-c and panel-d row alignment has not improved materially.
23. No visible separation exists between the three recipe groups in panel e.
24. Panel e has not been enlarged enough to materially improve number readability.
25. The final figure achieves polish by shrinking text instead of improving layout.

---

# 11. Success criterion
V3-4 should look like the same validated figure, but with the following editorial improvements immediately apparent:

- panel a has crisp, reference-style nested insets and cleaner resolution labeling;
- panel b reads cleanly, with an anchored legend and no title/annotation collisions;
- panel c is visually calmer and more consistent;
- panel d has a clearer analytical identity and visibly feeds into the fine-scale metrics;
- panel e is easier to parse because the recipe blocks are separated and the numbers are larger.

If the figure still feels crowded after these changes, continue improving spacing and local hierarchy rather than removing information.
