# Figure 3 — mixed-resolution training and high-resolution reconstruction

**Manuscript asset:** `Figure_MixedResolution.pdf`; D p. 9; LaTeX label `fig:mixed_resolution`.

**Scientific producer:** `1_SubTask_SuperResolution/Save_TrainedModel/_TrainedModels/_Process_Figures/Assembled/MixedResolution_unified_v3_7_20260914_1158.pdf`.

**V4_7 visual baseline:** `1_SubTask_SuperResolution/Save_TrainedModel/_TrainedModels/_Process_Figures/Assembled/MixedResolution_unified_v4_6_20260914_2227.pdf`.

**Latest art export:** `1_SubTask_SuperResolution/Save_TrainedModel/_TrainedModels/_Process_Figures/Assembled/MixedResolution_unified_v4_7_20260914_2318.pdf`.

Read the general contract and author checks A03/A04 first. The active V3-7 figure has **five panels, a–e**. The older six-panel a–f inventory, ten-column multi-recipe gallery and five-sweep main-figure layout are superseded and must not be restored during this art-only pass.

## Freeze the V4_4 visual inventory and V3-7 scientific state

Panel a contains the native 32×32, 64×64 and 128×128 representations with matched zoom connectors, followed by five stacked training recipes and their relative-exposure annotations. Panel b contains the four-method × five-recipe 512-sensor comparison above the three V4_4-carried sensor-count sweeps Mixed-HML, Zero-H-balanced and Zero-H-M-rich. Panel c contains five columns—Ground truth, DMF-Gen, FFM-Perceiver, Senseiver and MLP-RBF—with full fields, matched zooms and local-error tiles; the ground-truth lower tile contains the sensor layout. Panel d contains three scale rows—Large, Intermediate and Fine—across Truth component, DMF-Gen residual and Senseiver residual. Panel e contains two 4×9 numerical matrix blocks: spatial-pattern correlation and variance-allocation bias, each arranged as three recipes × three scales.

Keep every displayed model, recipe, sensor count, cell, number, interval, line, map and scale component. Preserve the five panel letters and their order. V4_5 carries forward V4_4's author-authorized same-size panel-a ROI translation, below-map resolution captions, five hidden redundant DMF-Gen bar annotations and restored saved Mixed-HML sweep; none is a new V4_5 scientific change. Do not move other historical sweeps, the expanded recipe gallery, fine-scale line charts or distribution summaries back from the SI; conversely, do not remove any evidence currently present in a–e. The active panel-e design intentionally has metric titles and no numerical colourbars. Do not add or remove a colourbar without a separately approved content change.

## V4_5 typography and compact geometry

Use the manuscript hierarchy literally at the 180-mm design width: panel
letters 11 pt bold; major group titles 9.5 pt regular; axis, colourbar and
ordinary subplot titles 8.5 pt regular; ticks, legends and method labels 7.8 pt
regular; and in-plot/matrix annotations 7.0 pt regular. At 162 mm these resolve
to 9.9, 8.55, 7.65, 7.02 and 6.3 pt. Only panel letters may be bold. Do not use
local size overrides in V4_5; any future reduction below 7.0 pt requires an
individual QA record and may not fall below 6.6 pt at design width.

V4_5 may shorten the canvas only by removing excess visible whitespace between
panel b and the shared c/d row. Translate a and b together so their mutual gap
and every panel rectangle, aspect ratio and internal layout remain unchanged.
Retain c/d row alignment, the V4_4 c/d-to-e gap and the V4_3 panel-e tick
geometry. Final renderer measurements at both 180 and 162 mm are authoritative.

The V4_5 b-to-c/d compaction is superseded because its previous content-extent
check did not include the complete rendered union of Panel b's multilevel
x-axis labels and the c/d titles/tags. Every later revision must apply the
general adjacent-major-row union-bbox gate to `a→b`, `b→c/d` and `c/d→e` and
must fail before export if any union intersects or scales below 3 mm clearance
at 162-mm insertion width.

V4_6 is the current corrected art baseline. It uses a 180 × 229.5-mm canvas,
retains the global typography hierarchy, assigns the requested Panel-a and
Panel-b headings to the 8.5-pt subplot-title role, enlarges the Panel-b grouped
bar axis to exactly 1.25 times its V4_5 physical height, and restores a clear
semantic boundary between Panel b and c/d. Its shared c/d row gap is 3 mm and
its 18-mm Panel-c colourbar strip retains both colourbars. These are visual
geometry and wording changes only; V4_5 arrays and mappings remain frozen.

V4_7 is the active art export. It removes Panel c's two numerical colourbars,
preserves all map arrays and normalization values in the source manifest, and uses the freed
band for an external annotation gutter. The five Panel-c bottom labels
(`Sensor layout` and the four frozen `Rel. L2` values) must share the exact
bottom-row baseline used by Panel d. Panel-d residual metrics must clear both
their owning image and the next image row. Apply the general final-aspect
pairwise text/evidence gate; a label merely moved before equal-aspect axes
settle is not compliant. Panel e may move upward, and the reclaimed vertical
space may enlarge Panel b's lower sweep windows without changing axes limits,
ticks, values or mappings.

## Main art issues

V3-7 has a coherent asymmetric evidence sequence, but most ordinary text is approximately 5.5–6.5 pt in the 183-mm PDF and therefore falls below the 7-pt floor after insertion at 162 mm. The major row gap is currently about 2 mm, several title/label clearances are tight, and panel c/d numerical labels sit directly over dense raster evidence. Increase typographic scale and measured clearance without changing arrays, crop geometry, axes, normalization or inventory.

The present model palette uses the older saturated red/blue/gray/teal family. Apply the general shared model palette consistently in b and c. Panel-a recipe components and panel-e metric colormaps encode different semantics and must not inherit model identity colours.

## a — resolutions and training recipes

Keep the native 32×32, 64×64 and 128×128 images, their visible cells, exact shared ROI and connector geometry. Preserve the renderer's interpolation mode. The discretization differences are scientific evidence: do not smooth, resample, sharpen or regenerate any map or inset.

Align the three resolution titles, parent maps, zoom boxes and insets on common baselines. Maintain a real blank spacer between the resolution block and the training-case chart, and verify that the vertical Training cases label clears the nearest inset. Thin zoom/connector strokes only uniformly and retain their paths.

Keep L/M/H stack order, all recipe totals, and the exposure annotations 1.00×, 0.34×, 0.44×, 0.16× and 0.19× exactly as supplied. Use the existing quiet ordered resolution-component family rather than the global model palette. Preserve contains-H and zero-H background groups and wrap Zero-H-balanced and Zero-H-M-rich consistently without changing their wording.

## b — 512-sensor comparison and three carried sweeps

The upper axis is a grouped **bar chart** containing four methods across five recipes. Retain all 20 bars, their values, intervals, category offsets, order and existing numerical annotations. Do not convert bars to points, change the linear y-axis, reorder recipes or tighten the axis to exaggerate the zero-H comparison.

Keep the lower three log-scale sweeps in the V4_4 order: Mixed-HML, Zero-H-balanced and Zero-H-M-rich. Preserve all four method traces, the five sensor counts, secondary H-grid-density labels, interval endpoints, line styles and marker coordinates (12 traces and 60 plotted sweep rows total). Do not restore any other superseded sweep axes to the main figure.

Use the global method palette and one shared, external legend whose handles exactly match the plotted lines and markers. Retain a dedicated legend strip at the 7.8-pt role. Keep all three sweep plot windows equal and align their two-level x labels and common semantic axis wording.

## c — five-column physical proof

Keep the fixed Zero-H-M-rich representative state, 512-sensor plan and five columns in their current order: Ground truth, DMF-Gen, FFM-Perceiver, Senseiver and MLP-RBF. Preserve the same full-field extent, matched zoom bounds, sensor positions, mask, contour levels, contour paths and display interpolation in every column.

Use one signed physical-field mapping across truth and predictions, retaining its exact source normalization. Use the V4_4 `magma` nonnegative absolute-error mapping across the four local-error tiles, again retaining the source normalization and mask. Fine lattice, checkerboard and striping structures in baseline predictions/errors are evidence, not rendering defects.

Keep the current local relative-L2 annotations and precision attached to their exact tiles. V4_7 requires a common gutter below the local-error row, including the `Sensor layout` label in the ground-truth column. If an annotation must remain in-data, register it in `LAYOUT_QA.json` and prove that it does not obscure a hotspot, sensor, contour or boundary. A03 governs the full-field versus zoom/local metric wording; do not reinterpret it during styling.

Align the five full-field, zoom and local-error rows precisely. Preserve the sensor-layout tile and its observed locations. V4_7 removes both Panel-c colourbars by explicit author authorization; retain their field/error normalization values in the manifest and do not rescale any map.

## d — multiscale qualitative components

Keep all three rows (Large, Intermediate, Fine) and all three columns (Truth component, DMF-Gen residual, Senseiver residual). Preserve V4_4's `Greys_r` truth-component mapping and `magma` absolute-residual mapping with `vmin=0`, together with the realized arrays, scale-specific positive limits, contour configuration and current row-to-value associations. Do not impose a shared range across scales or introduce a new zero-centred normalization.

Keep the current relative-L2 values 0.02/0.21/0.68 and 0.05/0.38/4.81 at their exact associated tiles. Move them to aligned gutters where feasible; otherwise apply the registered in-data-annotation gate. The large fine-scale Senseiver residual must remain visually honest and must not be clipped, blurred or given a private range.

Align d's three row boundaries exactly with c's full/zoom/error row grid, as in the active shared-parent layout. Increase the gap between c and d only by reallocating whitespace; do not distort tile aspect ratios.

## e — correlation and variance-allocation matrices

Retain both 4×9 matrices, all values, row/column order, three recipe groups and Large/Intermediate/Fine scale labels. Preserve the original normalization, lookup direction and signed values. Do not clip the negative correlation, normalize rows/columns, multiply values by 100, change signs or add best-cell markers.

Keep spatial-pattern correlation and variance-allocation bias visually distinct. Use adaptive black/white annotation text based on final background contrast. Maintain subtle real gaps between recipe groups and a larger central gap between the two metrics; avoid heavy cell borders.

The active V3-7 panel deliberately uses metric titles without numerical colourbars. Preserve this inventory. The title `Variance allocation bias [pp]` remains scientifically pending under A04: restyle the existing text but do not change its unit or numerical values without author approval.

## Acceptance and release gates

Confirm a–e; three native resolution maps; five recipe bars; 20 upper comparison bars; three complete five-count sensor sweeps; five panel-c columns; three panel-d rows × three columns; and both 4×9 panel-e matrices. Verify source arrays, category membership/order, sensor coordinates, ROI/zoom indices, interpolation, contour paths, interval endpoints, axes and normalization against the V4_4 visual manifest and V3-7 scientific source manifest.

At 162 mm, every ordinary label, legend entry and tick must meet the general 7-pt floor; matrix and in-plot annotations use the authorized 6.3-pt insertion tier. Verify a–e; three native resolution maps; five recipe bars; 20 upper comparison bars; all three complete five-count sensor sweeps; five panel-c columns; three panel-d rows × three columns; and both 4×9 panel-e matrices. Run the mandatory overlap, clipping, horizontal/vertical clearance, semantic-palette, grayscale and colour-vision-deficiency checks and save them in `LAYOUT_QA.json`. A03/A04 may remain unresolved during independent art work, but the final asset must be labelled **art reviewed, scientific release pending** until their approved mappings are recorded.
