# Figure revision guide: mixed-resolution training and zero-shot super-resolution

## Scope and editing principle

Preserve the existing six-panel composition, panel order, model set, training recipes, and quantitative content. The present evidence hierarchy is already effective:

- **a** defines the native resolutions and training-data recipes;
- **b** gives the headline 512-sensor population result;
- **c** localizes reconstruction errors in a matched field;
- **d** tests robustness across sensor density;
- **e--f** determine whether global accuracy extends to intermediate and fine spatial scales.

The edits below are primarily notation, labeling, visibility, and statistical-communication changes. Do not change numerical values or select a different representative field unless the source manifest is deliberately regenerated.

## Global revisions

1. **Standardize terminology across all panels.** Use exactly:
   - `H-only`
   - `H-limited`
   - `Mixed-HML`
   - `Zero-H-balanced`
   - `Zero-H-M-rich`
   - `DMF-Gen`
   - `FFM-Perceiver`
   - `Senseiver`
   - `MLP-RBF`

2. **Use one mathematical form for the error label.** Prefer `Physical relative $L_2$` or `$E_{\mathrm{rel}L_2}$` consistently in panels b and d. Render the 2 as a subscript.

3. **Retain model identity without relying on colour alone.** In panel d, add distinct marker shapes and/or line styles for the four models, while retaining the current colours. Keep the same model order in every legend and panel. Check the final figure with a colour-vision-deficiency preview and in grayscale.

4. **Align typography and panel labels.** Use the same panel-letter position, title weight, axis font size, tick font size, and line width throughout. Ensure that all text remains legible at the intended 183-mm final width.

5. **Do not imply formal discretization invariance.** Use `zero-shot H-resolution reconstruction`, `variable resolution`, or `unseen H-resolution field evaluations`. Avoid `arbitrary resolution` in this Results figure.

6. **Keep all field and error normalizations shared where the comparison requires them.** Do not introduce model-specific contrast adjustments. Display clipping may remain visual only and must not affect metrics.

## Panel a: native resolutions and training recipes

1. Keep the three density fields and the five stacked training-recipe bars.

2. Add a short label above or beside the ratio annotations:

   `Relative spatial-field exposure, $B_{\mathrm{DOF}}/B_{\mathrm{H-only}}$`

   This prevents the `1.00x`, `0.34x`, `0.44x`, `0.16x`, and `0.19x` labels from being mistaken for case-count or compute ratios.

3. The vertical axis currently says `Training cases`. Confirm the unit against the final data manifest. Use `Training trajectories` only if one active case is one trajectory; otherwise retain `Training cases`. Define training snapshots separately in Methods/SI.

4. Preserve the L/M/H legend, but place it close to the bars rather than between the example fields if this improves reading order.

5. Optionally add the recipe ratios in small text below the names, without changing the bar layout:
   - Mixed-HML: `1:1:1`
   - Zero-H-balanced: `1:1:0`
   - Zero-H-M-rich: `1:2:0`

   Add these only if they remain legible at final journal size.

6. Keep the same case/time for the L, M, and H examples. The zoom box must remain selected from the H-resolution ground truth, independently of model output.

## Panel b: headline error at 512 sensors

1. Retain the log-scale axis and the current recipe markers.

2. Increase the visibility of the 95% case-bootstrap confidence intervals with thin caps and sufficient contrast. Do not enlarge them artificially.

3. Retain the two intended grey paired-recipe connectors, but explain them visually or in the caption:
   - H-limited to Mixed-HML;
   - Zero-H-balanced to Zero-H-M-rich.

   The connectors should not suggest that all five recipes form a continuous trajectory.

4. Keep open/filled marker conventions consistent with the legend. Ensure that the two zero-H square markers remain visually distinct from H-containing recipes.

5. Add `lower is better` only if it can be placed unobtrusively; otherwise leave this to the caption.

## Panel c: matched H-resolution fields

1. Add a compact panel-level subtitle:

   `512 H-grid sensors; one matched held-out case--time field`

2. Change the left-side row labeling so that the first cell is clearly a **sensor layout**, while the model cells are **absolute-error maps**. The current combined label `Error / sensors` can be misread. A clean solution is:
   - over the reference bottom cell: `Sensor layout`;
   - over the model bottom cells: `Absolute error`.

3. Keep the three row roles:
   - full H-resolution field;
   - ground-truth-selected zoom;
   - absolute error / sensor layout.

4. Preserve common field limits across all reconstruction maps and a common absolute-error range across all error maps. State this in the caption or Methods, not as repeated panel text.

5. Standardize printed errors to three significant digits and typeset as `Rel. $L_2$`.

6. If sensors are overlaid on error maps, use one clearly defined marker style and explain it in a small legend. Otherwise show the full sensor plan only in the reference sensor-layout cell.

7. Do not call the qualitative field representative of the population. Keep the current configured snapshot unless the source manifest and all dependent panels are regenerated together.

## Panel d: sensor-density sweep

1. Retain the five recipe facets and the common log-scale y range.

2. Use the same model colours and the same model-specific marker/line identities in every facet. Do not encode model identity by colour alone.

3. Retain nested counts `64, 128, 256, 384, 512` with density percentages beneath. Use one shared x-axis title:

   `Sensor count / H-grid density (%)`

4. If confidence bands are added, use very light case-bootstrap ribbons and verify that they remain distinguishable. This is optional; do not add them if they obscure the curves.

5. Keep the ordering and y limits identical across facets so slopes and separations can be compared directly.

## Panel e: scale-resolved representative residuals

1. Add a visible subtitle or header:

   `Zero-H-M-rich training; 256 sensors`

   This condition is currently only recoverable from the caption/companion and should be visible in the panel.

2. Retain the large, intermediate, and fine wavelet rows and the reference/DMF-Gen/Senseiver columns.

3. Continue using signed residuals with symmetric limits about zero. Keep row-specific colour limits, but label them clearly as scale-specific display ranges.

4. Do not describe Senseiver as the universal `best non-DMF baseline` in the panel. Simply identify the model. Its role depends on the metric being considered.

5. Typeset component error consistently as `Rel. $L_2$` and retain two decimal places for the representative example.

## Panel f: population multiscale statistics

1. **Correct the spatial-correlation colour scale.** The table contains a negative value (`-0.09`), but the current colour bar begins at 0. Set the lower limit to at least `-0.10` (or slightly below) so the colour scale includes every displayed value. Keep the upper limit at 1.

2. Rename the colour-bar title to:

   `Spatial pattern correlation`

   This is an uncentered cosine similarity, not Pearson correlation.

3. Remove signed negative zero from the variance-bias annotations. Values that round to zero must display as `0.0` or `0.00`, never `-0.0`.

4. Prefer two decimal places for variance-allocation bias if legibility permits. This makes the near-zero DMF-Gen values visible rather than collapsing them all to `0.0`. If one decimal place is retained, state the full-precision values in a supplementary table.

5. Keep variance-allocation bias in **percentage points**, with a diverging colour map centred exactly at zero.

6. Keep the same model and recipe ordering in the upper and lower heatmaps.

7. Retain `Large`, `Interm.`, and `Fine` only if the abbreviations are defined in the caption/Methods. Otherwise use `Intermediate` where space permits.

## Experimental-provenance issues that the figure cannot solve

These points should be handled in Methods/SI and, where feasible, harmonized before final publication:

1. The H-containing DMF-Gen recipes currently use `GL_rbf_ENH`, whereas the two zero-H recipes use the base `GL_rbf`. A scientifically cleaner final comparison would use one canonical DMF-Gen backbone across all recipes. If retraining is not feasible, disclose the recipe-specific backbone and avoid interpreting recipe differences as a pure data-resolution effect.

2. DMF-Gen uses hard/default observation consistency, whereas FFM-Perceiver uses smooth endpoint consistency. Add a consistency-matched ablation in the SI or clearly report the inference contracts.

3. The figure reports case-bootstrap uncertainty, not variability across independent training seeds. Do not describe the error bars as model-training uncertainty.

4. `B_{\mathrm{DOF}}` measures spatial-value exposure, not elapsed training time, memory, energy, or total optimization cost.

5. Zero-shot super-resolution here means that H-resolution **training fields** are absent. Sparse measurements are still supplied at H-grid coordinates during reconstruction. State this definition in the Results text and caption.

6. Panel c and panel e use the configured snapshot 50. The wavelet exporter also records a separate truth-only snapshot 102, which is not displayed. Keep the artifact manifest authoritative and remove any stale documentation that says otherwise.

## Compact revised caption

```latex
\caption{\textbf{Mixed-resolution learning and zero-shot H-resolution reconstruction.}
\textbf{a}, Density fields on low- (L, $32\times32$), medium- (M, $64\times64$) and high-resolution (H, $128\times128$) grids and five training recipes; labels above the bars give spatial-field exposure relative to H-only.
\textbf{b}, Mean physical relative-$L_2$ error at 512 H-grid sensors across 300 held-out case--time pairs (error bars, 95\% case-bootstrap confidence intervals).
\textbf{c}, Matched H-resolution reconstructions, ground-truth-selected zooms and absolute errors at 512 sensors.
\textbf{d}, Error across nested 64--512-sensor sets.
\textbf{e}, Large-, intermediate- and fine-scale wavelet components of the reference and signed residuals from DMF-Gen and Senseiver under Zero-H-M-rich training at 256 sensors.
\textbf{f}, Median spatial pattern correlation and variance-allocation bias over 300 held-out fields for Mixed-HML and the two zero-H recipes. All outputs are evaluated on the H grid; zero-H models receive no H-resolution training fields.}
```
