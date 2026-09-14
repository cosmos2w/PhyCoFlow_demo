# Mixed-resolution Figure streamlining brief for Codex

## Objective

Rebuild the current mixed-resolution / zero-shot super-resolution main figure into a **leaner four-panel figure** without changing the scientific evidence, training runs, evaluation protocol, or the Results section's logical order.

The current six-panel figure contains four scientific arguments:

1. **Experimental design**: what L/M/H mean and how the five training recipes differ.
2. **Cross-resolution reconstruction performance**: how each method responds to training recipe and sensor budget, especially when H-resolution training fields are absent.
3. **Representative zero-H reconstruction**: what the quantitative difference looks like spatially.
4. **Multiscale fidelity**: whether zero-H reconstruction preserves intermediate/fine spatial structure rather than only the energetic large scales.

The revised figure must make those four arguments visually explicit. The current `b+d` and `e+f` pairs should be consolidated, and the contour-map burden in `a+c+e` should be reduced.

No new training or inference is requested. Use the existing validated caches/results and build an **additive new figure release**.

---

# 1. Problems to correct

## 1.1 Current panels b and d encode largely the same evidence

Current panel b is the **512-sensor slice** of the sensor-sweep data displayed in current panel d. It changes the visual grouping (method-first in b, recipe-first in d) but does not add a genuinely independent result.

This creates two problems:

- readers spend substantial figure area decoding the same dataset twice;
- the main zero-shot message is visually diluted because the sensor-count sweep gives equal space to all five recipes.

The revised figure must retain both ideas—**recipe adaptability at a standard sensor budget** and **sensor-budget robustness in the zero-H regime**—but place them inside one integrated performance panel.

## 1.2 Panels a, c and e overuse contour maps for different purposes

The three panels use similar warm/cool field images, but their scientific jobs are different:

- panel a only needs to explain **discretization level and training composition**;
- panel c needs to show **spatial reconstruction quality**;
- panel e needs to show **scale-specific residual structure**.

The current design makes them look visually repetitive and gives low-information setup images similar visual weight to actual reconstruction evidence.

The revision must make each image family serve a distinct role.

## 1.3 Current panel c repeats the recipe comparison already quantified elsewhere

Panel c displays three recipes for each of three models plus the reference, with full fields, zooms, and error maps. This creates a large gallery whose main message—DMF-Gen better preserves localized H-resolution structure in zero-H training—is already established quantitatively in b/d.

The qualitative panel should **illustrate one decisive zero-H setting**, not re-run the complete quantitative comparison as images.

## 1.4 Current panel e spends one third of its area on a non-discriminative large-scale component

The current Results already state that large-scale pattern correlation is approximately one for all methods. The large-scale maps therefore consume substantial area while contributing little discrimination.

The revised main figure should emphasize the **intermediate and fine scales**, where model differences are scientifically informative. Full large/intermediate/fine results remain in SI.

## 1.5 Current panel f is numerically dense

The two heatmaps display:

- 4 methods,
- 3 recipes,
- 3 scales,
- 2 metrics,

with numerical annotations in every cell.

Most large-scale correlation values are approximately 1 and most large-scale variance biases are approximately 0. This is valuable completeness for SI, but excessive for the main figure.

The main figure should retain the **fine-scale quantitative comparison across recipes** and move the complete three-scale tables/heatmaps to SI.

## 1.6 Visual hierarchy does not match the Results hierarchy

The primary claim of this section is **cross-resolution / zero-H reconstruction performance**, but contour galleries occupy more visual area than the central quantitative evidence.

The new composition must make the integrated performance panel the dominant panel.

---

# 2. Required final scientific structure

Use **four main panels** in this exact logical order:

- **a — Training-resolution design**
- **b — Reconstruction across training recipes and sensor budgets**
- **c — Representative zero-H H-resolution reconstruction**
- **d — Multiscale fidelity under mixed- and zero-H training**

This maps directly to the current Results narrative:

1. define the five recipes;
2. establish zero-H accuracy and sensor-budget behavior;
3. show what the reconstruction difference looks like;
4. establish that the gain extends to intermediate/fine spatial scales.

Do not introduce a new scientific storyline.

---

# 3. Recommended overall layout

Use a two-row scientific hierarchy plus a shallow design strip.

Preferred layout:

```text
┌──────────────────────────────────────────────────────────────────────┐
│ a  Training-resolution design                                      │
│    shallow full-width or near-full-width strip                     │
├──────────────────────────────────────────────────────────────────────┤
│ b  Integrated performance panel — visually dominant, full width     │
│    b1 recipe transfer @ 512 sensors                                │
│    b2 Zero-H-balanced sweep     b3 Zero-H-M-rich sweep             │
├───────────────────────────────────┬──────────────────────────────────┤
│ c Representative zero-H field     │ d Multiscale fidelity           │
│   compact qualitative comparison  │   qualitative + quantitative   │
└───────────────────────────────────┴──────────────────────────────────┘
```

If the existing assembler strongly favors a two-column top row, an acceptable alternative is:

```text
┌──────────────────────┬───────────────────────────────────────────────┐
│ a design             │ b integrated performance                    │
├──────────────────────┼───────────────────────────────────────────────┤
│ c reconstruction     │ d multiscale fidelity                       │
└──────────────────────┴───────────────────────────────────────────────┘
```

but only if panel b remains clearly the largest and most legible quantitative panel.

Target final width: the same journal-width contract already used by the project. Reduce total height relative to the current six-panel figure where possible.

---

# 4. Panel a — training-resolution design

## Scientific purpose

Panel a should answer only:

- what are L, M and H?
- how are the five training recipes composed?
- how much spatial-field exposure does each recipe contain relative to H-only?

It should **not** serve as another qualitative reconstruction panel.

## Required redesign

### Remove or strongly reduce the three large contour thumbnails

Do not keep the current three equal-size density fields as the dominant upper half of panel a.

Preferred replacement:

- use three compact resolution glyphs or very small thumbnails derived from the same field/crop:
  - L: `32 × 32`
  - M: `64 × 64`
  - H: `128 × 128`
- if data thumbnails are retained, keep them small and explicitly pixelated so they communicate discretization rather than field morphology;
- one shared label `same physical field, different discretization` is acceptable;
- no large colorbar is needed.

### Retain the recipe-composition bars

Keep the stacked L/M/H training-case bars for:

- H-only
- H-limited
- Mixed-HML
- Zero-H-balanced
- Zero-H-M-rich

Keep the relative spatial-field exposure values:

- 1.00×
- 0.34×
- 0.44×
- 0.16×
- 0.19×

Display these directly above the bars.

### Visually group the recipes

Create a subtle grouping:

- first three recipes: `contains H-resolution training fields`
- last two recipes: `zero-H training`

Use a bracket, background tint, or small group label. Do not add a large heading.

### Keep the panel compact

The training design is necessary context but not the principal result. It should occupy no more than approximately 20–25% of the total main-figure area.

---

# 5. Panel b — integrate current b and d

This is the most important revision.

## Scientific purpose

Panel b must simultaneously show:

1. performance as the training recipe changes from H-only to zero-H;
2. robustness to sensor count in the two zero-H settings.

This replaces both current panels b and d.

## b1 — recipe transfer at the standard 512-sensor budget

Replace the current method-grouped scatter plot with a **recipe-first trajectory plot**.

### Axes

- x-axis categories, in this exact order:
  1. H-only
  2. H-limited
  3. Mixed-HML
  4. Zero-H-balanced
  5. Zero-H-M-rich
- y-axis: physical relative-$L_2$, logarithmic scale
- sensor count fixed at 512

### Lines

One line per method:

- DMF-Gen
- FFM-Perceiver
- Senseiver
- MLP-RBF

Use the existing method colors and markers.

The x-axis now directly represents the scientific perturbation: **what resolution information was available during training**.

### Emphasis

- shade or bracket the last two x positions as `no H-resolution training fields`;
- visually emphasize DMF-Gen with a slightly thicker line;
- keep the other methods visible but lighter;
- do not add percentage annotations for every point;
- optionally label only the DMF-Gen endpoint values for the two zero-H recipes if space permits.

This plot should immediately show whether each method degrades, remains stable, or improves as H-resolution training is removed.

## b2 and b3 — zero-H sensor sweeps

Below b1, include only two compact sensor-sweep axes:

- **Zero-H-balanced**
- **Zero-H-M-rich**

Each plot shows:

- x-axis: sensor count 64, 128, 256, 384, 512
- optional secondary density labels 0.4, 0.8, 1.6, 2.3, 3.1%
- y-axis: physical relative-$L_2$, same logarithmic scale and limits
- same four method styles as b1

Share the y-axis between b2 and b3.

### Why only these two sweeps

The H-only, H-limited and Mixed-HML sensor sweeps are valid results but are not necessary in the main figure once b1 shows their 512-sensor endpoints. Export the complete five-recipe sweep as an SI figure.

### Main visual message

The reader should be able to see in one panel that:

- DMF-Gen remains strongest when complete H-resolution training fields are removed;
- its advantage persists from 64 to 512 sensors;
- M-rich lower-resolution training improves the zero-H result.

## Remove current panel b and current five-facet panel d after the new composite is built.

Do not display the same 512-sensor values twice.

---

# 6. Panel c — representative zero-H reconstruction

## Scientific purpose

Show what the zero-H performance difference looks like spatially.

Do not use panel c to compare all recipes.

## Fixed setting

Use the same validated representative state already selected by the current workflow.

Use:

- recipe: **Zero-H-M-rich**
- sensor count: **512**

If the existing representative selection is formally tied to another sensor count, preserve the validated representative and document the setting rather than silently changing it.

## Columns

Use at most:

1. Ground truth / reference
2. DMF-Gen
3. FFM-Perceiver
4. Senseiver

Do not include separate columns for Mixed-HML, Zero-H-balanced and Zero-H-M-rich within each method.

MLP-RBF does not need a qualitative column in the main figure.

## Rows

Use two visual rows rather than the current three:

### Row 1 — full H-resolution field

- ground truth
- reconstruction from each displayed method
- common field color scale
- mark the same zoom box on every map

### Row 2 — local error evidence

Preferred design:

- place a small zoom inset inside each Row-1 reconstruction;
- use Row 2 for absolute-error maps for the three methods.

Alternative design:

- Row 2 may show zoomed residual/error crops only.

Do not show both a full separate zoom row and a full separate error row unless the layout remains clearly lighter than the current panel.

## Sensor layout

Keep the sensor-layout thumbnail as a small inset associated with the reference, not as an equal-size main column.

## Labels

Keep one relative-$L_2$ value per method.

Avoid repeating recipe names because the entire panel has one recipe.

Use one shared field colorbar and one shared error colorbar.

---

# 7. Panel d — combine current e and f into one multiscale argument

## Scientific purpose

Demonstrate that zero-H reconstruction preserves spatial organization below the dominant large scale.

This panel should answer:

- what do intermediate/fine residuals look like?
- how do methods compare quantitatively at the fine scale across the key training recipes?

## d-left — compact qualitative wavelet evidence

Remove the large-scale row from the main figure.

Show only:

- Intermediate
- Fine

for:

- Ground-truth component
- DMF-Gen residual
- Senseiver residual

Use the same representative state and the same recipe currently used for the wavelet illustration (Zero-H-M-rich, unless the validated source specifies otherwise).

This reduces the qualitative wavelet block from 9 maps to 6.

Keep relative-$L_2$ labels for the residuals.

Use one shared signed-residual colorbar per scale only if necessary; otherwise use a consistent normalized visual range and document it.

## d-right — fine-scale quantitative summary

Replace the full three-scale, two-metric heatmaps with **two compact fine-scale plots** across:

- Mixed-HML
- Zero-H-balanced
- Zero-H-M-rich

and the four methods:

- DMF-Gen
- FFM-Perceiver
- Senseiver
- MLP-RBF

### Upper quantitative axis

Fine-scale spatial pattern correlation.

- x-axis: the three recipes
- y-axis: correlation
- horizontal reference at 1
- same method colors/markers as panel b

### Lower quantitative axis

Fine-scale variance-allocation bias.

- x-axis: the three recipes
- y-axis: bias
- horizontal reference at 0
- same method styles

Do not annotate every point with a number in the main figure.

The complete large/intermediate/fine numerical heatmaps must be regenerated as an SI figure/table, not discarded.

## Main-figure quantitative values to preserve

The new plot must preserve the existing validated medians, including the Zero-H-M-rich fine-scale pattern correlations:

- DMF-Gen: 0.815
- FFM-Perceiver: 0.368
- Senseiver: -0.089
- MLP-RBF: 0.165

and the corresponding validated fine-scale variance-allocation biases from the existing wavelet summary.

---

# 8. Results that move to SI rather than being deleted

Create a supplementary figure package containing:

## SI Figure Sx1 — complete sensor sweeps

The current five-recipe sensor-count sweep:

- H-only
- H-limited
- Mixed-HML
- Zero-H-balanced
- Zero-H-M-rich

for all four methods.

This is essentially the current panel d, cleaned up.

## SI Figure Sx2 — complete qualitative recipe gallery

Preserve the current multi-recipe qualitative comparison from panel c as an SI figure if all source images remain validated.

## SI Figure Sx3 — complete three-scale wavelet analysis

Preserve the current:

- large/intermediate/fine qualitative maps;
- full pattern-correlation heatmap;
- full variance-allocation-bias heatmap.

The main figure should show only the intermediate/fine qualitative evidence and fine-scale quantitative summary.

## SI tables

Export LaTeX-ready tables for:

- 512-sensor accuracy across all methods and recipes;
- 64–512 sensor sweeps;
- all large/intermediate/fine pattern correlations;
- all large/intermediate/fine variance-allocation biases.

---

# 9. Visual design rules

## Method encoding

Keep method colors and markers identical across all quantitative panels.

DMF-Gen should be visually primary but not use a different plotting grammar.

## Recipe encoding

Do not assign five equally prominent recipe colors if method color already carries the primary comparison.

Recipe should be encoded primarily by x-position/facet title.

Use a subtle common tint/bracket for the two zero-H recipes.

## Repetition

Do not repeat:

- the same method legend in multiple panels;
- recipe names inside panels where the setting is globally fixed;
- sensor-density labels on every sub-axis if a shared axis can carry them;
- separate field colorbars for maps with the same limits.

## Numeric annotations

Reserve direct numeric labels for:

- relative-$L_2$ values in the representative qualitative panel;
- optional zero-H DMF endpoints in b1.

Do not annotate every heatmap/dot with exact values in the main figure.

## Typography

Increase final-size readability by removing information rather than shrinking fonts.

No text smaller than the project's accepted Nature-style minimum.

---

# 10. Minimal manuscript-reference changes

The figure redesign should not require restructuring the Results section.

Use the following new reference logic:

- training recipes and exposure: `Fig.~3a`
- 512-sensor comparison and sensor-count behavior: `Fig.~3b`
- representative zero-H reconstructions: `Fig.~3c`
- wavelet / multiscale evidence: `Fig.~3d`

The existing Results progression can remain:

1. define L/M/H and five training recipes;
2. report zero-H 512-sensor errors and the 64–512 sensor trend;
3. describe the matched H-resolution reconstruction;
4. discuss intermediate/fine-scale fidelity.

Create a short `figure_reference_update.md` listing every old panel reference and its new panel letter.

---

# 11. Existing workflow and provenance

Use the existing validated mixed-resolution workflow rather than rebuilding data manually.

The repository already contains:

- `1_SubTask_SuperResolution/figures/scripts/render_mixed_resolution_best.py`
- the unified-v2 export/assembly/audit chain:
  - `96_export_unified_v2_panels.py`
  - `97_assemble_mixed_resolution_unified_v2.py`
  - `98_audit_unified_v2.py`

The technical companion identifies the validated source families including:

- `SensorSweepAllRecipes_summary_<run>.csv`
- `AllRecipeAccuracy_summary_<run>.csv`
- `MultiscaleWavelet_summary_<run>.csv`
- `MultiscaleWavelet_per_snapshot_<run>.csv`
- `ResolutionProtocol_budgets_<run>.csv`
- `ResolutionProtocol_fields_<run>.csv`
- `ResolutionProtocol_sensors_<run>.csv`
- `SensorSweep_per_snapshot_<run>.csv`
- the canonical reconstruction-cache manifest and representative-state metadata.

Do not modify validated source CSVs.

Prefer creating an additive new rendering layer, for example:

- `96_export_unified_v3_panels.py`
- `97_assemble_mixed_resolution_unified_v3.py`
- `98_audit_unified_v3.py`

or equivalent versioned scripts.

The new renderer may select subsets of validated rows for the streamlined main figure, but must document those selections in the figure manifest.

---

# 12. Required outputs

Generate a new timestamped/additive release with:

## Main figure

- composed streamlined Figure 3 in SVG
- PDF
- high-resolution PNG
- standalone panels a–d in SVG/PDF/PNG

## SI

- complete five-recipe sensor sweep
- complete recipe qualitative gallery
- complete three-scale wavelet figure
- LaTeX tables described above

## Documentation

- `figure_contract.md`
- `source_manifest.json`
- `qa.json`
- `completion_report.md`
- `figure_reference_update.md`
- detailed `quantitative_figure_report.md`

The quantitative report must state for each new panel:

- exact source file(s)
- row filters / recipes / sensor counts
- representative snapshot identity
- plotted statistic
- CI or dispersion definition
- whether the quantity is unchanged from unified-v2
- why the panel exists in the scientific argument

---

# 13. QA requirements

The new audit must fail if any of the following occur:

1. panel b duplicates the same 512-sensor data in two separate main-figure axes without a distinct purpose;
2. panel c contains more than one training recipe;
3. panel a retains three large field contours that dominate the training-design panel;
4. panel d main quantitative summary still includes the non-discriminative large-scale cells;
5. old large/intermediate/fine values disappear entirely rather than moving to SI;
6. recipe or method ordering differs from the validated source contract;
7. the zero-H-balanced / zero-H-M-rich values do not exactly match the validated summaries;
8. the representative snapshot is changed without explicit provenance;
9. source CSVs or cached reconstructions are modified;
10. the final rendered figure contains overlapping labels or typography smaller than the accepted project standard.

Perform final visual QA at manuscript print width, not only on a large monitor.

---

# 14. Preferred final interpretation

The streamlined figure should make the following reading effortless:

> The training protocol progressively removes H-resolution information (a). DMF-Gen maintains the strongest reconstruction when H-resolution training fields are absent, and this advantage persists across sparse sensor budgets (b). A representative H-resolution state shows that the quantitative improvement corresponds to better localized structure (c). The advantage extends below the dominant large scales to intermediate/fine spatial organization and energy allocation (d).

If the revised figure does not communicate this sequence within several seconds of inspection, continue simplifying it.
