# Mixed-resolution Figure V3-2 hybrid revision brief for Codex

## Objective
Revise the current streamlined mixed-resolution figure into a **hybrid V3-2** figure that preserves the macro-narrative clarity of the current version while restoring the most convincing micro-scale qualitative evidence from the earlier overloaded version.

The target is **not** to revert to the old six-panel dense layout. The target is to keep the current clean top-level logic and selectively reintroduce the **highest-value qualitative evidence**: the zoomed-in high-resolution flow structure and the corresponding local absolute error comparison.

The revised figure must therefore communicate, in a single inspection sequence:

1. what training information is available in each recipe;
2. how performance changes as H-resolution training fields are removed;
3. what the best zero-shot setting actually looks like in full-field and local fine-scale reconstruction;
4. why the multiscale quantitative summary supports that visual impression.

No retraining or reevaluation is requested. Use the existing validated sources and current rendering workflow.

---

# 1. Critical information-density audit and triage

## 1.1 What the previous overloaded version did well
The previous version contained two pieces of irreplaceable qualitative evidence:

- **zoomed-in H-resolution structure**, showing whether the model truly reconstructs fine local flow organization rather than only the broad global field;
- **local absolute error maps**, showing whether the apparent qualitative superiority corresponds to materially lower spatial error in the challenging local region.

These must be restored in the main figure.

## 1.2 What the previous overloaded version over-displayed
The previous version carried several information dimensions whose marginal value in the main figure is low once the current macro-structure is retained.

These should be demoted to SI:

### A. Full five-recipe sensor-sweep faceting
- H-only
- H-limited
- Mixed-HML
- Zero-H-balanced
- Zero-H-M-rich

Reason: once the 512-sensor recipe-transfer plot is kept in the main figure, the full five-facet sensor sweep becomes redundant. In the main figure, only the two zero-H sweeps are needed, because they address the core zero-shot generalization claim.

### B. Multi-recipe qualitative gallery in the reconstruction panel
Do **not** show Mixed-HML, Zero-H-balanced and Zero-H-M-rich simultaneously as separate qualitative columns for each model.

Reason: recipe effects are already quantified. Qualitative space should be spent on one decisive case.

### C. Full large / intermediate / fine matrix in the main figure
The complete 3-scale qualitative decomposition and the fully annotated 3-scale heatmaps should move to SI.

Reason: large-scale correlation is already saturated and provides little discrimination in the main figure.

### D. MLP-RBF qualitative reconstruction maps
Do not restore MLP-RBF as a main-figure contour column.

Reason: it contributes little additional interpretive value once its quantitative position is visible in the curves.

## 1.3 What the current streamlined version simplified too aggressively
The current version suppressed exactly the visual evidence that justifies the multiscale claim:

- no explicit local zoom comparison across models;
- only one row of local absolute error maps, without direct paired zoomed field evidence;
- wavelet panel reduced too far relative to the strong fine-scale narrative.

These should be strengthened, but only in one tightly controlled location.

---

# 2. Final scientific argument of V3-2
The figure should read in this exact order:

- **a** Training-resolution design: what is removed from training.
- **b** Cross-resolution performance: how each method responds to that removal and to sensor count in the zero-H regime.
- **c** Spatial proof at one decisive zero-H case: full field + zoomed-in region + local absolute error.
- **d** Multiscale confirmation: intermediate/fine residual evidence plus fine-scale quantitative metrics across the key recipes.

This is the same macro-narrative as the current clean version, but panel c must become more informative and panel d must visibly support it.

---

# 3. Concrete layout blueprint

## 3.1 Overall layout
Retain a four-panel figure, but rebalance panel heights so that the qualitative evidence has more room.

Recommended figure topology:

```text
Row 1:  a  (compact, full-width design strip)
Row 2:  b  (dominant quantitative block, full width)
Row 3:  c  (wide qualitative reconstruction panel, ~60–65% width)
        d  (multiscale support panel, ~35–40% width)
```

### Suggested GridSpec
If using Matplotlib `GridSpec`, a good starting contract is:

- outer grid: `nrows=3, ncols=12`
- height ratios: `[1.0, 2.2, 2.6]`
- row 1: `a = gs[0, :]`
- row 2: `b = gs[1, :]`
- row 3:
  - `c = gs[2, 0:8]`
  - `d = gs[2, 8:12]`

Alternative acceptable contract:
- width split row 3 = 7.5 : 4.5
- height ratios = 0.95 : 2.10 : 2.75

The critical point is that **panel c must be larger than in the current streamlined version** so the restored zoomed evidence is legible.

---

# 4. Panel-by-panel specification

## Panel a — training-resolution design
Keep the current improved structure.

### Preserve
- compact L / M / H resolution thumbnails/glyphs;
- stacked training-case bars for all five recipes;
- values above bars for relative spatial-field exposure;
- subtle grouping into:
  - contains H-resolution training fields;
  - zero-H training.

### Do not change substantially
This panel is already successful. Only minor spacing/typography cleanup is needed if required by row-height adjustments.

---

## Panel b — integrated quantitative performance
Keep the current integrated structure with three sub-axes:

- recipe-transfer at 512 sensors;
- Zero-H-balanced sensor sweep;
- Zero-H-M-rich sensor sweep.

### Preserve
- recipe-first narrative in the top axis;
- only the two zero-H sweeps in the lower axes;
- existing method colors/markers;
- shaded or bracketed zero-H region in the recipe-transfer axis.

### Minor refinement
If space permits, slightly enlarge the top recipe-transfer axis relative to the lower two sweeps, since it is the main quantitative summary.

No major scientific redesign is needed here.

---

## Panel c — restore high-value qualitative evidence without recreating the old visual wall
This is the core V3-2 update.

### 4.3.1 Scientific purpose
Panel c must show, for a single decisive zero-shot case, three linked views:

1. the **global H-resolution field**;
2. the **zoomed-in region** where fine-scale reconstruction matters most;
3. the **local absolute error** in that same region or over the field.

This replaces the current oversimplified qualitative evidence while avoiding the old matrix overload.

### 4.3.2 Fixed setting
Use one single validated setting:
- recipe: **Zero-H-M-rich**
- representative state: keep the current validated representative snapshot unless a different snapshot is already the formal representative in the existing manifest.

Do **not** vary recipes within panel c.

### 4.3.3 Column selection heuristic
Limit panel c to **four columns**:

1. Ground truth
2. DMF-Gen
3. FFM-Perceiver
4. Senseiver

#### Why these four
- **Ground truth** anchors all visual interpretation.
- **DMF-Gen** is the method to support.
- **FFM-Perceiver** is the most relevant generative contrast because it isolates backbone/conditioning differences under the same general mixed-resolution setting.
- **Senseiver** is the strongest deterministic reference and currently the most important non-generative comparator.

Do **not** include MLP-RBF as a main-figure qualitative column.

### 4.3.4 Row structure
Use **three compact qualitative rows** plus a small sensor-layout inset, instead of the previous bulky matrix.

#### Row 1 — Full H-resolution field
For each of the four columns:
- show the full field map;
- use a common field-value color scale;
- draw the same zoom box;
- for the three model columns, annotate the full-field relative-
  $L_2$.

#### Row 2 — Zoomed-in region
For each of the four columns:
- show the cropped zoomed-in field region only;
- use the same field-value color scale as Row 1;
- no extra colorbar here;
- align the crop exactly beneath the corresponding full-field map.

This is the **most important recovered evidence** from the old version.

#### Row 3 — Local absolute error maps
For the three model columns only:
- show absolute error maps aligned with the same zoom crop;
- optionally include a blank/reference cell or label under Ground truth;
- use one shared absolute-error colorbar;
- annotate the local or full-field relative-
  $L_2$ consistently with the existing validated source design.

Preferred choice: show **zoom-region absolute error**, not full-field error, because the zoom row already establishes the local challenge and this makes the panel more coherent.

If the validated pipeline only provides full-field absolute error maps, retain them, but still keep the local zoom row above.

### 4.3.5 Sensor layout
Keep the sensor layout as a **small inset** associated with the Ground-truth column or placed in the lower-left corner of panel c.
Do not allocate it a full tile of equal visual weight.

### 4.3.6 Colorbars
Use exactly two shared colorbars for panel c:
- one for field value;
- one for absolute error.

Place them compactly beneath the panel or along its bottom edge.
Do not repeat colorbars per row.

### 4.3.7 Visual principle
Panel c should now answer in one glance:

- globally, the methods reconstruct the same coarse field;
- locally, DMF-Gen restores the H-resolution structure more faithfully;
- this local advantage translates to materially lower absolute error.

---

## Panel d — multiscale support, aligned to panel c
Panel d should directly support the local intuition established in panel c.

### 4.4.1 Left half: compact qualitative scale evidence
Retain only the **intermediate** and **fine** rows, with three columns:

1. Ground-truth component
2. DMF-Gen residual
3. Senseiver residual

Do not reintroduce the large-scale row.

FFM-Perceiver does not need a qualitative residual column here; it is already visible in panel c and in the quantitative fine-scale plot.

### 4.4.2 Right half: fine-scale quantitative anchoring
Keep the current two line plots:
- fine-scale pattern correlation;
- fine-scale variance-allocation bias.

Use the three recipes:
- Mixed-HML
- Zero-H-balanced
- Zero-H-M-rich

Use the four methods:
- DMF-Gen
- FFM-Perceiver
- Senseiver
- MLP-RBF

### 4.4.3 Physical alignment requirement
The layout of panel d must make the link to panel c visually explicit.

Recommended arrangement:

```text
[d-left qualitative strips] | [d-right quantitative plots stacked vertically]
```

with the **fine-scale plot** vertically closer to the fine qualitative row than the intermediate-scale evidence.

More concretely:
- put the intermediate/fine qualitative residual strips on the left;
- on the right, stack:
  - top: fine-scale pattern correlation
  - bottom: fine-scale variance-allocation bias

This arrangement makes the quantitative plots feel like the numerical summary of the local residual evidence rather than a detached appendix.

### 4.4.4 Optional annotation refinement
To strengthen anchoring, optionally add subtle callout text or a small shared label such as:
- `Quantifies the fine-scale evidence shown at left`

but keep this extremely light.

---

# 5. Data-selection heuristics for the restored heatmap/contour evidence

## 5.1 Training recipe to visualize
Use **Zero-H-M-rich** as the single qualitative recipe.

### Why Zero-H-M-rich
- it is the most demanding and most relevant zero-shot setting;
- it is the strongest zero-H case for DMF-Gen, which avoids choosing a deliberately weak setting;
- it still directly supports the claim that the model reconstructs unseen H-resolution output from lower-resolution training exposure.

Zero-H-balanced remains represented quantitatively in panel b and panel d.

## 5.2 Baseline models to visualize
If limited to 3 model columns plus reference, use:
- DMF-Gen
- FFM-Perceiver
- Senseiver

### Rationale
- FFM-Perceiver is the most informative generative comparator.
- Senseiver is the strongest deterministic comparator.
- MLP-RBF adds little qualitative interpretive value beyond its quantitative placement.

## 5.3 Scale evidence to visualize
In the main figure, show only:
- intermediate qualitative residual evidence;
- fine qualitative residual evidence;
- fine quantitative metrics.

The complete large/intermediate/fine matrix remains in SI.

---

# 6. Information that must move to SI
To preserve workload visibility without overloading the main figure, explicitly export the following as SI rather than discarding them:

## SI Figure Sx1
Complete five-recipe sensor-sweep panel (essentially the old multi-facet sweep).

## SI Figure Sx2
Expanded qualitative reconstruction gallery including:
- Mixed-HML
- Zero-H-balanced
- Zero-H-Mrich
and additional model columns if already available.

## SI Figure Sx3
Full multiscale qualitative decomposition:
- large / intermediate / fine
- ground-truth component and residual maps.

## SI Figure Sx4
Full multiscale quantitative summary:
- pattern-correlation heatmap for all three scales;
- variance-allocation-bias heatmap for all three scales.

## SI tables
Export LaTeX-ready tables for:
- 512-sensor recipe-comparison summary;
- all sensor sweeps;
- all multiscale metrics.

This is important: the streamlined main figure must not make the work appear smaller than it was. The SI package should visibly preserve the complete evidence.

---

# 7. Detailed implementation notes

## 7.1 Panel-c topology suggestion
A practical inner GridSpec for panel c is:

- outer slot = panel c
- inner grid = `3 rows × 4 cols`
- height ratios approximately `[1.15, 0.90, 0.90]`
- column widths equal
- additional inset axes for sensor layout and bottom colorbars

Suggested semantics:
- row 1 = full field
- row 2 = zoom crop
- row 3 = absolute error crop

If space is tight, absolute error maps may use only the three model columns while the Ground-truth bottom cell is replaced by the sensor inset.

## 7.2 Panel-d topology suggestion
Use:
- inner grid `2 rows × 2 cols`
- left column wider than right column, e.g. width ratio `1.25 : 1.0`
- left upper/lower = intermediate and fine qualitative strips
- right upper = fine-scale correlation
- right lower = fine-scale bias

## 7.3 Typography and whitespace
- restore detail by increasing panel-c area, not by shrinking fonts;
- maintain the current clean whitespace discipline;
- no decorative subtitles inside panels unless strictly necessary;
- avoid dense numeric annotation beyond the existing rel.-
  $L_2$ values.

---

# 8. QA requirements specific to V3-2
The update fails if any of the following occur:

1. panel c still omits the zoomed-in field row;
2. panel c restores multiple recipes and becomes another gallery wall;
3. panel c includes more than Ground truth + 3 model columns in the main figure;
4. panel d reintroduces the full large/intermediate/fine heatmap matrix in the main figure;
5. panel d quantitative plots are visually detached from the qualitative scale evidence;
6. MLP-RBF is restored as a qualitative contour column in the main figure;
7. the main figure again duplicates the same recipe-transfer evidence in multiple axes;
8. SI outputs are not generated for the removed complete evidence;
9. the representative snapshot or recipe is changed without explicit provenance;
10. the final composition sacrifices readability to recover detail.

---

# 9. Required outputs
Create a new additive release for V3-2 containing:

## Main figure
- composed Figure 3 V3-2 in SVG / PDF / PNG;
- standalone panels a, b, c, d;

## SI outputs
- complete five-recipe sensor-sweep figure;
- expanded qualitative recipe gallery;
- complete three-scale qualitative figure;
- complete three-scale quantitative figure;
- LaTeX-ready SI tables.

## Documentation
- `source_manifest.json`
- `qa.json`
- `quantitative_figure_report.md`
- `figure_reference_update.md`
- `completion_report.md`

The quantitative report must explicitly describe:
- which evidence remained in the main figure;
- which evidence moved to SI;
- why each removal/addition improves information density;
- how the representative recipe/state were chosen;
- how panel-d quantitative plots anchor panel-c visual evidence.

---

# 10. Concise scientific success criterion
The revised V3-2 figure should make the following reading immediate:

> The training recipes progressively remove H-resolution information. DMF-Gen remains strongest under this removal and across zero-H sensor budgets. In the decisive Zero-H-M-rich case, its superiority is visible not only in the full field but also in the zoomed local H-resolution structure and corresponding local error. The multiscale metrics then confirm that this visual advantage reflects better intermediate/fine spatial fidelity rather than only smoother global contours.

If that sequence is not visually obvious within a few seconds, continue simplifying and rebalancing.
