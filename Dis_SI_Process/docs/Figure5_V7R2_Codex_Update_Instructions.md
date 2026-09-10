# Figure 5 V7R2 corrective brief for Codex

## Mission
Update the current Figure 5 ablation-enhanced composition by correcting content, layout, and styling issues **without rerunning training** and **without silently changing inherited validated Figure 5 quantities**. Work strictly within the `Dis_SI_Process/` workflow and create a new additive release rather than overwriting prior generated outputs.

The current figure already contains the right evidence families, but several presentation choices are not acceptable for the paper draft. The goal of this round is to produce a cleaner, logically ordered, publication-ready **Figure 5 V7R2** plus supporting SI artifacts.

---

## High-level principles

1. **Preserve inherited validated evidence unless explicitly replaced.**
   - Panels corresponding to the established UQ and cost scorecard should continue to use the accepted Figure 5 V6 sources unless a specific correction is requested here.
   - Do not replace the historical Figure 5 panel-f reconstruction-error value `0.117` with the newer ablation-A0 value `0.106321`. They belong to different evidence packages.

2. **Treat the new ablation evidence as a separate layer.**
   - Use the new ablation evaluation report as the scientific source for the new ablation/spectral panels.
   - Keep the source separation explicit in manifests, figure report, caption notes, and SI tables.

3. **Do not hide the training/checkpoint caveat.**
   - The ablation visualizations may proceed using the saved checkpoints despite unequal final epochs.
   - State clearly in the report and SI that the plotted ablation results are checkpoint-based comparisons of saved runs, not equal-budget retraining.

4. **Exclude A1 from the main ablation figure panels.**
   - The main ablation column/panels should use the five stochastic/architectural variants only:
     - Full model
     - No sensor feedback
     - No local conditioning
     - Local-only conditioning
     - IID Gaussian prior
   - A1 must remain available in the SI as a separate deterministic-objective control.

5. **Remove decorative headings.**
   - Eliminate unnecessary in-panel subheadings such as `Conditioning and source variants`.
   - Keep only panel letters, axis labels, and the necessary scorecard column headers in panel f.

---

## Scientific panel plan

Recompose Figure 5 so that the panel logic reads naturally from left to right and top to bottom.

### Final logical sequence and letter mapping

- **a**: normalized CRPS (existing validated panel)
- **b**: spread--error Spearman association (existing validated panel)
- **c**: selective reconstruction (existing validated panel; move to the top row so that panels a--c form the complete UQ story)
- **d**: ablation error distributions (new)
- **e**: scale-resolved fidelity summary (new; default main version uses `U1`)
- **f**: accuracy and computational footprint scorecard (existing validated panel)

This order is mandatory. The current arrangement, in which a/b/d form one group and c/e form another, must be retired.

---

## Required layout

### Global layout target
Use a compact multi-panel composition with the following structure:

```text
Top row:    a | b | c
Middle row: d (span two columns) | e
Bottom row: f (full width)
```

### Additional layout requirements

- Keep the full composed figure at Nature-compatible width; preserve editable SVG output.
- Reduce the unnecessary whitespace:
  - specifically the gutter between panels **b** and **c**
  - and the gutter between panels **d** and **e**
- Align panel widths visually so that the top row looks intentionally designed rather than patched together.
- Panel **e** may internally contain two vertically stacked sub-axes, but it should still count as one lettered panel.
- Panel **f** should remain a single full-width scorecard block.

---

## Panel-specific corrections

## Panel a
Use the existing validated normalized-CRPS content and style.

### Keep
- scatter cloud / statewise points
- box summary
- mean marker and interval
- method order

### Correct if needed
- no extra panel title
- spacing and font sizes should match the revised composition

---

## Panel b
Use the existing validated Spearman-association content and style.

### Keep
- dashed vertical zero line
- box/interval treatment
- method order

### Correct if needed
- no extra panel title
- spacing and font sizes should match the revised composition

---

## Panel c: selective reconstruction
This is the current selective-reconstruction plot, moved to the top row.

### Keep
- same validated curves and uncertainty bands
- same retained-fraction coordinates
- same normalization

### Correct
- remove any extra subplot title; panel letter plus axis labels are sufficient
- ensure its aspect ratio looks balanced next to panels a and b
- remove the standalone `DMF AURC = 0.741` annotation unless it is already required by the paper contract; if retained, it must not dominate the panel
- if an annotation is kept, prefer the more interpretable effect note `4.6% lower error at 80% retained`

---

## Panel d: ablation error distributions
This replaces the current point-only ablation panel.

### Scientific content
Plot the distribution of **statewise unobserved-field relative-L2** over the 1,000 held-out snapshots for the five main ablation models:

1. Full model
2. No sensor feedback
3. No local conditioning
4. Local-only conditioning
5. IID Gaussian prior

Use the **primary checkpoint policy: `last.pt`** for the main figure.

### Plot type
Use a horizontal distribution-oriented plot that matches the visual language of panels a and b.
One acceptable choice is:
- light jittered statewise points
- box summary
- clearly marked mean position
- mean numerical value placed near the marker

Equivalent alternatives such as violin + box + mean are acceptable only if they remain compact and visually consistent.

### Hard requirements
- show the **distribution**, not only the mean
- annotate only the **mean value** for each model
- **do not annotate percentage increase/decrease** on the panel
- no decorative subheading
- use the published labels above, not A0/A2/A3/A4/A5 in the visible panel
- preserve the performance order in a scientifically sensible display; the recommended order is the conceptual order listed above rather than sorting by the mean

### Reference values to verify against the report
For `last.pt`, mean unobserved-field relative-L2 should resolve to:
- Full model: `0.106321`
- No sensor feedback: `0.126974`
- No local conditioning: `0.144703`
- Local-only conditioning: `0.354951`
- IID Gaussian prior: `0.104294`

Round consistently in the figure labels.

---

## Panel e: scale-resolved fidelity summary
The current targeted RFF-vs-IID-only panel must be replaced.

### Scientific objective
Panel e must no longer look like a hand-picked comparison that presumes the conclusion. Instead, it should summarize how **all ablation variants** behave at the spectral population level and how accurately they reconstruct the selected high-frequency band.

### Default main-panel field
Use **`U1`** as the default main-panel field unless a later visual comparison shows another field is clearly more informative.

### Included methods in panel e
Include the following curves / comparisons:
- Truth
- Full model
- No sensor feedback
- No local conditioning
- Local-only conditioning
- IID Gaussian prior
- Senseiver

Senseiver should be shown as the deterministic reference because it is the baseline with the best L2 performance.

### Internal structure of panel e
Panel e contains two vertically stacked sub-axes:

#### e-top: population spectrum composite
- show the **population spectrum** for the selected field across the models above
- use the same estimator family that was already accepted in the ablation evaluation outputs
- truth must be clearly identifiable
- the high-band region should be visually locatable (for example, with a shaded band or vertical delimiters)
- the axis design must make over-supply and attenuation visible without requiring a separate ratio-only panel

Recommended display:
- x-axis: frequency shell / mode index
- y-axis: population summary of spectral power (likely logarithmic)
- use median curves; add IQR bands only if they remain readable

#### e-bottom: high-band relative-L2 across models
- show the **distribution across states** of the selected field's high-band relative-L2 for:
  - Full model
  - No sensor feedback
  - No local conditioning
  - Local-only conditioning
  - IID Gaussian prior
  - Senseiver
- use a compact horizontal distribution plot, again consistent with panel d
- mark the mean and its numerical value
- no percentage annotations

### Very important
Do **not** present only two values such as power/truth and high-band relative-L2 for two models. That design is rejected.

### Main-figure interpretation target
Panel e should let the reader see both:
- how the spectrum shape changes across models
- and which models actually reconstruct the high band more faithfully

---

## Alternative panel-e versions for field selection
Prepare additional full panel-e alternatives, each as a complete standalone artifact, for the following fields:
- `Y_CH4`
- `Y_CO`
- `T`
- `U1`
- `p`

### Deliverables
For each field, export:
- one standalone panel-e SVG
- one standalone panel-e PNG
- one short companion markdown note summarizing why that field may or may not be the strongest final choice

### Selection table
Create a compact comparison table that helps the author choose the final field. Include at least:
- qualitative readability of the population spectra
- separation among models in high-band relative-L2
- whether Senseiver provides a useful reference contrast
- any important caveat for that field

Do **not** make the final field choice automatically without surfacing this comparison.

---

## Panel f: accuracy and computational footprint
Retain the established panel-f scientific content and scorecard structure.

### Required corrections
- keep the updated x-axis title `Training update time`
- remove any separate memory legend/key
- revert to the compact original design by labeling **`Model`** and **`Peak`** directly on the inference-memory plot itself
- keep the inference-memory endpoint semantics:
  - filled = model
  - hollow = peak
- make sure the direct labels are compact and do not expand the panel unnecessarily

### Keep
- existing method markers and colors
- existing scorecard columns
- existing validated quantitative values

---

## Visual style requirements

1. Remove all unnecessary subplot titles/headings except the scorecard column headers in panel f.
2. Preserve consistent marker shapes and method colors with the established Figure 5 contract.
3. Maintain readable typography at final print width.
4. Ensure that jitter points stay light and do not overpower summaries.
5. Keep the composition balanced; panel e should not look like an afterthought.

---

## Data and provenance requirements

### Inputs
Use the accepted current-round figure as a style/provenance reference and the updated ablation evaluation outputs as the scientific source for new panels.

### Source separation
In the build manifest and the quantitative figure report, explicitly separate:
- inherited Figure 5 V6 sources for panels a, b, c, f
- new ablation sources for panels d and e

### Deterministic control
A1 must be retained in the SI package, not the main figure.

---

## Required SI outputs
Generate an SI package that at minimum includes the following.

### SI Figure S1
Ablation error distributions, including A1.

### SI Figure S2
Scale-resolved fidelity panel-e counterparts for all five physical fields under `last.pt`.

### SI Figure S3
Deterministic-objective control (A1) compared against the full model and Senseiver:
- unobserved-field relative-L2
- fieldwise physical relative-L2
- optional pressure-specific note if already available in the report

### SI Table S1
Machine-readable and LaTeX-ready summary of statewise unobserved-field relative-L2 for all six ablation runs under both checkpoint policies:
- mean
- 95% block-20 CI
- median
- IQR
- 95th percentile
- maximum

### SI Table S2
Fieldwise physical relative-L2 means for all six ablation runs under both policies.

### SI Table S3
High-band relative-L2 summary for the selected field across the ablation runs and Senseiver.

### SI Table S4
Population-spectrum diagnostic summary for the selected field, including any reported truth-relative high-band power statistic that is already available.

### SI Table S5
Checkpoint metadata table:
- displayed model label
- internal run label
- checkpoint policy
- epoch
- source path or hash if available
- note on unequal training endpoints

### SI text snippets
Provide copy-ready LaTeX figure captions and table captions for all SI outputs.

---

## Figure-making report requirements
Create a detailed quantitative figure-making report in markdown. This report is mandatory.

### It must include
1. exact source files used for each panel
2. panel-by-panel scientific intent
3. panel-by-panel quantitative definitions
4. any aggregation choices
5. which checkpoint policy is used where
6. how panel-e spectra were summarized
7. how high-band relative-L2 was computed / loaded
8. any deviations from the previous brief
9. visual design rationale for each panel
10. unresolved caveats that remain visible in the figure or SI

---

## Build/output requirements
Create a new additive timestamped release under `Dis_SI_Process/figures/generated/` and matching result/docs directories. Do **not** overwrite the prior release.

### Required main outputs
- full composed Figure 5 V7R2 in SVG
- full composed Figure 5 V7R2 in PNG
- standalone SVG and PNG for each panel a--f
- alternative standalone panel-e versions for `Y_CH4`, `Y_CO`, `T`, `U1`, `p`
- companion markdown files for each panel and the full figure

### Required derived outputs
- compact panel source tables
- build manifest
- QA file
- SI figure files
- SI LaTeX tables
- figure-making report
- field-selection comparison table

---

## QA checklist
Codex must explicitly check and document the following before finalizing:

1. panel letters follow the final order `a,b,c,d,e,f`
2. the old extra subheadings are gone
3. panel c is the selective-reconstruction plot
4. panel d is a **distribution** plot, not a mean-only plot
5. panel d shows mean values only, without percentage annotations
6. panel e includes all required ablation models plus Senseiver
7. panel e-top is a population-spectrum composite
8. panel e-bottom is a multi-model high-band relative-L2 plot
9. the whitespace between b/c and d/e has been reduced materially
10. panel f uses direct `Model` / `Peak` labels instead of a separate legend
11. panel f still uses the established validated scorecard values
12. A1 is excluded from the main figure but included in SI
13. last.pt is used in the main ablation panels
14. the report states that unequal final epochs were not corrected in this round
15. no source-package mixup occurred between the historical Figure 5 V6 results and the new ablation results

If any check fails, the release must be marked as incomplete rather than silently promoted.
