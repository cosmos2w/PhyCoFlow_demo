# Figure 5 V7: architectural controls and scale-resolved fidelity

## 0. Task and scientific decision

Continue the existing DMF-Gen paper post-processing workflow in `Dis_SI_Process/`. Deliver a revised Figure 5 with a **dedicated third, right-hand column for the stochastic architecture/prior comparisons**, all standalone panels, a focused supplementary package, LaTeX figure/table inserts, and a quantitative figure-making report.

Use the local Figure 5 release at:

```text
Dis_SI_Process/figures/generated/20260904_1200
```

as the visual and source-lineage starting point. Use the completed ablation evaluation at:

```text
0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910
```

as the new quantitative source. Although its report is named `Ablation_CondT_Evaluation_20260906.md`, its internal evaluation date is **2026-09-10**. You should check and use the latest checkpoints saved for A0 to do the evaluations and visualizations.

This is **post-processing and visualization only**. Do not train, resume training, change checkpoints, perform fresh model inference, benchmark GPUs, change solver steps, introduce smoothing/clipping, or rerun the old ensemble evaluation. CPU reductions, source validation, bootstrap of already saved per-state metrics, and plotting are in scope. Read existing processed artifacts in place and create only the compact derived tables needed for this build.

The new evidence supports two main messages:

> The saved full stochastic model reconstructs more accurately than its conditioning-route removal variants. The IID-source model has slightly lower whole-field error but substantially worse phase-sensitive fine-scale reconstruction, with strong excess high-band velocity power.

The second statement is a **reconstruction--fine-scale fidelity tradeoff**, not proof that the full model wins every metric. In particular, the full model also attenuates high-band velocity power.

### Scope of the main architectural/prior comparison

The main column contains five stochastic configurations: full model, no sensor feedback, no local conditioning, local-only conditioning, and IID Gaussian prior. The deterministic run is **not part of this main architectural/prior column**. Preserve it as a separately titled **deterministic objective control** in SI and in the complete provenance record. Do not delete its evaluated results or present the stochastic comparison as evidence that generation improves point-error over the same deterministic backbone.

### Training-budget treatment

Proceed with the available frozen checkpoints; no equal-epoch retraining is required for this task. Do not normalize errors by epoch, extrapolate to a common endpoint, truncate histories to imply matched training, or describe the comparison as budget-matched. State the endpoint differences in the caption/SI and retain both checkpoint policies. Best-versus-last robustness addresses checkpoint-selection sensitivity; it does not remove the difference in training duration or effective trained capacity.

---

## 1. Preflight: inspect the workspace before changing anything

1. Record the local repository root, branch, `git rev-parse HEAD`, `git status --short`, timestamp with timezone, and environment versions. Do not switch branches, reset, clean, commit, or push. Preserve unrelated working-tree changes.
2. Read any applicable `AGENTS.md`, then `Dis_SI_Process/README.md`, the current Figure 5 contracts, the local `20260904_1200` companions/manifests, and the ablation report in full, especially Sections 2, 3, 6b, 7, 9b and 10.
3. Inspect the renderer that actually produced `20260904_1200`; use its metadata to identify that script rather than assuming the oldest V6 script is current.
4. Resolve every required input path locally. Check schemas before implementing an adapter. Discover actual column names and metric IDs, and write the resolved mapping into a source manifest.
5. Verify that the ablation evaluation uses the frozen new reference, matching truth, sensor plan, original simulation time indices, and generation policy.

The remote paper-workflow branch inspected when this brief was prepared was `paper/postprocessing-multifield-superresolution`, at commit:

```text
a46c7b929e565faa72e639a81163cb27e8fa4752
```

Verified reusable code includes:

```text
Dis_SI_Process/README.md
Dis_SI_Process/configs/figure5_v6.yaml
Dis_SI_Process/figures/scripts/build_figure5_v6.py
Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format.py
Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v2.py
Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v3.py
Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v4.py
Dis_SI_Process/scripts/audit_figure5_v51_exploration.py
Dis_SI_Process/scripts/qa_figure5_outputs.py
Dis_SI_Process/tests/
Dis_SI_Process/utils/
```

The remote view did **not** expose the local `20260904_1200` release or the trained ablation evaluation directory. Their absence from GitHub is not evidence of absence on the workstation. This brief is grounded in the supplied report and image, not a claimed inspection of those local CSVs. The local preflight must resolve them. **Never silently fall back to an older Figure 5 release or older A0 evaluation.** If an exact input is unavailable, identify it and block only the affected formal output; do not substitute a visually plausible result.

The uploaded report used to prepare this brief has SHA-256:

```text
84d5c9fc708988f51f95d4668ef84786822367570e19bd20f71c466e632a1138
```

Record the hash of the local report as well. A different hash requires a content/provenance comparison, not an automatic assumption that the local file supersedes these frozen inputs.

---

## 2. Source registry and immutable evidence packages

Keep two separate evidence packages throughout the build.

### Package B: existing benchmark/ensemble evidence

Resolve this through `20260904_1200` and its accepted source manifests. The supplied prior V6 quantitative companion identifies these lineage sources:

```text
Dis_SI_Process/results/derived/20260831_1409/figure5_v5_source.csv
Dis_SI_Process/results/ValidationV51/PanelC/panel_c_exploration_20260902_1129/selective_risk.csv
Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_plot_source_common_b32.csv
Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_stage_source_common_b32.csv
Dis_SI_Process/results/ValidationV51/InferenceMemory/inference_memory_native_v51_20260903_094021/inference_memory_summary.csv
```

These are lineage pointers, not permission to replace missing current-release manifests. Preserve the existing normalized CRPS, spread--error association, selective-risk curve, benchmark accuracy and measured training/inference costs. The benchmark DMF-Gen row has error **0.117**; it must not become **0.106321** simply because a newer full-reference ablation run exists.

The prior companion's benchmark DMF checkpoint hash prefix is `857a505ff96c`. Confirm the exact current-release identity locally. Do not assign its 64-draw UQ or resource measurements to the newer ablation checkpoint.

### Package A: new saved-checkpoint comparisons

Use a resolved evaluation root `EVAL_ROOT`:

```text
0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910
```

Required inputs, as listed in the supplied report:

```text
EVAL_ROOT/checkpoint_inputs/A0/snapshot_manifest.json
EVAL_ROOT/evaluation_environment.json
EVAL_ROOT/validation_summary.json
EVAL_ROOT/checkpoint_sensitivity_vs_last.csv
EVAL_ROOT/A0_training_progress.csv

EVAL_ROOT/last/metrics/summary_metrics.csv
EVAL_ROOT/last/metrics/per_state_metrics.csv
EVAL_ROOT/last/metrics/paired_differences.csv
EVAL_ROOT/last/metrics/cache_audit.csv
EVAL_ROOT/last/metrics/analysis_metadata.json

EVAL_ROOT/best/metrics/summary_metrics.csv
EVAL_ROOT/best/metrics/per_state_metrics.csv
EVAL_ROOT/best/metrics/paired_differences.csv
EVAL_ROOT/best/metrics/cache_audit.csv
EVAL_ROOT/best/metrics/analysis_metadata.json

EVAL_ROOT/high_frequency/analysis_metadata.json
EVAL_ROOT/high_frequency/summary_high_frequency.csv
EVAL_ROOT/high_frequency/paired_high_frequency.csv
EVAL_ROOT/high_frequency/per_state_high_frequency.csv
EVAL_ROOT/high_frequency/population_U1_spectra.csv
EVAL_ROOT/high_frequency/U1_hann_robustness_summary.csv
```

Discover the exact locations of policy-specific `provenance_A*.json` and `manifest_A*.csv` from the evaluation tree. Read them rather than guessing their directory nesting. Resolve training histories through those run paths. Missing histories do not justify reconstructing curves from endpoint summaries.

The report identifies the original sensor plan as:

```text
/data/wanglz/Cache/PhyCoFlow_TurbulentCombustion_Process_Results/SensorPlans/SensorPlan_paper_full_20260711.csv
```

Use recorded hashes to validate identity. Do not create a new sensor plan. Heavy checkpoints, HDF5 files and reconstruction caches stay where they are; reference paths/hashes instead of copying them into `Dis_SI_Process/`.

### Identity rule

Use distinct internal keys such as:

```text
benchmark_dmf
ablation_full
ablation_no_sensor_feedback
ablation_no_local_conditioning
ablation_local_only_conditioning
ablation_iid_prior
deterministic_objective_control
```

Never join Package A and Package B merely on the display string `DMF-Gen`. Every derived row must carry `evidence_package`, checkpoint policy/hash, cohort identity, metric definition and source path.

---

## 3. Scientific display names and order

Use the following fixed scientific order, not a performance-sorted order. Internal A-codes remain only in manifests and the internal mapping file. Do not print A0--A5 as published method names.

| Internal run | Figure label | Expanded SI definition | Main column |
|---|---|---|---|
| A0 | Full model | DMF-Gen, newly trained full ablation reference with RFF Gaussian prior | Yes, first |
| A2 | No sensor feedback | Remove global-latent-to-sensor feedback; retain sensor-to-latent reasoning, local gathering and global query readout | Yes, second |
| A3 | No local conditioning | Zero local query-to-sensor Top-K/RBF conditioning; retain global conditioning | Yes, third |
| A5 | Local-only conditioning | Retain sensor-token Top-K/RBF evidence while bypassing global observation-conditioning routes | Yes, fourth |
| A4 | IID Gaussian prior | Replace the coherent RFF reference with IID Gaussian values; retain the architecture | Yes, fifth |
| A1 | Deterministic regression | Same-backbone direct-field objective; no prior draw or flow integration | Separate SI control |

Use two-line figure labels when necessary, for example `No sensor\nfeedback`, `No local\nconditioning`, `Local-only\nconditioning`, `IID Gaussian\nprior`. In the prior-focused panel, label A0 as **RFF (full)** and A4 as **IID**.

Place a subtle grouping gap or small separator before the IID-prior row, distinguishing conditioning-route variants from the source-prior change. Do not call A5 bare RBF interpolation or call A3 Perceiver. A5 removes multiple routes and has much less actively optimized capacity; it is not a single-edge, equal-capacity control.

---

## 4. Frozen numerical anchors: verify, do not hard-code into plots

Read full-precision values from the machine-readable sources. The following report values are regression anchors, not plot inputs. Compare at the precision displayed here, preferably by formatting the computed values to the same number of decimals. Record any disagreement and stop promotion of the affected panel.

### 4.1 Primary physical relative-L2

The unobserved mean is the equally weighted arithmetic mean of per-state CH4, CO, U1 and p relative errors. Temperature is excluded from that macro mean.

| Display label | Last mean | Last 95% block-20 interval | Last change vs full | Best mean | Best 95% block-20 interval |
|---|---:|---|---:|---:|---|
| Full model | 0.106321 | [0.104495, 0.108229] | reference | 0.107893 | [0.106044, 0.109840] |
| No sensor feedback | 0.126974 | [0.124868, 0.129248] | +19.4% | 0.129288 | [0.127076, 0.131656] |
| No local conditioning | 0.144703 | [0.142757, 0.146664] | +36.1% | 0.146284 | [0.144486, 0.148103] |
| Local-only conditioning | 0.354951 | [0.351422, 0.358696] | +233.8% | 0.356403 | [0.352764, 0.360126] |
| IID Gaussian prior | 0.104294 | [0.102412, 0.106302] | -1.9% | 0.104967 | [0.103069, 0.106949] |
| Deterministic regression, SI only | 0.078705 | [0.077359, 0.080195] | -26.0% | 0.078561 | [0.077237, 0.079971] |

For IID minus full, paired primary differences are:

- Last: **-0.002027**, interval **[-0.002634, -0.001416]**.
- Best: **-0.002926**, interval **[-0.003527, -0.002308]**.

The IID model is lower on **573/1000** last-policy states and **626/1000** best-policy states. These are state fractions, not independent training success rates.

### 4.2 Provenance and budget anchors

| Display label | Last epoch | Best epoch | Parameter elements with optimizer state, last |
|---|---:|---:|---:|
| Full model | 7520 | 7095 | 6,507,151 |
| No sensor feedback | 6000 | 5365 | 5,716,879 |
| No local conditioning | 6000 | 5615 | 5,650,572 |
| Local-only conditioning | 6000 | 5300 | 895,118 |
| IID Gaussian prior | 7310 | 7155 | 6,507,151 |
| Deterministic regression | 7180 | 7140 | 6,507,151 |

These are optimizer-state element counts, not automatically total parameters, trainable-parameter declarations, FLOPs or memory. Report them with their exact meaning. Extract actual update counts, learning-rate horizons and other budget metadata where recorded; mark missing entries as not recorded.

### 4.3 U1 fine-scale anchors

| Quantity | Full last | IID last | Full best | IID best |
|---|---:|---:|---:|---:|
| Canonical high-band power/truth | 0.4285 | 3.1199 | 0.4227 | 3.2379 |
| High-band relative-L2 | 0.873 | 1.825 | 0.874 | 1.857 |
| Whole-spectrum LSD, dB | 4.5602 | 3.7883 | 4.5819 | 3.8503 |
| High-band residual energy / full truth fluctuation energy, % | 0.0312 | 0.1317 | 0.0313 | 0.1363 |

The last-policy IID-minus-full high-band U1 relative-L2 difference is **+0.952 [0.910, 1.001]**. The best-policy difference is **+0.983 [0.942, 1.031]**.

The truth U1 high band contributes **0.0459%** of its complete centered fluctuation energy on average. This explains why a large relative error within this weak-energy band can coexist with a small change in the global metric. It is **not** an exact decomposition of the primary physical relative-L2.

### 4.4 All-field high-band residuals for SI

| Field | Full last | IID last | Paired difference, last | Full best | IID best |
|---|---:|---:|---|---:|---:|
| CH4 | 0.275 | 0.408 | +0.133 [0.128, 0.138] | 0.277 | 0.407 |
| CO | 0.833 | 1.115 | +0.282 [0.270, 0.295] | 0.837 | 1.115 |
| T | 0.544 | 0.801 | +0.257 [0.246, 0.268] | 0.548 | 0.801 |
| U1 | 0.873 | 1.825 | +0.952 [0.910, 1.001] | 0.874 | 1.857 |
| p | 0.545 | 0.611 | +0.065 [0.058, 0.074] | 0.547 | 0.608 |

The exact paired differences must come from the source, not subtraction of rounded table entries. All ten IID-minus-full differences remain positive at block lengths 5, 20 and 50 in the report.

---

## 5. Main-figure layout: create a real third ablation column

The existing figure already has three small upper plots and one full-width scorecard. Do not append a narrow fourth strip or shrink the existing complete SVG to two-thirds width. Recompose its panels from the underlying artists/renderers at readable final font sizes.

Use this six-panel layout as the primary delivery:

```text
┌──────────────────────┬──────────────────────┬─────────────────────────┐
│ a  CRPS              │ b  Spread–error      │ c  Conditioning and     │
│ existing evidence    │ existing evidence    │    source variants      │
├──────────────────────┴──────────────────────┼─────────────────────────┤
│ d  Selective reconstruction                 │ e  Fine-scale velocity  │
│ existing curve, spanning two columns        │    fidelity             │
│                                            │ two compact axes        │
├────────────────────────────────────────────┴─────────────────────────┤
│ f  Accuracy + training/inference footprint, all five original columns│
└──────────────────────────────────────────────────────────────────────┘
```

This gives the user a dedicated right-hand architectural/prior column while retaining the entire existing benchmark scorecard at full width. Start at **183 mm wide and approximately 190 mm high**, increasing height modestly if necessary rather than reducing font sizes. These are project design targets, not a statement about current journal page limits.

Keep all five scorecard columns, in the established order. Do not remove the original selective-risk panel. Its wider placement can use more horizontal whitespace; do not add a new metric merely to fill the available width.

### Panel-reference mapping

| Existing V6 label | V7 label | Role |
|---|---|---|
| a | a | Normalized CRPS |
| b | b | Spread--error association |
| c | d | Normalized selective risk |
| d | f | Five-column accuracy/resource scorecard |
| New | c | Saved-checkpoint stochastic architecture/prior comparison |
| New | e | RFF versus IID fine-scale velocity comparison |

Write `panel_reference_map.csv` and a commented LaTeX reference-update snippet. Do not automatically edit the manuscript.

### Inherited panels: preserve evidence, allow layout-only changes

Preserve the original source rows, central values, uncertainties, data transforms, method order, color/shape identity, and inference-memory encoding. Reposition and resize axes without changing scientific coordinates. Reusing an old scientific-data digest is preferable to requiring identical SVG pixel geometry after recomposition.

Two permitted wording edits:

- `Training time` becomes **Training update time**; the axis stays `ms / update`.
- Remove the isolated `DMF AURC = 0.741` annotation or replace it with **4.6% lower error at 80% retained**. Retain the complete five-method AURC table in the report/SI. The original area is over 0.2--1.0 and is not divided by 0.8.

Preserve filled **Model** and hollow **Peak** inference-memory endpoints. An older YAML contains an opposite open/filled mapping; the accepted renderer, image and companion establish the displayed meaning. Do not let that stale configuration reverse the key.

Do not rename benchmark methods such as Geo-FNO in this plotting task; record any outstanding naming audit separately rather than silently changing the inherited comparison.

### Typography and visual style

Use the latest local release's installed Arial registration and vector-text workflow; do not distribute font files. Preserve red for the DMF family. Use neutral slate/grey for route-removal variants and a distinct muted amber for the IID-prior variant. Define these in the new configuration rather than adding arbitrary colors in individual scripts. The separate ablation labels distinguish configurations from benchmark model identities.

At final width, aim for axis labels around 6.5--7 pt and tick/row text at least approximately 5.8--6 pt; do not make inherited annotations smaller than in the supplied release. Panel tags should remain approximately 8--9 pt and consistently aligned. Check all labels at the actual 183-mm output width. Use a pale red full-reference row highlight, thin guides, and restrained annotations. No radar plots, aggregate fidelity scores, significance stars, bar-area encodings, dual-y-axis charts, or unlabeled axis breaks.

If the third-column composition fails the text-overlap/readability checks after reasonable height adjustment, deliver it as a clearly labelled layout candidate and also provide a third-row fallback using exactly the same scientific panels. Do not change the metric selection to make a layout appear successful.

---

## 6. New main panel c: conditioning and source variants

### Question

How do the saved stochastic configurations compare in reconstruction error when sensor feedback, local evidence, global observation routes or prior coherence are changed?

### Data and statistic

- Package A only; the five stochastic configurations in the fixed order in Section 3.
- Primary policy: **last.pt** for every configuration.
- 1,000 matched evaluation states, 256 temperature observations, all 40,300 queries.
- One draw per generative state, two Euler steps, recorded hard observed-entry clamp, no EMA substitution.
- Horizontal coordinate: mean **physical unobserved-field relative-L2**.
- Interval: the accepted 95% circular moving-block-bootstrap interval of that mean, block length 20 and 2,000 replicates.

### Visualization

Use a **horizontal point-and-interval plot**, not a violin or a composite score.

- X-axis label: `Unobserved-field relative L2`.
- Begin with a linear range approximately 0.08--0.38; adjust only to include all valid intervals and annotations. No broken axis. A nonzero lower limit is acceptable because the marks are points, not bars.
- Large hollow red circle for Full model, grey points/shapes for the three route-removal variants, and a muted-amber diamond for IID Gaussian prior.
- Use two-line row labels where needed and a small grouping gap before IID.
- A thin vertical reference marks the full-model mean, **0.106321**. Label it as the full ablation reference, not the archived benchmark checkpoint.
- Print the full-model mean as **0.1063**. Print other row changes as **+19.4%**, **+36.1%**, **+233.8%**, **-1.9%**. Absolute means remain in the source table and companion; include them in the figure only if space permits without collision.
- A short caption clause states that intervals are across evaluation states, not training repeats.

The percent annotation is

```text
100 * (mean_variant / mean_full - 1).
```

It is a ratio of cohort means, not the average of statewise percent ratios. Error bars represent uncertainty in the displayed raw mean; they do not become CIs for that percent annotation. Paired effect intervals belong in the SI table. Do not derive them from overlap/non-overlap of the two marginal intervals.

### Interpretation

Show the IID advantage with its negative sign. State that feedback/local-route variants have larger error in their available trained implementations. The local-only model is about **3.34 times** the full model's error, not 3.34 times more accurate or a capacity-controlled estimate of the isolated global route's causal effect.

---

## 7. New main panel e: fine-scale velocity fidelity

### Question

What does the small IID-prior improvement in whole-field error conceal at the high-frequency end of the reconstructed U1 field?

### Layout and data

Use **two stacked compact horizontal point-and-interval axes** within one labelled panel. Both compare only **RFF (full)** with **IID**, using Package A, last.pt, all 1,000 states. This is a focused source-prior diagnostic, not a new overall model ranking.

Upper axis:

- Quantity: **canonical high-band power ratio**, prediction divided by truth.
- Full reference **0.4285**; IID **3.1199**.
- Axis label: `High-band power / truth`.
- Dashed vertical line at **1**, explicitly labelled `truth` or `ideal = 1`.
- Linear range approximately 0--3.8, extended if needed for valid intervals.
- Display annotations **0.43** and **3.12**.
- Do not add a `lower is better` arrow: values below one indicate attenuation.

Lower axis:

- Quantity: **phase-sensitive high-band relative-L2**.
- Full reference **0.873**; IID **1.825**.
- Axis label: `High-band relative L2`.
- Ideal value **0**; lower is better.
- Linear range approximately 0--2.1, extended for intervals if necessary.
- Display annotations **0.87** and **1.83**.

Use the same color/shape association in both axes. A thin neutral connector between the two model means may help show the contrast, but it is not a training trajectory or an intervention effect. Mean intervals are 95% block-20 intervals. Reuse accepted intervals where available; otherwise calculate the mean intervals from the saved per-state rows with the exact bootstrap procedure in Section 10, marking these as newly derived CPU summaries.

A large `3.12 × truth` label is acceptable if it fits, but pair it visibly with the full model's **0.43 × truth**. Do not present the RFF power as ideal. The high-band error axis is the stronger fidelity discriminator because it also detects phase mismatch.

### Exact metric boundary

These are **index-space high-band diagnostics**, not a physical-wavenumber turbulent kinetic-energy spectrum. The grid has 403 × 100 points and nonuniform physical spacing. The high band uses the existing strict rule:

```text
k > (2/3) * k_max
```

with the same retained-shell mask, isotropic cutoff, minimum four modes per shell and exclusion of DC/underpopulated shells. The report identifies 68 high-band shells and 17,704 Fourier modes. Preserve and verify this definition.

The canonical power ratio integrates **shell-mean power using the existing trapezoidal weighting**. The high-band residual uses **individual complex FFT coefficients over the high-band mode mask**. These share the same band but use different weighting conventions. The caption/SI must define both.

Do not substitute the mode-sum power ratios **3.206 / 3.327**, population medians, the ratio of pooled spectra, or Hann-windowed metrics for **3.1199 / 3.2379**. They answer related but distinct questions.

### Meaning

At the tested two-step sampler, the IID-source run has slightly smaller global error but excess fine-scale velocity power and roughly twice the high-band reconstruction error. The full model attenuates that band yet has lower phase-sensitive residual error. The SI must show that the high-band error increase also occurs in all five fields and survives both checkpoint policies; the main figure does not need five additional field panels.

---

## 8. Supplementary figure package

Create the following six focused SI figures, plus standalone subpanels and compact tables. Use clear names and stable LaTeX labels, not speculative manuscript S-numbers. All formal stochastic panels use the five-name order in Section 3. The deterministic comparison is isolated in SI-06.

### SI-01: complete reconstruction and conditioning-route effects

Suggested filename: `si_ablation_reconstruction_<timestamp>.svg`.

Show last/best fieldwise physical relative-L2 for CH4, CO, T, U1 and p, with temperature identified as the observed field. An annotated effect heatmap can display `log2(mean_variant / mean_full)` with zero as the reference; use a range containing all values rather than saturating the local-only row invisibly. Supply raw means and intervals in a companion table. Add paired macro/field effect intervals, and show the additional local-only versus no-feedback comparison separately.

Include standardized-field relative-L2 and truth-fluctuation-normalized error in the associated tables. They are different normalizations, not interchangeable versions of the primary metric. In particular, retain the pressure fluctuation check rather than relying solely on its small physical relative-L2.

Extract the strong spatial-localization detail: removing local conditioning raises CH4 relative-L2 from **0.057360 to 0.137979 (+140.6%)** under last.pt. Do not infer a physical length scale from this scalar error alone.

### SI-02: all-field scale-resolved source-prior comparison

Suggested filename: `si_ablation_highband_all_fields_<timestamp>.svg`.

Show full versus IID high-band relative-L2 across all five fields, separately for last and best; use paired point/interval or dumbbell plots with paired effect intervals in the table. Add canonical high-band power ratios with a clearly marked ideal value of one. Retain high-band results for the other stochastic configurations in the machine-readable and LaTeX tables.

Include the all-field whole-spectrum LSD and low/middle/high-band energy ratios in supplementary tables. The original whole-spectrum U1 LSD improves for IID; it must remain visible alongside its high-band deterioration. Do not construct an overall fidelity score to force a single winner.

### SI-03: U1 spectra, prevalence and leakage sensitivity

Suggested filename: `si_ablation_u1_spectral_audit_<timestamp>.svg`.

Use existing population spectra to plot the per-state **median shell-power ratio and IQR**, full versus IID, across normalized retained index-space wavenumber. Keep last/best distinguishable through separate facets or line styles. The horizontal truth ratio is one; shade the strict high band. State that IQR shading describes state dispersion, not a confidence band for a mean. Do not integrate the median-ratio curve to reproduce a mean integrated band ratio.

Add empirical cumulative distributions for the **mode-sum** high-band power ratio and high-band residual. Report the existing IID prevalence checks: mode-sum power exceeds truth in **98.3% / 98.8%** of states, and exceeds twice truth in **77.3% / 80.3%**, last/best. Do not attach those percentages to the canonical trapezoidal ratio.

Include a compact Hann-sensitivity display/table, explicitly separate from the primary untapered estimator:

| Policy | Full Hann power ratio | IID Hann power ratio | Full Hann high-band L2 | IID Hann high-band L2 |
|---|---:|---:|---:|---:|
| last | 0.075 | 10.306 | 1.009 | 3.222 |
| best | 0.074 | 10.685 | 1.009 | 3.276 |

Report the weak-energy-band explanation: mean truth high-band U1 energy fraction **0.0459%**, and residual-budget fractions from Section 4.3. Do not combine means of ratios into an exact global-MSE identity. Tapering does not remove the IID excess in this diagnostic; it also reveals stronger RFF attenuation. Neither result identifies a universal mechanism under other solvers or training seeds.

### SI-04: coupled distributions and histogram-range sensitivity

Suggested filename: `si_ablation_coupling_range_audit_<timestamp>.svg`.

For T--U1, CH4--U1 and p--U1, show the original truncated-range paper JSD, overflow-retaining JSD and retained predicted-pair fraction. Keep last/best policies separate. Use common original edges for all methods, not per-method or per-policy quantiles. Retain the original 64 × 64 and overflow 66 × 66 definitions.

The central audit is CH4--U1: full-model predicted retention is **65.447% last / 56.359% best**, against **99.000% truth**, while overflow JSD is **0.201011 / 0.320323**. A small truncated-range JSD does not establish fidelity of discarded mass. Show the ranking changes, including the local-conditioning removal's lower CH4--U1 overflow JSD and the policy-dependent IID comparison. Do not choose only the policy or histogram definition that favors the full model.

Keep this analysis in SI, not the new headline column. Include Pearson coupling error as a supplementary scalar check, identified as a coarse statistic rather than a substitute for JSD or spatial alignment.

### SI-05: checkpoint sensitivity and available training histories

Suggested filename: `si_ablation_checkpoint_sensitivity_<timestamp>.svg`.

Plot paired best-minus-last changes in macro physical relative-L2 and in the more sensitive coupling diagnostics using the existing 5/20/50-block outputs. Include both policy values rather than just claiming unchanged aggregate ranking.

Inspect recorded learning curves for the five stochastic variants. If present, show the actual recorded velocity-loss histories against optimizer update or epoch, with true endpoints and selected-checkpoint markers. Do not align unlike x-axes without converting from recorded metadata; do not divide losses by their own minima or visually truncate longer runs at 6000 to imply budget matching. Separate training from held-out validation objectives. Any display smoothing must be supplemental to visible raw histories and documented; prefer raw or binned summaries without altering the retained data.

The existing full-reference progress from **0.111397 at epoch 6135** to **0.106321 at epoch 7520** can be included as recorded historical reconstruction endpoints, not a fabricated dense error curve or an independent replicate. If histories are absent, deliver the checkpoint-sensitivity plot and provenance table, with the history panel explicitly unavailable in the completion report.

### SI-06: deterministic objective control

Suggested filename: `si_deterministic_objective_control_<timestamp>.svg`.

Compare Deterministic regression with Full model as a separate objective comparison, not one of the main stochastic architectural/prior rows. Include macro and fieldwise physical relative-L2 under both policies, pressure fluctuation error, and the coupling diagnostics with their retention/overflow companions.

State the observed aggregate result: **0.078705 versus 0.106321** for last and **0.078561 versus 0.107893** for best. Preserve the **26.0% / 27.2%** deterministic advantages. The full model has lower pressure error in both policies, and the last-policy truncated JSD is lower for all three pairs, but best-policy and overflow comparisons change parts of that interpretation.

This is a deterministic forward prediction versus one sampled two-step reconstruction. Do not call it an ensemble-mean comparison, infer calibration/diversity, or borrow the archived benchmark's 64-draw performance for the new full checkpoint. Do not add CRPS to this package from one generative draw as a substitute for the missing matched-ensemble test.

### Admissibility: required tables, optional gallery

Admissibility needs full tables but does not require another main figure. Retain negative-species frequency, frequency below -0.0001, negative-part magnitude/norm, exact nonpositive T/p counts, global minima, and affected state/time identities. Temperature sensor-excluded error and imposed sensor consistency belong here or in SI-01's tables.

Rounded `0.0000%` is not evidence of zero nonpositive-temperature predictions: IID has **three points** under each policy and local-only has **one** under last. All counts use **40,300,000** points per variant/policy. Record the IID minimum temperatures **-1648.564697** and **-1764.966797** exactly enough to expose severity. Do not clip predictions for any table or spectrum.

An optional cached-field gallery can illustrate one objective-selected typical high-band state and the recorded rare extreme states. It must use already saved fields, show the selection rule, preserve shared color limits and avoid any fresh inference. Missing full-field caches must not block the primary metric figures.

---

## 9. LaTeX tables and reusable inserts

Generate tables from validated derived data, not manual transcription of this brief. Use `booktabs`, ordinary tabular structures and, when helpful, existing project `siunitx` conventions. Split wide tables by policy/metric rather than using unreadably small `resizebox` text.

Required table groups:

| Table stem | Contents |
|---|---|
| `tab_ablation_provenance` | Scientific names, last/best epochs, exact checkpoint hashes in companion, update counts when known, LR horizon when known, optimizer-state element counts, selection-objective identity, endpoint/split caveats |
| `tab_ablation_reconstruction` | Per-field physical/standardized/fluctuation-normalized errors, macro errors, mean CIs, paired differences, fractions improved, both policies |
| `tab_ablation_spectral` | Whole-spectrum LSD and canonical total/middle/high-band power ratios for five fields and stochastic variants, with low-band values if present in source |
| `tab_ablation_high_frequency` | High-band residuals, paired prior effects, mode-sum quantiles/prevalence, fluctuation-energy budgets and separate Hann checks |
| `tab_ablation_coupling` | Paper JSD, overflow JSD, predicted/truth retention, paired effects, Pearson coupling-error check, both policies |
| `tab_ablation_admissibility` | Negative excursions, sign-error thresholds, exact event counts, minima, affected time IDs, temperature sensor-excluded error |
| `tab_deterministic_control` | Separate full-versus-deterministic objective comparison, both policies and supported tradeoffs |

Long tables can be split into several files retaining the table stem. Include all stochastic variants, including the advantageous IID results. Keep the deterministic objective group separate, while its checkpoint provenance remains in the full audit.

Provide:

```text
figure5_v7_caption.tex
figure5_v7_ablation_paragraph.tex
figure5_v7_panel_reference_updates.tex
si_ablation_methods.tex
si_ablation_results.tex
si_ablation_figures.tex
si_ablation_tables.tex
```

Use stable labels such as `fig:si-ablation-highband`, not fixed S-numbers. Supply LaTeX figure environments referencing the delivered high-resolution PNG renderings through `\includegraphics`; editable SVGs remain the authoritative artwork. This avoids requiring a new SVG-conversion toolchain for a copy-ready draft. Default output is SVG plus PNG; do not create a new PDF delivery unless separately requested.

### Suggested main-figure caption skeleton

Validate/update this from the actual V7 mapping and local inherited sources before delivery:

```latex
\caption{\textbf{Conditional ensembles, architectural controls and
scale-resolved reconstruction fidelity.}
All evaluations use temperature-only conditioning with 256 measurements
and 40,300 output points.
\textbf{a}, Normalized empirical CRPS for the benchmark generators.
\textbf{b}, Association between ensemble spread and ensemble-mean error.
\textbf{c}, Mean unobserved-field relative-$L_2$ for the separately
trained stochastic configurations; annotations give changes relative to
the new full reference. Intervals summarize 1,000 evaluation states.
\textbf{d}, Selective reconstruction using ensemble spread, normalized
by each method's full-cohort error. Panels a, b and d use the existing
200-state, 64-draw benchmark evaluation.
\textbf{e}, High-band $U_1$ power relative to truth and phase-sensitive
high-band relative-$L_2$ for the RFF and IID source models. Ideal values
are one and zero, respectively. The high band is the upper third of
retained index-space wavenumbers; power ratios use the canonical
shell-integration estimator.
\textbf{f}, The original 1,000-state benchmark accuracy and separately
measured training/inference footprints. Training uses common batch size
32 with method-native target workloads; inference uses batch size one.
Filled and hollow memory symbols denote model state and peak allocation.
Panels c and e use last checkpoints, one draw and two Euler steps, with
95\% circular moving-block intervals of length 20. Their full-reference
checkpoint differs from the benchmark checkpoint in a, b, d and f.
Training endpoints differ across configurations; the checkpoint ledger,
validation/test split, best-checkpoint sensitivity and separate
deterministic objective control are reported in the Supplementary
Information.}
```

The final caption and companion must also identify inherited block length 25 for the UQ panels. Do not describe the 1,000 states as independent trajectories or an untouched test set.

### Suggested ablation paragraph, to generate from the validated last-policy table

```latex
The saved stochastic configurations distinguish conditioning-route effects
from the source-prior tradeoff (Fig.~\ref{fig:uq_cost}c,e). The new full
reference achieved mean unobserved-field relative-$L_2$ of 0.1063.
Removing global-to-sensor feedback or local query conditioning increased
this error to 0.1270 and 0.1447, respectively; local-only conditioning
reached 0.3550. The IID-prior run instead reduced whole-field error to
0.1043, while its high-band velocity power reached 3.12 times the
reference-field power and its high-band relative-$L_2$ rose to 1.83,
compared with 0.43 and 0.87 for the full RFF model. The full model
therefore attenuated high-band power, whereas the IID configuration
combined a small improvement in the bulk error with larger fine-scale
residuals. The high-band residual increase occurred in all five fields
and retained its direction under both checkpoint policies
(Supplementary Fig.~\ref{fig:si-ablation-highband}). These comparisons
characterize the available trained configurations; their unequal
endpoints and the separate deterministic objective control are reported
in the Supplementary Information.
```

Do not add a sentence claiming that this proves the generative objective outperforms deterministic regression. Do not use the A1 control's results to assert a benefit of conditional variability that was not measured.

---

## 10. Statistical and metric contract

### State-level aggregation and pairing

For each state i and channel c:

```latex
E_{ic}
=\frac{\|\widehat u_{ic}-u_{ic}\|_2}
{\|u_{ic}\|_2+10^{-12}},\qquad
E_i=\frac14\sum_{c\in\{\mathrm{CH_4},\mathrm{CO},U_1,p\}}E_{ic}.
```

Use physical-unit fields for this primary metric. Do not average fields before taking their norms. Do not pool pixels or physical units across fields. Retain state IDs and original time indices; pair variants and checkpoint policies by exact identity, not CSV row order or rounded times.

For paired changes, use `d_i = E_variant,i - E_full,i` and bootstrap the paired differences. All 1,000 states receive equal weight.

### Bootstrap

Reuse accepted source intervals and tests wherever available. For new summaries only, use the established **circular moving-block bootstrap** of temporally sorted held-out states: 2,000 replicates, primary block length 20, sensitivities 5 and 50. Use common resampled state-index blocks across the paired configurations and across numerator/denominator reductions. Recover original seeds from metadata; the high-frequency audit records seed **20260910**. Record any newly introduced deterministic analysis seed separately.

The block length is a number of sorted held-out states, not a fixed number of original simulation timesteps. Do not silently fill gaps between held-out states, replace this with iid bootstrap, or reinterpret the intervals as variability across training seeds. Equal training seed strings do not prove identical initial backbone tensors when different priors consume different random streams.

No p-value stars or multiplicity-adjusted claim is needed. The source intervals are descriptive and conditional on the saved checkpoint, one sensor plan and one draw. Distinguish a mean CI from an IQR/p95 population interval in every panel and table.

### High-band residual and power

For the complex FFT coefficients P and T of centered predicted and true fields and the preserved high-band mode set H:

```latex
E_H=\left(
\frac{\sum_{q\in H}|P_q-T_q|^2}
{\sum_{q\in H}|T_q|^2}
\right)^{1/2},\qquad
R_{H,\mathrm{modes}}
=\frac{\sum_{q\in H}|P_q|^2}
{\sum_{q\in H}|T_q|^2}.
```

Canonical power instead uses the existing trapezoidal integral of shell-mean spectra over the same high-band shells. Import/reuse the existing definition rather than replacing it with an unweighted shell sum. Compute per-state ratios first and then their means. Check that no denominator is zero/nonfinite; do not quietly discard states or add a new regularization constant to change the published estimator.

The complete truth fluctuation denominator includes all centered FFT modes, including those outside the retained radial shells. Consequently, the reported high-band energy fractions are not interchangeable with percentages relative to retained-shell energy or the full non-centered physical-field norm.

Keep the untapered and Hann-window estimators separate. No smoothing, clipping of values, clipping of spectra or frequency-threshold optimization is allowed. The U1 focus is motivated by the reported concern; the SI all-field table documents the other channels and prevents selecting only a favorable metric.

### Joint-PDF metrics

Reuse base-2 JSD and its original edge/pseudocount policy. Do not confuse divergence with its square root. Original JSD is conditional on retained histogram range; underflow/overflow JSD includes all finite points. Every coupling table or claim must identify the estimator and checkpoint policy. Retention is not calibration or physical admissibility.

---

## 11. Implementation plan and standard output structure

Adapt the existing source loaders, plotting styles, SVG QA and manifest utilities instead of duplicating the entire V6 pipeline. Keep old scripts and releases unchanged; add new V7 files.

Suggested new files (these are requested interfaces to implement, not claims that they already exist):

```text
Dis_SI_Process/configs/figure5_v7_ablation.yaml
Dis_SI_Process/scripts/collect_figure5_v7_ablation_sources.py
Dis_SI_Process/figures/scripts/build_figure5_v7_ablation.py
Dis_SI_Process/figures/scripts/build_ablation_si_v7.py
Dis_SI_Process/scripts/audit_figure5_v7_ablation.py
Dis_SI_Process/tests/test_figure5_v7_ablation.py
```

Use one timestamp for the complete build and preserve the existing directory convention:

```text
Dis_SI_Process/
  results/derived/<timestamp>/
    source_manifest.json
    ablation_name_map.csv
    panel_reference_map.csv
    ablation_provenance.csv
    ablation_primary_summary.csv
    ablation_primary_paired.csv
    ablation_highband_summary.csv
    ablation_highband_paired.csv
    ablation_spectral_summary.csv
    ablation_coupling_audit.csv
    ablation_admissibility.csv
    deterministic_control_summary.csv
    checkpoint_sensitivity.csv
    figure5_v7_display_source.csv
    build_manifest.json
    qa.json
  figures/generated/<timestamp>/
    fig5_composed_v7_<timestamp>.svg
    fig5_composed_v7_<timestamp>.png
    fig5a_crps_<timestamp>.svg
    fig5b_spread_error_<timestamp>.svg
    fig5c_architecture_prior_<timestamp>.svg
    fig5d_selective_risk_<timestamp>.svg
    fig5e_highband_velocity_<timestamp>.svg
    fig5f_accuracy_footprint_<timestamp>.svg
    [matching standalone PNG renderings]
    si/
      [six SI composites, standalone subpanels, SVGs and PNGs]
  docs/generated/<timestamp>/
    figure_contract.md
    quantitative_figure_making_report.md
    fig5a_companion.md ... fig5f_companion.md
    si_companions/
    checkpoint_and_scope_notes.md
    figure5_v7_caption.tex
    figure5_v7_ablation_paragraph.tex
    figure5_v7_panel_reference_updates.tex
    si_ablation_methods.tex
    si_ablation_results.tex
    si_ablation_figures.tex
    si_ablation_tables.tex
    tables/
      [generated LaTeX tables]
    completion_report.md
```

Each new standalone subpanel needs a PNG preview, editable SVG and companion. Avoid writing a second 1.44-million-row high-frequency source copy: retain the existing raw file and store its hash/path plus the small summaries and displayed per-state columns that are actually needed. Do not modify raw metrics, cached reconstructions, checkpoint manifests or source reports.

### Requested command sequence

Implement compatible CLI options, then document and execute the actual resolved commands. Example interface:

```bash
STAMP=$(date +%Y%m%d_%H%M)
EVAL_ROOT=0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910
BASE_RELEASE=Dis_SI_Process/figures/generated/20260904_1200

conda run -n fig python Dis_SI_Process/scripts/collect_figure5_v7_ablation_sources.py \
  --evaluation-dir "$EVAL_ROOT" --benchmark-release "$BASE_RELEASE" \
  --timestamp "$STAMP" --strict-formal

conda run -n fig python Dis_SI_Process/figures/scripts/build_figure5_v7_ablation.py \
  --timestamp "$STAMP" --layout third-column --strict-formal

conda run -n fig python Dis_SI_Process/figures/scripts/build_ablation_si_v7.py \
  --timestamp "$STAMP" --strict-formal

conda run -n fig python Dis_SI_Process/scripts/audit_figure5_v7_ablation.py \
  --timestamp "$STAMP" --strict-formal

conda run -n fig python -m unittest Dis_SI_Process.tests.test_figure5_v7_ablation
```

Use the project's actual Python environment launcher, including `rtk proxy` if required locally. Do not assume a new command exists before implementing it, and do not install packages or trigger inference as an automatic fallback for missing metrics.

---

## 12. Quantitative figure-making report: required content

The report must be sufficient to write the main text and SI without reading plotting code. For every main and SI panel, record:

1. Scientific question and why this panel is included in the evidence sequence.
2. Package identity; exact input files and hashes; selected source columns and filters; checkpoint identity/policy; states, channels, measurements and draw count.
3. Equation or exact metric definition; units; ideal value; normalization; whether it is phase-sensitive, range-truncated, a marginal score or a population summary.
4. Aggregation order, source of intervals, bootstrap blocks/replicates/seed, and the distinction between confidence intervals and state dispersion.
5. All plotted numerical coordinates and interval endpoints at full source precision, plus the formatted annotation strings.
6. Derived comparisons and their definitions: ratio of means versus mean of ratios; paired changes; fractions improved; both checkpoint policies in SI.
7. Visual encoding: point/line/shape/color mapping, axes transformations and limits, guide lines, annotation placement, and every layout-only difference from the inherited figure.
8. Supported result and a separate boundary statement specific to that measurement. Include improvements by IID and deterministic regression, full-model high-band attenuation, unequal epochs/capacity, shared validation/test holdout and single-draw scope.
9. Relationship to the prior panel and to the central function-space reconstruction argument. Avoid treating all metrics as one rank.
10. QA status, missing inputs, any metric disagreement, and whether the output is a formal source-validated figure or a blocked/draft candidate.

Add a figure-level checkpoint map making the archived benchmark and new full ablation reference unmistakable. Include the prior-source tradeoff in one compact table, the old-to-new panel map, and the exact command/environment/git record. Link every reported number to a derived row or a raw source identity.

---

## 13. Strict-formal acceptance gates

`--strict-formal` certifies source integrity and faithful display, not matched-budget causal identification. Record scope flags explicitly, for example:

```yaml
comparison_scope:
  type: saved_checkpoint_comparison
  matched_evaluation_states: true
  matched_measurements: true
  matched_training_budget: false
  multiple_training_seeds: false
  independent_untouched_test_set: false
  matched_backbone_initialization_verified: false
  generative_draws_per_ablation_state: 1
  primary_checkpoint_policy: last
  sensitivity_checkpoint_policy: best
  new_ablation_reference_separate_from_benchmark: true
  deterministic_objective_control_retained_in_si: true
```

Validate at least the following:

- Exactly the expected 1,000 paired states per requested model/policy/metric; no duplicate key tuples; complete matching truth/sensor/time identities; all displayed values finite.
- Full-reference checkpoints match the frozen new A0 manifest and expected epochs. Do not follow live-run `last.pt` symlinks that changed after the snapshot.
- No pooling of last and best into 2,000 replicates, no mixing policies by which gives lower error, and no fake multi-seed uncertainty.
- All expected numerical anchors match report precision. Rounding is handled explicitly, not by lax tolerances that permit stale A0 values.
- IID has lower macro L2 in both policies and higher all-field high-band residuals; neither direction is lost through sign errors, relabelling or omitted rows.
- Canonical versus mode-sum versus Hann quantities are separately keyed; the power panel uses ideal 1 and the residual panel ideal 0.
- Old Figure 5 data digests and source identities are unchanged. The new full-reference error is never substituted into the old benchmark row; old CRPS and costs are not presented as remeasured for the new run.
- The deterministic control is present in the separate SI/source record, not mislabelled a failed run or a generative variant.
- No invalid local-only equal-capacity claim, no per-update-to-total-training-cost conversion without recorded update/time evidence, and no uncertainty-quality claim from the single ablation draw.
- JSD tables retain policy, retention and overflow context. Exact temperature event counts remain visible despite rounded percentages.
- SVGs parse, preserve editable text, contain the expected scientific labels, and have no clipped intervals or off-canvas annotations. Method labels do not display internal A-codes.
- Render at final width and inspect every main/SI panel. Check row alignment, crowded near-equal full/IID means, superscripts/subscripts, and the Model/Peak key. Confirm the scorecard remains readable rather than only looking good enlarged.
- LaTeX inserts have balanced environments/braces, resolved delivered figure paths, escaped underscores/percent signs, and consistent labels. Use the existing table toolchain for a compile smoke test if available, keeping temporary compilation products out of the delivery tree.
- Old artifacts and all source checkpoints/data remain unchanged; record before/after hashes for protected compact artifacts and use the existing checkpoint hash manifests without duplicating weights.

If source integrity fails, write a precise blocked-status report and retain valid independent outputs; do not mark the whole package as passing. The final completion report must distinguish produced artifacts from requested-but-unavailable ones.

---

## 14. Deliverable summary and completion response

Deliver one composed third-column Figure 5, six standalone main panels (panel e contains two separately exportable metric axes), six SI composites with their individual subpanels where applicable, all compact source tables, generated LaTeX tables and figure environments, the detailed numerical report, and passing source/visual QA or an explicit list of blocks.

Do not return only a plan or only screenshots. Report the paths of the completed composite, SI entry-point LaTeX, source manifest and quantitative figure-making report, together with the command to reproduce the bundle. Keep heavy data in place. No training or fresh inference is part of this release.
