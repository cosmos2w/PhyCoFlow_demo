# Author checks: keep scientific correction separate from art-only work

These are specific reconciliation tasks, not generic requests for more experiments. They arise from the supplied PDF and TeX. The art pass must preserve scientific labels and plotted data until the relevant author decision is recorded. A textual correction is not permission to alter a plotted array, cohort, mask or model identity.

## A01 — Fig. 1 architecture and notation (D p. 4; source lines 239–310, 663–710)

The rendered attention diagram appears to place the raw sensor tokens on **Q** and global tokens on **K/V** at the first cross-attention block, whereas the text says that learned global latents query the sensor tokens. The feedback block likewise appears to place enriched global tokens on Q and the sensor-side path on K/V, whereas the text describes sensors querying the global representation. Have the architecture owner trace the actual inputs and decide whether the labels or arrows need correction. This is a semantic architecture change, not an art adjustment.

The query-state annotation says “Measured field value u_tau”; the evolving state is not the observed scalar y_m. The formula in the top-K block uses sensor-location notation and an RBF expression that do not fully match Eq. (4), including the learned bias and apparent norm-power difference. The flow notation mixes t and tau, F and H_Omega, and several lower/upper-case state symbols. The global readout is labelled “Optional,” although the full Methods model includes it. Resolve the intended level of schematic abstraction and provide one approved text layer before final vector cleanup. Do not make Codex infer the implementation from the drawing.

The smooth-reference construction is described by Eq. (7), while the initial texture looks like uncorrelated noise. The current surface can be recoloured as art, but changing its realized texture to visually imply a different process is outside the style-only pass. The literal typo “on a either” can be corrected as typography; mathematical and causal labels require approval.

**Completion record:** approved Q/K/V assignment; approved notation dictionary; approved exact label replacements; decision on “Optional”; decision on whether the simplified RBF formula remains schematic. Put any semantic figure changes in a separate commit.

## A02 — Fourier baseline identity (D pp. 6–8, 12, 15, 18–19)

Plots use Geo-FNO; the Methods heading and several results use FNO; the stress-proxy paragraph uses FNO-R. Active Methods describe a Cartesian supervised Fourier baseline with a scatter/interpolation wrapper. A commented source paragraph distinguishes this from learned coordinate deformation, but comments do not establish the actual evaluated configuration.

Check the implementation and run metadata first. Establish a stable internal model ID, its correct display name and a precise description of the geometric adapter. Then propagate the name to all figures, captions, prose, SI and bibliography references. Do not simply rename every FNO to Geo-FNO or vice versa. The style specification retains the current displayed Geo-FNO entry until approved; its colour key is not an endorsement of model identity.

## A03 — Fig. 3 panel and example reconciliation (D pp. 9–11)

**Active-asset update:** the approved producer is now the V3-7 mixed-resolution figure with five panels a–e: a resolution/training-recipe design; b the 512-sensor four-method × five-recipe comparison plus two zero-H sensor sweeps; c the fixed five-column Zero-H-M-rich physical proof; d the three-scale qualitative component/residual grid; and e the correlation and variance-allocation matrices. The earlier six-panel a–f asset, ten-column multi-recipe montage and five-sweep main-figure layout are superseded. Revalidate E17–E22 and the manuscript caption against V3-7 rather than restoring the older inventory in the art pass.

For c, verify that the current numerical labels 0.032, 0.090, 0.086 and 0.138 are described consistently as the saved source intends—currently recorded by the V3-7 manifest as local/zoom-region relative-L2 values—and that the full-field values are not substituted in the caption. MLP-RBF is displayed in V3-7. Preserve all current figure values and restore detailed prose only from the corresponding saved evaluation output and approved metric mapping.

Confirm that the 95% case-bootstrap statement applies to the intervals displayed in the V3-7 panel-b bar comparison and sensor sweeps. The art pass must not add intervals to panels that do not currently contain them.

**Completion record:** approved V3-7 caption/panel map; approved full-field versus local/zoom metric wording; approved interval scope. Put any numerical or semantic figure change in a separate commit.

## A04 — Fig. 3 physical variable and heatmap units (D p. 9; Methods p. 18)

Methods identify the selected PDEBench variable as streamwise velocity V_x, while the wavelet residual colourbar says “density”; the Results opening only says “a scalar field.” Inspect the dataset key and plotting metadata. Neither prose nor figure should be treated as sufficient authority to resolve the conflict automatically.

The scale-bias heatmap contains values such as -1.45 and +1.21, while the prose calls the near-zero DMF-Gen biases “within 0.005 percentage points.” Confirm the underlying definition and any multiplier applied for plotting. Is the plotted quantity a fraction, percentage-point difference, relative bias or another normalization? Do not multiply by 100, renormalize per row or change the colour limits during styling. Record the approved unit and retain existing numerical values until the scientific correction is separately authorized.

## A05 — Temperature-only score across Figs. 4 and 5 (D pp. 13–16)

The multi-field section reports mean unobserved-channel error **0.117**; the ablation and cost sections report **0.1063**. Figure 4's rounded DMF-Gen row is also different from Fig. 5's scorecard. This may reflect a legitimate checkpoint, ensemble, conditioning or aggregation difference, or a synchronization error; the supplied materials do not establish which.

Create a small run ledger containing figure/panel, checkpoint, EMA versus ordinary weights, test-state IDs, observation IDs, solver/NFE, consistency correction, number of draws, sample aggregation and channel reduction order. If protocols differ, label them briefly and report them explicitly. If protocols are intended to match, update text and figure from the same authoritative result file. Do not silently select the smaller number, average the numbers or change only the caption.

E26 retains 0.1063 in the ablation/scorecard context and the preceding Results retain 0.117. The candidate is therefore an editorial review version, not a submission-ready reconciliation of these scores.

## A06 — Fig. 5 uncertainty and cost reporting (D pp. 14–16; commented Fig. 5 caption)

The PDF specifies 200 paired states and 64 draws for the generative evaluation. It does not fully define the uncertainty score, CRPS normalization, resampling procedure, interval encodings, selective-error normalization or computational measurement protocol. The TeX comments mention moving-block bootstrap, last.pt checkpoints, a high-band range, batch sizes and model-core timing. Treat these as clues to verify, not facts that can be silently restored.

Report what the 16.7-ms inference time measures: one draw or several, batch size, solver and number of function evaluations, output point count, warm-up and synchronization, inclusion of conditioning, neighbour search, interpolation and cache construction, and hardware. It must not be presented as the time for a 64-member uncertainty ensemble without evidence. The scorecard's training columns use native workloads according to source comments; clarify whether two-stage autoencoder training is included for Latent FM. A lower update time is not automatically lower total training cost.

Define the filled versus hollow memory entries and their scope. Model state is not the same object as peak allocated memory. Define the high-frequency band and whether the spectrum is taken on a structured index-space representation of the 40,300-point domain. Confirm what the violin internal dashed segments and all intervals denote. Keep means, medians and intervals unchanged in the art pass.

CRPS and spread/error correlation support distributional scoring and difficulty ranking. They do not by themselves establish calibrated credible intervals, exact posterior sampling, complete spatial joint-law accuracy or successful adaptive sensing. Use these boundaries to approve claims, not as a new defensive paragraph in the paper.

## A07 — Active Methods and unavailable SI (D pp. 16–19; source lines 750–779)

Training and observation-consistency subsections are commented out. Confirm and restore the configuration needed to interpret the reported models. Hard observation projection or endpoint correction, if used, affects the actual reconstruction procedure and must not be hidden by a description of velocity learning alone. Clarify which tasks and baselines use which consistency operation.

Specify data splits and the treatment of temporally related combustion states and PDEBench trajectories. State channel normalization, coordinate transformation, query/sensor sampling and checkpoint selection. Confirm whether the same source-function realization is used across query chunks. Define the reductions for all primary metrics and provide the appropriate sampling/resampling unit. None of these details can be filled in from general practice.

A short main Methods description plus a verified per-experiment SI table is preferable to a long architecture repetition. The SI was not supplied, so its completeness has not been checked.

## A08 — Attribution and references (D pp. 18–24)

The citation key `Jaegle2022PerceiverIO` resolves in the PDF to the 2021 Perceiver paper, while the active Methods call the backbone Perceiver IO. Check the intended citation and bibliography entry. No replacement BibTeX is fabricated because the `.bib` file and an external reference-verification request were not part of this attachment-based pass.

The source comments identify the sensor-anchored global–local pathway with earlier GLU work; the active citation to that work occurs in the dataset discussion. Have the authors decide whether the architecture Methods also need an explicit attribution sentence. Frame novelty around its integration into measurement-conditioned field generation, rather than silently implying that each inherited ingredient is new.

Verify the final bibliography and data/code availability statements. The supplied abstract and keywords were placeholders, replaced in E02/E03; the availability placeholders remain for the authors to supply real resources. Do not create release URLs or repository claims.

## A09 — Function-space language and notation (D pp. 5–6, 16–18)

Retain the conditional-law target, direct physical-field state and coordinate-query interface. Distinguish the inherited Hilbert-space result from a theorem about the expressivity or exact distributional recovery of this particular network. Eq. (8) conditions an ideal velocity on the function-valued interpolation, while the implemented query-wise head uses the local evolving value with measurement-dependent features. These are not automatically an expressivity guarantee for arbitrary joint conditional field laws. This observation follows from comparing the supplied equations, not from a separate proof of failure.

Likewise, a smooth finite-rank source and a shared query decoder do not alone justify a claim of arbitrary-resolution accuracy or globally smooth generated fields across hard-KNN neighbourhood changes. Report what is established: the specified function-valued construction and the tested 32²–128² transfer. A01/A07 should confirm the implementation details supporting coherent evaluation.

There are three notation collisions worth a light author pass: X as a query set versus X_tau as the interpolated function; U_1 as a terminal field versus the combustion streamwise-velocity channel; and E_X versus calligraphic E_X. Choose a small consistent notation scheme before editing equations or figure text. The patch set leaves the equations intact to avoid an unapproved global notation migration.

## Recording decisions

For each issue, record the responsible author, evidence file/run ID, approved wording or label mapping and completion status. Provide only the approved label mappings to the art agent. That agent may continue with independent spacing, font and colour work, but must report unresolved labels as release blockers rather than “fix” their scientific meaning.
