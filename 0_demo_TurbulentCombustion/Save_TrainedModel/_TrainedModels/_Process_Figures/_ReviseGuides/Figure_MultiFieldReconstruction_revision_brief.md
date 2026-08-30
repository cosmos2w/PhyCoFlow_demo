# Revision brief for `Figure_MultiFieldReconstruction`

## 1. Scientific role of the figure

This figure should serve as the paper's clearest test of **measurement-content variation through \(\mathcal{Y}\)**. The domain \(\Omega\), query set \(X\), five-channel target state, and test set remain fixed; what changes is which physical variables are present in the marked measurement set. The figure should establish five points, in this order:

1. A complete five-field state can be reconstructed when entire target channels are absent from the observations.
2. DMF-Gen's spatially registered error on unobserved channels is lowest among the evaluated methods in all three observation regimes.
3. The always-unobserved methane field improves as the supplied evidence becomes richer.
4. Pointwise, spectral, and joint-distribution metrics test different properties and should not be collapsed into one claim.
5. The result is not universal dominance: SiT has the lowest spectral LSD for all three fields in panel c and the lowest \(Y_{\mathrm{CH_4}}\)--\(U_1\) JSD in panel d.

The visual hierarchy should therefore read naturally as **example field -> test-set field error -> scale fidelity -> coupled-physics fidelity**.

## 2. Claim-critical checks before changing the layout

Resolve these items before the next publication export. They affect what the paper is allowed to claim, not only how the figure looks.

### 2.1 Confirm whether one checkpoint handles all channel regimes

Determine whether each method uses one jointly trained checkpoint for all three observation-channel combinations or separate condition-specific checkpoints.

- If one checkpoint is reused, record that fact in the figure manifest and Methods. This directly supports the claim that a single conditional operator accepts changing channel content.
- If separate checkpoints are used, do not write that “the same model” adapts to a changed \(\mathcal{Y}\). Describe the experiment as a comparison of condition-specific reconstructions under three missing-channel tasks.

### 2.2 Separate channel breadth from observation count

The current regimes contain 256, 512, and 1,024 scalar observations. The progression therefore changes both channel content and total evidence. It supports the conclusion that richer evidence improves reconstruction, but it does not isolate a pure channel-diversity effect at fixed measurement budget.

Retain the current progression in the main figure, but add a fixed-total-budget control in the Supplementary Information if feasible, for example:

- 256 \(T\) observations;
- 128 \(T\) + 128 \(U_1\) observations;
- 64 observations from each of \(Y_{\mathrm{CO}},T,U_1,p\).

A second useful control would hold the total at 1,024 while changing how those observations are distributed across channels.

### 2.3 Fix artifact provenance

The exported PDF and artifact manifest use snapshot 0, whereas the editable layout YAML currently records snapshot 500. Choose one source of truth, synchronize the YAML and manifest, and regenerate. Unless a predeclared selection rule is documented, label the field as **“one held-out state”**, not “representative.” A stronger option is to select the state closest to the median DMF-Gen temperature-only unobserved-channel error using a rule fixed before inspecting baseline maps.

### 2.4 Persist all panel-d metrics

Regenerate and save the full-precision \(Y_{\mathrm{CH_4}}\)--\(U_1\) JSD summary to the same finalized metric location used by the other pairs. The current value was computed transiently and only the two-significant-figure number in the PDF remains. Do not report extra digits until the persistent CSV and provenance record exist.

### 2.5 Synchronize the joint-PDF sample count

The reference figure pools 25 deterministically spaced truth states, not 100. Update the stale code docstring and all explanatory text to 25.

### 2.6 Remove hidden visual trimming

The gray panel-c points are currently omitted above each model/field-specific 90th percentile, although the bars and confidence intervals use all 1,000 states. This is easy to misread as the complete distribution. The preferred revision is to remove the gray scatter from the main figure. Put full, untrimmed distributions in a supplementary violin or ECDF panel.

### 2.7 Do not attribute the flow-baseline gap to one module without a control

DMF-Gen, FFM-FNO, and FFM-Perceiver use different observation-consistency choices in the recorded evaluation. The main text may conclude that the functional transport objective alone does not guarantee strong sparse multi-field reconstruction under the evaluated implementations. A causal claim that the sensor--global--sensor block alone creates the gap requires matched-consistency and component-ablation results.

## 3. Recommended overall composition

Keep the four-panel a--d structure, but simplify the visual encodings rather than adding more content.

- Preserve the current 7.2-inch two-column width.
- Use one model order everywhere: DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, SiT, MLP-RBF, Geo-FNO, Senseiver.
- Use one method palette in panels c and d. Emphasize DMF-Gen; keep all alternatives distinguishable but visually quieter.
- At final print size, use at least 7.5--8 pt for tick labels and 8.5--9 pt for headings. Panel letters should be bold and visibly separated from axes.
- Add a small shared banner above panels c and d: **“Temperature-only conditioning: 256 \(T\) sensors”**.
- Keep text and axes as vector objects. Rasterize only the dense contour and joint-density artists at publication resolution.
- Export a checking PNG at the exact intended print width and inspect it at 100% and 50% zoom before accepting the PDF.

## 4. Panel-specific revisions

### Panel a — held-out physical fields and absolute errors

**Retain** the three displayed fields and the current logical grouping: ground truth; DMF-Gen under three channel sets; selected temperature-only baselines.

Revise the panel as follows:

1. Replace “representative” with **“one held-out state”** unless a quantitative selection criterion is implemented and recorded.
2. Add unobtrusive sub-row labels on the far left: **field** and **\(|\mathrm{error}|\)**. The present stacked maps are not self-explanatory at first glance.
3. State **“256 sensors per observed channel”** directly under the DMF-Gen conditioning-progression heading.
4. Keep one field-specific value range and one field-specific absolute-error range across every method and condition. Never autoscale individual maps.
5. Label the paired colorbars explicitly. Recommended display units are:
   - \(Y_{\mathrm{CH_4}}\): mass fraction, shown with a \(\times10^{-2}\) multiplier;
   - \(p\): kPa for both the field and error colorbars, rather than \(10^5\) and \(10^4\) Pa notation;
   - \(U_1\): m s\(^{-1}\), with explicit ticks such as \(-120,0,120\).
6. The 256 sensor circles obscure the observed \(p\) and \(U_1\) fields. Preferred implementation: use a compact teal **observed-channel badge** in the corresponding column header and move the full sensor layout to a supplementary panel. Fallback: retain all sensor positions as very small hollow rings with no fill, thin stroke, and low alpha; never plot a subset without saying so.
7. Put the relative-error value in a clean strip immediately below each error map, using \(\varepsilon_{L_2}=\) and consistent three-significant-figure formatting. Avoid dark boxes that cover the error field.
8. Change the baseline heading to **“Selected baselines: conditioned on \(T\) only”**. The four maps are a representative family comparison, whereas panel b contains all eight methods.
9. Preserve the dashed separator between the conditioning progression and baselines, but align it with the full height of the panel and keep it lighter than the data.
10. Keep the current truth-derived clipping only for display, and preserve unclipped arrays for all metrics. Record the clipping ranges in the manifest.

### Panel b — all-test-state fieldwise errors

Panel b is the primary quantitative evidence and should be the easiest part of the figure to read.

1. Keep all eight models and all three conditions.
2. Add a compact panel heading: **“Mean physical relative-\(L_2\), \(n=1{,}000\)”**.
3. Replace the subtle teal top rule with a teal dot or small square inside observed-field cells, and add a one-line legend: **teal = directly observed**.
4. Make the changing definition of `Unobs.` explicit:
   - temperature only: mean of 4 unobserved channels;
   - \(T+U_1\): mean of 3 unobserved channels;
   - four-channel condition: methane only.
   A compact label such as `Unobs. mean (4)`, `Unobs. mean (3)`, and `Unobs. mean (1)` is preferable to leaving the denominator implicit.
5. Keep column-wise color normalization because the channel error scales differ greatly, but add **“column-wise scale”** beneath the matrix. Do not invite numerical comparison of color intensity between pressure and methane.
6. Use one number format throughout: three significant figures, switching to scientific notation only below \(10^{-3}\). Do not mix one-, two-, and four-decimal formatting arbitrarily.
7. Bold and thinly outline the lowest value in each condition-by-field column. Do not automatically bold the DMF-Gen row; let the data determine the emphasis.
8. Give the `Unobs.` column a slightly heavier left separator because it is the main cross-channel target.
9. Add a small side annotation beside the DMF-Gen methane cells, \(0.0645\rightarrow0.0588\rightarrow0.0454\), to make the always-unobserved comparison visible without asking readers to compare three distant blocks. Omit this annotation if it compromises legibility.
10. Retain complete-coverage means in the main figure; move standard deviations, medians, quartiles, and confidence intervals to a supplementary table.

### Panel c — spectral fidelity under temperature-only conditioning

The current panel is scientifically useful but visually overloaded and omits the method with the best mean LSD from the example spectra.

1. Replace bars plus visually trimmed gray points with a **mean-and-95% bootstrap-CI dot/whisker plot**. This removes hidden trimming and makes the ranking clearer.
2. Retain all eight methods in the summary plot, with a fixed model order and the same colors used in panel d.
3. In the lower spectra, show **GT, DMF-Gen, SiT, and Senseiver**:
   - SiT must appear because it has the lowest mean LSD in all three fields;
   - Senseiver is the strongest non-DMF pointwise baseline under temperature-only conditioning;
   - move Latent FM and the remaining model curves to the Supplementary Information.
4. Use the same held-out state as panel a and label it accordingly. Do not call it representative without a selection rule.
5. If the native shell curves are too jagged at print size, use deterministic logarithmic binning only for the displayed curves. Continue computing LSD on the documented native common shells, and state this distinction in Methods.
6. Keep the three fields explicitly marked **unobserved**.
7. Use consistent axis ranges where scientifically sensible, but do not force a common LSD range if it hides the velocity spread. Make the differing y-axis limits visually obvious.
8. Keep all 1,000 states in the means and confidence intervals. Put full untrimmed distributions in the Supplementary Information.

### Panel d — coupled-field distributions

Panel d should make the complementary wins unmistakable rather than visually suggesting universal dominance.

1. Use the same top-to-bottom model order as panel b.
2. Use a common logarithmic JSD axis range across the three pairs whenever the data permit, and label it **“base-2 JSD (log scale)”**. If a common range is impossible, state the pair-specific ranges clearly.
3. Replace free-floating \(\mu=\) text with a right-aligned mean-value column or a mean dot inside each violin. Keep the median and interquartile range encoding, but explain it once in a small legend.
4. Bold only the best mean for each pair. The intended pattern is:
   - DMF-Gen best for \(T\)--\(U_1\);
   - SiT best for \(Y_{\mathrm{CH_4}}\)--\(U_1\);
   - DMF-Gen best for \(p\)--\(U_1\).
5. Add a small shared colorbar to the top-row joint PDFs, labelled **“joint density (log)”**, because the three maps use a shared logarithmic normalization.
6. Optional but useful: overlay one or two contour levels from DMF-Gen and SiT on the ground-truth density maps. This gives the top row a direct comparative role without adding another full row of PDFs. Use the same 25 states for every pooled distribution and explain that pooled contours are illustrative, whereas JSD is computed per state.
7. Keep the physical interpretation visible in the pair titles: thermal--flow, chemistry--flow, and flow--flow consistency can be explained in the caption or nearby text, not repeated inside every axis.
8. Regenerate the \(Y_{\mathrm{CH_4}}\)--\(U_1\) values from a persistent finalized CSV before export.

## 5. Revised compact caption

```latex
\caption{\textbf{Reconstruction of missing channels in a coupled turbulent-combustion state.}
\textbf{a}, One held-out state: ground truth and DMF-Gen reconstructions under three observation-channel sets (256 sensors per observed channel), followed by selected temperature-only baselines; lower maps show absolute error and labels give physical relative-$L_2$ error.
\textbf{b}, Mean relative-$L_2$ error over 1,000 test states for each field and for the mean over channels absent from the condition (Unobs.; teal marks observed fields).
\textbf{c}, Temperature-only spectral fidelity of the unobserved $Y_{\mathrm{CH_4}}$, $p$ and $U_1$: mean LSD with 95\% snapshot-bootstrap confidence intervals and spectra for the state in \textbf{a}.
\textbf{d}, Ground-truth joint PDFs pooled over 25 test states and per-state base-2 JSD distributions over 1,000 states for $T$--$U_1$, $Y_{\mathrm{CH_4}}$--$U_1$ and $p$--$U_1$ under temperature-only conditioning. Lower values are better.}
```

## 6. Codex implementation sequence

1. Freeze and archive the current PDF, source manifest, metric CSVs, and reconstruction-cache manifest.
2. Resolve the checkpoint-sharing, snapshot-index, 25-state, and \(Y_{\mathrm{CH_4}}\)--\(U_1\) persistence issues above.
3. Refactor method ordering, palette, fonts, number formatting, and observed-channel markers into shared plotting constants.
4. Revise panel a without rerunning inference; read the existing caches and finalized display ranges.
5. Rebuild panel b from the complete 1,000-state summaries and verify every printed value against the CSV.
6. Rebuild panel c without percentile trimming; include SiT in the example spectra.
7. Rebuild panel d from finalized persistent metrics and use one consistent model order and axis policy.
8. Export PDF and PNG, then inspect at the intended two-column print width.
9. Write a new artifact manifest containing code revision, input paths and checksums, snapshot-selection rule, test count, pooled-PDF indices, field ranges, metric versions, bootstrap seed, and output checksum.
10. Run automated assertions that the plotted means match the finalized summaries and that every model--condition pair has 1,000 valid states.

## 7. Acceptance checklist

- [ ] The figure is readable at final journal width without zooming.
- [ ] No panel calls snapshot 0 “representative” unless a documented rule supports that term.
- [ ] One-checkpoint versus condition-specific training is stated correctly.
- [ ] `Unobs.` reports its changing number of channels.
- [ ] Panel c contains no hidden percentile trimming.
- [ ] SiT appears in the example spectra.
- [ ] The three panel-d pairs use persistent full-precision summaries.
- [ ] The model order and palette are consistent across panels.
- [ ] All colorbars state quantity and unit.
- [ ] The caption states 25 pooled states and 1,000 per-state metrics.
- [ ] Claims preserve the non-uniform outcome: DMF-Gen leads pointwise and two coupling metrics; SiT leads spectral LSD and methane--velocity JSD.
