# Figure 5 — conditional ensembles, mechanism and computational cost

**Manuscript asset:** `Figure_Evaluations.pdf`; D p. 15; LaTeX label `fig:uq_cost`.

**Active producer:** `Dis_SI_Process/figures/generated/figure5_precision_log_revision_20260910/figure5_log.pdf`.

Read the general contract and author checks A05/A06. The editorial revision moves this figure's evidence into Results without changing its figure number or panel contents.

## Preserve the inventory

Keep a: five-method normalized CRPS comparison; b: five-method spread/error Spearman correlation; c: five selective-reconstruction curves; d: two six-variant violin axes (whole unobserved-field error and U1 high-band error); e: truth plus the same six model/variant population spectra; f: eight-method accuracy and four computational-footprint columns. Preserve the existing sorted row order in f, which is different from the order in a/b.

The visual hierarchy should read ensemble utility → conditioning/source mechanism → accuracy–cost context. The existing arrangement already supports that sequence. Refine its labels, legend coordination and common row alignment; do not turn it into a new score dashboard, add a Pareto frontier or replace the charts.

## a and b — CRPS and spread/error association

Use the shared five-method identity colours and exactly the same vertical row positions across a and b. Keep every point, interval endpoint and existing per-state/background mark. The DMF-Gen row tint may be retained as a very pale neutral/red tint, but it must not make the other rows unreadable or imply a statistical selection criterion.

The original graphical layers include large central symbols, horizontal intervals and smaller/tinted marks. Preserve all layers. A06 must identify their statistics; the art pass cannot decide that an interval is a 95% confidence interval or that a symbol is a median. Use cap/stroke contrast sufficient for print, with no shift in x coordinate.

Retain the zero-correlation reference line in b at its current position and preserve the negative portion of the axis. Do not crop it away to make all generators look positively informative. Keep the CRPS axis scale/limits and the correlation axis scale/limits unchanged. Align x-axis label heights and keep the long normalized-CRPS label readable without abbreviating it into an undefined metric.

## c — selective reconstruction

Retain all five curves, all retention fractions, their paired error values and every uncertainty-band array. Use model colours and existing marker distinctions, approximately 1.6-pt DMF-Gen and 1.1-pt alternative lines. All alternatives must remain visible near the upper error level; do not merge them into a single “baselines” trace.

Keep the y-axis normalization, the range and the retained-fraction direction exactly as in the source. Do not plot discarded fraction instead, switch to absolute error, change the denominator or mark a new threshold. Any reported 80%-retention result belongs to the text; do not add a new improvement arrow as a decoration.

Place a compact shared legend for a–c in external whitespace only if it retains every method and does not duplicate/obscure a's existing method labels. A figure-wide legend must not confuse the later ablation variants with the baseline models. If a already supplies a clear method key, colour/marker consistency across the three panels can suffice.

## d — conditioning and source ablations

Keep the six categories in their current order: full model, no sensor feedback, no local conditioning, local-only conditioning, IID Gaussian prior, Senseiver. Preserve both violin geometries, the original log/linear transformations, all internal dashed statistics and any tails. The two metrics must stay vertically separate.

Use a coordinated ablation palette: full model uses DMF-Gen red; three conditioning variants use distinguishable blue-gray tones; IID source uses warm ochre; Senseiver uses its shared gray-blue identity. Apply exactly the same variant colours and line styles in e. Do not accidentally reuse a baseline colour and imply that a conditioning ablation is another benchmark architecture.

Increase the label size and either wrap long category names consistently or retain one common modest rotation with sufficient bottom gutter. Keep the six tick centres fixed. Do not rotate each name by a different angle or change to horizontal violins as an easier layout. The upper metric label and lower high-band label should share an x offset and sit outside the data region.

Use thin violin outlines and gentle fills. Internal dashed mean/summary marks should remain at their exact positions; they must not be re-centred visually. The IID source's lower whole-field error and worse high-band error are a central trade-off, so do not make its first violin visually subdued or its second violin excessively saturated.

## e — population spectrum

Keep truth and all six variants, preserving every frequency-shell coordinate, power value, oscillation and the original high-frequency tail. This panel contains visible shell-scale oscillations; do not smooth or downsample them away. Keep log power scale, existing y limits, frequency axis and predefined band.

Use dark truth, red full model, the conditioning-variant blue-grays, ochre IID and shared Senseiver gray-blue. Maintain line-style distinctions so overlapping variants remain traceable in grayscale. Use no new per-shell markers. Adjust line weights within the general hierarchy, not to make high-frequency attenuation disappear.

Retain the high-band shaded region, vertical boundary and existing label/arrow exactly in data space. Its range is a metric definition, not a drawing convenience. Reduce the shading opacity if necessary, uniformly and without moving its boundary. Place the legend in whitespace without covering the shaded-band annotation or the portion where the IID curve separates. Prefer a compact external legend above/adjacent to the axis if an internal one cannot avoid covering data.

A06 must confirm the band definition and index-space/physical-space frequency interpretation. Do not rename “Frequency shell” to a physical wavenumber with invented units.

## f — accuracy and computational footprint

Keep five aligned columns: unobserved-field relative-L2, training-update time, training memory, inference time and inference memory. Preserve all eight rows and their current order (DMF-Gen, Senseiver, SiT, Geo-FNO, FFM-Perceiver, FFM-FNO, MLP-RBF, Latent FM in the rendered figure). Do not independently sort each cost column.

Use a single row-coordinate system across all axes, equal bar height, consistent padding and common model-label alignment. The accuracy column's numerical labels should remain associated with the same points and retain their exact values/precision. Use the same palette as a–c and Figs. 2–4. Model names may be dark neutral with a colour key rather than tiny coloured text, but identity must remain unambiguous.

Keep the horizontal bars and point comparison types as currently drawn. Do not add a connected accuracy–cost trade-off curve or a new “best” badge. Preserve zero baselines and original scales for all cost axes. Use clear units: milliseconds per update, GiB, milliseconds, MiB, preserving the source's binary memory units rather than silently converting to GB/MB.

The final memory column has filled and hollow encodings for model state and peak memory. Keep both exact lengths and their association. They are not additive stacked components: do not stack or sum them. Make the hollow outline visible at about 0.8 pt and ensure the small filled model-state entry does not vanish. Align the “Model” and “Peak” labels/key outside the bars without changing their meaning.

Preserve any timing/memory error bars and their definitions; the style pass cannot infer their confidence level. The equal-looking row design must not imply identical native training workloads or one-draw/64-draw equivalence. These are reporting questions under A06.

## Acceptance and release gates

A05 must reconcile 0.1063 here with 0.117 in the preceding multi-field evaluation. A06 must confirm all uncertainty and timing/statistical encodings. Retain the current scientific values until approved. A02 still governs the Fourier baseline identity in f.

Check that all five generative methods are present in a–c, both six-category violin axes align, e contains truth plus six variant traces, and every scorecard row matches across its five columns. Ensure no coloured baseline is nearly invisible, no legend covers the high-band tail and no bar/interval geometry changes. Review the complete figure at 162 mm and record the latency/ensemble-scope question as a scientific release gate, not an art defect.
