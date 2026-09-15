# Figure 5 — conditional ensembles, mechanism and computational cost

**Manuscript asset:** `Figure_Evaluations.pdf`; D p. 15; LaTeX label `fig:uq_cost`.

**Active producer:** `Dis_SI_Process/figures/generated/figure5_precision_log_revision_20260910/figure5_log.pdf`.

**Active art export:** `Dis_SI_Process/figures/generated/art_style_review/Figure_Evaluations_art_v3_20260915_0900/figure5_log.pdf`.

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

For V3, both violin axes are `41 mm` high and touch at one shared boundary
(`0 mm` hspace). Display each existing mean above its violin tip using two
significant digits and at least `1 mm` rendered tip clearance. The sole lower
y-limit extension is an author-approved display-headroom exception; preserve
the log transform and every violin/mean coordinate, and record the change.

Use thin violin outlines and gentle fills. Internal dashed mean/summary marks should remain at their exact positions; they must not be re-centred visually. The IID source's lower whole-field error and worse high-band error are a central trade-off, so do not make its first violin visually subdued or its second violin excessively saturated.

## e — population spectrum

Keep truth and all six variants, preserving every frequency-shell coordinate, power value, oscillation and the original high-frequency tail. This panel contains visible shell-scale oscillations; do not smooth or downsample them away. Keep log power scale, existing y limits, frequency axis and predefined band.

Use dark truth, red full model, the conditioning-variant blue-grays, ochre IID and shared Senseiver gray-blue. Maintain line-style distinctions so overlapping variants remain traceable in grayscale. Use no new per-shell markers. Adjust line weights within the general hierarchy, not to make high-frequency attenuation disappear.

Retain the high-band shaded region, vertical boundary and existing label/arrow exactly in data space. Its range is a metric definition, not a drawing convenience. Reduce the shading opacity if necessary, uniformly and without moving its boundary. Place the legend in whitespace without covering the shaded-band annotation or the portion where the IID curve separates. Prefer a compact external legend above/adjacent to the axis if an internal one cannot avoid covering data.

A06 must confirm the band definition and index-space/physical-space frequency interpretation. Do not rename “Frequency shell” to a physical wavenumber with invented units.

## f — accuracy and computational footprint

Keep five aligned columns: unobserved-field relative-L2, training-update time, training memory, inference time and inference memory. Preserve all eight rows and their current order (DMF-Gen, Senseiver, SiT, Geo-FNO, FFM-Perceiver, FFM-FNO, MLP-RBF, Latent FM in the rendered figure). Do not independently sort each cost column.

Use a single row-coordinate system across all axes, equal bar height, consistent padding and common model-label alignment. The accuracy column's numerical labels should remain associated with the same points and retain their exact values/precision. Use the same palette as a–c and Figs. 2–4. Model names may be dark neutral with a colour key rather than tiny coloured text, but identity must remain unambiguous.

The exact model name **DMF-Gen** is deliberately bold in the visible model-name
rails of a and f; retain this emphasis and keep all other model names regular.
The point symbols in the left-most f axis must be large hollow symbols with the
same 5.8-pt size, 0.9-pt edge and method-specific shapes used in a/b.

Keep the horizontal bars and point comparison types as currently drawn. Do not add a connected accuracy–cost trade-off curve or a new “best” badge. Preserve zero baselines and original scales for all cost axes. Use clear units: milliseconds per update, GiB, milliseconds, MiB, preserving the source's binary memory units rather than silently converting to GB/MB.

The final memory column has filled and hollow encodings for model state and peak memory. Keep both exact lengths and their association. They are not additive stacked components: do not stack or sum them. Make the hollow outline visible at about 0.8 pt and ensure the small filled model-state entry does not vanish. Align the “Model” and “Peak” labels/key outside the bars without changing their meaning.

Preserve any timing/memory error bars and their definitions; the style pass cannot infer their confidence level. The equal-looking row design must not imply identical native training workloads or one-draw/64-draw equivalence. These are reporting questions under A06.

V3 author override: in the right-most inference-memory axis only, retain the
error-bar artists and source intervals in memory but hide them from export.
Keep the paired filled model-state and hollow peak bars; use a `0.9 pt` outline
on all eight hollow peak bars. This does not authorize hiding uncertainty in
the other four scorecard axes.

## Acceptance and release gates

A05 must reconcile 0.1063 here with 0.117 in the preceding multi-field evaluation. A06 must confirm all uncertainty and timing/statistical encodings. Retain the current scientific values until approved. A02 still governs the Fourier baseline identity in f.

Check that all five generative methods are present in a–c, both six-category violin axes align, e contains truth plus six variant traces, and every scorecard row matches across its five columns. Ensure no coloured baseline is nearly invisible, no legend covers the high-band tail and no bar/interval geometry changes. Review the complete figure at 162 mm and record the latency/ensemble-scope question as a scientific release gate, not an art defect.

For the active art V3 export, use a fixed `179.8 x 244 mm` MediaBox rather than
the baseline's approximately `258 x 249 mm` tight crop. Apply the established
hierarchy at the final physical size: panel labels `11.0 pt` bold, axis titles
approximately `8.5 pt`, ticks and legends `7.8 pt`, and ordinary in-plot
annotations `7.0 pt`. At 162-mm insertion width, ordinary ticks and legends
therefore remain at least `7 pt`. Keep every scientific artist coordinate and
scale type unchanged. Use one measured visible left rail for the longest
model names, both Panel-d y-axis titles and tags a/d/f, and one common right
axes boundary for all three major rows. Use an exact `2:1` Panel-d/e plotting
width ratio with a `14 mm` inter-panel gutter. Both Panel-d violin axes must be
`41 mm` high with zero hspace; wrap the two y-axis titles consistently and verify at
least `3 mm` title-to-title clearance. Wrap the four long Panel-d categories
onto two lines and retain one common `45 degree` rotation. Give Panel f four
equal `4 mm` physical gaps, match its categorical row pitch to Panel a within
5%, and align its method-label spine with Panel a. Align the baselines of a/b/c
to the top-row axes, place d/e and f on their respective axes-top baselines,
and crop the outer top/bottom/right rendered margins to at most `1.5 mm`.
Use the shared model palette and `5.8 pt` hollow markers in a/b/f; enlarge only
Panel-a raw-point clouds to `0.95 pt2`. Rename the spectrum annotation
`High-band`, place it at the upper edge, and keep the wrapped legend down/right
with zero sampled curve intersections. The release
must include fixed-size PDF/SVG/PNG outputs, a true vector-derived 162-mm render,
grayscale and deuteranopia reviews, an exact before/after scientific data-artist hash,
an enumerated lower-violin view-limit exception,
and zero clipped text, text-to-text overlaps, text-to-nonowned-axes overlaps,
tick-label overlaps or panel-tag/data-window overlaps.
