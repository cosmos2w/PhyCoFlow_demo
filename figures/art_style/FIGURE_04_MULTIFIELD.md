# Figure 4 — completion of unobserved combustion fields

**Manuscript asset:** `Figure_MultiFieldReconstruction.pdf`; D p. 12; LaTeX label `fig:MultiFieldReconstruction`.

**Scientific baseline:** `0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Process_Figures/Assembled/Composite/CoupledFieldReconstruction_round3_spacing_20260828_0810.pdf`.

**Active art export:** `0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Process_Figures/Assembled/Composite/CoupledFieldReconstruction_art_v3_20260914_2343.pdf`.

Read the general contract. Preserve panels a–d and their existing scientific contents.

## Inventory

Panel a contains three displayed fields (methane mass fraction, pressure, streamwise velocity), three DMF-Gen observation regimes and four temperature-only baseline columns. Predictions have associated absolute-error maps and numerical labels. Panel b displays three 8×5 mean-error heatmaps; the frozen source still contains the sixth Unobs. aggregate column. Panel c has three eight-method LSD bar charts with their original intervals/state points and three example spectra. Panel d has three reference joint-PDF maps and three eight-method JSD violin groups.

The current author-approved display inventory is seven Panel-a columns in this exact order: T only, T+U1, CO+T+U1+p, FFM-FNO, Latent FM, SiT and Senseiver. The former Ground truth column is omitted from display only. Keep all three heatmaps and all spectra/violins. Do not add temperature or carbon-monoxide rows to the montage or remove the four baseline comparisons. The physical field subset in a is deliberate: it shows variables unobserved in the temperature-only regime.

## Main art issues

The figure is evidence-rich but currently uses very small error labels inside dark gray rectangles, faint map features, dense white contours, saturated table fills and tiny observation-status marks. The left heatmap block and right statistics block need stronger alignment and more room for text. These are visual issues, not reasons to change the metrics or select fewer snapshots.

A useful first layout pass is to reserve approximately 40–43% of the lower figure width for b and the remainder for c/d, then tune to the actual text. This modest panel-width redistribution preserves the current arrangement and panel membership. Do not enlarge b by distorting the physical domains in a or by deleting right-side methods.

## a — physical fields and conditioning progression

Keep the two existing column groups distinct: DMF-Gen conditioning progression and temperature-only baseline comparison. Align the group headings and maintain a slightly wider inter-group gutter. Wrap long observed-channel headings over two lines at the same font size. Do not abbreviate the four-channel condition in a way that hides one of the variables or changes CO into CH4.

The Ground truth display column is intentionally absent in V3; do not delete or rewrite its source arrays. Every displayed prediction/error pairing must remain clear through a small common gap and consistent border treatment. Keep the physical-field labels at far left and the repeated vertical `Recon.`/`Error` row-type labels in the adjacent inner label column.

Methane and pressure should use the same nonnegative-magnitude colormap family where their original data and norms permit, while signed streamwise velocity should retain a signed diverging colour role. Use identical mappings within each variable across truth, DMF-Gen regimes and baselines. The current source's range, offsets and normalization are frozen; do not autoscale a method to reveal more detail. Keep absolute errors on a common light-to-dark nonnegative palette per variable, matching the error semantics adopted in Fig. 3.

Preserve all spatial structures, domain masks, sensor markers and contour levels. The channel-added panels include visible measurement markers; do not remove them as clutter or infer new marker locations. A modest reduction of contour opacity/width can help, but apply it to every comparable tile and keep the same contours. Do not smooth flame interfaces, sharpen methane boundaries or denoise pressure/velocity textures.

The current error values include scientific notation. Retain their precision and exponent, but move the label to an aligned gutter or use a much lighter compact annotation background. Do not obscure the flame edge or the strongest error band. Use a readable relative-L2 symbol rather than tiny fragmented mathematical spans; the metric is not squared L2 error.

For the current compact art revision, remove the six far-right field/error
colourbars as an explicit author-authorized exception to the general retention
rule.  Preserve every underlying limit and normalization in the source
manifest and scientific-state comparison.  Reallocate the freed width to equal
inter-column gaps. V3 additionally removes the Ground truth display column and
horizontally expands the seven retained display axes without changing their
coordinates, crop bounds, interpolation, masks, contour paths or source arrays.
Use the active baseline's exact scalar lookups: `crest` for methane,
`viridis` for pressure, `RdBu_r` for signed streamwise velocity and `OrRd` for
absolute errors.  These Figure 4 assignments override the general scalar
defaults for continuity with the approved round-3 figure.

Retain the earlier explicit contour-GridSpec `prediction_error_hspace` reduction
from `0.16` to `0.08`. V3 separately halves the inter-field spacer ratio from
`0.25` to `0.125`; this distinction is mandatory because both controls exist in
the producer. Reduce the physical Panel-a content height accordingly so the
saved field-gap space compresses the canvas rather than stretching the map rows.

## b — three error heatmaps

Preserve all 144 source entries (three 8×6 tables), including their existing precision. V3 displays only the five individual-field columns, for 120 visible values; the author-approved `Unobs.` omission is a display operation after source loading. Do not delete the aggregate values from the manifest or source comparison. Retain the physical cell width of the five visible columns and shrink/translate the matrix block rather than stretching the cells.

Use the active round-3 Panel-b `Reds` palette and preserve its existing
normalization exactly. If the source already uses shared normalization, keep it.
If it does not, flag the difference rather than silently impose one. **Never
introduce a new column-wise normalization** to make pressure errors visually
comparable with species/velocity errors. Keep every number visible at the
7.0-pt annotation role with renderer-derived black/white contrast; a darker
cell must never receive dark text and a lighter cell must never receive white
text.

Lighten the most saturated table backgrounds enough through the colormap choice to make the numeric values primary, but do not alter the metric-to-colour normalization or clip large entries. Use light cell separation rather than thick grid borders. Keep every method tick label regular under the active typography hierarchy; do not conceal lower errors belonging to other methods or add best-in-column stars.

Strengthen the existing teal observed-channel marks so they can be seen at print scale. Preserve their exact column memberships under T; T+U1; and CO+T+U1+p. A colour accent for “observed” is an information encoding and must not be reassigned to the DMF-Gen brand colour. Keep the regime headings aligned at identical distances above their tables.

## c — LSD and spectra

The upper three panels are **bar charts**, not a redesign opportunity. Retain all eight bars, original heights, widths in data/category space, ordering, intervals and raw point overlays. Use the shared method palette, moderate fill saturation and a thin readable edge. Draw error bars above both bars and state points. Raw dots may become smaller and more transparent uniformly, but no subsampling, fresh jitter or clipping is allowed.

Keep the current LSD units in dB and every existing y-axis range. Do not rescale the three variables to make their ranks appear more similar. Preserve SiT's lower LSD scores; the styling should not imply that DMF-Gen leads the spectrum metric.

Place an eight-method bar legend above the bar group with compact square
swatches and sufficient separation from the variable titles.  Give the spectra
their own four-entry line-style legend (ground truth plus the three existing
selected methods) in one horizontal row centred in the gap between bar and
spectrum axes.  Neither legend may cover data.  Retain the original spectrum
membership and GT reference trace. Use thin distinguishable line styles without inserting markers on every frequency shell. Preserve all wiggles and the high-frequency tail: do not smooth, average or re-bin the spectra.

Align each spectrum with its corresponding LSD axis above. Keep log axes,
spectral units and the existing wavenumber caption.  Thin the displayed
spectral-energy y ticks to at most four labelled log levels per subplot while
retaining the exact scale and axis limits. The 256 T sensors and coverage annotation are source facts, not an invitation to recompute or round the percentage differently.

## d — joint distributions and JSD violins

Keep the T–U1, CH4–U1 and p–U1 pairs in their current order. Preserve the exact binned ground-truth density arrays, original colour normalization and empty/masked-bin treatment. Use a restrained sequential colormap. Do not add new contour estimates, smooth the histogram or replace it with a scatter cloud. Preserve x/y variable assignments and domains.

Keep the three JSD groups aligned with the corresponding PDFs. Retain all eight method violins, their row order, density paths and mean values. Use the same method colours as c and the other figures. V3 removes only the `mu=` prefix, widens every violin body by one shared display-only width factor without changing its KDE path/support, and preserves every numeric value, including the better SiT result for CH4–U1.

Use a neutral base-2 JSD axis label. Attempt one common baseline for the retained
numerical tick values; if the final renderer still detects any collision, hide
all three tick-label sets consistently while preserving the ticks, transforms
and axis limits. Do not turn JSD into a percentage or force identical x limits if the existing comparisons use different ranges. The PDF maps are pooled over 25 states while the JSD distributions use 1,000 states according to the caption; no pooling or sampling change is allowed.

For the current V3 renderer, use one shared violin width of `0.66`, reserve the
violin data to the left `0.66` axes fraction, and align the numeric mean column
at `0.71` axes fraction. Use a 6-mm rendered inter-column gap for Panel d. All
24 mean labels must remain within their owned axes and the final tick-label
overlap count must be zero.

## Acceptance and gates

A02 governs the Fourier baseline's scientific display name. A05 governs the difference between the temperature-only Unobs. result here and the Fig. 5 score; preserve this figure's numbers until the authors reconcile the run ledger. A07 governs units, metric reduction and source protocols. Do not copy Fig. 5's smaller score into b.

This file and the active general art contract supersede conflicting redesign
recommendations in `_Process_Figures/_ReviseGuides/Figure_MultiFieldReconstruction_revision_brief.md`
for this art-style-only pass. In particular, do not replace Panel-c bars, alter
spectral membership, add/remove scientific annotations, change normalization,
or restore Panel-a colourbars from that older brief.

Verify seven displayed montage columns × three physical-field reconstruction/error pairs, three displayed 8×5 heatmaps backed by complete 8×6 source matrices, three LSD/spectrum pairs and three PDF/JSD pairs. The final-aspect gate must report zero text/text, text/non-owned-data-window and clipped-mean collisions. All observed-channel marks, scalar observations, histogram bins, state dots and distributions must retain their memberships. Review ordinary heatmap and error-label text at 162 mm; if it cannot be read, adjust panel allocation and label wrapping rather than scientific content.
