# DMF-Gen figure system — general Codex contract

## Assignment

Refine the **art style only** of the existing DMF-Gen figure system. The current authorized scope is limited to Figs. 3–5: `Figure_MixedResolution.pdf`, `Figure_MultiFieldReconstruction.pdf` and `Figure_Evaluations.pdf`. Figs. 1–2 are context only and must not be edited in this pass. Read this file, `style_contract.json`, the relevant `FIGURE_0N_*.md` and `../editorial/02_AUTHOR_CHECKS.md` before editing. Use the existing plotting code, numerical inputs and editable vector objects. The supplied manuscript PDF is the visual reference, not a substitute for scientific source data.

The desired character is a restrained, technically legible scientific figure system: generous separation of major blocks, compact internal alignment, clear physical maps, a consistent red DMF-Gen identity, muted but distinguishable alternatives and quiet axes. The FBNO reference motivates the staging of evidence and the repeated reference/prediction/error grammar. It does not authorize copying its layouts, changing DMF-Gen's panel contents or adding perspective surfaces and radial charts.

All numerical values, scientific meanings and comparisons are frozen. Make no model, training, inference, data-processing or evaluation changes. Resolve uncertainties through the author-check ledger, not through aesthetic judgment.

## 1. Content boundary: what is and is not art

**Allowed:** font family/size/weight; label wrapping and equivalent mathematical typesetting; colours and colormaps; uniform line/marker/border weights; transparent fills; spacing; legend placement; colourbar geometry; panel alignment; modest resizing of existing panels; removal of decorative shadows or backgrounds; equivalent number formatting that preserves precision and value; vector/raster export settings. Correcting the unambiguous brand spelling `DMFGen` to `DMF-Gen` is allowed. Expanding `FFM-Perc.` to `FFM-Perceiver` is allowed when there is space and the underlying model ID is unchanged.

**Frozen:** panel inventory and panel letters; which model/case/recipe/channel appears; row/column ordering of scientific categories; plotted coordinates, samples and numerical annotations; error definitions and sign conventions; observed/unobserved membership; sensors and masks; field support; zoom windows; contour levels; interpolation/resampling and triangulation; KDE bandwidth and support; violin/box statistics; mean versus median; confidence or quantile intervals; spectral or wavelet processing; axis scale/limits/ticks and colour normalization/limits; cohort sizes and aggregation; crop extents of physical data. Preserve existing display transforms even when they seem imperfect.

Do not use per-model contrast enhancement, percentile clipping, colourbar rescaling, curve smoothing, median filtering, cosmetic denoising, extra contours, resampled surfaces, different seeds or automatic outlier removal. Do not change a bar chart into dots, a violin into a box, a ridgeline into another density estimate or a planar field into a 3-D surface. Do not add significance stars, rankings, arrows asserting improvement, error reductions or uncertainty semantics.

Scientific corrections belong to a separate approved change set. In particular, do not resolve Geo-FNO/FNO identity, density/V_x, the Fig. 4/5 score mismatch, Fig. 1 Q/K/V routing or uncertainty-interval definitions. An unresolved label can be restyled in place but blocks final scientific approval. The active Fig. 3 source is the five-panel V3-7 asset; do not restore the superseded a–f inventory or change its a–e panel letters as a caption repair.

## 2. Begin from the actual source

Locate the scripts or vector documents producing these assets:

| Figure | Asset named in `main-4.tex` | Manuscript page |
|---|---|---:|
| 1 | `Figure_1_Model.pdf` | 4 |
| 2 | `Figure_IrregularGeometry.pdf` | 7 |
| 3 | `Figure_MixedResolution.pdf` | 9 |
| 4 | `Figure_MultiFieldReconstruction.pdf` | 12 |
| 5 | `Figure_Evaluations.pdf` | 15 |

For the current three-figure art pass, the approved producer binding is:

| Manuscript asset | Active producer PDF | Content guidance |
|---|---|---|
| `Figure_MixedResolution.pdf` | `1_SubTask_SuperResolution/Save_TrainedModel/_TrainedModels/_Process_Figures/Assembled/MixedResolution_unified_v3_7_20260914_1158.pdf` | `FIGURE_03_MIXED_RESOLUTION.md` |
| `Figure_MultiFieldReconstruction.pdf` | `0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Process_Figures/Assembled/Composite/CoupledFieldReconstruction_round3_spacing_20260828_0810.pdf` | `FIGURE_04_MULTIFIELD.md` |
| `Figure_Evaluations.pdf` | `Dis_SI_Process/figures/generated/figure5_precision_log_revision_20260910/figure5_log.pdf` | `FIGURE_05_ENSEMBLE_COST.md` |

These are content-based bindings. Do not select an instruction file from a producer filename alone. If a later export supersedes one of these PDFs, update this table and `style_contract.json` together before styling.

The assets above are source references, not files included in this handoff. Locate their true producers in the working repository. If the numerical/vector sources are missing, report the missing inputs and stop the affected reconstruction work. Do not regenerate maps from screenshots, trace scientific boundaries from rendered pixels or invent an illustrative substitute. Existing vectors may be restyled without changing their paths; mathematical correction still needs author approval.

Before changing anything, preserve the original outputs and record file hashes, input paths, plot configuration and command. Render a baseline. Identify each axes object by semantic panel ID rather than by its incidental position in a Python list. For example, `fig3.c.Senseiver.Zero-H-M-rich.error` is a better identifier than `axes[29]`.

## 3. Publication-scale typography and canvas

Use **180 mm** as the production design width for these multi-panel figures. This is a proposed house design width, not a claim about a current journal requirement. The supplied TeX has A4 width 210 mm and 24-mm left/right margins, so its current insertion width is **162 mm**. Review every export at both widths. A font designed at 180 mm shrinks by 0.9 at 162 mm. Increasing export DPI will not repair undersized text.

The following five-level hierarchy is mandatory at the 180-mm design width.
Values are final-size roles, not Matplotlib defaults to be scaled again.  Use
regular weight throughout unless the table explicitly says otherwise; method
identity, category labels and major titles do not become bold merely to fill
space.

| Role | Design size | At 162 mm |
|---|---:|---:|
| Bold lowercase panel letter | 11 pt | 9.9 pt |
| Major group heading / figure-internal major title, regular | 9.5 pt | 8.55 pt |
| Axis title / colourbar title / ordinary subplot title, regular | 8.5 pt | 7.65 pt |
| Tick / standard legend / method label, regular | 7.8 pt | 7.02 pt |
| In-plot annotation / matrix value / mean label, regular | 7.0 pt | 6.3 pt |

The role ordering is strict: `panel label > major title > axis/subplot title >
tick/legend > in-plot annotation`.  A renderer must resolve every visible text
artist to one of these roles and record its role, resolved size, weight and
semantic panel owner in `LAYOUT_QA.json`. **Only panel letters may be bold; all
other visible text must use regular weight.** The sole current author-approved
exception is the exact model name **DMF-Gen** in the visible model-label rails
of Figure 5 panels a and f; that repeated identity must be bold while every
other model name remains regular. Record this exception per artist in the
layout QA so it cannot spread to titles, values, legends or other figures. Do not
promote dense values to tick size, or allow a legend to compete with a major
title.  A constrained annotation may be reduced below 7.0 pt only when the
complete 7.0-pt string cannot fit after wrapping, equivalent number formatting
and geometry reallocation have all been attempted.  Such a reduction requires
a per-artist record and may not go below 6.6 pt at 180 mm (5.94 pt at 162 mm).

These floors apply to ordinary base text, not the naturally smaller glyphs in superscripts/subscripts. Several current DMF-Gen ordinary labels measure about 4–5 pt in the PDF. Do not reproduce that size merely because the standalone figure looks readable when enlarged. Use one sans-serif family throughout, preferably Arial; use Liberation Sans or DejaVu Sans only as available substitutes, and use the same substitution in every figure. Record the resolved family. Never ship font files.

Set font sizes at the final physical export size. Preserve that size in the PDF MediaBox: indiscriminate `bbox_inches='tight'` can change the physical width and therefore the subsequent scaling. Use explicit margins and axes positions; validate PDF dimensions after export. Increase room, wrap long labels and reallocate whitespace before reducing font size. If all panels cannot be retained at the minimum readable size in an acceptable page footprint, report a layout conflict rather than drop data or shrink labels invisibly.

Suggested starting canvas heights at 180 mm: Fig. 1 175–190 mm, Fig. 2 185–205 mm, Fig. 3 215–230 mm, Fig. 4 220–235 mm and Fig. 5 175–195 mm. These are layout starting points, not constraints to stretch physical domains. Check the reduced height plus the LaTeX caption against the actual manuscript page; adjust whitespace and floats in a separate manuscript layout pass if necessary.

Use ordinary upright text for words and units and italic mathematical symbols for variables. Render consistent subscripts in `Y_CH4`, `Y_CO`, `U_1`, `C_D`, `K_t` and relative-L2. Preserve whether the data are dimensional or normalized; do not add Pa, m/s or another unit without source confirmation. Replace cramped inline mantissa/exponent fragments with a coherent scientific-notation label without changing its numerical meaning or precision.

## 4. Colour system: model identity is not physical magnitude

Use the following model identity colours consistently across Figs. 2–5. The palette evolves the present warm-red hero / muted alternatives, rather than replacing the paper's identity with the FBNO palette.

| Model display identity | Hex | Default marker for already-marked series |
|---|---|---|
| DMF-Gen / full model | `#C94053` | circle |
| FFM-Perceiver | `#4C86A6` | diamond |
| FFM-FNO | `#425B76` | square |
| SiT | `#9B83C1` | downward triangle |
| Latent FM | `#725591` | upward triangle |
| Geo-FNO (identity pending A02) | `#DA9A66` | x |
| Senseiver | `#8D9BAD` | rightward triangle |
| MLP-RBF | `#4C9E91` | plus |
| Reference / truth | `#252525` | none unless already present |

Use colour plus an existing line/marker distinction, not colour alone. Do not apply markers to dense spectra merely because a model has a default marker. Check grayscale and colour-vision-deficiency simulations; the palette is a proposal, not a claim of independently certified accessibility. If two traces remain hard to distinguish, adjust stroke/marker treatment uniformly and document it without changing the data.

Keep reference fields and predictions on the same **physical-value** colormap within each existing comparison group. Never colour DMF-Gen's physical map red because red is its model identity. Put model identity in the header, an existing rule or the quantitative series instead.

Default scalar semantics: nonnegative magnitudes use `viridis`; signed physical fields and signed residuals use `RdBu_r` with a light zero when the existing normalization maps zero to the midpoint; nonnegative absolute errors and nonnegative error heatmaps use `YlOrRd`; ordered correlation heatmaps use `cividis`; joint-density maps use `cividis` with the original bin mask retained. Preserve all current norms, limits, contour levels and ranges. Do not extend the Fig. 3 correlation range to [-1,1] or rescale Fig. 4 heatmap columns.

Changing a colormap is allowed; changing the mathematical normalization is not. If a signed map's original normalization does not put zero at its midpoint, do not silently introduce `TwoSlopeNorm` or change the limits. Either use an appropriate colour lookup with its neutral point at the existing normalized zero, or report the conflict. Similarly, do not consolidate independent colourbars by imposing new shared limits. The physical normalization comes from the existing source configuration, not from this style document.

For Fig. 1, use separate semantic accents: measurement/sensor anchor `#C94053`, global reasoning `#4C86A6`, local evidence `#DA9A66`, fusion/output `#4C9E91`. These colours denote computational roles in that schematic; they are not legends for baseline models.

## 5. Lines, fills, axes and small marks

Use 0.6–0.7-pt axes and ticks at design size, and 0.35–0.45-pt gridlines in light gray. Quantitative plots normally retain only left/bottom spines, with frame changes applied uniformly. Physical panels may use a thin neutral 0.4–0.5-pt frame where it helps preserve the boundary of an existing crop. Domain outlines are data: retain their exact paths and use approximately 0.7–0.9 pt, not a heavy black ring that obscures a narrow airfoil.

A useful hierarchy is truth 1.5–1.7 pt, DMF-Gen 1.5–1.6 pt and alternatives 1.0–1.2 pt. Never reduce an alternative's contrast to near invisibility. Use comparable bar widths and fill strength across methods; a red DMF-Gen bar and readable label are sufficient emphasis. Markers should generally be 3–3.5 pt with approximately 0.6-pt edges. Use the original marker coordinates and cadence.

Confidence/quantile ribbons: retain the original lower/upper arrays and interval interpretation, use a common 0.12–0.18 alpha and draw the central trace above them. Error-bar caps/lines should remain readable at final size, roughly 0.7–0.8 pt. For existing per-state dots, retain every point, its existing jitter and ordering; a common small size and alpha around 0.10–0.18 can reduce clutter. Do not regenerate jitter or select a prettier subset.

Violin, box and ridge fills should be soft enough to reveal internal summaries, but keep the actual density path and statistic locations. Never recompute a KDE to make it smoother, more symmetric or narrower. White internal summary strokes can work on medium-dark fills; use a dark stroke where white would disappear. Preserve whether a mark denotes mean, median, quartile or a confidence interval.

Apply gridlines only where already appropriate; adding a very light major grid is allowed as reference decoration, but do not add scientific threshold/reference lines. Keep all axis scales, limits, ticks and reference thresholds from the original. A bar axis that currently begins at zero must continue to do so. Do not change linear/log axes or truncate ranges to magnify differences.

## 6. Layout grammar

Preserve major panel order and keep each figure's existing narrative. Within a comparison matrix, make the plot windows equal in size, align row/column headings and place every error annotation in the same relative location. Group information by whitespace rather than heavy rectangles. Separate major blocks by about 4–6 mm, adjacent maps by about 1–1.5 mm and a colourbar from its map group by about 1.5–2 mm. These are adjustable spacing targets.

Panel letters sit outside data regions at a consistent left/top offset. Do not put bold letters over colour maps. Align a shared legend with the group it explains and use a compact multi-column layout where needed. Preserve legend memberships and category order unless the legend alone is being harmonized; the ranking/order of the plotted rows remains frozen. Never move a legend over a scientifically important boundary, uncertainty ribbon or high-frequency tail.

Do not duplicate a label merely to decorate the canvas. Conversely, do not remove labels, sensors, plots or colourbars that contain information. Existing numerical labels can move from an opaque in-map box into a small adjacent gutter if the association remains unambiguous and the complete map remains visible. Do not enlarge a box so that it hides an error hotspot.

Latest author-authorized Figure 4 exception: the far-right Panel-a colourbars may
be removed in the compact art revision because the field/error limits remain
frozen and recorded in the source manifest.  This exception does not authorize
rescaling, shared-limit substitution or colourbar removal in another figure.
Figure 4 V3 may also omit the displayed Ground truth montage column, omit the
displayed Panel-b `Unobs.` column, remove the `mu=` prefix from Panel-d mean
annotations, and horizontally expand the seven retained Panel-a axes. These
are display-inventory/geometry exceptions only: the complete source matrices,
truth arrays, numeric mean values, coordinates, transforms, masks, limits,
normalizations, distributions and scientific ordering remain frozen and must
be retained in the source-state comparison.
Panel-c spectral y ticks may be thinned to at most four labelled levels without
changing the log scale or axis limits, and Panel-d tick labels may be staggered
or repositioned without changing their numerical values. For Figure 4 V3 only,
all Panel-d tick labels may instead be hidden consistently when a final-render
single-line collision test fails; hiding labels does not authorize changing the
tick locations, log transforms or axis limits.
Figure 4 V4 supersedes that fallback for its active export: retain the three JSD
tick-label sets, place them on one baseline, and resolve crowding through wider
subplot columns and outward anchoring only. Its c/d left spines and panel-letter
offsets must be measured in page coordinates and match within `0.01 mm`.

### Mandatory layout-integrity and cross-figure consistency gates

The spacing guidance above is a release contract, not a visual suggestion. Every proposed figure must pass the following checks at both the 180-mm design width and the manuscript's current 162-mm insertion width. A figure with a known collision, clipped object, ambiguous label ownership or unexplained palette mismatch is **not complete**.

1. **Zero unintended overlap.** Titles, panel letters, axis labels, tick labels, legends, colourbar labels/exponents, method names, numerical annotations and callouts must not overlap one another or cross into a neighbouring panel. Ordinary headings, legends and panel letters must remain outside data regions. Existing data-anchored annotations may remain inside a map or plot only when their scientific association is frozen and moving them would make that association ambiguous; even then, their box must not cover a contour, sensor, boundary, hotspot, curve, interval, bar, violin tail or other evidence-bearing mark. Record every such intentional in-data annotation by semantic panel ID in the layout QA record; an unrecorded intersection fails.
2. **Measured clearance.** Preserve the 4–6-mm target between major blocks, 1–1.5 mm between adjacent maps and 1.5–2 mm between a map group and its colourbar. After final text extents are known, use hard minimum clearances of 3 mm between major panel blocks, 1 mm between adjacent plot/map windows, 1 mm between ordinary text and a non-owned axes/data region, and 1.5 mm between a panel letter and the nearest non-owned text or data region. Horizontal and vertical gaps must both be measured; whitespace in only one direction does not excuse a collision in the other. If frozen content makes a minimum impossible, report the layout conflict rather than shrink text below the font floor.
3. **Canvas containment and clipping.** Every visible artist, including superscripts, minus signs, rotated labels, error-bar caps, marker edges, legends and colourbar exponents, must lie inside the intended PDF MediaBox with a positive clearance. No text may be clipped by an axes patch, rasterization boundary or tight bounding box. The exported MediaBox must retain the declared physical dimensions.
4. **Semantic colour consistency.** Build and retain a figure-to-figure style table keyed by semantic identity. The same model, recipe, observation status, physical quantity, residual sign and ablation variant must use the same approved colour/line/marker treatment everywhere that role recurs. Do not reuse one colour for two conflicting roles within a figure. Validate exact resolved RGBA/hex values from the artists, then inspect grayscale and a colour-vision-deficiency simulation; every comparison must remain distinguishable without relying on colour alone.
5. **Balanced local rhythm.** Comparable rows, columns, repeated maps, heatmaps and scorecard axes must have equal plot-window dimensions and consistent internal padding unless the evidence hierarchy explicitly requires a different size. Shared titles, legends and colourbars must be centred on the complete group they explain. Long labels must be wrapped consistently, with equal line spacing and baseline offsets across peer panels.
6. **Adjacent-major-row union clearance.** For every vertically adjacent major
   panel row, form a rendered union bounding box from every visible owned
   element—not only the nominal panel/container rectangle. The union must
   include data axes, axis `tightbbox` extents, both levels of tick labels,
   axis titles, panel letters, legends, colourbars and their titles/exponents,
   annotations, and any figure-level text assigned to that panel. The upper
   row's union bottom must remain at least 3 mm above the lower row's union top
   at final insertion width, and the two unions must have zero intersection.
   Missing or unowned figure-level text is itself a failed audit. A positive
   nominal GridSpec/container gap cannot override a failed rendered-union gate.
   The QA record must retain the renderer-derived box for every required artist
   class, the component union boxes, each adjacent-row pair, and the final row
   union boxes in PDF-MediaBox millimetres. Unknown visible artists, missing
   boxes, clipped artists, or unexplained intersections are failures. The
   162-mm preview must be rasterized anew from the vector export at the target
   physical scale; resizing the 180-mm PNG is not an insertion-width check.
7. **Pairwise text/evidence collision gate.** After the final aspect-ratio and
   axes-position adjustments, recompute every visible text bounding box with
   the renderer. Compare each ordinary annotation, callout and free-standing
   label against every data-window/evidence bounding box, including its owning
   axes, and compare all such text boxes pairwise. External annotations must
   have zero data-window intersections and at least 1 mm clearance from every
   data window at the 180-mm design width; peer text boxes must have zero
   intersection. An annotation may intersect its owning data window only when
   its semantic ID is present in the intentional in-data registry and the
   high-zoom evidence review confirms that it covers no contour, sensor,
   boundary, hotspot, curve, interval, bar, violin or matrix value. The audit
   must store each tested text box, every nearest evidence box, the measured
   clearance, all pairwise text intersections and the exception registry.
   Running this gate before equal-aspect axes settle is invalid because the
   final renderer can move the plot window after the annotation was placed.

Self-check these gates after the final draw and after the final aspect-ratio pass, not from configuration values alone. Use the selected plotting backend's renderer to collect the final display bounding boxes and semantic owners for all text, axes, legends, colourbars and data-bearing artists. Run pairwise forbidden-intersection, annotation-to-all-data-window, adjacent-major-row union and minimum-clearance checks on the 180-mm render and again on a true 162-mm render. Then inspect full-page and high-zoom raster previews because bounding boxes alone cannot detect a label covering a contour, dense point cloud or hotspot. Record the tested dimensions, collision count, clipped-artist count, per-annotation nearest-evidence clearance, row-union owners and extents, minimum horizontal and vertical clearances, intentional in-data annotations, resolved semantic style table and grayscale/CVD results in `LAYOUT_QA.json`. Passing requires zero unexplained collisions and zero clipped artists; visual inspection cannot waive a failed geometric check, and a geometric pass cannot replace visual inspection.

## 7. Physical maps and image fidelity

Preserve physical aspect ratios, coordinate transforms and domain masks. For grid comparisons, use the original sampling/rendering interpolation; do not make low-resolution panels look high-resolution with a new bicubic display. For irregular grids, preserve the existing triangulation, connectivity and mask. Keep the same realized source surfaces in Fig. 1.

Truth and prediction windows must share the original spatial extent. Matched zooms must show the same cells/coordinates across models. Contour line levels, positions and count are frozen. Their colour, opacity and stroke can be softened uniformly, but they cannot be moved, resampled or deleted. Low contrast in a residual map is not permission for per-model limits.

Out-of-domain regions should be visibly neutral using the existing mask. Do not turn true zero-valued physical regions into “no data,” or fill missing bins with a plausible-looking value. In density plots, retain the distinction between empty bins, masked bins and small nonzero densities.

## 8. Export, tests and delivery

Deliver an editable source and PDF for each figure, plus a high-resolution PNG preview; SVG is useful when the source can preserve its text and geometry reliably. Keep text, axes, curves, boundaries and schematic shapes vector where possible. Large contour fields or dense point layers may be rasterized at sufficient effective resolution while keeping surrounding text vector. Export at least a 600-dpi preview for detailed inspection and verify that rasterized layers remain adequate after insertion. Do not rasterize the whole figure as a shortcut.

Keep source text editable and fonts embedded. Do not convert every glyph to outlines solely to conceal a font issue. Inspect at 100% print scale and at high zoom for clipping, missing superscripts, hairline disappearance, cropped colourbar exponents and transparency seams.

For each figure, write a `STYLE_CHANGELOG.md` entry: input/source paths; style settings changed; preserved data hashes and plotted-array checks; approved label-only changes, if any; unresolved author checks; final dimensions and font family; output paths. Include before/after renders, `LAYOUT_QA.json`, and report whether the plot arrays, geometry, category order, normalization and statistic locations are identical. Run a scientific-state comparison separately from image-difference review: a large pixel difference is expected when the palette changes, but changed scientific arrays are not.

No final asset is approved until every original panel and annotation can be matched to its after-version, all ordinary text is readable at 162 mm, and no scientific change has entered the style commit. Use `06_QA_AND_CODEX_HANDOFF.md` for the final checklist and agent prompt.
