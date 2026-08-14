# Mixed-resolution unified-v2 figure contract

Core conclusion: The effect of lower-resolution training is architecture dependent; DMF-Gen remains the most robust model when high-resolution training is reduced or removed, while scale-resolved evidence distinguishes global accuracy from fine-structure fidelity.

Figure archetype: asymmetric mixed-modality figure.

Target/output: double-column publication figure; 183 mm fixed-width editable SVG and PDF plus a high-resolution PNG preview.

Backend: Python (Matplotlib) for plotting, export, preview rendering, and visual QA.

Panel map:

- a: establishes the multi-resolution sampling and resolution protocol.
- b: compares all recipe/model reconstruction errors at a fixed sensor budget.
- c: provides representative full-field, zoomed-region, and error-map evidence.
- d: tests robustness across sensor counts.
- e: compares representative orthogonal-wavelet field components.
- f: quantifies scale-wise pattern correlation and variance-fraction bias.

Evidence hierarchy:

- Hero evidence: panel c qualitative reconstruction plate.
- Quantitative validation: panels b, d, and f.
- Controls/robustness: panel a; panel e is the truth-selected qualitative counterpart to panel f.

Statistics: 300 canonical held-out cases; mean and bootstrap 95% confidence intervals for reconstruction/sensor panels; wavelet summaries include mean, standard deviation, median, quartiles, and bootstrap 95% confidence intervals, with median/IQR shown in panel f.

Source data: the canonical reconstruction-cache manifest and its derived CSV/JSON result tables.

Image integrity: field layers use the physical grid; each wavelet scale row has one zero-centred normalization shared across truth and models; no model-specific smoothing, contrast adjustment, or sensor overlay.

Reviewer risks: missing recipe coverage, stale checkpoint-derived cache rows, inconsistent axes/normalization, unreadable final-size text, and text–figure or text–text collisions (especially panel c headers, annotations, and colorbars).

Refresh provenance: the `20260806_1124` artifact selectively refreshes the 1,500 FFM-Perceiver / H-limited canonical rows from the updated `best.pt` checkpoint. Artifact inspection, recipe-manifest validation, and CUDA loading passed; all regenerated arrays are finite and no dummy substitution was required. The other 35,700 canonical manifest rows are unchanged. The all-recipe sweep retains 100/100 summary-cell coverage. Multiscale panels reuse the unchanged audited `20260802_1250` tables because H-limited is not part of their displayed recipe set.

Scoped quantitative change: relative to `20260802_1250`, FFM-Perceiver / H-limited mean physical relative L2 increased by 3.12%, 10.11%, 14.58%, 13.02%, and 10.64% at 64, 128, 256, 384, and 512 sensors, respectively. No other all-recipe accuracy or plotted sensor-sweep cell changed.

QA caveat: Arial was unavailable in the plotting environment, so the declared Liberation Sans/DejaVu Sans fallback was used. All 49 numerical, provenance, geometry, typography-role, editable-text, and collision checks passed for `20260806_1124`.
