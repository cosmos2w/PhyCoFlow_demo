# Figure contract

Core conclusion: coupled physical fields can be reconstructed from incomplete channels, and reconstruction fidelity and joint-field structure can be compared fairly as observed channels increase.

- Archetype: asymmetric mixed-modality figure, with fieldwise L2 as the hero quantitative grid.
- Backend: Python/matplotlib only.
- Target/output: Nature-family double-column, 183 mm; editable SVG primary, PDF and PNG supported.
- Evidence: condition matrix defines the intervention; contours localize representative behavior; Field-L2 quantifies fidelity; representative PDFs show cross-field structure; JSD violins quantify test-set robustness; channel-wise spectral energy and dB LSD test preservation of spatial-scale energy.
- Statistics: physical relative L2, base-2 JSD, and channel-wise dB/natural-log LSD; per-snapshot values; mean, standard deviation, median, quartiles, and snapshot-bootstrap 95% confidence intervals.
- Traceability: every panel is sourced from timestamped CSV/cache artifacts and records checkpoint, sensor plan, split, solver, NFE, consistency policy, snapshot, and seeds.
- Integrity: truth/reconstruction share physical limits; errors use one pooled robust upper limit; joint PDFs share global physical-space bins; spectra use identical native FFT shells and a recorded physical/topological coordinate decision; missing data are never imputed.
- Reviewer risks: model relocation dependencies, inconsistent point ordering, direct sensor entries lowering observed-field L2, and single-generation stochastic uncertainty. These are exposed in status/metadata columns.
