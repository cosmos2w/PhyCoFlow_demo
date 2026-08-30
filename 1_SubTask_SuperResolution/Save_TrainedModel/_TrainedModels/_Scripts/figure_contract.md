# Figure contract: mixed-resolution learning and zero-H transfer

Core conclusion: Lower-resolution training cases improve H-resolution reconstruction through better coarse and fine-scale recovery, while zero-H training transfers fine structure with measurable frequency- and sensor-dependent penalties.

Figure archetype: asymmetric mixed-modality figure. Backend: Python/matplotlib exclusively. Target: Nature-family double-column figure, 183 × 170 mm. Primary output is editable SVG, accompanied by PDF and PNG preview. Dense field maps are rasterized inside the otherwise vector figure to control file size; text, axes, markers, and lines remain vector/editable.

Evidence hierarchy:

- Hero evidence: panel b paired Question-A estimates, panel c aligned Question-A fields/errors, panel e separately grouped Question-B estimates, and panel f cross-model zero-H fine-structure views.
- Mechanistic evidence: panel d H-to-M coarse/detail decomposition and panel g radial frequency error.
- Protocol and robustness: panel a native grids/training compositions and panel h nested test-time sensor sweep.

Panel map:

- a: one identical case/time at native L, M, and H grids; actual training-case compositions and spatial-DOF budgets.
- b: H-limited versus Mixed-HML physical relative L2, paired within each model, bootstrap 95% confidence intervals; H-only is a reference marker only.
- c: H truth, H-limited/Mixed-HML reconstructions, and absolute errors for an algorithmically selected median-like case.
- d: paired coarse and detail errors using `P_M u = upsample_M_to_H(area_average_H_to_M(u))`.
- e: H-only, Zero-H-balanced, and Zero-H-M-rich estimates, kept visually and statistically separate from panel b.
- f: H-resolution hybrid field/zoom comparison for available deterministic models with H-only checkpoints.
- g: radial spectral error with actual L/M-grid Nyquist boundaries.
- h: error versus nested 64/128/256/384/512 sensor sets; 256 is the formal default.

Statistics: the default screen is 300 distinct held-out CFD cases, one deterministic time per case drawn from the intersection of every available run's usable time window. A case-time pair is the sampling unit. Curves/estimates report snapshot means with case-level bootstrap 95% confidence intervals; frequency curves report median/IQR. All formal errors are evaluated at H in physical units. Sensor coordinates and stochastic generation seeds are paired across models, recipes, and sensor counts.

Source-data integrity: every quantitative panel is backed by timestamp-matched CSV. Qualitative fields are hydrated from float32 compact caches. Ground truth and H-grid coordinates are stored once as shared immutable arrays. The representative case minimizes robust distance to the median recipe errors and paired difference; the zoom ROI is selected from ground-truth gradient magnitude only.

Image integrity: field and error limits are shared within each invited comparison. No local contrast, gamma, or selective filtering is applied. Zoom crops use nearest native cells. Missing checkpoints remain explicitly marked `Missing`.

Reviewer risks: pseudo-replication across adjacent times, recipe/config mismatch, normalization leakage, cache quantization, unequal sensor masks or stochastic draws, conflating spatial DOF with wall-clock cost, and cherry-picked qualitative cases. The workflow contains explicit guards for each risk.
