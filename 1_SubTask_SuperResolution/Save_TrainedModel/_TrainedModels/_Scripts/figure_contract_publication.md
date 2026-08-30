# Figure contract: mixed-resolution learning and zero-H reconstruction

Core conclusion: Mixed-resolution training improves high-resolution reconstruction and fine-structure recovery, while models trained without H-resolution examples retain measurable transfer that depends on spatial frequency and test-time sensor count.

Figure archetype: asymmetric image-plate plus quantitative composite.

Backend: Python/Matplotlib only.

Target output: 183 mm × 210 mm; editable SVG and TrueType PDF; 600 dpi PNG preview.

Panel map:

- a — enlarged, equal-aspect, strictly discrete native L/M/H cells, with a shared high-gradient zoom that exposes the 32²/64²/128² density difference, and the five active-case training compositions; the redundant field colorbar and state subtitle are omitted.
- b — paired H-limited versus Mixed-HML mean physical relative L2 with bootstrap 95% CI.
- c — common-normalization, multi-model qualitative evidence for Question A.
- d — M-aligned coarse/detail decomposition of the same comparison.
- e — H-only and zero-H transfer estimates, preserving genuine missing references.
- f — Zero-H-balanced, shared contrast-aware fine-structure ROI with full field, bordered magnification, vector observation schematic cropped to that ROI, absolute-error ROI, and full-field physical relative-L2 annotations.
- g — median spectral error and IQR for Mixed-HML, Zero-H-balanced, and Zero-H-M-rich.
- h — sensor-count efficiency from 64 to 1,024 observations with sensor-density labels and bootstrap 95% CI for the same three recipes.

Evidence hierarchy:

- Hero evidence: panels c and f.
- Aggregate validation: panels b and e.
- Mechanistic/scale evidence: panels d and g.
- Protocol and robustness: panels a and h.

Statistics: 300 distinct canonical test cases; aggregate means use bootstrap 95% confidence intervals; spectral curves use median and interquartile range.

Image integrity: fields are read in physical units from finalized compact caches. No Gaussian filtering, sharpening, or model-dependent normalization is permitted. Panel-a fields use finite Cartesian-grid validation and exact flat-shaded native cells; the shared inset is selected only from integrated H-resolution ground-truth gradient. Contours elsewhere are computed from the cached native grid. Panel-f uses one shared fixed-size ROI for every column. It maximizes the direction-neutral absolute prediction difference between DMF-Gen and Senseiver among windows at or above the median integrated ground-truth gradient; this model-dependent selection is recorded in the source manifest. Observation markers show only the actual sensors inside that ROI and remain vector artwork in SVG/PDF.

Reviewer risks addressed: native-resolution ambiguity, inconsistent color limits, undisclosed ROI selection, fabricated H-only references, sensor-plan misalignment, blurred rasterized observation markers, and illegible final-size legends.
