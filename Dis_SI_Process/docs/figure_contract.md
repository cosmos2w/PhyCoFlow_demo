# Figure 5 draft contract

Core conclusion: repeated conditional generations should expose non-collapsed empirical spread associated with missing-channel error, while the adopted DMF workflow should show a transparent and tractable accuracy–latency–memory trade-off.

- Figure archetype: asymmetric mixed-modality figure.
- Target/output: Nature Machine Intelligence-style, 183 mm double-column, editable SVG only during the draft stage.
- Backend: Python/Matplotlib exclusively.
- Final draft size: 183 mm × 208 mm; standalone panels use the same typography and color vocabulary.
- Hero evidence: panel a, a spatial truth/ensemble-mean/error/ensemble-standard-deviation plate for a predeclared unobserved field.
- Validation evidence: panels b–d, empirical coverage, interval width, and spread–error association.
- Computational evidence: panels e–h, native-mesh error–latency, query scaling, memory scaling, and error versus measured NFE.
- Statistics: state-level coverage and width; temporal block-bootstrap intervals over states; Spearman association as primary; warm synchronized latency median/IQR; peak allocated memory; physical relative L2.
- Source-data policy: prefer frozen ValidationV2 CSV/NPZ products. Existing single-reconstruction and architecture/system benchmarks may only appear as visibly labelled draft proxies.
- Image integrity: truth and reconstruction share color limits; absolute error and sensitivity have independent robust non-negative limits; no smoothing, resampling, or local contrast manipulation.
- Reviewer risks: solver-sensitivity is not predictive uncertainty; held-out-case bootstrap intervals are not predictive intervals; architecture benchmarks are not the planned eight-method native-checkpoint comparison; throughput-extension coordinates are not primary accuracy evidence; all timing is hardware- and protocol-specific.

## Panel map

- a — spatial UQ example: truth, ensemble mean, absolute error, ensemble standard deviation. Until UQ visual maps exist, use a real deterministic reconstruction and label cross-NFE solver sensitivity as a non-UQ proxy.
- b — empirical versus nominal coverage from the S=64 calibration cohort. Remain visibly pending until `coverage_by_level.csv` exists.
- c — fieldwise interval width from the same S=64 cohort, in physical units. Remain visibly pending until the formal width columns exist.
- d — state-level spread versus ensemble-mean error with binned trend and Spearman association. A one-state cross-NFE diagnostic may preview layout only.
- e — eight-method native-mesh latency versus adopted error. An existing architecture/throughput Pareto may preview layout only.
- f — DMF median latency versus query count.
- g — DMF peak allocated memory versus query count.
- h — DMF error versus measured NFE.

## Draft gate

No proxy panel may be described as manuscript evidence. A panel becomes formal only when its source is a frozen ValidationV2 run with the identities, row counts, QA, and temporal-dependence rules specified in `bk/ToDos/Process_Plan_V2.md`.
