# Figure5b quantitative companion

Does larger ensemble spread track harder states?

Package B: archived benchmark checkpoint identities; a/b/d use200 paired states and64 draws with moving-block25/2,000 intervals; a/b bootstrap seed20260830, d method-specific recorded bootstrap_seed in benchmark_main_d.csv. f uses original1,000-state accuracy and separately measured resource protocols. These results were not remeasured for the new ablation reference.

Metric and aggregation: Spearman rank association between macro normalized spread and macro ensemble-mean relative-L2 across states; range[-1,1], positive is informative, zero reference. The cloud contains accepted bootstrap estimates, not independent models.

Result and boundary: Positive DMF association supports uncertainty informativeness. Association does not establish calibration, prospective prediction or causality.

Coordinates at source precision:

| method | metric_value | ci_low | ci_high |
| --- | --- | --- | --- |
| DMF-Gen | 0.65362334058351468 | 0.56047581468082919 | 0.72080807102831324 |
| FFM-FNO | 0.183048076201905 | -0.0037361788058276999 | 0.3588600901546215 |
| FFM-Perceiver | 0.2147093677341933 | 0.079897547375348807 | 0.3476554288011155 |
| Latent FM | -0.033017325433135798 | -0.16405559930810179 | 0.10635111043614311 |
| SiT | 0.26120753018825471 | 0.10345463889629911 | 0.38370677275261189 |

Source rows: `Dis_SI_Process/results/derived/20260910_1540/benchmark_main_b_summary.csv`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.
