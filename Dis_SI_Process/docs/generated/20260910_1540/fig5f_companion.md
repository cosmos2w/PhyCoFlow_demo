# Figure5f quantitative companion

What accuracy and measured update/inference footprints accompany the archived benchmark methods?

Package B: archived benchmark checkpoint identities; a/b/d use200 paired states and64 draws with moving-block25/2,000 intervals; a/b bootstrap seed20260830, d method-specific recorded bootstrap_seed in benchmark_main_d.csv. f uses original1,000-state accuracy and separately measured resource protocols. These results were not remeasured for the new ablation reference.

Metric and aggregation: Five columns: mean physical macro error with stateCI; median synchronized training-update ms with IQR; peak allocated trainingGiB; median warm inference ms with IQR; Model stateMiB and Peak allocatedMiB. TrainingB32 retains native workloads; inferenceB1.

Result and boundary: The archived accuracy/resource comparison is preserved. IQR timing variation is not a confidence interval; per-update time is not total training time; stage maxima are not additive concurrent costs.

Coordinates at source precision:

| method | mean_unobserved_relative_l2 | error_ci_low | error_ci_high | training_update_ms | training_peak_allocated_mib | inference_latency_ms | model_state_mib | inference_peak_allocated_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DMF-Gen | 0.1170851006737255 | 0.1152293016262132 | 0.119002979177899 | 112.19590157270432 | 8121.091796875 | 16.690303802490234 | 24.826839447021484 | 417.64208984375 |
| Senseiver | 0.1429897546212702 | 0.1410956958960351 | 0.1448327279410096 | 49.854734912514687 | 3916.97412109375 | 8.3015680313110352 | 31.813251495361328 | 536.671875 |
| SiT | 0.21025289583977849 | 0.20776769035228751 | 0.21295689960316319 | 661.80231189355254 | 14881.0029296875 | 20.991935729980469 | 39.92413330078125 | 82.40625 |
| Geo-FNO | 0.22989554701189391 | 0.22698744199371221 | 0.2330580060310197 | 235.67476449534297 | 9493.8974609375 | 3.4067521095275879 | 19.700458526611328 | 122.521484375 |
| FFM-Perceiver | 0.34789035148214847 | 0.34502624571286561 | 0.35085766775255439 | 109.16404239833356 | 4761.90771484375 | 23.087104797363281 | 20.124530792236328 | 312.5302734375 |
| FFM-FNO | 0.38981520058406238 | 0.38731610805910138 | 0.39233818100936141 | 249.84606495127079 | 9449.73974609375 | 8.7019357681274414 | 19.709735870361328 | 149.7373046875 |
| MLP-RBF | 0.3961935395853694 | 0.39296591322811109 | 0.39931462262486739 | 24.946504272520546 | 1955.02490234375 | 3.1447041034698486 | 2.2735786437988281 | 280.70654296875 |
| Latent FM | 0.45310385726020469 | 0.44905155392021529 | 0.45709821546386181 | 90.725332032889114 | 4233.2724609375 | 10.172415733337402 | 337.75148391723633 | 392.1171875 |

Source rows: `Dis_SI_Process/results/derived/20260910_1540/benchmark_main_f.csv`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.
