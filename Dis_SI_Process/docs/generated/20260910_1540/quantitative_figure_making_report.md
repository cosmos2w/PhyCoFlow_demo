# Quantitative figure-making report: Figure5 V7

Conclusion: the available full stochastic model reconstructs more accurately than its conditioning-route variants; IID slightly improves whole-field error while producing excess fine-scale velocity power and larger phase-sensitive residuals.

Archetype: quantitative grid; dedicated right-hand architecture/prior column. Python/Matplotlib in fig only. Main canvas183×205mm; editable SVG plus600dpiPNG, no new PDF delivery. Panel map aCRPS, bspread/error, cstochastic controls, dselective risk across two columns, etwo high-band axes, ffull-width five-column scorecard.

All old benchmark coordinates and method identities are preserved by accepted-renderer scientific geometry digests. Layout changes: larger canvas; repositioned axes; standalone bmethod labels and dlegend; minimum inherited annotations5.8pt; Training time wording becomes Training update time; isolated AURC annotation removed; filled Model/hollow Peak retained. Full red; route-removal slate; IID muted amber. New axes use6pt row/tick text and6.6pt labels. No smoothing/clipping/inference.

Evidence packages and source identity are defined in source_manifest.json. Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats. Package B: archived benchmark checkpoint identities; a/b/d use200 paired states and64 draws with moving-block25/2,000 intervals; a/b bootstrap seed20260830, d method-specific recorded bootstrap_seed in benchmark_main_d.csv. f uses original1,000-state accuracy and separately measured resource protocols. These results were not remeasured for the new ablation reference.


## Prior-source tradeoff

| policy | model | macro_L2 | U1_canonical_power | U1_highband_L2 | U1_whole_LSD_dB | U1_residual_full_fluctuation_percent |
| --- | --- | --- | --- | --- | --- | --- |
| last | Full model | 0.10632076677317751 | 0.42853620860075847 | 0.87340655840460224 | 4.5602247498930222 | 0.031224090091379997 |
| last | IID Gaussian prior | 0.1042938161915426 | 3.1199122893463591 | 1.8252839107455296 | 3.7882606088325219 | 0.13171897511812 |
| best | Full model | 0.10789321497517119 | 0.42269102757974808 | 0.87430889076201579 | 4.5819063382288601 | 0.031341397871500004 |
| best | IID Gaussian prior | 0.1049672037365057 | 3.2378984039698624 | 1.8574287413417121 | 3.850324137711671 | 0.13634545041411 |

## Checkpoint map

Archived benchmark DMF SHA256: 857a505ff96cc3512c20f45641250d9db9d448c1f1e33b8c4a1c792acb1e4a06; its mean error remains0.1170851006737255 (display0.117). Full ablation hashes below are distinct; no archived64-draw UQ or cost is reassigned to them.

| display_label | policy | checkpoint_epoch | global_step | parameter_elements_with_optimizer_state | scheduler_t_max | checkpoint_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| Full model | last | 7520 | 533920 | 6507151 | 10000 | 71a9e1004010558f7137f4140eb976d51d367d39e9bc3473334912257cbaa541 |
| No sensor feedback | last | 6000 | 426000 | 5716879 | 10000 | 5037246ef415d5d0b78d253b5a4ecdde76b549f2436541f32dd5d1aa54e401cf |
| No local conditioning | last | 6000 | 426000 | 5650572 | 10000 | 15de1f603be5580e51379675296dfb218e7a34f983272f081abd442510683cda |
| Local-only conditioning | last | 6000 | 426000 | 895118 | 10000 | 0e6184a6b9c5634d79005529744f98ebc0f3d016a084ec1133987b987c1ea2a3 |
| IID Gaussian prior | last | 7310 | 519010 | 6507151 | 10000 | 21a57fbc86dc7eef7b498c8a1110b1ac9c1a679f73dd5f344f8d446c4de454a9 |
| Deterministic regression | last | 7180 | 509780 | 6507151 | 10000 | f5fb90759ae30eebcfcaebac7f12ce38d52b4e85bac13adde1b864bec3314ecc |
| Full model | best | 7095 | 503745 | 6507151 | 10000 | 634b839b1be4abdb976fe848524bb81708254df65b1e8cd29b3111a9f63ec654 |
| No sensor feedback | best | 5365 | 380915 | 5716879 | 10000 | d311ef1f17aa9ad7bba72ace94560402743e02644a0ab91deb466587b658fb3e |
| No local conditioning | best | 5615 | 398665 | 5650572 | 10000 | 7db5a8b4c7afd11ec9954b7f2891f81b83b6f18b4134596a4dca37fd9817abf1 |
| Local-only conditioning | best | 5300 | 376300 | 895118 | 10000 | b8549181d245b0e0d8ddf57aac4bb9a0246aebe1b0523bb265b1491b00976d0f |
| IID Gaussian prior | best | 7155 | 508005 | 6507151 | 10000 | 7e50d6260677b8f725f6354f8d24c481877f49e7a813ff6660554fa7c275c1ae |
| Deterministic regression | best | 7140 | 506940 | 6507151 | 10000 | 73275866949b4975719afb9545e005746db87890d015b2fad47c4202479895dd |

## Exact command and environment record

```json
{
  "repository_root": "/home/wanglz/Desktop/src/PhyCoFlow",
  "branch": "paper/postprocessing-multifield-superresolution",
  "head": "9b1ef32978edd49cfddd55365ea54b4b7bcac57f",
  "timestamp": "2026-09-10T15:36:42-04:00",
  "git_status_short": "?? Dis_SI_Process/docs/Figure5_V7_Ablation_Codex_Instructions_20260910.md",
  "environment": {
    "python": "3.11.15",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "matplotlib": "3.10.9",
    "conda_environment": "fig"
  }
}
```

```bash
rtk proxy conda run -n fig python Dis_SI_Process/scripts/build_figure5_v7_bundle.py --timestamp 20260910_1540 --strict-formal
rtk proxy conda run -n fig python -m unittest Dis_SI_Process.tests.test_figure5_v7_ablation
```

No training, inference, GPU timing, new solver settings or fresh bootstrap was run for this release. Raw historical curves are descriptive recordings, not dense reconstruction-error trajectories or independent repeats. All rows available at 2026-09-10 20:38:23 UTC were frozen in the SI history snapshot; the full-model history contains8026 rows, while its evaluated last checkpoint is epoch7520. The snapshot manifest records each captured source hash and compact CSV hash; reruns use these frozen histories.

## Panel-reference map

| old_panel | new_panel | role |
| --- | --- | --- |
| a | a | Normalized CRPS |
| b | b | Spread–error association |
| c | d | Selective reconstruction |
| d | f | Accuracy and footprint |
| new | c | Conditioning and source variants |
| new | e | Fine-scale velocity fidelity |

## All five benchmark selective-risk areas

| method | risk_auc |
| --- | --- |
| DMF-Gen | 0.74104139459652396 |
| FFM-FNO | 0.7946414715826795 |
| FFM-Perceiver | 0.79153763995209825 |
| Latent FM | 0.80140378710766413 |
| SiT | 0.78096790066959643 |

These areas integrate the normalized curve over0.2–1.0 and are not divided by0.8. At80%retention the DMF within-method normalized risk is 0.95391194830325221.

## Figure5a quantitative companion

Does the archived empirical conditional ensemble assign useful marginal predictive distributions?

Package B: archived benchmark checkpoint identities; a/b/d use200 paired states and64 draws with moving-block25/2,000 intervals; a/b bootstrap seed20260830, d method-specific recorded bootstrap_seed in benchmark_main_d.csv. f uses original1,000-state accuracy and separately measured resource protocols. These results were not remeasured for the new ablation reference.

Metric and aggregation: Normalized empirical CRPS = mean over points and four unobserved fields of [mean_k|X_k-y| - (1/(2K²))sum_kl|X_k-X_l|] / frozen training-field standard deviation. Ideal0; marginal, not phase-sensitive. Per-state values then equal-weight cohort mean. Boxes show state dispersion; mean intervals are separate.

Result and boundary: DMF-Gen has the lowest accepted normalized CRPS. This is not a demonstration of perfect calibration; inherited reliability remains underdispersed.

Coordinates at source precision:

| method | metric_value | ci_low | ci_high |
| --- | --- | --- | --- |
| DMF-Gen | 0.066717025886039102 | 0.063976462251938995 | 0.069406044216142104 |
| FFM-FNO | 0.39890982557542259 | 0.37387576073651241 | 0.43068150110374848 |
| FFM-Perceiver | 0.2596316273288849 | 0.2476113821739177 | 0.27228474277365861 |
| Latent FM | 0.3710934321594932 | 0.35437878333588901 | 0.38962117648171268 |
| SiT | 0.0999081668476242 | 0.0969710697923373 | 0.1030436406476502 |

Source rows: `Dis_SI_Process/results/derived/20260910_1540/benchmark_main_a_summary.csv`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.


## Figure5b quantitative companion

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


## Figure5c quantitative companion

How do saved stochastic route/prior configurations compare in physical reconstruction?

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Metric and aggregation: Per-state physical field relative-L2 followed by equal-weight four-field macro and cohort mean; ideal0. Raw mean95%block20 intervals. Annotation100*(variant cohort mean/full cohort mean-1), not a mean of statewise ratios or percent-effect CI.

Result and boundary: Route-removal implementations have larger error. IID improves whole-field error. Local-only differs in optimized capacity and multiple routes, so this is not an isolated equal-capacity causal estimate.

Coordinates at source precision:

| display_label | policy | mean | block20_ci95_low | block20_ci95_high | annotation |
| --- | --- | --- | --- | --- | --- |
| Full model | last | 0.10632076677317751 | 0.10449524238782081 | 0.10822937860009681 | 0.1063 |
| No sensor feedback | last | 0.12697441583241301 | 0.12486830285874979 | 0.12924839657481199 | +19.4% |
| No local conditioning | last | 0.1447029438194137 | 0.14275715989634091 | 0.14666356018545801 | +36.1% |
| Local-only conditioning | last | 0.35495072396035698 | 0.35142188007896308 | 0.35869606554277261 | +233.8% |
| IID Gaussian prior | last | 0.1042938161915426 | 0.10241220413913921 | 0.10630154872937481 | -1.9% |

Source rows: `Dis_SI_Process/results/derived/20260910_1540/ablation_primary_summary.csv`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.


## Figure5d quantitative companion

Does uncertainty support selecting states with lower relative reconstruction error?

Package B: archived benchmark checkpoint identities; a/b/d use200 paired states and64 draws with moving-block25/2,000 intervals; a/b bootstrap seed20260830, d method-specific recorded bootstrap_seed in benchmark_main_d.csv. f uses original1,000-state accuracy and separately measured resource protocols. These results were not remeasured for the new ablation reference.

Metric and aggregation: Retain least-uncertain fraction q=0.2,...,1.0, divide retained-set mean error by same method full-cohort mean; endpoint1. Rank is recomputed in each accepted resample. AURC is integral over0.2–1.0, not divided by0.8.

Result and boundary: DMF retained-set relative error is lower; the metric is within-method normalized and must be read with absolute accuracy. No new metric fills the wider layout.

Coordinates at source precision:

| method | coverage_fraction | risk | ci_low | ci_high | risk_auc |
| --- | --- | --- | --- | --- | --- |
| DMF-Gen | 0.20000000000000001 | 0.87757090598023568 | 0.84535298910860079 | 0.90191838277810121 | 0.74104139459652396 |
| DMF-Gen | 0.29999999999999999 | 0.87943634455076281 | 0.85726019544573417 | 0.90684543755504921 | 0.74104139459652396 |
| DMF-Gen | 0.40000000000000002 | 0.89626753208343657 | 0.87652068163141772 | 0.92060466983381983 | 0.74104139459652396 |
| DMF-Gen | 0.5 | 0.90725710421013417 | 0.89009156584251636 | 0.9270762065436432 | 0.74104139459652396 |
| DMF-Gen | 0.59999999999999998 | 0.92181683535650139 | 0.90874966652032196 | 0.93887573736388041 | 0.74104139459652396 |
| DMF-Gen | 0.69999999999999996 | 0.94302779860388919 | 0.92939077753556465 | 0.95685056148176084 | 0.74104139459652396 |
| DMF-Gen | 0.80000000000000004 | 0.95391194830325221 | 0.94536804524463525 | 0.96753054243494885 | 0.74104139459652396 |
| DMF-Gen | 0.90000000000000002 | 0.96991092986714644 | 0.96330457010847526 | 0.97806182526518637 | 0.74104139459652396 |
| DMF-Gen | 1 | 1 | 1 | 1 | 0.74104139459652396 |
| FFM-FNO | 0.20000000000000001 | 0.98981987607952282 | 0.9646738617778976 | 1.0085715085733069 | 0.7946414715826795 |
| FFM-FNO | 0.29999999999999999 | 0.99059195671566203 | 0.97293092728540598 | 1.0082590243585496 | 0.7946414715826795 |
| FFM-FNO | 0.40000000000000002 | 0.99338577327567235 | 0.97794205257793998 | 1.00886036529937 | 0.7946414715826795 |
| FFM-FNO | 0.5 | 0.99208608329904957 | 0.97942512729155284 | 1.0073113975518111 | 0.7946414715826795 |
| FFM-FNO | 0.59999999999999998 | 0.99119304407302722 | 0.97854274102605943 | 1.0023319828542476 | 0.7946414715826795 |
| FFM-FNO | 0.69999999999999996 | 0.99288281235703557 | 0.98645585747085363 | 0.99849196574236976 | 0.7946414715826795 |
| FFM-FNO | 0.80000000000000004 | 0.9945911753542388 | 0.98965435348289155 | 0.99822129435204243 | 0.7946414715826795 |
| FFM-FNO | 0.90000000000000002 | 0.99677393271234815 | 0.99411245408534477 | 0.99975124474487365 | 0.7946414715826795 |
| FFM-FNO | 1 | 1 | 1 | 1 | 0.7946414715826795 |
| FFM-Perceiver | 0.20000000000000001 | 0.96702960354563761 | 0.95360058705813922 | 0.99002236526105103 | 0.79153763995209825 |
| FFM-Perceiver | 0.29999999999999999 | 0.97485085847032482 | 0.95917764308249998 | 0.99592583198330242 | 0.79153763995209825 |
| FFM-Perceiver | 0.40000000000000002 | 0.98543920554203179 | 0.97010873292193922 | 0.99853817337013762 | 0.79153763995209825 |
| FFM-Perceiver | 0.5 | 0.98791692372685525 | 0.97692925736202396 | 1.0006052348545329 | 0.79153763995209825 |
| FFM-Perceiver | 0.59999999999999998 | 0.99056000564097435 | 0.98272791938744963 | 0.99945597271341957 | 0.79153763995209825 |
| FFM-Perceiver | 0.69999999999999996 | 0.9949166846130536 | 0.98868867414941519 | 1.0009784003392554 | 0.79153763995209825 |
| FFM-Perceiver | 0.80000000000000004 | 0.99884136643613464 | 0.99379676376490755 | 1.0029196796029849 | 0.79153763995209825 |
| FFM-Perceiver | 0.90000000000000002 | 0.99933655331878957 | 0.99662965125167036 | 1.0013900160184692 | 0.79153763995209825 |
| FFM-Perceiver | 1 | 1 | 1 | 1 | 0.79153763995209825 |
| Latent FM | 0.20000000000000001 | 0.99801186181099399 | 0.9831163623508008 | 1.0189212401349943 | 0.80140378710766413 |
| Latent FM | 0.29999999999999999 | 0.9965934587837032 | 0.98502556693584165 | 1.012105965807699 | 0.80140378710766413 |
| Latent FM | 0.40000000000000002 | 1.0030264903512831 | 0.99130352997775284 | 1.0158737663022452 | 0.80140378710766413 |
| Latent FM | 0.5 | 1.0040600054706046 | 0.9939247127201768 | 1.0162493372640726 | 0.80140378710766413 |
| Latent FM | 0.59999999999999998 | 1.0027226607597004 | 0.99603344299154717 | 1.0119480746077167 | 0.80140378710766413 |
| Latent FM | 0.69999999999999996 | 1.0020547964770536 | 0.99661124026269321 | 1.0094685614138428 | 0.80140378710766413 |
| Latent FM | 0.80000000000000004 | 1.0036871229284623 | 0.99847935626648365 | 1.0092045572677408 | 0.80140378710766413 |
| Latent FM | 0.90000000000000002 | 1.0028874054003372 | 0.99908945493726964 | 1.0074211459564919 | 0.80140378710766413 |
| Latent FM | 1 | 1 | 1 | 1 | 0.80140378710766413 |
| SiT | 0.20000000000000001 | 0.94849724378258315 | 0.9149178738872128 | 0.9827171474601456 | 0.78096790066959643 |
| SiT | 0.29999999999999999 | 0.95551792558381476 | 0.93182481068738443 | 0.98623170226064816 | 0.78096790066959643 |
| SiT | 0.40000000000000002 | 0.96536716213158824 | 0.94303189110939878 | 0.99297386926961484 | 0.78096790066959643 |
| SiT | 0.5 | 0.97369410355619956 | 0.9603076595402632 | 0.99044957347828555 | 0.78096790066959643 |
| SiT | 0.59999999999999998 | 0.97977182935272078 | 0.96699488138946876 | 0.98790574485234639 | 0.78096790066959643 |
| SiT | 0.69999999999999996 | 0.97590236991064283 | 0.97068200176775599 | 0.98921829702302999 | 0.78096790066959643 |
| SiT | 0.80000000000000004 | 0.99082281978389564 | 0.9795790871465242 | 0.99694765240974603 | 0.78096790066959643 |
| SiT | 0.90000000000000002 | 0.9943541744858112 | 0.99052148704565679 | 0.99983391097833663 | 0.78096790066959643 |
| SiT | 1 | 1 | 1 | 1 | 0.78096790066959643 |

Source rows: `Dis_SI_Process/results/derived/20260910_1540/benchmark_main_d.csv`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.


## Figure5e quantitative companion

What does IID’s small global-error improvement conceal in fine-scale velocity?

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Metric and aggregation: Two distinct per-state high-band quantities: canonical trapezoidal shell-mean power ratio (ideal1) and complex-mode residual relative-L2 (ideal0, phase-sensitive). Mean block20 intervals; full/IID only, last policy.

Result and boundary: IID has excess high-band velocity power and roughly twice the high-band residual. Full power is attenuated and is not ideal. This is an index-space, one-draw, two-step diagnostic.

Coordinates at source precision:

| display_label | metric | mean | block20_ci95_low | block20_ci95_high | annotation |
| --- | --- | --- | --- | --- | --- |
| Full model | canonical_shellmean_high_energy_ratio | 0.42853620860075847 | 0.41199518450000661 | 0.44606586716251873 | 0.43 |
| IID Gaussian prior | canonical_shellmean_high_energy_ratio | 3.1199122893463591 | 2.9472808122108289 | 3.321242409458756 | 3.12 |
| Full model | highband_error_relative_l2 | 0.87340655840460224 | 0.8621901011860641 | 0.88485340755612574 | 0.87 |
| IID Gaussian prior | highband_error_relative_l2 | 1.8252839107455296 | 1.7793718541521102 | 1.877885185559828 | 1.83 |

Source rows: `Dis_SI_Process/results/derived/20260910_1540/main_ablation_plot_coordinates.csv`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.


## Figure5f quantitative companion

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


## SI-01: Complete reconstruction and conditioning-route effects

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Physical relative-$L_2$ and fieldwise effects under both policies. Temperature is observed. Effects use log$_2$ ratios of cohort means with a nonsaturating color range; raw means, intervals, paired effects, pressure-normalization checks and the additional local-only versus no-feedback contrast are supplied in tables and standalone diagnostics.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si01a_fieldwise_physical_l2

```json
{
  "id": "si01a_fieldwise_physical_l2",
  "role": "fieldwise primary reconstruction error under both checkpoint policies",
  "metric": "physical_relative_l2",
  "normalization": "physical field norm per state; fieldwise summary",
  "statistics": "mean with 95% circular moving-block interval, block length 20, 2,000 replicates, n=1,000 states",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "visual_encoding": "method color/marker from shared YAML; horizontal point with block-20 interval; observed T labelled",
  "axes": {
    "x": "Physical relative L2",
    "xlim": [
      0.0,
      0.86
    ],
    "policy_facets": [
      "last.pt",
      "best.pt"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si01a_fieldwise_physical_l2](si_companions/si01a_fieldwise_physical_l2_companion.md).

## si01b_fieldwise_effect_heatmap

```json
{
  "id": "si01b_fieldwise_effect_heatmap",
  "role": "fieldwise effect relative to the full ablation reference",
  "metric": "physical_relative_l2",
  "definition": "log2(mean variant / mean A0) computed from fieldwise cohort means; not a mean of statewise ratios",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "visual_encoding": "diverging RdBu_r heatmap; zero is the full-model reference; T is identified as observed",
  "axes": {
    "color_limits": [
      -3.0,
      3.0
    ],
    "policies": [
      "last",
      "best"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si01b_fieldwise_effect_heatmap](si_companions/si01b_fieldwise_effect_heatmap_companion.md).

## si01c_normalization_checks

```json
{
  "id": "si01c_normalization_checks",
  "role": "fieldwise standardized and truth-fluctuation normalizations retained alongside physical L2",
  "metrics": [
    "normalized_relative_l2",
    "truth_fluctuation_normalized_l2"
  ],
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "visual_encoding": "method color/marker; fieldwise horizontal point and block-20 interval; observed T labelled",
  "axes": {
    "x": "fieldwise relative L2",
    "xlim_standardized": [
      0,
      1.0
    ],
    "xlim_truth_fluctuation": [
      0,
      2.45
    ],
    "policy_facets": [
      "last.pt",
      "best.pt"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si01c_normalization_checks](si_companions/si01c_normalization_checks_companion.md).

## si01d_local_conditioning_ch4_detail

```json
{
  "id": "si01d_local_conditioning_ch4_detail",
  "role": "reported local-conditioning sensitivity detail",
  "metric": "physical_relative_l2",
  "source_keys": [
    "last_summary_metrics"
  ],
  "derived_comparison": {
    "definition": "100*(A3/A0-1) from cohort means",
    "percent": 140.55070970045347,
    "formatted": "+140.6%"
  },
  "visual_encoding": "horizontal point and block-20 interval",
  "axes": {
    "x": "CH4 physical relative L2",
    "xlim": [
      0.035,
      0.16
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si01d_local_conditioning_ch4_detail](si_companions/si01d_local_conditioning_ch4_detail_companion.md).

## si01

```json
{
  "id": "si01",
  "role": "SI entry-point composite si01",
  "linked_subpanels": [
    "si01a_fieldwise_physical_l2",
    "si01b_fieldwise_effect_heatmap"
  ],
  "layout": {
    "rows": 2,
    "columns": 1,
    "height_ratios_in": [
      2.85,
      2.75
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.


## SI-02: All-field scale-resolved source-prior comparison

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Full RFF versus IID phase-sensitive high-band residuals and canonical shell-integrated high-band power in all five fields, separately for last and best. Intervals are mean block-20 confidence intervals; power and residual ideal values are one and zero. All stochastic variants remain in the tables.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si02a_all_field_highband_l2

```json
{
  "id": "si02a_all_field_highband_l2",
  "role": "all-field phase-sensitive high-band residual",
  "metric": "highband_error_relative_l2",
  "source_keys": [
    "high_frequency_summary"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000 states per method/policy",
  "visual_encoding": "RFF hollow red circles; IID filled amber diamonds; policy facets",
  "axes": {
    "x": "High-band relative L2",
    "ideal": 0,
    "xlim": [
      0,
      2.25
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si02a_all_field_highband_l2](si_companions/si02a_all_field_highband_l2_companion.md).

## si02b_all_field_canonical_power

```json
{
  "id": "si02b_all_field_canonical_power",
  "role": "canonical shell-mean high-band power ratio",
  "metric": "canonical_shellmean_high_energy_ratio",
  "definition": "per-state shell-mean spectrum integrated with existing trapezoidal weighting over strict retained high band; not mode-sum",
  "source_keys": [
    "high_frequency_summary"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000 states per method/policy",
  "visual_encoding": "RFF hollow red circles; IID filled amber diamonds; truth=1 dashed guide",
  "axes": {
    "x": "High-band power / truth",
    "ideal": 1,
    "xlim": [
      0,
      3.8
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si02b_all_field_canonical_power](si_companions/si02b_all_field_canonical_power_companion.md).

## si02

```json
{
  "id": "si02",
  "role": "SI entry-point composite si02",
  "linked_subpanels": [
    "si02a_all_field_highband_l2",
    "si02b_all_field_canonical_power"
  ],
  "layout": {
    "rows": 2,
    "columns": 1,
    "height_ratios_in": [
      2.9,
      2.9
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.


## SI-03: Velocity spectra, prevalence and leakage sensitivity

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Median shell-power ratios with IQR state-dispersion shading; empirical distributions of mode-sum high-band power and phase-sensitive residuals; separate Hann-window sensitivity. Both policies are shown. These index-space spectra do not identify a physical-wavenumber inertial range. Canonical, mode-sum and tapered quantities are distinct estimators.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si03a_u1_population_spectra

```json
{
  "id": "si03a_u1_population_spectra",
  "role": "population shell-power ratio distribution across retained index-space",
  "metric": "population_U1_spectra median/IQR",
  "source_keys": [
    "population_U1_spectra"
  ],
  "statistics": "per-state median and IQR; shading is state dispersion, not a confidence band; n=1,000 per shell/method/policy",
  "visual_encoding": "RFF solid red; IID dashed amber; IQR alpha=0.15; truth=1; strict high band shaded",
  "axes": {
    "x": "wavenumber/kmax",
    "xlim": [
      0,
      1
    ],
    "high_band": ">2/3",
    "y_ideal": 1
  }
}
```

Individual panel companion with every plotted coordinate: [si03a_u1_population_spectra](si_companions/si03a_u1_population_spectra_companion.md).

## si03b_u1_mode_sum_ecdf

```json
{
  "id": "si03b_u1_mode_sum_ecdf",
  "role": "empirical distributions of mode-sum power and high-band residual",
  "metrics": [
    "reconstruction_to_truth_high_energy_ratio",
    "highband_error_relative_l2"
  ],
  "source_keys": [
    "high_frequency_per_state"
  ],
  "statistics": "ECDF over 1,000 states; this mode-sum power is separate from canonical trapezoidal ratio",
  "visual_encoding": "RFF solid red; IID dashed amber; dotted power guide at truth=1",
  "axes": {
    "power_ideal": 1,
    "residual_ideal": 0,
    "y": "ECDF"
  }
}
```

Individual panel companion with every plotted coordinate: [si03b_u1_mode_sum_ecdf](si_companions/si03b_u1_mode_sum_ecdf_companion.md).

## si03c_u1_hann_sensitivity

```json
{
  "id": "si03c_u1_hann_sensitivity",
  "role": "Hann-window sensitivity kept separate from primary untapered estimators",
  "metrics": [
    "hann_reconstruction_to_truth_high_energy_ratio",
    "hann_highband_error_relative_l2"
  ],
  "source_keys": [
    "U1_hann_robustness_summary",
    "high_frequency_summary"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000",
  "prevalence_mode_sum_primary": {
    "last_A0": {
      "fraction_gt_1": 0.042,
      "fraction_gt_2": 0.0
    },
    "last_A4": {
      "fraction_gt_1": 0.983,
      "fraction_gt_2": 0.773
    },
    "best_A0": {
      "fraction_gt_1": 0.037,
      "fraction_gt_2": 0.0
    },
    "best_A4": {
      "fraction_gt_1": 0.988,
      "fraction_gt_2": 0.803
    }
  },
  "visual_encoding": "RFF hollow red; IID filled amber; policy facets",
  "axes": {
    "x": "Hann diagnostic",
    "primary_untapered": false
  }
}
```

Individual panel companion with every plotted coordinate: [si03c_u1_hann_sensitivity](si_companions/si03c_u1_hann_sensitivity_companion.md).

## si03

```json
{
  "id": "si03",
  "role": "SI entry-point composite si03",
  "linked_subpanels": [
    "si03a_u1_population_spectra",
    "si03b_u1_mode_sum_ecdf",
    "si03c_u1_hann_sensitivity"
  ],
  "layout": {
    "rows": 3,
    "columns": 1,
    "height_ratios_in": [
      2.65,
      3.0,
      2.65
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.


## SI-04: Coupled distributions and histogram-range sensitivity

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Fixed-range paper JSD, overflow-retaining JSD, and predicted/truth retention for three coupled field pairs under both policies. All stochastic configurations use the same archived edges. Low conditional JSD does not establish fidelity of discarded mass or spatial alignment.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si04_joint_pdf_jsd_base2

```json
{
  "id": "si04_joint_pdf_jsd_base2",
  "role": "Paper-range JSD (base 2)",
  "metric": "joint_pdf_jsd_base2",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000; common paper edges",
  "visual_encoding": "method color/marker; policy facets; JSD base 2",
  "axes": {
    "x": "Paper-range JSD (base 2)",
    "pair_targets": [
      "T-U1",
      "CH4-U1",
      "p-U1"
    ],
    "histogram_definition": "64x64 original edges"
  }
}
```

Individual panel companion with every plotted coordinate: [si04_joint_pdf_jsd_base2](si_companions/si04_joint_pdf_jsd_base2_companion.md).

## si04_joint_pdf_jsd_with_overflow_base2

```json
{
  "id": "si04_joint_pdf_jsd_with_overflow_base2",
  "role": "Overflow-retaining JSD (base 2)",
  "metric": "joint_pdf_jsd_with_overflow_base2",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000; common paper edges",
  "visual_encoding": "method color/marker; policy facets; JSD base 2",
  "axes": {
    "x": "Overflow-retaining JSD (base 2)",
    "pair_targets": [
      "T-U1",
      "CH4-U1",
      "p-U1"
    ],
    "histogram_definition": "66x66 overflow-retaining extension"
  }
}
```

Individual panel companion with every plotted coordinate: [si04_joint_pdf_jsd_with_overflow_base2](si_companions/si04_joint_pdf_jsd_with_overflow_base2_companion.md).

## si04c_coupling_retention

```json
{
  "id": "si04c_coupling_retention",
  "role": "retained-pair context for truncated and overflow JSD",
  "metrics": [
    "joint_pdf_reconstruction_retained_fraction",
    "joint_pdf_truth_retained_fraction"
  ],
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000; retention is not calibration",
  "visual_encoding": "predicted points offset below pair row; truth ticks above; method color; 0.99 guide",
  "axes": {
    "x": "retention fraction",
    "xlim": [
      0.42,
      1.01
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si04c_coupling_retention](si_companions/si04c_coupling_retention_companion.md).

## si04

```json
{
  "id": "si04",
  "role": "SI entry-point composite si04",
  "linked_subpanels": [
    "si04_joint_pdf_jsd_base2",
    "si04_joint_pdf_jsd_with_overflow_base2",
    "si04c_coupling_retention"
  ],
  "layout": {
    "rows": 3,
    "columns": 1,
    "height_ratios_in": [
      2.65,
      2.65,
      2.65
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.


## SI-05: Checkpoint sensitivity and recorded training histories

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Paired best-minus-last sensitivity and temporal block-length diagnostics alongside available raw training/validation loss histories. Longer histories are retained at their actual recorded endpoints with selected checkpoint markers. The comparison is not budget matched; validation and test share a holdout. Direct-field and flow-velocity training objectives are not pooled.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si05a_checkpoint_sensitivity

```json
{
  "id": "si05a_checkpoint_sensitivity",
  "role": "best-minus-last checkpoint-selection sensitivity",
  "source_keys": [
    "checkpoint_sensitivity"
  ],
  "statistics": "paired block-20 intervals from saved policy comparison; policy values are not pooled",
  "visual_encoding": "horizontal signed point/interval; zero guide; fixed stochastic order",
  "axes": {
    "x": "best-minus-last",
    "metrics": [
      "physical_relative_l2",
      "joint_pdf_jsd_with_overflow_base2"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si05a_checkpoint_sensitivity](si_companions/si05a_checkpoint_sensitivity_companion.md).

## si05b_block_length_sensitivity

```json
{
  "id": "si05b_block_length_sensitivity",
  "role": "paired best-minus-last sensitivity across circular block lengths",
  "source_keys": [
    "checkpoint_sensitivity"
  ],
  "statistics": "saved paired best-minus-last mean with source block-5/20/50 intervals; n=1,000",
  "visual_encoding": "method-colored horizontal line across block lengths with block-specific vertical intervals and zero guide",
  "axes": {
    "x": "block length in sorted held-out states",
    "blocks": [
      5,
      20,
      50
    ],
    "metrics": [
      "physical_relative_l2",
      "joint_pdf_jsd_with_overflow_base2"
    ],
    "policy_comparison": "best-minus-last"
  }
}
```

Individual panel companion with every plotted coordinate: [si05b_block_length_sensitivity](si_companions/si05b_block_length_sensitivity_companion.md).

## si05c_recorded_training_histories

```json
{
  "id": "si05c_recorded_training_histories",
  "role": "actual recorded train/validation loss histories",
  "source_keys": [
    "history_A0",
    "history_A2",
    "history_A3",
    "history_A5",
    "history_A4"
  ],
  "statistics": "raw recorded history points; no smoothing, normalization, truncation or endpoint reconstruction",
  "visual_encoding": "method color; train/validation separate axes; selected checkpoint markers: circle=last, diamond=best",
  "axes": {
    "x": "epoch",
    "y": "loss (log scale)",
    "missing_methods": []
  }
}
```

Individual panel companion with every plotted coordinate: [si05c_recorded_training_histories](si_companions/si05c_recorded_training_histories_companion.md).

## si05

```json
{
  "id": "si05",
  "role": "SI entry-point composite si05",
  "linked_subpanels": [
    "si05a_checkpoint_sensitivity",
    "si05b_block_length_sensitivity",
    "si05c_recorded_training_histories"
  ],
  "layout": {
    "rows": 3,
    "columns": 1,
    "height_ratios_in": [
      2.65,
      2.65,
      2.65
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.


## SI-06: Deterministic objective control

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Separate deterministic objective control against the full stochastic reference: macro/fieldwise physical error and coupling diagnostics under both policies. Pressure fluctuation error and complete retention/overflow companions are tabulated. Lower deterministic point error does not quantify diversity or calibration.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si06a_deterministic_macro_control

```json
{
  "id": "si06a_deterministic_macro_control",
  "role": "separate deterministic objective control",
  "metric": "physical_relative_l2",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "derived_comparison": {
    "last_deterministic_minus_full_percent": -25.974040046966994,
    "best_deterministic_minus_full_percent": -27.186320349262317
  },
  "visual_encoding": "horizontal point/interval; A0 hollow red, A1 filled slate",
  "axes": {
    "x": "unobserved physical relative L2"
  }
}
```

Individual panel companion with every plotted coordinate: [si06a_deterministic_macro_control](si_companions/si06a_deterministic_macro_control_companion.md).

## si06b_deterministic_fieldwise_control

```json
{
  "id": "si06b_deterministic_fieldwise_control",
  "role": "fieldwise deterministic-versus-full control",
  "metric": "physical_relative_l2",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "visual_encoding": "A0 hollow red; deterministic A1 filled slate; policy facets",
  "axes": {
    "x": "physical relative L2"
  }
}
```

Individual panel companion with every plotted coordinate: [si06b_deterministic_fieldwise_control](si_companions/si06b_deterministic_fieldwise_control_companion.md).

## si06c_deterministic_coupling_control

```json
{
  "id": "si06c_deterministic_coupling_control",
  "role": "deterministic coupling diagnostics with range companions",
  "metrics": [
    "joint_pdf_jsd_base2",
    "joint_pdf_jsd_with_overflow_base2"
  ],
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "statistics": "mean with 95% block-20 interval; common paper edges and 66x66 overflow edges",
  "visual_encoding": "circle/policy points for paper JSD; x markers for overflow JSD; A0 red/A1 slate",
  "axes": {
    "x": "JSD base 2",
    "pair_targets": [
      "T-U1",
      "CH4-U1",
      "p-U1"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si06c_deterministic_coupling_control](si_companions/si06c_deterministic_coupling_control_companion.md).

## si06

```json
{
  "id": "si06",
  "role": "SI entry-point composite si06",
  "linked_subpanels": [
    "si06a_deterministic_macro_control",
    "si06b_deterministic_fieldwise_control",
    "si06c_deterministic_coupling_control"
  ],
  "layout": {
    "rows": 3,
    "columns": 1,
    "height_ratios_in": [
      2.65,
      2.65,
      2.65
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.

## Admissibility event ledger

| display_label | policy | target | snapshot | time_index | event_count | minimum |
| --- | --- | --- | --- | --- | --- | --- |
| Local-only conditioning | last | T | 129 | 1258 | 1 | -4.2698974609375 |
| IID Gaussian prior | last | T | 291 | 2769 | 1 | -1648.564697265625 |
| IID Gaussian prior | last | T | 753 | 7604 | 1 | -1397.456298828125 |
| IID Gaussian prior | last | T | 935 | 9436 | 1 | -966.496826171875 |
| IID Gaussian prior | best | T | 291 | 2769 | 1 | -1764.966796875 |
| IID Gaussian prior | best | T | 753 | 7604 | 1 | -1429.140625 |
| IID Gaussian prior | best | T | 935 | 9436 | 1 | -970.839599609375 |

Denominator40,300,000 points per configuration/policy. Tables retain species negative frequencies, below−0.0001 frequencies, negative-part norms/magnitudes, exact T/p counts, global minima and their state/time identities. Rounded zero percentages are not evidence of zero events.

## All-field paired fine-scale residual sensitivity

IID minus full; all ten saved comparisons retain positive lower bounds at every block length.

| policy | field | method_mean_minus_baseline | block5_ci95_low | block5_ci95_high | block20_ci95_low | block20_ci95_high | block50_ci95_low | block50_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| last | CH4 | 0.1326745840657946 | 0.1282812706512988 | 0.13741093018358069 | 0.1277697442437904 | 0.1380869230373428 | 0.1273263335580575 | 0.13839497975961751 |
| last | CO | 0.28194059020732171 | 0.27168790525923092 | 0.2929090056202282 | 0.27010450343056691 | 0.29487155967509199 | 0.26912724749156169 | 0.29464777892492039 |
| last | T | 0.2568557021840116 | 0.24769952668887071 | 0.26694162719130338 | 0.2462752015336965 | 0.2680638913143929 | 0.24530121387515449 | 0.26887188463298128 |
| last | U1 | 0.95187735234092763 | 0.9160281964959498 | 0.99067201353066381 | 0.90973477630772437 | 1.0013135005359033 | 0.90574131815080239 | 1.0007649266496601 |
| last | p | 0.065486120914424295 | 0.059271739476034098 | 0.073777147619730504 | 0.058383387124711202 | 0.074321773435861896 | 0.057467896027642601 | 0.076836002584367097 |
| best | CH4 | 0.1296187597185269 | 0.12541924485858341 | 0.13426097753779381 | 0.1249637080663102 | 0.13454501305854949 | 0.1245555549853864 | 0.13506781317689279 |
| best | CO | 0.27861398174786112 | 0.26864134008128021 | 0.2894466427404791 | 0.26744713522760238 | 0.29061812523574809 | 0.2665321808834536 | 0.29102262789426048 |
| best | T | 0.25337175262714662 | 0.2446537427702675 | 0.26293948359232272 | 0.2432522030844004 | 0.26407663174125318 | 0.24281519908604349 | 0.26457612010118042 |
| best | U1 | 0.98311985057969598 | 0.94603725697474039 | 1.0208463077371659 | 0.94213656030471438 | 1.0307745430639073 | 0.93759253613440463 | 1.0298412201395442 |
| best | p | 0.061610521345861598 | 0.055713690364293002 | 0.069653685416150704 | 0.0546850590346179 | 0.070269847991882697 | 0.053931124135257399 | 0.0726242188398602 |

## Scope and measurement boundaries

Paragraph {Saved-checkpoint evaluation and scope.}
We retain two separate evidence packages. The archived benchmark provides the 200-state, 64-draw uncertainty results and the original 1,000-state accuracy/resource scorecard. Architectural comparisons use six saved implementations on 1,000 paired held-out snapshots, with five stochastic configurations and a separately reported deterministic objective. Every state has 256 temperature observations and 40,300 queries. The stochastic sampler uses one draw and two Euler steps with a hard observed-entry clamp, live weights and no EMA substitution. The full reference is frozen at epoch 7520 (last) and 7095 (best). The validation and test loaders share the same holdout; these are temporally related snapshots from one simulation. Endpoints and parameter elements with optimizer state differ across configurations; these element counts do not denote FLOPs or measured memory. Equal seed strings do not verify matched backbone initialization. Best-versus-last sensitivity does not make training budgets equal.
Paragraph {Reconstruction and uncertainty.}
For state $i$ and field $c$, $E_{ic}=\|\widehat u_{ic}-u_{ic}\|_2/(\|u_{ic}\|_2+10^{-12})$ in physical units. The primary macro error is $E_i=\frac14\sum_{c\in\{\mathrm{CH}_4,\mathrm{CO},U_1,p\}} E_{ic}$; temperature is excluded. States are equally weighted. Standardized-field and truth-fluctuation-normalized errors use distinct denominators and remain separate. The latter uses $\|u_{ic}-\overline u_{ic}\|_2$. The clamp imposes sensor consistency; sensor-excluded temperature error evaluates the remaining points. Percent effects are $100(\overline E_{\rm variant}/\overline E_{\rm full}-1)$, a ratio of cohort means. Paired effects resample differences after pairing exact snapshot and original time identities. Accepted intervals use 2,000 circular moving-block resamples of temporally sorted held-out states, block 20, with paired sensitivities at 5 and 50. Reconstruction uses recorded seed 20260906; high-frequency summaries use 20260910. Blocks count sorted held-out states, not consecutive simulation steps. These are conditional mean intervals, not training-seed variability or state dispersion. No new resampling was necessary for the plotted means.
Paragraph {Scale-resolved metrics.}
Fields on the $403\times100$ grid are centered spatially. The retained index-space radial shells exclude DC, shells with fewer than four modes, and modes beyond the isotropic cutoff. The strict high band is $k>(2/3)k_{\max}$, containing 68 shells and 17,704 Fourier modes. Canonical power integrates shell-mean spectra with the existing trapezoidal weights. The mode-sum ratio instead uses $R_H=\sum_{q\in H}|P_q|^2/\sum_{q\in H}|T_q|^2$. Phase-sensitive high-band error is $E_H=(\sum_{q\in H}|P_q-T_q|^2/\sum_{q\in H}|T_q|^2)^{1/2}$. Ratios are calculated per state before population averaging. Power has ideal one and attenuation below one; residual error has ideal zero. Whole-spectrum LSD is the root mean square of $10\log_{10}[(E_P(k)+\epsilon)/(E_T(k)+\epsilon)]$ over retained shells, with the original truth-relative numerical floor. Population spectra display median per-state shell ratios and IQR, which are state dispersion. Integrating that median curve does not recover a mean integrated power ratio. Full truth fluctuation budgets include all centered Fourier modes, including modes outside retained radial shells. Hann-window diagnostics apply the recorded separable taper and mean-square correction with unchanged band membership; they remain a separate estimator. No fields or spectra were smoothed or clipped.
Paragraph {Coupling and admissibility.}
Paper JSD is base-2 Jensen--Shannon divergence, not its square root, of fixed $64\times64$ joint histograms for T--$U_1$, CH$_4$--$U_1$ and p--$U_1$. Common archived truth-derived 0.5--99.5\% edges and the original $10^{-12}$ probability pseudocount are reused across all methods/policies. Paper JSD normalizes retained mass; the overflow extension uses $66\times66$ bins including all finite underflow/overflow mass. Predicted and truth retention accompany every coupling interpretation. Pearson correlation error is a coarse scalar complement. These distributions do not verify spatial alignment. Admissibility uses CH$_4$,CO $<0$ and T,p $\leq0$ before clipping, with exact counts out of 40,300,000 points per model/policy, minima and affected time IDs. None of these diagnostics establishes PDE satisfaction, conservation or thermodynamic closure.

The saved stochastic configurations distinguish conditioning-route effects from the source-prior tradeoff (Fig.~\ref{fig:uq_cost}c,e). The new full reference achieved mean unobserved-field relative-$L_2$ of 0.1063. Removing global-to-sensor feedback or local query conditioning increased this error to 0.1270 and 0.1447, respectively; local-only conditioning reached 0.3550. The IID-prior run reduced whole-field error to 0.1043, while high-band velocity power reached 3.12 times truth and high-band relative-$L_2$ rose to 1.83, compared with 0.43 and 0.87 for the full RFF model. The full model attenuated high-band power, whereas IID combined a small bulk-error improvement with larger fine-scale residuals. The high-band residual increase occurred in all five fields under both checkpoint policies (Supplementary Fig.~\ref{fig:si-ablation-highband}). These comparisons describe available trained configurations with unequal endpoints and effective optimized capacity; the separate deterministic objective control is reported in the Supplementary Information.

\paragraph{Distributional and objective controls.}
Full-model CH$_4$--$U_1$ predicted retention is 65.447\% last and 56.359\% best, versus 99.000\% truth; overflow JSD is 0.201011 and 0.320323. Removing local conditioning yields lower CH$_4$--$U_1$ overflow JSD despite larger spatial error. IID coupling comparisons depend on policy and histogram definition. Deterministic regression achieves macro errors 0.078705 and 0.078561, versus full 0.106321 and 0.107893, advantages of 26.0\% and 27.2\%. Full has lower pressure error under both policies. Full last has lower truncated JSD for all three pairs, while best-policy and overflow comparisons alter parts of that result. This is deterministic forward prediction versus one sampled reconstruction, not an ensemble-mean test or evidence about conditional diversity.
\paragraph{Fine-scale and admissibility limits.}
The mean truth high-band velocity energy fraction is 0.0459\% of full centered fluctuation energy. IID high-band residual-budget fractions are 0.1317\% and 0.1363\%, versus full 0.0312\% and 0.0313\%. These explain a weak-band/global-error separation without forming an exact decomposition of primary physical relative-$L_2$. IID mode-sum power exceeds truth in 98.3\% / 98.8\% of states. Hann tapering retains IID excess and stronger RFF attenuation. The exact admissibility table preserves three nonpositive-temperature points for IID under each policy and one for local-only last, despite percentages rounding to zero. These results remain conditional on the saved checkpoints, sensor plan and two-step protocol. IID mode-sum high-band velocity power exceeds twice truth in 77.3\% / 80.3\% of states (last/best). All ten IID-minus-full high-band residual comparisons (five fields, two policies) have positive mean differences and positive interval lower bounds for circular blocks 5, 20 and 50. These are paired checkpoint-conditional contrasts, not training-seed evidence.

## Validation status

Source QA: pass; 262 source checks. Final package QA at document generation: pass. The live final status is results/derived/20260910_1540/qa.json. SVG visual review is bound to output hashes; source integrity does not establish equal-budget causal identification. The optional cached-field gallery is not needed for these metric figures.

## Source ledger

| role | path | sha256 |
| --- | --- | --- |
| protected accepted release | Dis_SI_Process/figures/generated/20260904_1200/fig5d_accuracy_computational_footprint_20260904_1200.svg | fcfa4982f0e5811c327dee74083a86547efe4061a970476ae9263b0a63bff21f |
| protected accepted release | Dis_SI_Process/figures/generated/20260904_1200/fig5_composed_v6_20260904_1200.pdf | a9e3b29a00aa4d5c0cc0cc4e401ad06ce99c6847b4316b913458c656941fd62a |
| protected accepted release | Dis_SI_Process/figures/generated/20260904_1200/fig5_composed_v6_20260904_1200.svg | d552cc75772ed949d64dc0c569cbc921c3fac0874c1c2e9129cfd8cf20d54a50 |
| protected accepted release | Dis_SI_Process/figures/generated/20260904_1200/fig5b_uncertainty_tracks_difficult_states_20260904_1200.svg | 293b867c8d33218366530f68f8f6131b1e8154fae0df3f17415eff8b01be247c |
| protected accepted release | Dis_SI_Process/figures/generated/20260904_1200/fig5a_probabilistic_reconstruction_20260904_1200.svg | de4248f97c4041bbf468dbf1f1da64da476fc8d082baf8326d357137ca00bb55 |
| protected accepted release | Dis_SI_Process/figures/generated/20260904_1200/fig5c_selective_reconstruction_20260904_1200.svg | 0bc60a04464956b4230d8427c8d0ef70854ea48205636e1ff78788fd34ac7543 |
| protected accepted release | Dis_SI_Process/results/derived/20260904_1200/build_manifest.json | 07fbbeca451e72ce3bd57ef0e3b66fa35653cdd62cf9030a1f59211faee36011 |
| protected accepted release | Dis_SI_Process/results/derived/20260904_1200/qa.json | 2136b9d290b1e140eb0cc6a47a5ccbd7ea8c308b49901a6196d6978a6a65c286 |
| protected accepted release | Dis_SI_Process/docs/generated/20260904_1200/figure_contract.md | 4e5da308bd39f12f7b8d36477e68e286f80c61eeb90ee80a674cd67b2c2ba4ea |
| protected accepted release | Dis_SI_Process/docs/generated/20260904_1200/completion_report.md | 1aea2580ff57329ac470c171b22820c4890043822be8b9b42c5673a8bee3359b |
| saved evidence | Dis_SI_Process/configs/figure5_v6.yaml | 18a8cee9d6feed672c9c48c2c79fc7278a33decb6dadba7165116259ab303e50 |
| Package B accepted lineage | Dis_SI_Process/results/derived/20260831_1409/figure5_v5_source.csv | 8753eec426d6fbfc83ff5f3a469d05b8b2596dab114f9c28aaa192b973ce1615 |
| Package B accepted lineage | Dis_SI_Process/results/derived/20260831_1409/build_manifest.json | 71094d5d8867ed8fade32025fe937a325c608edc4e9747b9b22de159064bf693 |
| Package B accepted lineage | Dis_SI_Process/results/derived/20260831_1409/qa.json | 62f7bcb60a909a36c634cfbc34e11d4594ec75cbf26fdc7238b144bace85dcd6 |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/PanelC/panel_c_exploration_20260902_1129/selective_risk.csv | 6144766d3b9c89027971b310c8a1ba982c1af06ac534826bac4616cacb3b3679 |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/PanelC/panel_c_exploration_20260902_1129/manifest.json | 85339b5ea6e9a792162695172f25fdd5ee65a9a9bd871d2323237faa4897c542 |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/PanelC/panel_c_exploration_20260902_1129/qa.json | 056e602ff21d43d53287c3917b0abbdf30826558943792efd21f010b46d2280a |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_plot_source_common_b32.csv | 3c966c0f75a0b5c927267aceb77bb3625e006af086aaaac69b96df52984cd6e8 |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_stage_source_common_b32.csv | 79d3491f9268ba668a2a8a68535aade89e2df7e73f4e6d716cfae71f34407ac4 |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_manifest_common_b32.json | f38268a9ccc685cd8f403b9744a8aa5edad3781c73d809d8cdd611de1d1629bb |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_qa_common_b32.json | 5424de2e06cfe9c3fc8dc6ba7613442316a175bdd5bfab97aaf320b2eb4dd9cc |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/InferenceMemory/inference_memory_native_v51_20260903_094021/inference_memory_summary.csv | 60af88d21ca33e3585ba9e8b142de1f42f267169a64a56b018c5a2b84529c6b2 |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/InferenceMemory/inference_memory_native_v51_20260903_094021/manifest.json | 6d297961d4c2180c25019a76b16ebafd1c7f8f776341058e369a4e020361f5f4 |
| Package B accepted lineage | Dis_SI_Process/results/ValidationV51/InferenceMemory/inference_memory_native_v51_20260903_094021/qa.json | 7bc953318e4f6a58f0fe74e5865c0572f3270461f91933499eac79192bda0484 |
| Package B formal 200-state 64-draw UQ provenance | Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/manifest.json | 05c5daccf7238d7644b34842bcc5ed751a93bb5b08ace92cfd3f0fd0d433aaae |
| Package B formal 200-state 64-draw UQ provenance | Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/qa.json | a83b32c7e9b67bfdac00431b2738163d2cb14b726659eadd43012475a8ef3ba0 |
| Package B formal 200-state 64-draw UQ provenance | Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/method_draw_audit.csv | 549f69b599aaca8e937de30cae7ecccf373ee79a6d3009d35e9bdb8581391e6f |
| Package B formal 200-state 64-draw UQ provenance | Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/crps_summary.csv | 195f2cf43ce3176d241cac9c15bb780552e0fbeea209964b5ea918a2b05e9e86 |
| Package B formal 200-state 64-draw UQ provenance | Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/spread_error_summary.csv | 836e6228444ff7d35ab8f4cabef9a37369515eb87d266645fd4013ee79ce6a6e |
| Package B formal 200-state 64-draw UQ provenance | Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/per_state_method.csv | db308cc56025ec150b7a4c6f8ce410a917bef454a1fa03e0b6ae7d30927cd6e8 |
| protected inherited renderer | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format.py | f1527e590e5c48ad1976ff9e18154bf591b360948369735d171479c2a34c95fc |
| protected inherited renderer | Dis_SI_Process/figures/scripts/build_figure5_v6.py | 25f0f670c01a8267d590a2acc5423ab638ce56856cbde9de8182cba99a5a5afc |
| protected inherited renderer | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v4.py | 463c78e81e32686fbb2e8101bf78aab32ac5c3e101552564d5921c980001696d |
| protected inherited renderer | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v3.py | 4f73f67a6a93909293a16396f287813075587f4663f44c57240b0f3cdee457f2 |
| protected inherited renderer | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v5.py | 502e1af41f438bc2026b3681bdefcbb4ed7514caf87188db0390a075aa9bca3a |
| protected inherited renderer | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v2.py | 8dd43c7721bc98ea50120c86dd5f1ea1308ba4681307bb41d672828d279f09c9 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/Ablation_CondT_Evaluation_20260910.md | 84d5c9fc708988f51f95d4668ef84786822367570e19bd20f71c466e632a1138 |
| identical report alias | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/Ablation_CondT_Evaluation_20260906.md | 84d5c9fc708988f51f95d4668ef84786822367570e19bd20f71c466e632a1138 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/checkpoint_inputs/A0/snapshot_manifest.json | 376e662144ac6c6df4d7eac6862c8948a1f50d17c32bf05b9e48d3b2fc36304f |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/checkpoint_audit.json | b4719f1a00cb98d9ebc562f7360f79f73670e8f7bdee26cbce108f003dd2ef5f |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/evaluation_environment.json | 076f1423f7fea085e7e3a2ebb864337ef7041498605c72035a1a18be8cd62da6 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/validation_summary.json | 48d75d141a5f144673fe7fa1ac4c5bd7a78c01921b586a61c1c73fb191f6fe02 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/A0_training_progress.csv | 5378d4a5e7f89cac2314a068b78bdcd9ca2c626b8b90f8e5535093bb5d60cbed |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/checkpoint_sensitivity_vs_last.csv | e15e32a1bc89c96ecfc78d1c3fbe52241004e1c739b6df8ee3acf9c56da3d1d9 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/source_hashes.json | 6d2cf20c462b777928b6aaba8f17ee2be64e5fdf389a146d57e91571194e659a |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/coverage_audit.json | f0a74b10044caa952ba00f62f9f9e2991f3dfd5a26c3d317e37390cdb9b1de03 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/metrics/analysis_metadata.json | d399645e8f490569cd1de57c9dd78ab7bd0b67b89c8522a9641e58a5541a79cd |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/provenance_A0.json | f36b6dae247446b7d5054dfda35c5afad066636f8194ea4f3dc8d0b1dbc39983 |
| saved evidence | /data/wanglz/Cache/PhyCoFlow_TurbulentCombustion_Process_Results/SensorPlans/SensorPlan_paper_full_20260711.csv | c4c3233b2a5d87c8ae59f0673c85368d33d78df5b07adf4240654f5f7fb0b5aa |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/checkpoint_inputs/A0/run_config.yaml | a7463065a62a2f8ea9b892c0d945125f3b8dc9c476211672c45c4661a5ff5974 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/manifest_A0.csv | 0669d9786abc11f1076cd0583f63bf67c8448a4b8f42e06a3d5ff46d6237c834 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/provenance_A2.json | cebb92524f93b7c16bb89a4d977f14c26c5cc8509bd9c9d2e4f564d9c5b1e662 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A2_no_sensor_global_feedback_DemoN602_20260831_182527/run_config.yaml | ebcad181a97244452f672bee7932b5f650f2f025867fc1c4411bea341ed37437 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/manifest_A2.csv | 1440f0cd9d70ab4d59350b84067abe814d6a8a0c1ba5abd2e69e9d774c2322ad |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/provenance_A3.json | f43eaceb610f8b8c9a4b3c9ea358ffa175c65b9a64cf03a7bec824bbafa9262b |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A3_no_local_query_conditioning_DemoN603_20260831_182718/run_config.yaml | 3a4ade9cf75ae6bf9250f2b3972decb7cdbec8be2f7f8774515d8775a2e62369 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/manifest_A3.csv | 857f8d0dffa2e50860747b30b0b0bb76fcac4eb4f2967985d875798891571251 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/provenance_A5.json | 116131feef486f75bc38b7894abc7ffa7ef402bc94973b46a8f87367188dcc44 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A5_local_sensor_tokens_only_DemoN605_20260901_211321/run_config.yaml | 0a2ec89dafce2de7a014ba07e4034901db431ccf01affc7e870e3746de310fc1 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/manifest_A5.csv | 9680267c9d1f071368e73cf2d8d64e0620ee15f903d1b3881674d93c4a622469 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/provenance_A4.json | d20a430ab8ffed70294fe0872d5c9a0bf968bf77ad6703bb6dceb01146f45ddb |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A4_iid_prior_DemoN604_20260903_120841/run_config.yaml | ecaec764904721a3d1bc7854a15c4f71c7181e505a3c314aa9c2268fb4265a27 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/manifest_A4.csv | 9af843c5df3c0d375f52f09f98feeca11a6a9f56a73e4bf91e41230f633e3e16 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/provenance_A1.json | cd85838b3caea6b5f1bd9c42ecf98ac6ce037293b81c6c7cad77cc439f039c13 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A1_deterministic_DemoN601_20260903_171000/run_config.yaml | 301397cac6cef4f726deb53f025666ea69a20fb5da2f2e1cdc42aff9c2720c4b |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/manifest_A1.csv | 93c3b8e28960f1d0cc64e45539517fa3d73411e44693a1ee9498aff2ae3592ba |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/metrics/cache_audit.csv | 0e794076f8ae9fc4a7ca6150c075981de3a2dde7f4c9b5bf7f98b81bae6b0a7a |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/metrics/analysis_metadata.json | 24159b272c1c333d20153b1a538313ace140ee130fb8dc64ee8a13f804717464 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/provenance_A0.json | 82fcf65ad6ecb5c79a9fa48db58b4ed10823b6a5ed19a0332b2249a1c8013155 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/manifest_A0.csv | 07b7d931131026686f8d29435bfdb46ab4484268d86280c7197d0f4d9381e87a |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/provenance_A2.json | 314154ed2d546060b0624b06e86742aae2f26122b69ec00741177865bbc37ff5 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/manifest_A2.csv | ac0614ea5cab2d5faa2f34765f87e80671dc71406596ad46d52b0ed28751daed |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/provenance_A3.json | d18e7ee4e5d0974e42ffd4a15dd0a427ac3df9d2f0521839b431674ad37762ec |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/manifest_A3.csv | 2a351bf58e617055cc980fca8f4a41c76a0743505d1bcd0a2867fe9706e227ab |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/provenance_A5.json | 2506ae093c0df972ed56aceed6837fbf4dfb394a42dc668fa599f30bb1763d07 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/manifest_A5.csv | 367da38c7c85f3b00f82bb8a0c5fd8171a8fd8fbd6f3df5274bce6e69efd89b4 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/provenance_A4.json | 5b9455cb1f2bcc3841dee5eac9d6e4a4995cc4b5d3f7ee1b3160a31acfc6d880 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/manifest_A4.csv | 2a8e406b48f1a290ff9856be5b041c78e889073bbbfce970460000f268b87914 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/provenance_A1.json | 18f02696bc94732919e604e523b84b0975c9836daa897baf7a2f9b8a8a17c739 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/manifest_A1.csv | 99602891bc70575ce788172bfc58319853387d13419b4239a32e0d8a56a352c6 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/metrics/cache_audit.csv | 77737a5c3e0cfb2e6cdf892b3a7af64440bcc95e5dcc2796ddaeb08ddcfbc93f |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/metrics/summary_metrics.csv | ac6efc744ec055cae71d3b19714f0b0205b2e26aa77a6c7a89f7e9ad7f68c2ab |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/metrics/paired_differences.csv | 868421436db5e4f2c8e4a116f11cadedb7d2c3557cb61d139e74f7d2bf35fb36 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/metrics/summary_metrics.csv | 560886a2979f011160b625de4ac262bedb339a70e27ee8b68495cb7870e68c2c |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/metrics/paired_differences.csv | 3612bf4c95ac6527679a153d248d6512e9665644ede412bb6420fa511a831ffc |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency/summary_high_frequency.csv | 95bff4d9bfcbe037da08397c8ff3241b3e03897abc1a1be218d49798d6977a52 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency/paired_high_frequency.csv | c7dd99011e7b08795d0d880d356a7fbbf169fbd2e6f015ca15f0d3136e69164a |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/metrics/per_state_metrics.csv | 648e5aeb0c5ee3ff97a6f715593023d7c05051aa9102c12e76c79f478a8e9d75 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/best/metrics/per_state_metrics.csv | ec4af7e0a28802b52c7234e6d11856876ac5b1fcf9ef94e8fd86c22c2387c74b |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency/analysis_metadata.json | 72a91f4bec57360c9c1bdb225a0de32cde405a61117de44bc065d8b7774d0d1a |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency/per_state_high_frequency.csv | 2e93c0f202a2506fcf6fff1cd79b630ce34dc667f7235ba9c04ea93289870584 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency/population_U1_spectra.csv | 2d4d5b3cfb3addccf94f9e66c13e6a6f1e66b5151e66642b1d967112cef2b8a5 |
| saved evidence | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency/U1_hann_robustness_summary.csv | ca5113e75ac9c4754cf81e6befd94b4dddfe661d67736dc0aacd811ba49a7154 |
| available recorded history; live file may continue | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A0_baseline_DemoN600_20260906_233723/loss_history.csv | c3ff7dfc872ae1c4f5bd02449fe310ffe33d8a7c74fbb7273f748b40c44bab14 |
| available recorded history; live file may continue | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A2_no_sensor_global_feedback_DemoN602_20260831_182527/loss_history.csv | 7b8b1ae1fea5dcc5c96d3b3e5ec85e535ed8899030d1e31fec439a9f6206a2ab |
| available recorded history; live file may continue | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A3_no_local_query_conditioning_DemoN603_20260831_182718/loss_history.csv | 20f2ce6a0700d0c6d74da2bb2993e25c8b2b1172815be4de817303583617ca11 |
| available recorded history; live file may continue | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A5_local_sensor_tokens_only_DemoN605_20260901_211321/loss_history.csv | 44e8e55c69647013b6afd04b9a4e8935eb07b3471e3543eb448fdcb4d738e79c |
| available recorded history; live file may continue | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A4_iid_prior_DemoN604_20260903_120841/loss_history.csv | 70effc352617f992d8e366f09b8b995ac0d08c11b4b0694dfdaacdf2c351434e |
| available recorded history; live file may continue | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A1_deterministic_DemoN601_20260903_171000/loss_history.csv | c86d9f31cec7a198a9dab3eef99dd270ff32cd668ee4a7e22994dc01e69459f0 |
| saved evidence | Dis_SI_Process/figures/generated/20260910_1540/si/history_snapshot_manifest.json | 59a359f82952516f7198f9c114d941dd2911803a07a1a8a6b62a3248503d8366 |
| timestamp-frozen recorded history used by SI; all captured rows retained | Dis_SI_Process/figures/generated/20260910_1540/si/training_history_A0.csv | 9f17fb2a39c5bf98a211dc4a5d6e95cd05c054a918917194db30a29efd42df11 |
| timestamp-frozen recorded history used by SI; all captured rows retained | Dis_SI_Process/figures/generated/20260910_1540/si/training_history_A2.csv | 6e292a5c528f921a42e644b3c55901de220ad21d1ef12edf84453f3a10a055c2 |
| timestamp-frozen recorded history used by SI; all captured rows retained | Dis_SI_Process/figures/generated/20260910_1540/si/training_history_A3.csv | 769cfba86dc4a54342d0a6be992b04b0cb506c0356ed7cc195e50066f2036a55 |
| timestamp-frozen recorded history used by SI; all captured rows retained | Dis_SI_Process/figures/generated/20260910_1540/si/training_history_A5.csv | 14a0e44624c2f5d2699c4a0f7fa625d08760be8096f82f87e45ccf934e02d7ca |
| timestamp-frozen recorded history used by SI; all captured rows retained | Dis_SI_Process/figures/generated/20260910_1540/si/training_history_A4.csv | e37ad839e52970781361a6a13050ce926137c8badb89553f76eff737d10e184e |

## Complete precision and display audit

`figure5_v7_display_source.csv` carries the evidence-package/checkpoint/cohort/metric/source identities. `main_artist_coordinates.json` records every main artist line, scatter offset, confidence-band path, annotation position and axis transform. `main_ablation_plot_coordinates.csv` records new point/interval values and formatted annotations. `si_plot_manifest.json` records every standalone SI plotted coordinate, scientific definition, source hash/filter, interval versus dispersion, and encoding; composites use the same complete scientific subpanels. Full summary and paired-effect tables preserve source precision. Histogram context stays attached in ablation_coupling_audit.csv; no policies are pooled.
