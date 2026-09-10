# Figure 5 V7R2 quantitative figure-making report

Release: `20260910_1707`. Documentation status: **complete**. Source QA status: **pass**.

## Scientific claim and evidence boundary

The saved full stochastic configuration and its conditioning-route controls provide a checkpoint-conditional comparison of reconstruction error, while the coherent RFF prior and IID prior expose a whole-field versus fine-scale fidelity tradeoff. The historical Figure 5 V6 uncertainty and cost evidence remains an inherited evidence package. The new ablation layer is not a matched-budget causal study: endpoints, effective optimized capacity and training histories differ, and the validation/test holdout is shared. For the five stochastic ablation generators, each state contributes one draw generated with two Euler steps and the recorded observed-entry clamp. Senseiver is a direct deterministic forward reference; those generator operations are not imposed on it.

## Source registry and package separation

Package B supplies panels a, b, c and f from the accepted Figure 5 V6 release. Package A supplies panels d and e from the source-gated V7R2 reductions. No Package A row is joined to Package B by the display string `DMF-Gen`; the new full model is carried by its internal run key. The source gate requires all new compact files, source QA pass, explicit last.pt policy, matching state/time identities, and a Senseiver last.pt reference for the spectral panel. Numeric values in this report are read from the source reductions and inherited source ledger; a newer prose report is not silently substituted for them.

## Exact source files used

The following source mappings are read from `source_manifest.json`; entries without a panel mapping are treated as unresolved rather than guessed.

### Panel a: Normalized empirical CRPS

| key | path | sha256 | role |
| --- | --- | --- | --- |
| Dis_SI_Process/figures/generated/20260904_1200/fig5a_probabilistic_reconstruction_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5a_probabilistic_reconstruction_20260904_1200.svg | de4248f97c4041bbf468dbf1f1da64da476fc8d082baf8326d357137ca00bb55 | Package B inherited accepted 20260904_1200 source |

### Panel b: Spread-error Spearman association

| key | path | sha256 | role |
| --- | --- | --- | --- |
| Dis_SI_Process/figures/generated/20260904_1200/fig5b_uncertainty_tracks_difficult_states_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5b_uncertainty_tracks_difficult_states_20260904_1200.svg | 293b867c8d33218366530f68f8f6131b1e8154fae0df3f17415eff8b01be247c | Package B inherited accepted 20260904_1200 source |

### Panel c: Selective reconstruction

| key | path | sha256 | role |
| --- | --- | --- | --- |
| Dis_SI_Process/figures/generated/20260904_1200/fig5c_selective_reconstruction_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5c_selective_reconstruction_20260904_1200.svg | 0bc60a04464956b4230d8427c8d0ef70854ea48205636e1ff78788fd34ac7543 | Package B inherited accepted 20260904_1200 source |

### Panel d: Ablation error distributions

| key | path | sha256 | role |
| --- | --- | --- | --- |
| reconstruction_states.csv | Dis_SI_Process/results/derived/20260910_1707/reconstruction_states.csv | 535e300ed9b52f61503f265153e56ba78a9cbcda1128d069c4d6515397fbd7fe | Package A V7R2 saved-checkpoint reduction |
| reconstruction_summary.csv | Dis_SI_Process/results/derived/20260910_1707/reconstruction_summary.csv | f6478058231d28d2e48184bb3d0363ac1448bd7c4fa0adf105bf5836152b180a | Package A V7R2 saved-checkpoint reduction |

### Panel e: Scale-resolved fidelity

| key | path | sha256 | role |
| --- | --- | --- | --- |
| highband_states.csv | Dis_SI_Process/results/derived/20260910_1707/highband_states.csv | 58b819c80e54001d0cd8e34669a76d25eaab3cf253a0011525dfa30624e0239c | Package A V7R2 saved-checkpoint reduction |
| highband_summary.csv | Dis_SI_Process/results/derived/20260910_1707/highband_summary.csv | acea4813cab3d3a9a2abdd4ef2217a93857fb2588e82cb3fb4c76f4ad0abb9f6 | Package A V7R2 saved-checkpoint reduction |
| spectra_population.csv | Dis_SI_Process/results/derived/20260910_1707/spectra_population.csv | ba3fb130125040c2e81aa1420b084ba02394d6d857041fe73b8f4969e0f65ad3 | Package A V7R2 saved-checkpoint reduction |
| spectral_diagnostics.csv | Dis_SI_Process/results/derived/20260910_1707/spectral_diagnostics.csv | 6ec66706008e20a52ff3e5f8c44f31f9571425b08e9d8aa655731ec57b04f731 | Package A V7R2 saved-checkpoint reduction |

### Panel f: Accuracy and computational footprint

| key | path | sha256 | role |
| --- | --- | --- | --- |
| Dis_SI_Process/figures/generated/20260904_1200/fig5d_accuracy_computational_footprint_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5d_accuracy_computational_footprint_20260904_1200.svg | fcfa4982f0e5811c327dee74083a86547efe4061a970476ae9263b0a63bff21f | Package B inherited accepted 20260904_1200 source |


### Compact V7R2 source reductions

| file | sha256 | rows |
| --- | --- | --- |
| reconstruction_states.csv | 535e300ed9b52f61503f265153e56ba78a9cbcda1128d069c4d6515397fbd7fe | 49000 |
| reconstruction_summary.csv | f6478058231d28d2e48184bb3d0363ac1448bd7c4fa0adf105bf5836152b180a | 49 |
| highband_states.csv | 58b819c80e54001d0cd8e34669a76d25eaab3cf253a0011525dfa30624e0239c | 35000 |
| highband_summary.csv | acea4813cab3d3a9a2abdd4ef2217a93857fb2588e82cb3fb4c76f4ad0abb9f6 | 35 |
| spectra_population.csv | ba3fb130125040c2e81aa1420b084ba02394d6d857041fe73b8f4969e0f65ad3 | 7920 |
| spectral_diagnostics.csv | 6ec66706008e20a52ff3e5f8c44f31f9571425b08e9d8aa655731ec57b04f731 | 35 |
| checkpoint_provenance.csv | 445628023dad37fffc0bd79e4cd4bf76609427602612ecd3a421e0fb2d82d944 | 7 |


### Complete source-manifest hash ledger

| key | path | sha256 | role |
| --- | --- | --- | --- |
| sources[0] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/Ablation_CondT_Evaluation_20260910.md | 84d5c9fc708988f51f95d4668ef84786822367570e19bd20f71c466e632a1138 | Ablation_CondT_Evaluation_20260910.md |
| sources[1] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/metrics/per_state_metrics.csv | 648e5aeb0c5ee3ff97a6f715593023d7c05051aa9102c12e76c79f478a8e9d75 | per_state_metrics.csv |
| sources[2] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/metrics/summary_metrics.csv | ac6efc744ec055cae71d3b19714f0b0205b2e26aa77a6c7a89f7e9ad7f68c2ab | summary_metrics.csv |
| sources[3] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency/per_state_high_frequency.csv | 2e93c0f202a2506fcf6fff1cd79b630ce34dc667f7235ba9c04ea93289870584 | per_state_high_frequency.csv |
| sources[4] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency/summary_high_frequency.csv | 95bff4d9bfcbe037da08397c8ff3241b3e03897abc1a1be218d49798d6977a52 | summary_high_frequency.csv |
| sources[5] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency/per_state_U1_spectra.csv | 470d8e2caaf147a931c3370a5681528cb7d09cc1aa8d042d3b7602918a530e24 | per_state_U1_spectra.csv |
| sources[6] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/last/metrics/snapshot0_spectra.csv | e4d0a2088ffa994474529791541ec52bd95b029c4eca905cc007e0628d1d5ad2 | snapshot0_spectra.csv |
| sources[7] | /data/wanglz/Cache/PhyCoFlow_TurbulentCombustion_Process_Results/ReconstructionCache/ReconstructionCache_manifest_paper_full_20260711.csv | 2b77e4a31554126e23b3e1a43de931a36797a67e70211380681dd205d9cf9b57 | ReconstructionCache_manifest_paper_full_20260711.csv |
| sources[8] | 0_demo_TurbulentCombustion/src/analyze_ablation_high_frequency.py | 64ac6d20bfefa094751d83588ad958510d65c519c9f05d9cc9f8730f866cf82b | analyze_ablation_high_frequency.py |
| sources[9] | 0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Scripts/common/spectral.py | 186e84beb9a4419f14d381b71cea9e09c91597589eb91b12e0f068f92538a9bc | spectral.py |
| sources[10] | Dis_SI_Process/results/derived/20260910_1540/source_manifest.json | a6cd3cd371d7ef68a422f3064f10a88148309caadc06d6cc833bdb3e7baa6403 | source_manifest.json |
| sources[11] | Dis_SI_Process/results/derived/20260910_1540/build_manifest.json | 7d0916346bf5b8a994f536c117d60f0d7f19fd15ae2d8f6a422c034e1b464a83 | build_manifest.json |
| sources[12] | Dis_SI_Process/results/derived/20260904_1200/build_manifest.json | 07fbbeca451e72ce3bd57ef0e3b66fa35653cdd62cf9030a1f59211faee36011 | build_manifest.json |
| sources[13] | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_a_samples.csv | 8e82151c63469032b612afced64da87a379915b86edbbd53d998d50b5a4ca984 | benchmark_main_a_samples.csv |
| sources[14] | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_a_summary.csv | 06919008c2c8c96e1f99ce0fce6eeffbbed7a94d30a0697b1668fca448b5ff2e | benchmark_main_a_summary.csv |
| sources[15] | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_b_samples.csv | 1dfa807d4f1082bcba8e32707b841be5203cc7404f76350b67cc8fbea710d41c | benchmark_main_b_samples.csv |
| sources[16] | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_b_summary.csv | b6d60652e3bf43de3cd85770da2c0901c8f41df7bed8ccd47699fe63277f1a4c | benchmark_main_b_summary.csv |
| sources[17] | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_d.csv | 5160e7eea425c53c33734cd78240cbbfbc55ec608712779e7c2bdb82d264c4b1 | benchmark_main_d.csv |
| sources[18] | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_f.csv | 47b595a64b856da33667be5eeae11e5667c91c6c9de61eb9f16a8f44034003d3 | benchmark_main_f.csv |
| sources[19] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/evaluation_20260910/checkpoint_inputs/A0/last.pt | 71a9e1004010558f7137f4140eb976d51d367d39e9bc3473334912257cbaa541 | A0_last_checkpoint |
| sources[20] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A1_deterministic_DemoN601_20260903_171000/last.pt | f5fb90759ae30eebcfcaebac7f12ce38d52b4e85bac13adde1b864bec3314ecc | A1_last_checkpoint |
| sources[21] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A2_no_sensor_global_feedback_DemoN602_20260831_182527/last.pt | 5037246ef415d5d0b78d253b5a4ecdde76b549f2436541f32dd5d1aa54e401cf | A2_last_checkpoint |
| sources[22] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A3_no_local_query_conditioning_DemoN603_20260831_182718/last.pt | 15de1f603be5580e51379675296dfb218e7a34f983272f081abd442510683cda | A3_last_checkpoint |
| sources[23] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A4_iid_prior_DemoN604_20260903_120841/last.pt | 21a57fbc86dc7eef7b498c8a1110b1ac9c1a679f73dd5f344f8d446c4de454a9 | A4_last_checkpoint |
| sources[24] | 0_demo_TurbulentCombustion/Save_TrainedModel/ablation_condT/A5_local_sensor_tokens_only_DemoN605_20260901_211321/last.pt | 0e6184a6b9c5634d79005529744f98ebc0f3d016a084ec1133987b987c1ea2a3 | A5_last_checkpoint |
| sources[25] | 0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/Senseiver/Cond_T/last.pt | b055ccc8ad86ab2d4d44a527620f4c0db06009c26de26d8d52db44065fc48503 | Senseiver_last_checkpoint |
| sources[26] | 0_demo_TurbulentCombustion/src/analyze_ablation_high_frequency.py | 64ac6d20bfefa094751d83588ad958510d65c519c9f05d9cc9f8730f866cf82b | accepted_spectral_analysis_source |
| sources[27] | 0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Scripts/common/spectral.py | 186e84beb9a4419f14d381b71cea9e09c91597589eb91b12e0f068f92538a9bc | accepted_common_spectral_estimator |
| sources[28] | /data/wanglz/Cache/PhyCoFlow_TurbulentCombustion_Process_Results/FieldL2/FieldL2_per_snapshot_paper_full_20260711.csv | eb21baf7cee50d3453c31b19ca38b05c35c06ea2bb17a95c8f0aeb63ceebbfc7 | optional_Senseiver_FieldL2_crosscheck |
| Dis_SI_Process/figures/generated/20260904_1200/fig5d_accuracy_computational_footprint_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5d_accuracy_computational_footprint_20260904_1200.svg | fcfa4982f0e5811c327dee74083a86547efe4061a970476ae9263b0a63bff21f | Package B inherited accepted 20260904_1200 source |
| Dis_SI_Process/figures/generated/20260904_1200/fig5_composed_v6_20260904_1200.pdf | Dis_SI_Process/figures/generated/20260904_1200/fig5_composed_v6_20260904_1200.pdf | a9e3b29a00aa4d5c0cc0cc4e401ad06ce99c6847b4316b913458c656941fd62a | Package B inherited accepted 20260904_1200 source |
| Dis_SI_Process/figures/generated/20260904_1200/fig5_composed_v6_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5_composed_v6_20260904_1200.svg | d552cc75772ed949d64dc0c569cbc921c3fac0874c1c2e9129cfd8cf20d54a50 | Package B inherited accepted 20260904_1200 source |
| Dis_SI_Process/figures/generated/20260904_1200/fig5b_uncertainty_tracks_difficult_states_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5b_uncertainty_tracks_difficult_states_20260904_1200.svg | 293b867c8d33218366530f68f8f6131b1e8154fae0df3f17415eff8b01be247c | Package B inherited accepted 20260904_1200 source |
| Dis_SI_Process/figures/generated/20260904_1200/fig5a_probabilistic_reconstruction_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5a_probabilistic_reconstruction_20260904_1200.svg | de4248f97c4041bbf468dbf1f1da64da476fc8d082baf8326d357137ca00bb55 | Package B inherited accepted 20260904_1200 source |
| Dis_SI_Process/figures/generated/20260904_1200/fig5c_selective_reconstruction_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5c_selective_reconstruction_20260904_1200.svg | 0bc60a04464956b4230d8427c8d0ef70854ea48205636e1ff78788fd34ac7543 | Package B inherited accepted 20260904_1200 source |
| Dis_SI_Process/results/derived/20260904_1200/build_manifest.json | Dis_SI_Process/results/derived/20260904_1200/build_manifest.json | 07fbbeca451e72ce3bd57ef0e3b66fa35653cdd62cf9030a1f59211faee36011 | Package B inherited accepted 20260904_1200 source |
| Dis_SI_Process/results/derived/20260904_1200/qa.json | Dis_SI_Process/results/derived/20260904_1200/qa.json | 2136b9d290b1e140eb0cc6a47a5ccbd7ea8c308b49901a6196d6978a6a65c286 | Package B inherited accepted 20260904_1200 source |
| Dis_SI_Process/docs/generated/20260904_1200/figure_contract.md | Dis_SI_Process/docs/generated/20260904_1200/figure_contract.md | 4e5da308bd39f12f7b8d36477e68e286f80c61eeb90ee80a674cd67b2c2ba4ea | Package B inherited accepted 20260904_1200 source |
| Dis_SI_Process/docs/generated/20260904_1200/completion_report.md | Dis_SI_Process/docs/generated/20260904_1200/completion_report.md | 1aea2580ff57329ac470c171b22820c4890043822be8b9b42c5673a8bee3359b | Package B inherited accepted 20260904_1200 source |


### Inherited Package B benchmark and V6 renderer inputs

The inherited panel SVGs above are the protected visual evidence. These exact benchmark tables are the corresponding numeric inputs for panels a--c and f; the V6 configuration and renderer chain are listed so the layout/style inheritance is reproducible.

| key | path | sha256 | role |
| --- | --- | --- | --- |
| Dis_SI_Process/configs/figure5_v6.yaml | Dis_SI_Process/configs/figure5_v6.yaml | 18a8cee9d6feed672c9c48c2c79fc7278a33decb6dadba7165116259ab303e50 | Package B inherited V6 configuration/renderer input |
| Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format.py | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format.py | f1527e590e5c48ad1976ff9e18154bf591b360948369735d171479c2a34c95fc | Package B inherited V6 configuration/renderer input |
| Dis_SI_Process/figures/scripts/build_figure5_v6.py | Dis_SI_Process/figures/scripts/build_figure5_v6.py | 25f0f670c01a8267d590a2acc5423ab638ce56856cbde9de8182cba99a5a5afc | Package B inherited V6 configuration/renderer input |
| Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v4.py | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v4.py | 463c78e81e32686fbb2e8101bf78aab32ac5c3e101552564d5921c980001696d | Package B inherited V6 configuration/renderer input |
| Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v3.py | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v3.py | 4f73f67a6a93909293a16396f287813075587f4663f44c57240b0f3cdee457f2 | Package B inherited V6 configuration/renderer input |
| Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v5.py | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v5.py | 502e1af41f438bc2026b3681bdefcbb4ed7514caf87188db0390a075aa9bca3a | Package B inherited V6 configuration/renderer input |
| Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v2.py | Dis_SI_Process/figures/scripts/build_figure5_v6_nature_format_v2.py | 8dd43c7721bc98ea50120c86dd5f1ea1308ba4681307bb41d672828d279f09c9 | Package B inherited V6 configuration/renderer input |
| benchmark_main_a_samples | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_a_samples.csv | 8e82151c63469032b612afced64da87a379915b86edbbd53d998d50b5a4ca984 | Package B inherited accepted benchmark source |
| benchmark_main_a_summary | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_a_summary.csv | 06919008c2c8c96e1f99ce0fce6eeffbbed7a94d30a0697b1668fca448b5ff2e | Package B inherited accepted benchmark source |
| benchmark_main_b_samples | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_b_samples.csv | 1dfa807d4f1082bcba8e32707b841be5203cc7404f76350b67cc8fbea710d41c | Package B inherited accepted benchmark source |
| benchmark_main_b_summary | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_b_summary.csv | b6d60652e3bf43de3cd85770da2c0901c8f41df7bed8ccd47699fe63277f1a4c | Package B inherited accepted benchmark source |
| benchmark_main_d | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_d.csv | 5160e7eea425c53c33734cd78240cbbfbc55ec608712779e7c2bdb82d264c4b1 | Package B inherited accepted benchmark source |
| benchmark_main_f | Dis_SI_Process/results/derived/20260910_1540/benchmark_main_f.csv | 47b595a64b856da33667be5eeae11e5667c91c6c9de61eb9f16a8f44034003d3 | Package B inherited accepted benchmark source |


## Panel-by-panel intent, definitions and design rationale


### Unobserved-field reconstruction statistics (A1 shown for SI context)

| display_label | n | mean | block20_ci95_low | block20_ci95_high | median | q25 | q75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full model | 1000 | 0.1063207667731775 | 0.1044952423878208 | 0.1082293786000968 | 0.1030912992551808 | 0.093292122502138 | 0.1157107128801152 | 0.1417722829556948 | 0.1993688889266365 |
| No sensor feedback | 1000 | 0.126974415832413 | 0.1248683028587497 | 0.1292483965748119 | 0.1231441562381441 | 0.1112348486599389 | 0.1386332136155527 | 0.1698346021311825 | 0.2327402376299946 |
| No local conditioning | 1000 | 0.1447029438194137 | 0.1427571598963409 | 0.146663560185458 | 0.1424522189166816 | 0.1310104364357827 | 0.1552986109427989 | 0.1795970117087757 | 0.2470756333724128 |
| Local-only conditioning | 1000 | 0.3549507239603569 | 0.351421880078963 | 0.3586960655427726 | 0.3527352661229739 | 0.3311490396646599 | 0.375609375201343 | 0.4079201848082557 | 0.4660544655595336 |
| IID Gaussian prior | 1000 | 0.1042938161915426 | 0.1024122041391392 | 0.1063015487293748 | 0.1013600002157301 | 0.0918786724320902 | 0.1133550642409659 | 0.1382193193027015 | 0.1931664183056083 |
| Deterministic regression | 1000 | 0.07870496823327 | 0.0773590733579577 | 0.0801945762271615 | 0.0753831781609645 | 0.0694897007915977 | 0.0835807727568816 | 0.1065157299487969 | 0.1885206539588684 |


### U1 high-band residual statistics (A1 shown for SI context)

| display_label | n | mean | block20_ci95_low | block20_ci95_high | median | q25 | q75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full model | 1000 | 0.8734065584046022 | 0.8621901011860641 | 0.8848534075561257 | 0.9028338688204152 | 0.793965733919316 | 0.9695989432056396 | 1.0601388861558607 | 1.708936004548555 |
| No sensor feedback | 1000 | 0.9174607939236632 | 0.9044547275353452 | 0.9304104691771716 | 0.9417155229068774 | 0.8251423600681551 | 1.0159156033445744 | 1.1419330482834065 | 1.5529098252650606 |
| No local conditioning | 1000 | 1.3556050263233643 | 1.3313876762691537 | 1.3787654206253817 | 1.281672945542213 | 1.163784445893124 | 1.4722030235178218 | 1.9083718753692689 | 2.911778121741502 |
| Local-only conditioning | 1000 | 1.4717447282306744 | 1.447890313043029 | 1.4989682959483266 | 1.4519593397877135 | 1.3048079206937473 | 1.621782412381501 | 1.9489347569820648 | 3.2856744905374167 |
| IID Gaussian prior | 1000 | 1.8252839107455296 | 1.7793718541521102 | 1.877885185559828 | 1.7955915689114563 | 1.524025146233135 | 2.0848713507922527 | 2.590504216702332 | 4.30829007020632 |
| Deterministic regression | 1000 | 0.8342534285920838 | 0.822634511260981 | 0.8462381251251659 | 0.8743405564026758 | 0.7581256851746283 | 0.938838551571652 | 1.0141467657552836 | 1.2698036622628923 |
| Senseiver | 1000 | 1.2773966062562505 | 1.2635123952530078 | 1.2918634234143016 | 1.2401908228336287 | 1.1701154937385672 | 1.3519103796930516 | 1.5923504890323916 | 2.0634788941526527 |

### a. Normalized empirical CRPS

Intent: Inherited marginal uncertainty evidence; preserve the accepted source coordinates, method order, points, box and interval treatment.

Definition: For K=64 accepted ensemble draws, CRPS is K^{-1} sum_k |X_k-y| - (2K^2)^{-1} sum_{k,l} |X_k-X_l|, normalized by the frozen training-field standard deviation. Points comprise the four unobserved fields and the equal macro-field mean over the accepted 200-state cohort.

Evidence package: Package B, inherited Figure 5 V6 release.

Design rationale: Compact scatter and box summary; no decorative title; layout-only resizing and repositioning.
### b. Spread-error Spearman association

Intent: Inherited uncertainty-informativeness evidence; preserve the zero guide, interval treatment and method order.

Definition: Across accepted held-out states, compute Spearman association between macro normalized ensemble spread and macro ensemble-mean relative-L2 error. The plotted cloud contains accepted bootstrap estimates; it is not a set of statewise correlations.

Evidence package: Package B, inherited Figure 5 V6 release.

Design rationale: Compact box/interval treatment with the dashed zero line; no decorative title.
### c. Selective reconstruction

Intent: Inherited within-method selective-risk evidence; move it to the top row without changing coordinates or normalization.

Definition: For each method, retained-set reconstruction error is normalized by that method's full-cohort error while retaining the lowest-spread states; the inherited retained-fraction coordinates and accepted area are preserved.

Evidence package: Package B, inherited Figure 5 V6 release.

Design rationale: Balanced top-row aspect ratio; optional effect annotation remains subordinate to the curves.
### d. Ablation error distributions

Intent: Show the distribution of unobserved-field relative-L2 across 1,000 held-out snapshots for the five stochastic variants.

Definition: Statewise relative-L2 values loaded from reconstruction_states.csv and summarized by reconstruction_summary.csv; mean and block-20 interval are source values.

Evidence package: Package A, new ablation source reductions; last.pt only.

Design rationale: Horizontal light jitter points plus compact box summary and mean marker; mean numbers only and no percentage annotations.
### e. Scale-resolved fidelity

Intent: Show population-spectrum shape and the distribution of selected-field high-band relative-L2 across all main variants plus Senseiver.

Definition: Median shell spectra are loaded from spectra_population.csv; source IQR values remain available for audit but are not plotted. High-band statewise relative-L2 and summary statistics are loaded from highband_states.csv and highband_summary.csv.

Evidence package: Package A, new ablation source reductions; last.pt only, with Truth and Senseiver reference rows.

Design rationale: One lettered panel containing two vertically stacked axes; all 198 median shell points are connected, sparse markers every 42 shells identify methods, the high-band locator is shown on the spectrum, and the lower distribution plot retains all states.
### f. Accuracy and computational footprint

Intent: Inherited scorecard evidence for accuracy and measured training/inference resources.

Definition: Preserve the accepted scorecard values and endpoint semantics; update the x-axis wording to Training update time and label Model and Peak directly on the inference-memory plot.

Evidence package: Package B, inherited Figure 5 V6 release.

Design rationale: Single full-width scorecard block; direct memory endpoint labels remove the separate memory legend.

## Quantitative aggregation and checkpoint policy

Table S1 loads the `unobserved_mean` field from `reconstruction_summary.csv` for all six saved runs and retains n, mean, block-20 confidence limits, median, interquartile range, 95th percentile and maximum. Table S2 retains fieldwise physical relative-L2 means for CH4, CO, T, U1 and p; temperature remains visible as a fieldwise diagnostic and is excluded from the unobserved macro. Table S3 loads selected-field U1 high-band relative-L2 for all six ablation runs plus Senseiver. Table S4 preserves the selected source spectral diagnostics, including any source-reported truth-relative high-band power statistic. Table S5 records internal run key, display label, last.pt policy, epoch, source path and checkpoint hash.

All new SI figures and tables use last.pt only. The document writer does not merge best.pt rows or display a best-versus-last split. New d/e distributions retain every state and use the source Q25, median, Q75, 1.5-IQR whiskers, mean and block-20 confidence limits; intervals are not intervals over training seeds. The main ablation comparison excludes A1; A1 remains in SI Figure S1/S3 and the complete provenance table. Inherited a--c use 200 held-out states and 64 ensemble draws with the accepted block-25/2,000-resample procedure; new d/e use 1,000 held-out states with block-20/2,000 resamples.

## Spectral summary and high-band computation

For each centered field $u-\bar{u}$ on the native structured grid, the source estimator computes $F=\operatorname{FFT}_2(u-\bar{u})$ and the absolute spectral density $E=|F|^2/(n_x n_y)$, then takes native shell means. Panel e-top displays the per-shell median absolute energy at all 198 retained shells; source IQR values are retained in `spectra_population.csv` for audit but are not plotted. Continuous curves use all shell points; sparse method markers every 42 shells, with method-specific shapes/colors and a separate Truth legend entry, are presentation-only and do not filter, smooth or subsample the data. The high-band is the strict upper third $k>2k_{\max}/3$, which is 68 shells and 17,704 Fourier modes for this grid. Panel e-bottom uses the phase-sensitive high-frequency residual $L_{2,\mathrm{HF}}=\sqrt{\sum_{k\in H}|F_{\mathrm{recon}-\mathrm{truth}}(k)|^2 / \sum_{k\in H}|F_{\mathrm{truth}}(k)|^2}$ from `highband_states.csv` and its source summary. The canonical high-band power ratio is a separate trapezoidal integral of shell-mean spectra, $\int_H E_{\mathrm{recon}}(k)\,dk / \int_H E_{\mathrm{truth}}(k)\,dk$; it is distinct from the Fourier mode-sum and the phase-sensitive residual and is retained in Table S4 where supplied. Truth is an explicit population row and Senseiver is a direct deterministic reference. No shell, frequency, smoothing or energy normalization is invented here.

## Bootstrap, cohort and budget notes

The new d/e held-out cohort contains 1,000 matched snapshots, with original snapshot/time identities carried in the state tables; state-level aggregation is equal-weighted. New d/e circular moving-block intervals use block length 20 and 2,000 resamples, with the accepted reconstruction summary reusing the recorded primary bootstrap seed 20260906 and the new Senseiver/high-frequency-derived summaries using seed 20260910. Blocks count sorted held-out states rather than simulation time steps. The five stochastic ablation generators use one draw per state, two Euler steps and the observed-entry clamp. Senseiver is direct deterministic forward evaluation with no Euler integration and no imposed observed-entry clamp. Inherited a--c use the accepted 200-state, 64-draw cohort and block length 25 with 2,000 resamples; panel f preserves its separately measured scorecard quantities. Training endpoints are unequal and were not corrected or truncated. A shared validation/test holdout means the intervals are checkpoint-conditional. Panel f preserves the inherited separately measured training-update and inference-memory quantities, with direct Model and Peak endpoint labels; memory values are not recomputed from the new ablation checkpoints.

Recorded seed/bootstrap metadata:

- `source_manifest.cohort.sensor_seed_policy` = `per-state seeds retained from A0 manifest and matched across all caches`
- `source_manifest.bootstrap.seed` = `20260910`
- `source_manifest.summary_sources.ablation_reconstruction.bootstrap_seed` = `20260906`
- `source_manifest.summary_sources.ablation_highband.bootstrap_seed` = `20260910`
- `source_manifest.summary_sources.Senseiver_reconstruction_and_highband.bootstrap_seed` = `20260910`
- `source_manifest.method_provenance.A0.generation_seed_policy` = `stable_seed(20260711,'generation','DMF-Gen','Cond_T',snapshot); shared across variants`
- `source_manifest.method_provenance.A1.generation_seed_policy` = `stable_seed(20260711,'generation','DMF-Gen','Cond_T',snapshot); shared across variants`
- `source_manifest.method_provenance.A2.generation_seed_policy` = `stable_seed(20260711,'generation','DMF-Gen','Cond_T',snapshot); shared across variants`
- `source_manifest.method_provenance.A3.generation_seed_policy` = `stable_seed(20260711,'generation','DMF-Gen','Cond_T',snapshot); shared across variants`
- `source_manifest.method_provenance.A4.generation_seed_policy` = `stable_seed(20260711,'generation','DMF-Gen','Cond_T',snapshot); shared across variants`
- `source_manifest.method_provenance.A5.generation_seed_policy` = `stable_seed(20260711,'generation','DMF-Gen','Cond_T',snapshot); shared across variants`

## Field-selection comparison

The default main field is U1. The alternatives remain full panel-e candidates and are not collapsed to a single score or selected automatically.

| field | population_spectrum_readability | highband_relative_l2_separation | senseiver_reference_contrast | source_method_coverage | caveat |
| --- | --- | --- | --- | --- | --- |
| Y_CH4 | Shellwise sawtooth structure; retain the unsmoothed source curve. It is readable as a species-scale diagnostic but needs print-width inspection. Spectrum coverage 7/7. | required mean range 0.2753--1.899 (span 1.624) | Full mean 0.275332; Senseiver mean 1.72729; delta (Senseiver minus Full) +1.45195 | high-band 6/6; spectrum 7/7 including Truth | Merit: strong Senseiver-versus-Full high-band separation. Caveat: shellwise sawtooth structure and species scaling can dominate visual interpretation. |
| Y_CO | More overlapping shell curves and the least high-band mean spread among the required methods; retain the unsmoothed source curve. Spectrum coverage 7/7. | required mean range 0.8328--1.179 (span 0.3464) | Full mean 0.832801; Senseiver mean 1.12569; delta (Senseiver minus Full) +0.292886 | high-band 6/6; spectrum 7/7 including Truth | Merit: the overlapping curves expose a conservative comparison. Caveat: the small high-band mean spread makes candidate ranking weak. |
| T | Shellwise sawtooth structure; retain the unsmoothed source curve. It remains a useful observed-field diagnostic. Spectrum coverage 7/7. | required mean range 0.5444--1.887 (span 1.343) | Full mean 0.544406; Senseiver mean 1.67587; delta (Senseiver minus Full) +1.13147 | high-band 6/6; spectrum 7/7 including Truth | Merit: strong Senseiver-versus-Full high-band separation. Caveat: temperature is observed conditioning input and is excluded from the unobserved macro. |
| U1 | Smooth, interpretable roll-off; the IID Gaussian prior visibly oversupplies high-frequency power. Spectrum coverage 7/7. | required mean range 0.8734--1.825 (span 0.9519) | Full mean 0.873407; Senseiver mean 1.2774; delta (Senseiver minus Full) +0.40399 | high-band 6/6; spectrum 7/7 including Truth | Merit: smooth roll-off and visible IID high-frequency oversupply make the scale tradeoff easiest to read. Caveat: index-space wavenumber is not a physical turbulence wavenumber, and weak truth high-band energy can magnify relative residuals. |
| p | Shellwise sawtooth structure; retain the unsmoothed source curve. Pressure-scale behavior remains visible without a smoothing choice. Spectrum coverage 7/7. | required mean range 0.5454--1.756 (span 1.21) | Full mean 0.54539; Senseiver mean 1.56671; delta (Senseiver minus Full) +1.02132 | high-band 6/6; spectrum 7/7 including Truth | Merit: strong Senseiver-versus-Full high-band separation. Caveat: pressure dynamic range and sign/admissibility should be checked against the source normalization. |


## Deviations from the preceding V7 brief

The final V7R2 logical order is a, b, c selective reconstruction; d ablation state distributions; e all-model population spectra plus selected-field high-band distributions; f the inherited scorecard. A1 is excluded from the main figure. New SI figures/tables are last.pt-only. The old decorative conditioning/source heading is removed from the contract, and percentage increase/decrease annotations are excluded from panels d and e. The historical 0.117 benchmark row is retained separately from the 0.106321 new full-ablation row.


## Resolved QA history

The initial Senseiver epoch probe could not read checkpoint metadata in the lightweight figure environment because torch was unavailable; the authoritative phycoflow_env torch.load probe verified the recorded Senseiver last.pt epoch as 5000, and that value is retained in Table S5. An earlier documentation attempt collided with the pandas `row.mean` method while formatting a paragraph; the named mean column is now used and the stamped writer reran successfully. Main-figure inspection removed false bounds caused by an invisible out-of-range logarithmic tick while preserving inherited a/b/c/f geometry. The SI visual review covered the eight required source/figure pairs and passed. These transient failures remain represented by the stamped QA/source history where applicable; no scientific source was substituted.


## Unresolved issues and failed checks

- None recorded by the document writer; renderer and final visual review remain authoritative.

The full main caption initially exceeded the LaTeX float height; the final concise caption retains the interpretation and sampling details while full mathematical definitions remain in this report. Standalone c required an explicit legend built from the validated curve order. SI column groups now retain their S1--S5 table identities, and a page flush keeps S3 before the tables. See `results/derived/20260910_1707/qa_history.json` and `execution_record.json` for failed stages and their resolutions.


## Reproduction

```bash
rtk proxy conda run -n fig python Dis_SI_Process/scripts/write_figure5_v7r2_documents.py --timestamp 20260910_1707 --strict-formal
```
