# Cond_T ablation config-diff audit

A0: `Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/run_config.yaml`
A0 SHA-256: `e24de2d9b8909daa520b7b390a4a62396832ceb6e6810cd67f90673d9cf2f6c1`

Operational additions (not scientific changes): unique Demo/output paths, device 1, the archived dataset-stat path, explicit 10,000-epoch scheduler horizon, explicit legacy execution defaults, model identity, and ablation provenance.
A2/A3/A5 additionally use the documented user-selected 6,000-epoch stop while retaining the 10,000-epoch scheduler horizon.

## A1

Expected intervention: `deterministic_same_backbone`.

- `Demo_Num` (operational/provenance): `17` → `601`
- `ablation` (operational/provenance): `None` → `{'enabled': True, 'id': 'A1', 'variant': 'deterministic_same_backbone', 'reference_name': 'DMF_Gen/Cond_T', 'reference_config': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/run_config.yaml', 'reference_config_sha256': 'e24de2d9b8909daa520b7b390a4a62396832ceb6e6810cd67f90673d9cf2f6c1', 'reference_checkpoint': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/best.pt', 'reference_checkpoint_sha256': 'af43f763fa91b2204a9413c650bd329bb01cfaeea64af325315339d64e2258a6'}`
- `condition_attention_execution` (operational/provenance): `None` → `'legacy_mha'`
- `coord_dim` (operational/provenance): `None` → `3`
- `data_path_mode` (operational/provenance): `None` → `'legacy'`
- `dataset_stats_path` (operational/provenance): `None` → `'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/dataset_stats.pt'`
- `device_ids` (operational/provenance): `[2]` → `[1]`
- `initialization` (operational/provenance): `None` → `'scratch'`
- `model_name` (operational/provenance): `None` → `'GL_rbf_ENH'`
- `save_dir` (operational/provenance): `'Save_TrainedModel/ffm_tc_pointcloud'` → `'Save_TrainedModel/ablation_condT/A1_deterministic'`
- `scheduler_t_max` (operational/provenance): `None` → `10000`
- `sensor_attention_buckets` (operational/provenance): `None` → `[256, 320, 384]`
- `sensor_attention_padding_mode` (operational/provenance): `None` → `'full'`
- `training_mode` (operational/provenance): `None` → `'standard'`

## A2

Expected intervention: `no_sensor_global_feedback`.

- `Demo_Num` (operational/provenance): `17` → `602`
- `ablation` (operational/provenance): `None` → `{'enabled': True, 'id': 'A2', 'variant': 'no_sensor_global_feedback', 'reference_name': 'DMF_Gen/Cond_T', 'reference_config': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/run_config.yaml', 'reference_config_sha256': 'e24de2d9b8909daa520b7b390a4a62396832ceb6e6810cd67f90673d9cf2f6c1', 'reference_checkpoint': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/best.pt', 'reference_checkpoint_sha256': 'af43f763fa91b2204a9413c650bd329bb01cfaeea64af325315339d64e2258a6'}`
- `condition_attention_execution` (operational/provenance): `None` → `'legacy_mha'`
- `coord_dim` (operational/provenance): `None` → `3`
- `data_path_mode` (operational/provenance): `None` → `'legacy'`
- `dataset_stats_path` (operational/provenance): `None` → `'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/dataset_stats.pt'`
- `device_ids` (operational/provenance): `[2]` → `[1]`
- `epochs` (approved matched-budget override): `10000` → `6000`
- `initialization` (operational/provenance): `None` → `'scratch'`
- `model_name` (operational/provenance): `None` → `'GL_rbf_ENH'`
- `save_dir` (operational/provenance): `'Save_TrainedModel/ffm_tc_pointcloud'` → `'Save_TrainedModel/ablation_condT/A2_no_sensor_global_feedback'`
- `scheduler_t_max` (operational/provenance): `None` → `10000`
- `sensor_attention_buckets` (operational/provenance): `None` → `[256, 320, 384]`
- `sensor_attention_padding_mode` (operational/provenance): `None` → `'full'`
- `training_mode` (operational/provenance): `None` → `'standard'`

## A3

Expected intervention: `no_local_query_conditioning`.

- `Demo_Num` (operational/provenance): `17` → `603`
- `ablation` (operational/provenance): `None` → `{'enabled': True, 'id': 'A3', 'variant': 'no_local_query_conditioning', 'reference_name': 'DMF_Gen/Cond_T', 'reference_config': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/run_config.yaml', 'reference_config_sha256': 'e24de2d9b8909daa520b7b390a4a62396832ceb6e6810cd67f90673d9cf2f6c1', 'reference_checkpoint': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/best.pt', 'reference_checkpoint_sha256': 'af43f763fa91b2204a9413c650bd329bb01cfaeea64af325315339d64e2258a6'}`
- `condition_attention_execution` (operational/provenance): `None` → `'legacy_mha'`
- `coord_dim` (operational/provenance): `None` → `3`
- `data_path_mode` (operational/provenance): `None` → `'legacy'`
- `dataset_stats_path` (operational/provenance): `None` → `'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/dataset_stats.pt'`
- `epochs` (approved matched-budget override): `10000` → `6000`
- `initialization` (operational/provenance): `None` → `'scratch'`
- `model_name` (operational/provenance): `None` → `'GL_rbf_ENH'`
- `save_dir` (operational/provenance): `'Save_TrainedModel/ffm_tc_pointcloud'` → `'Save_TrainedModel/ablation_condT/A3_no_local_query_conditioning'`
- `scheduler_t_max` (operational/provenance): `None` → `10000`
- `sensor_attention_buckets` (operational/provenance): `None` → `[256, 320, 384]`
- `sensor_attention_padding_mode` (operational/provenance): `None` → `'full'`
- `training_mode` (operational/provenance): `None` → `'standard'`

## A4

Expected intervention: `iid_gaussian_prior`.

- `Demo_Num` (operational/provenance): `17` → `604`
- `ablation` (operational/provenance): `None` → `{'enabled': True, 'id': 'A4', 'variant': 'iid_gaussian_prior', 'reference_name': 'DMF_Gen/Cond_T', 'reference_config': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/run_config.yaml', 'reference_config_sha256': 'e24de2d9b8909daa520b7b390a4a62396832ceb6e6810cd67f90673d9cf2f6c1', 'reference_checkpoint': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/best.pt', 'reference_checkpoint_sha256': 'af43f763fa91b2204a9413c650bd329bb01cfaeea64af325315339d64e2258a6'}`
- `condition_attention_execution` (operational/provenance): `None` → `'legacy_mha'`
- `coord_dim` (operational/provenance): `None` → `3`
- `data_path_mode` (operational/provenance): `None` → `'legacy'`
- `dataset_stats_path` (operational/provenance): `None` → `'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/dataset_stats.pt'`
- `device_ids` (operational/provenance): `[2]` → `[1]`
- `initialization` (operational/provenance): `None` → `'scratch'`
- `model_name` (operational/provenance): `None` → `'GL_rbf_ENH'`
- `prior` (scientific): `'rff'` → `'iid'`
- `save_dir` (operational/provenance): `'Save_TrainedModel/ffm_tc_pointcloud'` → `'Save_TrainedModel/ablation_condT/A4_iid_prior'`
- `scheduler_t_max` (operational/provenance): `None` → `10000`
- `sensor_attention_buckets` (operational/provenance): `None` → `[256, 320, 384]`
- `sensor_attention_padding_mode` (operational/provenance): `None` → `'full'`
- `training_mode` (operational/provenance): `None` → `'standard'`

## A5

Expected intervention: `local_sensor_tokens_only`.

- `Demo_Num` (operational/provenance): `17` → `605`
- `ablation` (operational/provenance): `None` → `{'enabled': True, 'id': 'A5', 'variant': 'local_sensor_tokens_only', 'reference_name': 'DMF_Gen/Cond_T', 'reference_config': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/run_config.yaml', 'reference_config_sha256': 'e24de2d9b8909daa520b7b390a4a62396832ceb6e6810cd67f90673d9cf2f6c1', 'reference_checkpoint': 'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/best.pt', 'reference_checkpoint_sha256': 'af43f763fa91b2204a9413c650bd329bb01cfaeea64af325315339d64e2258a6'}`
- `condition_attention_execution` (operational/provenance): `None` → `'legacy_mha'`
- `coord_dim` (operational/provenance): `None` → `3`
- `data_path_mode` (operational/provenance): `None` → `'legacy'`
- `dataset_stats_path` (operational/provenance): `None` → `'Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/dataset_stats.pt'`
- `device_ids` (operational/provenance): `[2]` → `[0]`
- `epochs` (approved matched-budget override): `10000` → `6000`
- `initialization` (operational/provenance): `None` → `'scratch'`
- `model_name` (operational/provenance): `None` → `'GL_rbf_ENH'`
- `save_dir` (operational/provenance): `'Save_TrainedModel/ffm_tc_pointcloud'` → `'Save_TrainedModel/ablation_condT/A5_local_sensor_tokens_only'`
- `scheduler_t_max` (operational/provenance): `None` → `10000`
- `sensor_attention_buckets` (operational/provenance): `None` → `[256, 320, 384]`
- `sensor_attention_padding_mode` (operational/provenance): `None` → `'full'`
- `training_mode` (operational/provenance): `None` → `'standard'`

## Result

PASS: protected fields match A0 except documented approved budget overrides.
