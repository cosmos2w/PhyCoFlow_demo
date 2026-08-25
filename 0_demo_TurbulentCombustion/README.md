# Turbulent Combustion Demo

## Recommended PointCloudFFM interface

`GL_rbf_CQ` is the public name of the validated balanced model. Stage 7 freezes
its scientific architecture and release weights; Stage 8 makes cached K/V with
full sensor padding its preferred execution. The historical internal name
`GL_rbf_ENH_CQ` remains accepted so existing YAML files, imports, configs, and
checkpoints load without key translation.

Run these commands from `0_demo_TurbulentCombustion/` in the `phycoflow_env`
environment:

```bash
python src/train_pointcloud_ffm.py --config configs/gl_rbf_cq.yaml

python src/reconstruct_pointcloud.py \
  --config configs/gl_rbf_cq.yaml \
  --checkpoint ReleaseArtifacts/GL_rbf_CQ_rc1/GL_rbf_CQ_v0.9.0-rc1_e1000_ema_resolved_portable.pt

python src/evaluate_fixed_manifest.py \
  --config configs/gl_rbf_cq.yaml \
  --checkpoint ReleaseArtifacts/GL_rbf_CQ_rc1/GL_rbf_CQ_v0.9.0-rc1_e1000_ema_resolved_portable.pt
```

Use `--data`, `--dataset-stats-path`, `--save-dir`, `--device-ids`, or
`--set KEY=VALUE` with the training command to override portable defaults.
`--dry-run` validates identity, paths, and model shape without reading data or
starting training.

### Public profiles

| Profile | Purpose | Key settings |
|---|---|---|
| `configs/gl_rbf_cq.yaml` | recommended balanced model | latent 256; CQ 128/low-rank-64/additive; EMA; sinusoidal FiLM; measurement/support; cached K/V + full padding |
| `configs/gl_rbf_cq_fast.yaml` | lowest CQ training/inference cost | latent 128; CQ 128/low-rank-64/additive; historical scalar time; no EMA shortcut features; cached K/V + full padding |
| `configs/legacy_gl_rbf_enh.yaml` | F0 reproduction/checkpoint loading | latent 128; full enhanced GL-RBF query path; historical execution |

All three preserve K=32 Top-K RBF/GLRES semantics and default to the validated
optimized data path. The binary release checkpoint is intentionally ignored by
Git; obtain or generate it according to [the artifact manifest](artifacts/MANIFEST.md),
then verify it with:

```bash
python scripts/verify_release_artifacts.py
```

New CQ runs use:

```yaml
condition_attention_execution: cached_kv
sensor_attention_padding_mode: full
```

Set `condition_attention_execution: legacy_mha` only for historical
reproduction, numerical debugging, or compatibility diagnosis. It remains fully
supported. Static bucketing and dynamic trimming are not release defaults.

`src/train_pointcloud_ffm.py` is now the single canonical training command for
both public profiles and historical YAML files. The stable reusable Python
boundary remains `src/phycoflow_pointcloud/`; `Model.GL_rbf_ENH_CQ` and the
historical evaluation entry points remain compatibility interfaces. The
remainder of this README documents the detailed workflow and baselines.

### Checkpoints and cache lifecycle

Research checkpoints may contain live weights plus EMA state. Validation and
reconstruction select EMA when the checkpoint requests it and repair historical
EMA shadows with live frozen parameters/buffers. The portable release instead
contains one already-resolved `model` state and is inference-only; use the
research checkpoint to resume training.

Persistent Top-K geometry is valid only for the same query/sensor coordinate
tensors and observation mask (including shape, storage identity/version,
device, and dtype). Rebuild it whenever those change. Sensor values may change
while reusing geometry. At `static_features` cache level, the validated path
performs no new KNN search after geometry construction.

This demo currently contains two related but distinct pipelines:

- the main conditional point-cloud flow-matching workflow for turbulent combustion field reconstruction
- unified baseline pipelines for alternative generative and deterministic models

The previous `README.md` mostly described the baseline side. This version documents both.

## 1. Main Workflow: Point-Cloud Flow Matching

The primary reconstruction workflow is centered on these files:

- `src/Model.py`
- `src/train_pointcloud_ffm.py`
- `src/evaluate_ffm.py`
- `src/helpers.py`
- `Save_config/config_pointcloud_ffm.yaml`

At a high level, the main model learns to reconstruct full multi-field turbulent-combustion states from sparse sensor observations. The workflow is:

1. Load a turbulent-combustion snapshot from the HDF5 dataset.
2. Randomly build sparse observations from one or more conditioned physical fields.
3. Train a flow-matching model to predict the velocity field that transports a Gaussian prior sample toward the target physical state.
4. During evaluation, integrate the learned dynamics for a small number of sampling steps to reconstruct the full field.
5. Save reconstruction figures, JSON metrics, and optional extra spatial/spectral diagnostics.

### 1.1 What `src/Model.py` contains

`src/Model.py` is the core model definition file for the main workflow. It includes:

- prior models:
  - `IIDGaussianPrior`
  - `RFFGaussianPrior`
- point-cloud backbones:
  - `ConditionalPointMLPRBF`
  - `ConditionalPointPerceiver`
  - `ConditionalPointHybridLocalGlobalRBF`
- grid baseline backbone used inside the same FFM wrapper:
  - `FNO`
- flow-matching wrappers:
  - `PointCloudFFM`
  - `FNOFFM`

Conceptually, the main data path is:

1. A prior sample is generated on the query coordinates.
2. Sparse observations are encoded from `(obs_coords, obs_values, obs_field_ids)`.
3. The chosen backbone combines:
   - query-point information `(coords, x_t, t)`
   - sparse sensor information
4. The backbone predicts the instantaneous velocity field.
5. `PointCloudFFM` or `FNOFFM` uses that predictor for training loss computation and multi-step sampling.

### 1.2 Backbone choices in the main workflow

The `backbone` argument in `train_pointcloud_ffm.py` selects the model family:

- `mlp_rbf`: local RBF aggregation baseline on point clouds
- `perceiver`: latent cross-attention backbone for global interaction
- `GL_rbf`: hybrid local-global point-cloud backbone
- `GL_rbf_ENH`: enhanced hybrid local-global point-cloud backbone. It keeps the same RBF/top-K local gather choices as GL_rbf, but adds Senseiver-inspired coordinate tokenization, optional sensor-to-latent re-injection, query-to-latent readout, and normalized fusion.
- `GL_rbf_ENH_CQ`: Stage-6 compact-query sibling. It keeps the complete GL_rbf_ENH condition/global/local core while using a 128-wide point-state path, additive fusion, compact residual/coarse heads, and either the full latent readout (`cq_readout_mode=full`) or cached rank-64 lightweight readout (`lowrank`).
- `fno`: grid-based FNO baseline wrapped in the same flow-matching interface

So although the folder also has a separate "baseline pipeline", the main point-cloud workflow itself already supports several backbone variants through one training script.

### 1.3 What `src/train_pointcloud_ffm.py` does

`src/train_pointcloud_ffm.py` is the training entrypoint for the main workflow.

Its responsibilities are:

- parse CLI arguments and load overrides from `Save_config/config_pointcloud_ffm.yaml`
- normalize generalized conditioning arguments such as:
  - `cond_fields`
  - `n_obs_min_list`
  - `n_obs_max_list`
  - visualization-specific conditioning settings
- back up the resolved YAML config for reproducibility
- create train/validation datasets with `TurbulentCombustionH5Dataset`
- randomly build sparse observations with `build_sparse_condition(...)`
- optionally subsample query points during training for memory/computation savings
- instantiate the selected backbone and wrap it in `PointCloudFFM` or `FNOFFM`
- run training/validation epochs
- save `best.pt`, logging CSVs, `args.json`, normalization stats, and reconstruction previews

Important training details visible in the script:

- multiple conditioned fields are supported, not just one field
- each conditioned field can have its own sensor-count range
- visualization conditioning can be different from training conditioning
- `fno` requires explicit regular-grid dimensions `Num_x` and `Num_y`
- `RELOAD` resumes from the latest matching saved run

### 1.4 What `src/evaluate_ffm.py` does

`src/evaluate_ffm.py` is the standalone evaluator for a trained point-cloud FFM run.

Its responsibilities are:

- recover the latest backed-up YAML config for a given `Demo_Num`
- reconstruct the matching model directory and checkpoint
- rebuild the model architecture from the saved config
- load either `best.pt` or `last.pt`
- run reconstruction on the requested split and snapshot
- call `visualize_reconstruction(...)` to save figures and core metrics
- optionally compute extra structured-grid analysis:
  - `ssim`
  - gradient-based metrics
  - spectral metrics
- optionally save `.npz` analysis payloads for offline inspection

This script is especially useful after training because it can:

- test different sampling step counts with `--n-steps-generation`
- change the evaluated split without retraining
- override visualization conditioning settings at evaluation time

### 1.5 Main workflow outputs

The main point-cloud workflow writes artifacts to several places:

```text
Save_TrainedModel/ffm_tc_pointcloud_DemoN<id>_<timestamp>/
```

Inside each run directory, training stores checkpoints, logs, normalization
stats, and reconstruction previews together:

Typical files include:

- `best.pt`
- `last.pt`
- `args.json`
- `run_config.yaml`
- `dataset_stats.pt`
- `loss_history.csv`
- `loss_history.json`
- `loss_history.png`
- `Evaluation/epoch_XXXX/`

Backed-up launch configs are also stored under:

- `Save_config/pointcloud_ffm/`

The older shared `Save_loss_csv/` and `Save_reconstruction_files/` roots are no
longer used by `src/train_pointcloud_ffm.py` for new training runs.

### 1.6 Main workflow example commands

Run commands below from the repository root:

```bash
cd /home/wanglz/Desktop/src/PhyCoFlow
```

Train the main point-cloud FFM model:

```bash
python src/train_pointcloud_ffm.py \
  --config Save_config/config_pointcloud_ffm.yaml \
  --Demo-Num 0
```

Train with a specific backbone:

```bash
python src/train_pointcloud_ffm.py \
  --config Save_config/config_pointcloud_ffm.yaml \
  --Demo-Num 0 \
  --backbone perceiver
```

Train the enhanced GL_rbf backbone:

```bash
python src/train_pointcloud_ffm.py \
  --config Save_config/config_pointcloud_ffm.yaml \
  --Demo-Num 10 \
  --backbone GL_rbf_ENH \
  --gather-mode topk_rbf
```

Use `gather_mode=topk_rbf` for the clean enhanced local-global model. Use `gather_mode=topk_rbf_glres` for the strongest residual-enhanced variant.

Resume the latest matching run:

```bash
python src/train_pointcloud_ffm.py \
  --config Save_config/config_pointcloud_ffm.yaml \
  --Demo-Num 0 \
  --RELOAD
```

### 1.7 Direct Coherence Post-Training

Standard point-cloud FFM training still uses the ordinary rectified-flow data
loss path:

```bash
python src/train_pointcloud_ffm.py \
  --config Save_config/config_pointcloud_ffm.yaml \
  --Demo-Num 19
```

Direct coherence post-training uses the same training entrypoint, but switches
to a scheduled differentiable clean rollout before applying global coherence
losses:

```bash
python src/train_pointcloud_ffm.py \
  --config Save_config/config_pointcloud_ffm_direct_posttrain.yaml \
  --Demo-Num 40
```

In direct post-training, the model first computes the usual RF velocity MSE on
the normal query subset. On scheduled steps, it also differentiably rolls out a
clean sample and compares its empirical field distribution to the reference
with differentiable self, mutual, and cross coherence terms. `coherence_every_n_steps`
reduces the expense by running that rollout only every N optimizer steps.
Post-training writes a live monitor next to the usual `loss_history.*` files:
`direct_coherence_history.csv`, `direct_coherence_history.json`, and
`direct_coherence_history.png`. The PNG is refreshed every epoch and shows
total/data/coherence losses plus the self/mutual/cross physical coherence
components.

Gradient balancing can use either `weighted_sum` or `config`. ConFIG follows
the separate-gradient conflict-free update pattern and requires the optional
package:

```bash
pip install conflictfree
```

This direct method requires differentiable coherence terms because gradients
flow through the terminal rollout and Wasserstein sorting/top-k operations.
That differs from RAM, where detached scalar coherence rewards are used for
posterior-style policy fine-tuning.

Evaluate a trained model:

```bash
python src/evaluate_ffm.py \
  --Demo-Num 0 \
  --split test \
  --snapshot-index 0 \
  --checkpoint best
```

Evaluate with extra diagnostics:

```bash
python src/evaluate_ffm.py \
  --Demo-Num 0 \
  --split test \
  --snapshot-index 0 \
  --extra-metrics ssim grad spectrum \
  --save-analysis-npz
```

## RAM Fine-Tuning

RAM, or Reinforce Adjoint Matching, is a post-training fine-tune path for an
existing PointCloudFFM checkpoint. It keeps the rectified-flow convention used
by the base model:

```text
x_t = (1 - t) * z + t * x
target velocity = x - z
```

Run RAM fine-tuning from `0_demo_TurbulentCombustion/`:

```bash
python src/train_finetune.py \
  --config Save_config/config_pointcloud_ffm_ram.yaml \
  --Demo-Num 20
```

RAM has two implementation modes:

- Full-copy RAM keeps separate `ref_model`, `policy_model`, `old_model`, and
  optional `eval_model` roles. This preserves the original reference behavior
  and supports `finetune_mode: head_glres` or `finetune_mode: all`.
- LoRA RAM keeps one pretrained GL_rbf model and attaches three adapters:
  `default` for the trainable policy, `old` for endpoint sampling and old
  velocity targets, and `evaluation` for validation/checkpoint export. Use
  `finetune_mode: lora_head_glres` as the recommended default for
  memory-efficient GL_rbf/topk_rbf_glres fine-tuning. Use
  `finetune_mode: lora_all_linear_glrbf` to adapt all linear layers under the
  GL_rbf model. The LoRA target scope is derived from `finetune_mode`;
  `lora_target_scope` is optional and must be `null`, `auto`, or match the
  selected mode.

The RAM algorithm is unchanged in both modes: endpoints are sampled, scalar
coherence rewards produce group-relative advantages, endpoints are analytically
re-noised, and the policy is trained against a detached velocity MSE target.
No reward gradients, SDE rollouts, or adjoint sweeps are used.

RAM separates the raw coherence cost from the reward shaping used for
fine-tuning. `reward_mode` chooses the lower-is-better raw cost, such as
`global_dist`, `marginal_only`, or `field_l2`. `reward_transform` then maps
that cost into scalar rewards before the usual group-relative advantage step.
`negative_cost` is the historical smooth posterior-tilting behavior. For
stronger posterior-pruning behavior, use scale-free or thresholded transforms
such as `group_rank`, `top_bottom`, or `softplus_barrier`.

RAM has additional memory controls beyond the base trainer. Endpoint sampling
still produces full fields, but reward/coherence and velocity matching can use
separate point subsets:

- `ram_n_query_points`: number of query points used for the RAM velocity loss,
  analogous to base `n_query_points`.
- `ram_reward_n_points`: optional uniform point subset used only for scalar
  reward/coherence. `null` keeps full-grid reward evaluation.
- `fixed_reward_points_for_eval` / `fixed_reward_points_for_rollout`: reuse a
  deterministic reward/coherence point subset for comparable validation and
  rollout curves. Rollout caches the indices under `Rollout/` unless
  `rollout_reward_point_path` is set.
- `align_ram_and_rollout_obs_consistency`: when `true`, RAM endpoint sampling
  uses the same sparse-observation consistency mode as rollout monitoring.
- `train_ratio_downsample`: fresh random fraction of the training split used in
  each RAM epoch; validation and test sets are unchanged.
- `ram_query_sampling`: query selection mode, usually `obs_mix` to match base
  PointCloudFFM training.
- `ram_endpoint_microbatch_size`: splits old-policy full-field endpoint
  rollouts along the repeated condition batch.
- `ram_loss_microbatch_size`: splits RAM velocity matching and accumulates
  gradients across chunks.
- `global_include_pairwise`: pairwise coherence is more expensive; keep it
  `false` for first formal RAM runs and enable it later if needed.

The default RAM config uses `lora_all_linear_glrbf`, `batch_size: 32`,
`num_samples_per_condition: 8`, `num_loss_targets_per_endpoint: 2`,
`ram_n_query_points: 1024`, and `ram_reward_n_points: 4096`. If memory is still
very low, reduce `ram_endpoint_microbatch_size`, `ram_loss_microbatch_size`, or
the reward/query point counts first before changing the RAM batch structure.

Evaluate a RAM-finetuned checkpoint with the same evaluator:

```bash
python src/evaluate_ffm.py \
  --run-dir Save_TrainedModel/ram_tc_pointcloud_DemoN20_<timestamp> \
  --split test \
  --snapshot-index 0 \
  --checkpoint best \
  --n-steps-generation 4 \
  --obs-consistency-mode endpoint_smooth \
  --extra-metrics ssim grad spectrum \
  --save-analysis-npz
```

RAM outputs are stored under:

```text
Save_TrainedModel/ram_tc_pointcloud_DemoN<id>_<timestamp>/
```

By default RAM metric histories are saved once in CSV form:
`loss_history.csv`, `ram_metrics.csv`, and `rollout_metrics.csv`.
`loss_history.png` plots only the RAM velocity-matching objective, while
`validation_history.png` plots the separate validation rollout score. Set
`save_history_json: true` if JSON copies of the history tables are also needed.

For full-copy RAM, the `model` entry in `best.pt` and `last.pt` is the
evaluation EMA model. For LoRA RAM, `model` is a normal merged state dict with
the `evaluation` adapter folded into the original architecture, while
`lora_state` stores the adapter payload separately. In both cases,
`src/evaluate_ffm.py` loads RAM checkpoints without special adapter runtime
logic.

Train LoRA RAM:

```bash
python src/train_finetune.py \
  --config Save_config/config_pointcloud_ffm_ram.yaml \
  --Demo-Num 30
```

Evaluate:

```bash
python src/evaluate_ffm.py \
  --run-dir Save_TrainedModel/ram_tc_pointcloud_DemoN30_<timestamp> \
  --split test \
  --snapshot-index 0 \
  --checkpoint best \
  --n-steps-generation 4 \
  --obs-consistency-mode endpoint_smooth \
  --extra-metrics ssim grad spectrum \
  --save-analysis-npz
```

## 2. Unified Generative Baselines

In addition to the main point-cloud FFM workflow, this directory also contains a separate unified pipeline for several generative baselines:

- `s3gm`
- `latent_fm`
- `sit`

The baseline pipeline is mainly organized around:

- `Save_config/config_baseline_Gen.yaml`
- `src/model_baseline.py`
- `src/train_Gen_Baseline.py`
- `src/evaluate_Gen_Baseline.py`
- `src/helpers_baseline.py`

The goal of this baseline pipeline is different from the main workflow: it provides a common configuration and run structure so baseline switching becomes mostly a config change.

### 2.1 What `src/model_baseline.py` contains

`src/model_baseline.py` consolidates the model implementations and utilities for the baseline experiments. It includes:

- baseline model definitions
- adapters and wrappers
- checkpoint and run utilities
- sampling utilities
- visualization helpers used by the baseline evaluator

It is the baseline counterpart to the main workflow's `src/Model.py`.

### 2.2 Baseline families

The unified baseline config supports:

- `s3gm`: score-based sparse-sensing generative model
- `latent_fm`: two-stage latent flow matching
- `sit`: Scalable Interpolant Transformer

The main selectors are:

- `baseline_model`
- `training_stage`

Stage behavior:

- `s3gm`: only Stage 1 is used
- `sit`: only Stage 1 is used
- `latent_fm`:
  - Stage 1 trains the autoencoder
  - Stage 2 trains the latent flow model on top of the Stage 1 encoder

### 2.3 Baseline workflow files

- `src/train_Gen_Baseline.py`: unified baseline training launcher
- `src/evaluate_Gen_Baseline.py`: unified baseline evaluation launcher
- `src/helpers_baseline.py`: baseline dataset loading, sparse conditioning, plotting, and shared utilities
- `src/sit_transport/`: transport and ODE/SDE helpers for the SiT baseline

### 2.4 Baseline outputs

Baseline runs are saved under:

```text
Save_TrainedModel/Baseline_<baseline>_Stage<stage>_DemoN<demo_num>_<timestamp>/
```

Typical contents:

- `best.pt`
- `last.pt`
- `run_config.yaml`
- `run_metadata.json`
- `final_summary.json`
- `loss_history.csv`
- `loss_history.json`
- `dataset_stats.pt`
- `Evaluation/`

Config backups are stored under:

- `Save_config/gen_baseline/`

### 2.5 Baseline example commands

Train S3GM:

```bash
python src/train_Gen_Baseline.py \
  --config Save_config/config_baseline_Gen.yaml \
  --baseline-model s3gm \
  --training-stage 1 \
  --device cuda:1
```

Train latent FM Stage 1:

```bash
python src/train_Gen_Baseline.py \
  --config Save_config/config_baseline_Gen.yaml \
  --baseline-model latent_fm \
  --training-stage 1
```

Train latent FM Stage 2:

```bash
python src/train_Gen_Baseline.py \
  --config Save_config/config_baseline_Gen.yaml \
  --baseline-model latent_fm \
  --training-stage 2
```

Train SiT:

```bash
python src/train_Gen_Baseline.py \
  --config Save_config/config_baseline_Gen.yaml \
  --baseline-model sit \
  --training-stage 1
```

Evaluate the latest matching baseline run:

```bash
python src/evaluate_Gen_Baseline.py \
  --config Save_config/config_baseline_Gen.yaml \
  --baseline-model s3gm \
  --training-stage 1
```

## 3. Unified Deterministic Baselines

The deterministic baseline workflow uses the same dataset loading, sparse sensor
conditioning, logging, checkpoint layout, and reconstruction plotting conventions
as the unified generative baseline workflow, but trains direct supervised
regressors instead of generative samplers.

The deterministic baseline files are:

- `Save_config/config_baseline_Det.yaml`
- `src/train_Det_Baseline.py`
- `src/evaluate_Det_Baseline.py`
- `src/model_baseline.py`
- `src/helpers_baseline.py`

### 3.1 Deterministic baseline families

The config selector is:

- `baseline_model`

Supported deterministic values are:

- `mlp_rbf`: the original point-cloud FFM MLP-RBF sparse-gather backbone used in direct supervised mode
- `senseiver`: Perceiver-IO / Senseiver sparse-sensor reconstruction model
- `geofno`: supervised Geo-FNO / neuraloperator FNO sparse-to-full regressor

All deterministic baselines currently use:

- `training_stage: 1`
- direct MSE supervised training
- the unified run naming pattern:

```text
Save_TrainedModel/Baseline_<baseline>_Stage1_DemoN<demo_num>_<timestamp>/
```

### 3.2 Default comparison settings

`Save_config/config_baseline_Det.yaml` is set up to match the generative
baseline defaults for fair comparison:

- default conditioning is temperature-only:
  - `cond_fields: [2]`
  - `n_obs_min_list: [192]`
  - `n_obs_max_list: [384]`
  - `vis_n_obs_list: [256]`
- 10,000 epochs by default
- `eval_every: 5`
- `save_every: 500`
- grid size:
  - `num_x: 403`
  - `num_y: 100`

The config comments also include launch variants for `T + U_1` and
`CO + T + U_1 + p`, matching the generative baseline template.

The default batch sizes are chosen for a 48 GB GPU:

- `mlp_rbf`: `batch_size: 128`
- `senseiver`: `batch_size: 64`
- `geofno`: `batch_size: 96`

For four-field conditioning, reduce `mlp_rbf` and `senseiver` batch sizes if
you hit memory limits. `geofno` processes full grids rather than subsampled
query points, so its memory profile is different from the point-cloud models.

### 3.3 Deterministic model notes

`mlp_rbf` reuses the existing `ConditionalPointMLPRBF` block from the FFM
model family, but wraps it for direct deterministic regression. Internally it
passes `t=0` and a zero field state to the original backbone, so the sparse
conditioning API remains compatible with the main FFM model.

`senseiver` uses sparse sensor tokens and query coordinates directly. It
subsamples query points during training via `n_query_points` and reconstructs
the full field during visualization/evaluation.

`geofno` uses `neuraloperator`. In this workspace, `phycoflow_env` has
`neuralop` available; if another environment is used, install
`neuraloperator` or use the environment that already provides it.

The default Geo-FNO spectral modes are aligned with the current FFM-FNO launch:

- `fno_modes_x: 24`
- `fno_modes_y: 12`

### 3.4 Deterministic example commands

Run commands below from `0_demo_TurbulentCombustion/`.

Train MLP-RBF deterministic baseline:

```bash
python src/train_Det_Baseline.py \
  --config Save_config/config_baseline_Det.yaml \
  --baseline-model mlp_rbf \
  --training-stage 1 \
  --device cuda:1
```

Train Senseiver:

```bash
python src/train_Det_Baseline.py \
  --config Save_config/config_baseline_Det.yaml \
  --baseline-model senseiver \
  --training-stage 1 \
  --device cuda:1
```

Train Geo-FNO:

```bash
python src/train_Det_Baseline.py \
  --config Save_config/config_baseline_Det.yaml \
  --baseline-model geofno \
  --training-stage 1 \
  --device cuda:1
```

Resume the latest matching deterministic run:

```bash
python src/train_Det_Baseline.py \
  --config Save_config/config_baseline_Det.yaml \
  --baseline-model mlp_rbf \
  --training-stage 1 \
  --reload
```

Evaluate the latest matching deterministic run:

```bash
python src/evaluate_Det_Baseline.py \
  --config Save_config/config_baseline_Det.yaml \
  --baseline-model mlp_rbf \
  --training-stage 1 \
  --split test \
  --snapshot-index 0
```

Override visualization sensors at evaluation time:

```bash
python src/evaluate_Det_Baseline.py \
  --config Save_config/config_baseline_Det.yaml \
  --baseline-model senseiver \
  --training-stage 1 \
  --vis-cond-fields 2 3 \
  --vis-n-obs-list 256 256
```

## 4. Relationship Between the Pipelines

To avoid confusion:

- `src/Model.py` + `train_pointcloud_ffm.py` + `evaluate_ffm.py` describe the main point-cloud FFM workflow used for sparse-conditioned reconstruction
- `src/model_baseline.py` + `train_Gen_Baseline.py` + `evaluate_Gen_Baseline.py` describe the unified generative baseline benchmarking workflow
- `src/model_baseline.py` + `train_Det_Baseline.py` + `evaluate_Det_Baseline.py` describe the unified deterministic baseline benchmarking workflow

They live in the same demo folder because they use the same turbulent combustion dataset and related sparse-conditioning ideas, but they are not the same training/evaluation stack.

## 5. Recommended Starting Point

If you want to understand the primary workflow of this demo, read the files in this order:

1. `src/train_pointcloud_ffm.py`
2. `src/Model.py`
3. `src/evaluate_ffm.py`
4. `src/helpers.py`

If you want to compare against alternative baselines, then move to:

1. `src/train_Gen_Baseline.py`
2. `src/model_baseline.py`
3. `src/evaluate_Gen_Baseline.py`
4. `src/helpers_baseline.py`

For deterministic baseline comparisons, read:

1. `src/train_Det_Baseline.py`
2. `Save_config/config_baseline_Det.yaml`
3. `src/model_baseline.py`
4. `src/evaluate_Det_Baseline.py`
