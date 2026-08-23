# Stage-7 commands

Run from the Stage-7 project worktree:

```text
/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/.worktrees/pointcloud-smart-cq-stage7/0_demo_TurbulentCombustion
```

## Validate configs

```bash
conda run -n phycoflow_env python \
  _CheckNotes/Stage7_smart_cq/configs/validate_configs.py
```

## Pre-training benchmark on physical GPU 1

The command intentionally refuses to run with less than 30 GiB free unless explicitly overridden. Its formal Stage-7 rows use the configured 2048-query exact-gradient execution microbatch, and persistent 1M-query inference uses a shared 32768-query chunk for all candidates.

```bash
CUDA_VISIBLE_DEVICES=1 KEOPS_CACHE_FOLDER=/tmp/keops_stage7_benchmark_gpu1 \
conda run --no-capture-output -n phycoflow_env \
python src/benchmark_pointcloud_stage7.py \
  --device cuda:0 \
  --f0-config _CheckNotes/Stage6_formal_baseline/F0_frozen_current.yaml \
  --f0-checkpoint /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_formal_baseline/runs/F0_frozen_current_DemoN9300_20260821_075633/best.pt \
  --cq-config _CheckNotes/Stage6_clean_ab/CQ_LR_1000ep_b128.yaml \
  --output _CheckNotes/Stage7_smart_cq/benchmarks/pretraining_cost.json
```

## Formal screens

The efficiency gates passed and both screens fit concurrently on physical GPU 1. These remain direct Python launches; no repository wrapper script is required.

S7-A:

```bash
CUDA_VISIBLE_DEVICES=1 KEOPS_CACHE_FOLDER=/tmp/keops_stage7_s7a_gpu1 \
conda run --no-capture-output -n phycoflow_env \
python src/train_pointcloud_ffm.py \
  --config _CheckNotes/Stage7_smart_cq/configs/S7_A_Cond128_200ep_b128.yaml
```

S7-B:

```bash
CUDA_VISIBLE_DEVICES=1 KEOPS_CACHE_FOLDER=/tmp/keops_stage7_s7b_gpu1 \
conda run --no-capture-output -n phycoflow_env \
python src/train_pointcloud_ffm.py \
  --config _CheckNotes/Stage7_smart_cq/configs/S7_B_All256_200ep_b128.yaml
```

For background execution, append distinct output redirections and `&` to the two commands. Do not launch isolated EMA-only, FiLM-only, measurement-only, or latent256-only runs.

## Selected continuation

S7-B was the clear epoch-200 winner. Resume the same run in place on physical
GPU 1 with the direct Python command:

```bash
CUDA_VISIBLE_DEVICES=1 KEOPS_CACHE_FOLDER=/tmp/keops_stage7_s7b_continue_gpu1 \
conda run --no-capture-output -n phycoflow_env \
python src/train_pointcloud_ffm.py \
  --config _CheckNotes/Stage7_smart_cq/configs/S7_B_All256_1000ep_resume.yaml
```

## Final fixed-manifest evaluation

The final command evaluates S7-B milestones and best with exact live frozen
state plus EMA trainable weights:

```bash
CUDA_VISIBLE_DEVICES=1 KEOPS_CACHE_FOLDER=/tmp/keops_stage7_eval_final_gpu1 \
PYTHONPATH=src conda run --no-capture-output -n phycoflow_env \
python src/evaluate_pointcloud_fixed_manifest.py \
  --config _CheckNotes/Stage7_smart_cq/screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/run_config.yaml \
  --manifest /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint _CheckNotes/Stage7_smart_cq/screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/epoch_0400.pt \
    _CheckNotes/Stage7_smart_cq/screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/epoch_0600.pt \
    _CheckNotes/Stage7_smart_cq/screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/epoch_0800.pt \
    _CheckNotes/Stage7_smart_cq/screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/epoch_1000.pt \
    _CheckNotes/Stage7_smart_cq/screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/best.pt \
  --device cuda:0 --batch-size 1 --repeats 3 --rf-seed 1729 \
  --output _CheckNotes/Stage7_smart_cq/evaluation_1000/S7_B_milestones_and_best.json
```

## Separate kernel benchmark

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
conda run --no-capture-output -n phycoflow_env \
python _CheckNotes/Stage7_smart_cq/benchmark_attention_kernels.py \
  --config _CheckNotes/Stage7_smart_cq/screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/run_config.yaml \
  --checkpoint _CheckNotes/Stage7_smart_cq/screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/epoch_1000.pt \
  --device cuda:0 --batch-size 128 --query-size 4096 \
  --query-microbatch-size 2048 --n-obs 256 --warmup 2 --iterations 7 \
  --output _CheckNotes/Stage7_smart_cq/benchmarks/attention_kernel_comparison.json
```

## Final analysis and figure

```bash
conda run --no-capture-output -n fig \
python _CheckNotes/Stage7_smart_cq/evaluation_1000/analyze_stage7_final.py
conda run --no-capture-output -n fig \
python figures/scripts/plot_stage7_final_pareto.py
```
