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

The command intentionally refuses to run with less than 30 GiB free unless explicitly overridden.

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

Launch only after the efficiency gate passes. These are direct Python launches; no wrapper script is required. Whether both fit concurrently on physical GPU 1 must be decided from the measured peak memory.

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
