# Stage-7 epoch-200 evaluation commands

Run these only after S7-A and S7-B have produced `epoch_0200.pt`. Replace `<S7_A_RUN>` and `<S7_B_RUN>` with their timestamped run directories.

All fixed-manifest evaluations use the clean reference manifest checksum, batch 1, three repeats, and RF seed 1729. Stage-7 checkpoint loading automatically selects EMA weights because `model_ema_eval: true` is recorded in the checkpoint.

## S7-A fixed manifest

```bash
CUDA_VISIBLE_DEVICES=1 KEOPS_CACHE_FOLDER=/tmp/keops_stage7_eval_s7a_gpu1 \
conda run --no-capture-output -n phycoflow_env \
python src/evaluate_pointcloud_fixed_manifest.py \
  --config <S7_A_RUN>/run_config.yaml \
  --manifest /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint <S7_A_RUN>/epoch_0001.pt <S7_A_RUN>/epoch_0020.pt \
    <S7_A_RUN>/epoch_0040.pt <S7_A_RUN>/epoch_0060.pt \
    <S7_A_RUN>/epoch_0100.pt <S7_A_RUN>/epoch_0150.pt \
    <S7_A_RUN>/epoch_0200.pt \
  --device cuda:0 --batch-size 1 --repeats 3 --rf-seed 1729 \
  --output _CheckNotes/Stage7_smart_cq/screen_200/evaluation/S7_A_fixed_manifest.json
```

## S7-B fixed manifest

```bash
CUDA_VISIBLE_DEVICES=1 KEOPS_CACHE_FOLDER=/tmp/keops_stage7_eval_s7b_gpu1 \
conda run --no-capture-output -n phycoflow_env \
python src/evaluate_pointcloud_fixed_manifest.py \
  --config <S7_B_RUN>/run_config.yaml \
  --manifest /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint <S7_B_RUN>/epoch_0001.pt <S7_B_RUN>/epoch_0020.pt \
    <S7_B_RUN>/epoch_0040.pt <S7_B_RUN>/epoch_0060.pt \
    <S7_B_RUN>/epoch_0100.pt <S7_B_RUN>/epoch_0150.pt \
    <S7_B_RUN>/epoch_0200.pt \
  --device cuda:0 --batch-size 1 --repeats 3 --rf-seed 1729 \
  --output _CheckNotes/Stage7_smart_cq/screen_200/evaluation/S7_B_fixed_manifest.json
```

## Existing latent-256 reference

The clean latent-256 CQ run already contains `epoch_0200.pt`; do not retrain it.

```bash
CUDA_VISIBLE_DEVICES=1 KEOPS_CACHE_FOLDER=/tmp/keops_stage7_eval_cql256_gpu1 \
conda run --no-capture-output -n phycoflow_env \
python src/evaluate_pointcloud_fixed_manifest.py \
  --config /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_clean_ab/runs/CQ_LR_L256_1K_B128_DemoN9561_20260822_144624/run_config.yaml \
  --manifest /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_clean_ab/runs/CQ_LR_L256_1K_B128_DemoN9561_20260822_144624/epoch_0200.pt \
  --device cuda:0 --batch-size 1 --repeats 3 --rf-seed 1729 \
  --output _CheckNotes/Stage7_smart_cq/screen_200/evaluation/CQ_LR_L256_fixed_manifest.json
```

## Reconstruction

The training configs generate matched snapshot-0 Euler NFE1/2/4 reconstruction at epoch 200 using EMA weights. Reference F0/CQ-LR-128 epoch-200 reconstruction must be generated with the same snapshot, observation layout, consistency behavior, and RF seed before cross-model reconstruction claims are made. Do not compare the existing epoch-1000 images to Stage-7 epoch-200 outputs.
