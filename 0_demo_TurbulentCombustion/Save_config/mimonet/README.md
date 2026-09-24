# MIMONet Cond_T baseline

This is one deterministic, 5,000-epoch MIMONet comparison to the saved A0 DMF-Gen run. The upstream MIMONet branch–trunk network is in `src/mimonet_upstream/`; `src/train_mimonet_condT.py` only prepares combustion tensors and runs training.

- Data: the A0 HDF5 file, seed 42, time stride 1, and exact 90%/10% shuffled snapshot split. The A0 code names the same 1,000-snapshot holdout `val` and `test`; there is no separate third split.
- Inputs: one random set of 192–384 temperature observations per training state, using the A0 `build_sparse_condition` sampler. Two branches receive paired standardized T values with a validity mask and paired normalized 3D locations with the same mask. The sorted sensor order and padding retain the value/location pairing. No other field or time value enters the branches.
- Targets: standardized CH4, CO, T, U1, and p at 4,096 uniformly sampled query points per state. The trunk receives the exact A0 normalized coordinates. The loss is five-channel normalized MSE; physical values use the A0 training mean and standard deviation.
- Model: released MIMONet `mul` branch fusion, ReLU FCNs, basis 256, branch hidden width 512, trunk hidden width 256. The only dimension changes are branch input dimensions for variable sensor values/locations and five output fields.
- Training: Adam, initial learning rate 1e-4, weight decay 1e-6, cosine annealing over 5,000 epochs, batch 128, full 9,000-snapshot training exposure per epoch, and validation every five epochs (also epoch 1). The optimizer, schedule, batch and validation cadence match A0 where applicable. MSE replaces the A0 rectified-flow loss because MIMONet is a direct deterministic regressor.
- Validation: full 1,000-snapshot holdout MSE for checkpoint selection. Startup diagnostics at epochs 1, 10, 25, 50 use four fixed holdout snapshots with 256 T sensors and full-mesh physical relative-L2. These diagnostics are preliminary and never select model settings.
- Loss figure: `loss_history.png` is saved by default after epoch 1 and every 50 epochs. Editable SVG and PDF plus a PNG preview and figure contract are written under `figures/generated/mimonet_condT_loss/<run_name>/` by `figures/scripts/plot_mimonet_loss.py` in the `fig` environment.
- Reconstruction diagnosis: at every 500 epochs, the live epoch model reconstructs A0's held-out snapshot 0 with the archived 256-T-sensor plan. `Evaluation/epoch_XXXX/` gets A0-style truth/reconstruction/absolute-error PNGs for all five fields, `mimonet_metrics.json` with the A0 normalized L2 and sensor-consistency convention, and `quality_diagnosis.json` with physical relative-L2, ranges, spatial variation and physical-boundary checks. Deterministic MIMONet has one reconstruction per epoch; the A0 `euler_nfe2/4/8` variants do not apply.
- Post-training evaluation: `src/evaluate_mimonet_condT.py` uses the archived A0 paper sensor plan (256 T sensors per held-out snapshot), hard-clamps observed T exactly as A0 does, and computes per-snapshot physical relative-L2 per field and the mean across CH4, CO, U1 and p. Do not use held-out metrics for tuning.

Run from `0_demo_TurbulentCombustion/`:

```bash
conda run -n phycoflow_env --no-capture-output python -u src/train_mimonet_condT.py --config Save_config/mimonet/config_MIMONet_condT.yaml --smoke
conda run -n phycoflow_env --no-capture-output python -u src/train_mimonet_condT.py --config Save_config/mimonet/config_MIMONet_condT.yaml
```

The smoke path writes no result artifacts. The main run writes its configuration, metadata, model summary, histories, loss figure, recurring reconstruction evaluation, and `best.pt`/`last.pt` under `Save_TrainedModel/deterministic_baselines/MIMONet_condT/`.
