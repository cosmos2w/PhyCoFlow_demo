# MIMONet combustion baselines

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

## Additional saved conditions

`src/train_mimonet_multicond.py` trains two further 5,000-epoch comparisons with the same released branch/trunk architecture, 256-dimensional basis, five outputs, Adam settings, checkpoint cadence, default loss plot, and every-500-epoch five-field reconstruction diagnosis:

| Condition | Branch sensors per state | Training query sampling | Unobserved fields |
| --- | --- | --- | --- |
| `Cond_TU1` | 192–384 T and 192–384 U_1 | A0 `obs_mix` | CH4, CO, p |
| `Cond_COTU1P` | 192–384 each of CO, T, U_1, p | A0 `uniform` | CH4 |

The sensor value and normalized coordinate branches give each conditioned field a fixed 384-slot segment. The A0 sampler still selects independent random sensors for each field; the adapter only moves the padded observations into stable segments. No unobserved field enters either branch. Both configurations use their own saved DMF-Gen reference config and statistics. Their train and holdout split is the same 9,000/1,000 split as Cond_T. Training query sampling follows each reference config; validation uses uniform queries as in A0. Every 500 epochs, the archived 256-sensors-per-field paper plan is used for a single holdout snapshot reconstruction, with A0-style plots and normalized metrics plus physical relative-L2 and field-range diagnostics. These held-out diagnostics do not change training settings.

Run from `0_demo_TurbulentCombustion/`:

```bash
conda run -n phycoflow_env --no-capture-output python -u src/train_mimonet_multicond.py --config Save_config/mimonet/config_MIMONet_condTU1.yaml --smoke
conda run -n phycoflow_env --no-capture-output python -u src/train_mimonet_multicond.py --config Save_config/mimonet/config_MIMONet_condCOTU1P.yaml --smoke
conda run -n phycoflow_env --no-capture-output python -u src/train_mimonet_multicond.py --config Save_config/mimonet/config_MIMONet_condTU1.yaml
conda run -n phycoflow_env --no-capture-output python -u src/train_mimonet_multicond.py --config Save_config/mimonet/config_MIMONet_condCOTU1P.yaml
```

The output roots are `Save_TrainedModel/deterministic_baselines/MIMONet_condTU1/` and `Save_TrainedModel/deterministic_baselines/MIMONet_condCOTU1P/`. Their output folders contain configuration, source hashes, reference hashes, exact launch command, model summary, histories, best/latest/milestone checkpoints, loss plots, and `Evaluation/epoch_XXXX/` diagnoses. The normalized five-field MSE is a necessary deviation from A0's rectified-flow loss because MIMONet directly predicts fields. No hyperparameter sweep or uncertainty module is used.
