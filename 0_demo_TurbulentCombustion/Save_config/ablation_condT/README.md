# DMF-Gen Cond_T ablations A1–A5

These are complete launch configs for the adopted multi-field Cond_T experiment. They do not inherit from the newer generic/CQ configs.

## Authoritative A0

- Config: `Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/run_config.yaml`
- Config SHA-256: `e24de2d9b8909daa520b7b390a4a62396832ceb6e6810cd67f90673d9cf2f6c1`
- Selected checkpoint: `Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/best.pt`
- Best-checkpoint SHA-256: `af43f763fa91b2204a9413c650bd329bb01cfaeea64af325315339d64e2258a6`
- Last checkpoint: `Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/last.pt`
- Last-checkpoint SHA-256: `857a505ff96cc3512c20f45641250d9db9d448c1f1e33b8c4a1c792acb1e4a06`
- Dataset-stat SHA-256: `87ffa529644766ac85d0d58869517b2a9062afeb0a10c450f9f25fff4a8f03b3`

The intended A0 schedule is 10,000 epochs with a 10,000-epoch cosine horizon. The archived best checkpoint is epoch 5,365 / update 380,915. The archived last checkpoint is epoch 6,005 / update 426,355 (71 optimizer updates per epoch), with a learning rate consistent with the 10,000-epoch cosine schedule. The archived checkpoint predates scheduler serialization, but the saved LR corroborates that horizon. Validation occurred at epoch 1 and every 5 epochs; `best.pt` is the minimum validation-loss live-weight checkpoint. No EMA metadata/state is present.

Important comparability warning: A0 did not reach the configured 10,000 epochs. A2, A3, and A5 currently use the user-selected reduced stopping budget of 6,000 epochs while retaining A0's 10,000-epoch cosine horizon; A1 and A4 remain configured for 10,000 epochs. The reduced group is close to, but five epochs shorter than, archived A0's last checkpoint at epoch 6,005. Results across the two budget groups are not strictly causal comparisons unless the stopping policy is harmonized before interpreting outcomes.

## Scientific interventions

- A1, `deterministic_same_backbone`: retains the full GL_rbf_ENH conditioning backbone and predicts the normalized five-field target directly from zero query state at `tau=0`, using one MSE-supervised evaluation. It neither stores nor calls a generative prior. Its inference is deterministic. If evaluation requests observation consistency and supplies point indices, the same final hard observation clamp is applied; use the same policy for A0 and A1.
- A2, `no_sensor_global_feedback`: `_refine_sensor_tokens` returns the original masked sensor tokens. `sensor_back_attn` remains instantiated with identical keys/shapes but is not executed. Sensor-to-latent attention, latent reinjection/blocks, `sensor_out_proj`, global/query readout, local top-K/RBF gathering, final head, RFF prior, and RF loss remain active.
- A3, `no_local_query_conditioning`: the direct query-local feature is an exact `[B,N,cond_dim]` zero tensor. `aggregate_sparse_obs`, `_aggregate_chunk`, and `_aggregate_topk_from_geometry` are bypassed to cover uncached and cached reconstruction. All local modules/parameters and head widths remain instantiated; sensor feedback, latent/global reasoning, query-to-latent readout, global summary/scaffold, RFF prior, and RF loss remain active.
- A4, `iid_gaussian_prior`: the full A0 backbone and RF objective are unchanged. Only `prior: rff` becomes `prior: iid`; the RFF YAML keys remain as unused provenance.
- A5, `local_sensor_tokens_only`: a stronger local-only control based on A2. Observation-dependent latent encoding, latent self-processing/reinjection, latent-to-sensor feedback, global summary, and query-to-latent readout are bypassed. Original sensor tokens are projected by `sensor_out_proj` and provide the only observation-dependent query conditioning through top-K/RBF gathering. The ordinary point/state branch and final/coarse heads remain active, but receive no observation-dependent global feature. All bypassed global modules stay instantiated for schema fairness.

All variants have 6,507,151 trainable parameters. A0/A2/A3/A5 have identical 144-key wrapper state schemas. A1 and A4 retain the identical 142-key backbone schema and parameter count, but their wrapper schemas omit the RFF prior buffers `prior.omega` and `prior.phase`; this is intentional. A1 is an objective-level control, not a parameter-perfect wrapper identity.

## Launch commands

Run from `0_demo_TurbulentCombustion/`. Each config selects physical GPU 1 and writes to a distinct `Save_TrainedModel/ablation_condT/..._DemoN..._<timestamp>` directory.

```bash
conda run --no-capture-output -n phycoflow_env python src/train_pointcloud_ffm.py \
  --config Save_config/ablation_condT/config_A1_deterministic_condT.yaml

conda run --no-capture-output -n phycoflow_env python src/train_pointcloud_ffm.py \
  --config Save_config/ablation_condT/config_A2_no_sensor_global_feedback_condT.yaml

conda run --no-capture-output -n phycoflow_env python src/train_pointcloud_ffm.py \
  --config Save_config/ablation_condT/config_A3_no_local_query_conditional_condT.yaml

conda run --no-capture-output -n phycoflow_env python src/train_pointcloud_ffm.py \
  --config Save_config/ablation_condT/config_A4_iid_prior_condT.yaml

conda run --no-capture-output -n phycoflow_env python src/train_pointcloud_ffm.py \
  --config Save_config/ablation_condT/config_A5_local_sensor_tokens_only_condT.yaml
```

Validate without reading data or launching training:

```bash
conda run --no-capture-output -n phycoflow_env python src/train_pointcloud_ffm.py \
  --config Save_config/ablation_condT/config_A1_deterministic_condT.yaml --dry-run
```

Repeat with the other four config names. Regenerate and enforce the config audit with:

```bash
conda run --no-capture-output -n phycoflow_env python src/audit_ablation_configs.py
```

## Validation completed during preparation

- Config audit: PASS for all protected A0 fields.
- Trainer dry-run: PASS for A1–A5.
- Unit routing/schema/objective tests: 6 passed.
- Real-data CUDA:1 smoke: PASS. Every variant completed two optimizer steps on two real HDF5 snapshots, one inference call, finite-loss/output checks, and checkpoint/metadata save-load round trip. The smoke used 64 query points, 192 temperature sensors, batch size 1, and the torch neighbor backend. These smoke overrides were not written into the launch configs.

## Later primary evaluation (not run here)

Use the same 1,000 held-out states, fixed 256-temperature-sensor plans, and Figure 4 evaluation identities. Report mean physical relative-L2 across the four unobserved fields `Y_CH4`, `Y_CO`, `U1`, and `p`; also report fieldwise relative-L2, paired per-state differences versus A0, temporal block-bootstrap 95% confidence intervals, and training/validation curves.

Only after the primary result is interpretable, compute normalized CRPS on the fixed 200 states × 16 draws for A0/A2/A3/A4/A5. A1 is deterministic and must not be turned into an artificial stochastic ensemble. Do not initially run LSD/JSD.

See `config_diff_audit.md` for the exact raw YAML differences from A0.
