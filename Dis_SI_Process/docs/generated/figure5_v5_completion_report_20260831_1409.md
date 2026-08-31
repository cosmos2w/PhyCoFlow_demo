# Figure 5 V5 completion report

- Generated: `20260831_1409`
- Starting branch HEAD: `ec8c2b27ee2716d0b11b8dd991c9fd90c56d9e84`
- Strict SVG/data QA: **PASS**
- Print-size visual QA: **PASS**
- ValidationV5 retained-file payload: **1022.9 KiB** (`1047433` bytes); final filesystem usage: **1.1M** (`1067913` bytes from `du -sb`, including directory metadata)

## Reuse matrix

| Quantity | Formal source | Action |
|---|---|---|
| State-wise normalized CRPS | V3 `uq_compare_formal_20260830_v3r6` | Reused in place; no inference/bootstrap rerun |
| State-level spread/error Spearman | V3 `uq_compare_formal_20260830_v3r6` | Reused in place; exact bootstrap reconstruction only |
| Spatial error-capture curves | V5 `uq_localization_formal_v5` | New streaming repeated inference; compact reducer only |
| Native inference latency + frozen relative-L2 | V3 `formal_cost_clean_v3_20260830_v3` | Reused in place |
| Canonical update timing | V4 `training_replay_formal_v4r2` + V4.2 Geo-FNO DDP | Reused in place; no replay rerun |
| Replay-equivalent GPU-hours | V5 `lifecycle_formal_v5` | Newly derived from adopted updates/GPU counts; Latent-FM stages summed |

## Main quantitative results

- Panel a: DMF-Gen: 0.0667; SiT: 0.0999; FFM-Perceiver: 0.2596; Latent FM: 0.3711; FFM-FNO: 0.3989
- Panel b: DMF-Gen: ρ=0.654; SiT: ρ=0.261; FFM-Perceiver: ρ=0.215; FFM-FNO: ρ=0.183; Latent FM: ρ=-0.033
- Panel c: SiT: C(0.20)=0.631, EC-AUC=0.283; DMF-Gen: C(0.20)=0.570, EC-AUC=0.241; FFM-Perceiver: C(0.20)=0.555, EC-AUC=0.232; FFM-FNO: C(0.20)=0.522, EC-AUC=0.215; Latent FM: C(0.20)=0.510, EC-AUC=0.212
- Panel d: DMF-Gen: 16.69 ms, 62.5 GPU h, L2=0.117; FFM-FNO: 8.70 ms, 102.8 GPU h, L2=0.390; FFM-Perceiver: 23.09 ms, 55.9 GPU h, L2=0.348; Latent FM: 10.17 ms, 71.2 GPU h, L2=0.453; SiT: 20.99 ms, 516.0 GPU h, L2=0.210; MLP-RBF: 3.14 ms, 26.9 GPU h, L2=0.396; Geo-FNO: 3.41 ms, 119.7 GPU h, L2=0.230; Senseiver: 8.30 ms, 24.0 GPU h, L2=0.143

## Cleanup and storage

The panel-c runner created no scratch directory, per-draw file, repeated CSV/NPZ product, or saved ensemble stack. Each in-memory stack was discarded before advancing to the next state. No checkpoint, HDF5 dataset, cache or older result bundle was copied. No new arrays were retained; only compact CSV summaries, manifests and QA remain.

Removed after QA:

- temporary panel-c smoke run `uq_localization_smoke_v5`;
- temporary GPU shards `uq_localization_shard_ffm_v5`, `uq_localization_shard_latent_sit_v5` and `uq_localization_shard_sit_v5`;
- ten temporary Python-rendered PNG previews used for print-size visual inspection.

## Verification

- Strict-formal V5 build and SVG/data contract checks: PASS.
- Composed figure, four standalone panels and five SI figures inspected at 240 dpi: PASS.
- Targeted V3–V5 regression suite: 13 tests passed.

## Unavailable values

None. All eight lifecycle methods have accepted native latency, adopted update counts, GPU counts, canonical timing and frozen Figure-4 error. Latent FM uses both required sequential stages; Geo-FNO uses the formal 2-GPU DDP replay.

## Narrative checks and limitations

- DMF-Gen is not best in spatial error capture; SiT has the largest C(0.20).
- State-level association is not distinguishable from zero for FFM-FNO, Latent FM.
- Formal reliability results remain underdispersed; panels a–c must be described as empirical conditional ensemble uncertainty, not Bayesian posterior uncertainty, perfect calibration or prospective error prediction.
- Replay-equivalent model-core training GPU-hours are not historical training wall time and do not establish a matched-budget causal efficiency ranking.
