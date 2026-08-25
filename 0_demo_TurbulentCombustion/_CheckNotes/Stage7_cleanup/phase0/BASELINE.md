# Phase 0: frozen cleanup baseline

## Git anchors

- Cleanup base: `65d6be5cb6b9eecb45df5fcb4f1798bb33ed11fa`
- Cleanup branch: `cleanup/gl-rbf-cq-rc1`
- Numerical/checkpoint oracle: annotated tag `gl-rbf-cq-v0.9.0-rc1`
- Oracle commit: `1b9a6d47f6c248364df6ba54155b5eac3d5e6e67`
- The project `src/` tree is identical between the oracle commit and cleanup base.

## Environment

- Python 3.10.19
- PyTorch 2.5.1+cu121
- CUDA runtime 12.1; cuDNN 90100
- PyKeOps 2.3
- GPU: NVIDIA RTX 6000 Ada Generation, driver 570.207, 49140 MiB

## Frozen artifacts

| Artifact | SHA256 |
|---|---|
| `GL_rbf_CQ_v0.9.0-rc1_e965_research.pt` | `e4c97bcb6385b7ec666baff652009068bfba2fa473c472f8b49469bdc40d7fc9` |
| `GL_rbf_CQ_v0.9.0-rc1_e1000_research.pt` | `31e59110258d4cc4715e13a5c92efb01d1eec72bdac7da3a3cec384da6f2042a` |
| `dataset_stats.pt` | `a3f3efb8a552af5804315e15ea21afb871585c88574f81e9a4ea4b59ee3f999a` |
| `run_config_training.yaml` | `11685bc3662e99177ba20f1aed5d518d6631ab4f72139ed38e6494ec978e7ca5` |

## Regression and numerical oracle

- Pre-refactor full suite: **143 passed** in 10.72 s on GPU 1.
- `capture_rc1_oracle.py` loads the three real reference checkpoints, resolves
  live/EMA state through the historical loader, and runs fixed synthetic inputs.
- Captured oracle JSON: `rc1_oracle.json`
- Oracle JSON SHA256: `56484012df5d4208b106fb30fa8eecaca0b24b5284eaf19677d1bdac4f1ad395`

| Candidate | Resolved weights | State SHA256 | Output SHA256 |
|---|---|---|---|
| `GL_rbf_CQ` e1000 | EMA trainable + live frozen | `f1c92d4bcf1b9e0ac90ad20b3b3468764f6cca200dde1575b914d9ab68d7b99f` | `63e4e5189f54e659aa84f0fff0552080bee4b5e2dac15dc5e25f9f06761cb90a` |
| CQ-LR fast e845 | live | `703ff488f0a641109b285bfcbcf5c3e33445ba5cbdf524e45db86313259bd845` | `94e510f137f3b73afd0c3c2410f1009f65b53c8abab103c3c7ac5da4497aa15b` |
| F0 e845 | live | `998807f793863f69b37019669fcdf1555c028769f8e9c8413c8ec7950910e58e` | `3c973f9429f410b28945cdc68b06fb5cfe181ad3f28dc1d7deb5659c2cae0da7` |

## Coordinate baseline

The production data coordinates have shape `(40300, 3)`. The third coordinate is
currently the constant plane value 0.5. The cleanup therefore treats the current
case as a 2-D slice represented in a three-coordinate interface and separately
tests varying-x/y/z synthetic inputs. Plotting remains explicitly 2-D.
