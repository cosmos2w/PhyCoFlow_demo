# Stage 6 — CQ-Balanced quality recovery

Branch: `perf/pointcloud-cq-balanced`

Base: `d95b7803ad2e0850072369fc9d9928a024c7e7e9`

## Outcome

Implementation, backward compatibility, cache equivalence, gradient equivalence,
and the complete regression suite pass. The 192-D structured-concat candidate
fails the mandatory pre-training efficiency gate, and the sole allowed 224-D
fallback also fails. Therefore no 200-epoch training, continuation, or kernel
benchmark was launched.

The stopping decision is intentional: the runbook requires approximately 1.15x
training-step speedup and at least 10% lower allocated or reserved memory versus
F0 before spending training time.

## Implemented candidate

`cq_fusion_mode` accepts `additive` and `structured_concat`; the default is
`additive` when absent.

For `structured_concat`:

- point state: 192-D;
- global stream: `global_q + cq_readout_scale * query_global_q`, 192-D;
- native local RBF condition: 128-D;
- head input: 512-D concatenation;
- head: `LayerNorm(512) -> 192 -> 192 -> fields`;
- GLRES coarse branch remains 192-D.

The additive module names, shapes, and initialization order are unchanged. The
real clean CQ-LR checkpoint strict-loads from its legacy config with zero missing
or unexpected keys.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| Focused implementation/equivalence | 14 passed | `implementation/gate_b_equivalence.log` |
| Full regression | 129 passed, 1 skipped | `implementation/full_regression.log` |
| Real old CQ checkpoint strict load | pass | `implementation/legacy_cq_strict_load.log` |
| 192 clean B128/Q4096 efficiency | fail | `cost_benchmark/clean_b128_q4096.json` |
| 224 sole-fallback efficiency | fail | `cost_benchmark/fallback_224_clean_b128_q4096.json` |
| Batch-1 4k/16k/65k + persistent 1M NFE-4 | complete | `cost_benchmark/cost_benchmark.json` |
| 200-epoch quality screen | not run by gate | `RESULTS.md` |
| 1000-epoch continuation | not eligible | `RESULTS.md` |
| MHA-mask/SDPA and fused AdamW | not run; no scientific candidate selected | `RESULTS.md` |

## Reproduction

Run tests:

```bash
conda run --no-capture-output -n phycoflow_env pytest -q
```

Run batch-1 scaling and persistent inference:

```bash
CUDA_VISIBLE_DEVICES=0 KEOPS_CACHE_FOLDER=/tmp/keops_stage6_cq_balanced_gpu0 \
conda run --no-capture-output -n phycoflow_env \
python src/benchmark_pointcloud_cq.py \
  --device cuda:0 --query-sizes 4096 16384 65536 --batch-size 1 \
  --n-obs 256 --iterations 5 --warmup 2 --component-iterations 5 \
  --million-query-count 1000000 --million-chunk-size 8192 \
  --million-iterations 3 \
  --f0-config /path/to/F0_ENH_1000ep_b128.yaml \
  --f0-checkpoint /path/to/F0/best.pt \
  --output _CheckNotes/Stage6_CQ_balanced_quality_recovery/cost_benchmark/cost_benchmark.json
```

For the decisive clean-protocol cost gate, add:

```text
--query-sizes 4096 --batch-size 128 --skip-million
```

Use `--balanced-query-dim 224` only for the sole documented fallback.

Generate the decision report and figure:

```bash
conda run --no-capture-output -n phycoflow_env \
  python _CheckNotes/Stage6_CQ_balanced_quality_recovery/summarize_cost.py
conda run --no-capture-output -n fig \
  python figures/scripts/plot_stage6_cq_balanced_gate.py
```

## Artifacts

- `RESULTS.md`: human-readable gate decision and recommendation.
- `cost_benchmark/gate_decision.json`: machine-readable thresholds and outcomes.
- `CQ_Balanced_192_Full_200ep.yaml`: validated primary clean protocol (not launched).
- `CQ_Balanced_224_Full_200ep.yaml`: validated sole fallback (not launched).
- `figures/cq_balanced_efficiency_gate.{svg,pdf,png}`: rendered evidence.
- `figures/figure_contract.md`: scientific claim, evidence map, and review risks.
