# Figure 5 V4.1 source and statistical contract

## Figure contract

- Core conclusion: probabilistic reconstruction quality and uncertainty informativeness are shown with their underlying sampling distributions, while inference, training, and high-resolution memory costs remain transparent and hardware-qualified.
- Archetype: quantitative comparison grid with paired distribution panels, paired log–log accuracy–cost planes, and one full-width scalability axis.
- Backend: Python/Matplotlib in the `fig` environment for all rendering and QA previews.
- Export: editable SVG, 183-mm composed main width; all V2–V4 products remain immutable provenance.
- Reviewer risks: panel-a states and panel-b bootstrap estimates are different statistical units; two-GPU wall time must not be compared directly with one-GPU wall time; no accuracy is claimed above N=40,300; fixed-grid reconstruction followed by slicing is forbidden as query scaling.

## Main source tables

### a — state-wise normalized empirical CRPS

Input: V3 `per_state_method.csv`, exactly five methods × 200 paired held-out states. Scatter units are states. The box summarizes those 200 values. The open method marker and thin line retain the formal mean and temporal moving-block-bootstrap 95% CI from `crps_summary.csv`.

### b — spread/error Spearman association

Input: the same paired V3 state table. Exactly 2,000 moving-block-bootstrap Spearman estimates are rebuilt with the original SHA-256 stable seed, block length 25, and method salt. The distribution must reproduce the adopted CI to absolute tolerance `5e-13`. Scatter units are bootstrap estimates, not independent states. The open marker is the full-sample Spearman rho.

### c — native inference accuracy–latency

Input: unchanged formal V3 clean native table. x is median warm model-core latency with IQR at N=40,300; y is the frozen mean unobserved-field relative L2 with temporal-bootstrap CI. Both axes are logarithmic.

### d — canonical training accuracy–memory

One-GPU rows reuse process-local peak allocations from the formal V4 preloaded-batch update replay. Latent FM uses the maximum allocation of its two non-concurrent required stages. Geo-FNO is accepted only from a formal two-rank DDP memory replay at global batch 192 on two distinct GPUs. The common x metric is total simultaneous peak allocated memory: one-device peak for the one-GPU rows and the sum of both rank peaks for Geo-FNO. The Geo-FNO source separately records rank-wise and maximum per-device values. PyTorch allocated-memory counters are process-local, so foreign allocations are recorded but do not enter the metric; shared-load wall time is inadmissible and unused. Both axes are logarithmic.

### e — peak allocated memory scaling

Input: unchanged formal V4 scale-stress memory table plus the validated V3 native prefix. Only methods with canonical variable-query support receive curves. Fixed-grid methods receive native-size open markers only. The V4 latency axis is removed from V4.1 but preserved in V4 provenance.

## Zero-H-balanced backup

Input: audited `QuestionB_per_snapshot_2026-08-06_11-24.csv`, recipe `4_ZeroH_Balanced`, 256 sensors, exactly 300 valid canonical snapshots for each of DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver. Its per-snapshot physical-error means must reproduce the summary joined by `UnifiedV2DataManifest_20260806_1124.json`; every check in `UnifiedV2Audit_20260806_1124.json` must pass.

The four backup panels show physical, gradient, sensor-excluded, and normalized relative-L2 distributions. The archive has no five-method CRPS, ensemble spread/error association, or checkpoint-matched clean cost evidence, so the backup must never be described as a metric-matched a–d replacement.

## Strict-formal failure policy

`--strict-formal` fails on a missing/non-passing Geo-FNO DDP memory run, any non-formal inherited V3/V4 source, a Zero-H audit mismatch, a bootstrap reproduction mismatch, a proxy, or missing data. No V2 timing, contaminated wall time, or single-GPU Geo-FNO OOM extrapolation is accepted.
