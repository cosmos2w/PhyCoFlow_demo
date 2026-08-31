# Figure 5 V4 completion report

- Generated: `20260831_0110`
- Panel status: `a=formal, b=formal, c=formal, d=formal, e=formal`
- V3 provenance preserved: `true`

## Panel-d training-cost gate

Formal panel-d run: `training_replay_formal_v4r2`. Metric: `training_update_time_ms` = median synchronized canonical forward/loss/backward/gradient-clip/optimizer update at each method's adopted batch/query configuration, after 20 warmups and across 100 measured updates (10 blocks × 10) for each successful stage; all nine required stages were attempted. Promoted single-stage methods: DMF-Gen, FFM-FNO, FFM-Perceiver, SiT, MLP-RBF, Senseiver. Method-level unavailable: Latent FM, Geo-FNO. Reasons: Latent FM: method has multiple required training stages; no single update-time scalar is defensible; Geo-FNO: adopted batch 192 exceeded the 47.38-GiB GPU capacity. Latent FM stage-level medians: Latent FM shared autoencoder=166.90 ms/update; adopted_checkpoint_training_stage=390.36 ms/update. Historical GPU-hours and filesystem timestamps were not used.

Different archived training budgets are not a causal architectural comparison. Historical file timestamps are never used as training time. Replay-equivalent GPU-hours require a passing predeclared validation gate; otherwise a directly measured update-time metric or SI-only result is required.

## Panel-e scale-stress gate

Variable-query methods: DMF-Gen, FFM-Perceiver, MLP-RBF, Senseiver. Fixed-grid/native-only methods: FFM-FNO, Latent FM, SiT, Geo-FNO. DMF-Gen: largest success 8,000,000; first failure none (8M global cap); FFM-Perceiver: largest success 4,000,000; first failure 8,000,000; MLP-RBF: largest success 1,000,000; first failure 2,000,000; Senseiver: largest success 1,000,000; first failure 2,000,000. V3 native prefix sources: `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV3/CostClean/formal_cost_clean_v3_20260830_v3/query_latency_summary.csv` and `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV3/CostClean/formal_cost_clean_v3_20260830_v3/memory_summary.csv`.

The native validated endpoint is N=40,300. Values above it are throughput-only and use one common frozen query specification/hash. Fixed-grid methods remain native-only markers. OOM/runtime-cap events must remain visible in the source table.

## Scope and provenance

V3 UQ and clean native inference products are reused only when their original schema, checkpoint identity, and QA pass. V3 query-latency/memory tables and all V2 cost products are explicitly excluded as V4 fallbacks. No NFE panel or ablation training is part of this workflow.

The V2 DMF latency of approximately 127 ms remains superseded by the V3 clean warm model-core value of 16.69 ms and is not reused in V4.

## Results that qualify or contradict a simple efficiency narrative

DMF-Gen has the lowest reconstruction error and normalized CRPS. At the differing adopted training configurations, its 527.51-ms update is faster than FFM-FNO and SiT but slower than FFM-Perceiver, MLP-RBF, and Senseiver; this per-update footprint is not normalized for batch size or a matched training budget. Latent FM cannot receive a defensible single method-level update-time coordinate because the adopted model requires two unlike training stages, and Geo-FNO's adopted batch cannot be replayed within the 47.38-GiB device capacity. In high-N inference, DMF-Gen reaches the 8M safety cap; FFM-Perceiver fails at 8M after succeeding at 4M, while MLP-RBF and Senseiver first fail at 2M after succeeding at 1M. These hardware-specific boundaries are capacity evidence, not accuracy evidence.
