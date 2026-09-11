# Figure 5 V4.2 completion report

- Generated: `20260831_1021`
- Starting commit: `6d4f26a83604db42b29de374fda26a49ab385cbf`
- QA: **PASS**

## Correction

Panel d's x axis is again canonical training update time (`ms/update`), not training memory. The six valid single-stage V4 coordinates are bit-for-bit unchanged, including DMF-Gen at `527.508987113833 ms/update`. Latent FM remains unavailable because its two unlike required stages do not support one scalar.

## New Geo-FNO formal result

Run `geofno_ddp_timing_formal_v42r2` used clean physical GPUs 1 and 2, true DDP, global batch 192 (96/rank), 20 warmups, and 10×10 measured updates. Median synchronized wall time is `723.615075 ms/update` (IQR `722.945709–724.650422`); block-drift fraction is `0.001262`. GPU-ms and peak allocation are retained only as provenance.
