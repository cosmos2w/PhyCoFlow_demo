# Figure 5 V4.1 panel d

- Generated: `20260831_0952`
- SVG: `fig5d_accuracy_training_loglog_v41_20260831_0952.svg`
- Evidence status: **FORMAL**

## Protocol and visual statistic

One-GPU rows retain process-local peak allocations from the clean V4 canonical replay. Geo-FNO uses two-GPU DDP at global batch 192; x is the sum of simultaneous per-rank peak allocated memory. Both axes are logarithmic; shared-load wall timing is not used.

## Main quantitative result

DMF-Gen: error=0.1171, peak=32.02 GiB total; FFM-FNO: error=0.3898, peak=36.66 GiB total; FFM-Perceiver: error=0.3479, peak=13.69 GiB total; Latent FM: error=0.4531, peak=9.17 GiB total; SiT: error=0.2103, peak=14.51 GiB total; MLP-RBF: error=0.3962, peak=23.66 GiB total; Geo-FNO: error=0.2299, peak=55.29 GiB total; Senseiver: error=0.1430, peak=30.65 GiB total.

## Exact source

`Dis_SI_Process/results/ValidationV41/GeoFNOMultiGPU/geofno_ddp_memory_formal_v41/geofno_ddp_summary.csv`

## Interpretation limit

Panel a shows the distribution across paired states, whereas panel b shows uncertainty in a method-level association statistic. Panel d compares adopted configurations, not a causal matched-budget training experiment. Panel e carries no accuracy claim beyond 40,300 points.
