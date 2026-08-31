# Figure 5 V4.1 Zero-H-balanced backup panel c

- Generated: `20260831_0952`
- SVG: `fig5c_zeroh_sensor_excluded_l2_backup_v41_20260831_0952.svg`
- Evidence status: **AUDITED FORMAL SOURCE**

This backup uses 300 canonical snapshots for each of DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver under recipe `4_ZeroH_Balanced` with 256 sensors. It is an accuracy-distribution backup, not a CRPS or ensemble-UQ substitute.

## Main quantitative result

DMF-Gen: median=0.0363, mean=0.0408; FFM-Perceiver: median=0.0992, mean=0.1031; MLP-RBF: median=0.1731, mean=0.1813; Senseiver: median=0.0824, mean=0.0863.

## Exact source

`/data/wanglz/Cache/PhyCoFlow_SuperResolution_Process_Results/QuestionB_ZeroH/QuestionB_per_snapshot_2026-08-06_11-24.csv`

Audit: `/data/wanglz/Cache/PhyCoFlow_SuperResolution_Process_Results/UnifiedPublicationV2/UnifiedV2Audit_20260806_1124.json`.
