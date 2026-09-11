# Figure 5 V4.1 Zero-H-balanced backup panel a

- Generated: `20260831_0952`
- SVG: `fig5a_zeroh_physical_l2_backup_v41_20260831_0952.svg`
- Evidence status: **AUDITED FORMAL SOURCE**

This backup uses 300 canonical snapshots for each of DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver under recipe `4_ZeroH_Balanced` with 256 sensors. It is an accuracy-distribution backup, not a CRPS or ensemble-UQ substitute.

## Main quantitative result

DMF-Gen: median=0.0360, mean=0.0405; FFM-Perceiver: median=0.0984, mean=0.1023; MLP-RBF: median=0.1727, mean=0.1808; Senseiver: median=0.0823, mean=0.0862.

## Exact source

`/data/wanglz/Cache/PhyCoFlow_SuperResolution_Process_Results/QuestionB_ZeroH/QuestionB_per_snapshot_2026-08-06_11-24.csv`

Audit: `/data/wanglz/Cache/PhyCoFlow_SuperResolution_Process_Results/UnifiedPublicationV2/UnifiedV2Audit_20260806_1124.json`.
