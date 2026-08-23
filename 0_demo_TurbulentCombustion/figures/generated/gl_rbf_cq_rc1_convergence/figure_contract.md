Core conclusion: latent-256 Stage 7 recovers and surpasses F0 fixed-manifest RF quality while retaining the compact 128-D CQ query decoder.
Figure archetype: quantitative grid
Target journal/output: technical release note; Nature-style double-column SVG/PDF/TIFF/PNG
Backend: Python (matplotlib, Agg)
Final size: 183 mm × 92 mm
Panel map:
  a: validation RF loss versus epoch for clean F0, CQ-LR-128, CQ-LR-256, S7-A, and S7-B
  b: matched fixed-manifest RF loss at epoch 1000 (or explicitly labeled partial checkpoint)
  c: CQ-LR-to-F0 gap closure and final Stage 7 improvement
Evidence hierarchy:
  hero evidence: 64-layout × 3-repeat fixed-manifest RF comparison
  validation evidence: full training trajectories
  controls/robustness: explicit epoch and live/EMA labels
Statistics needed: n=192 fixed-manifest evaluations per formal candidate; mean loss; no cross-seed confidence interval
Source data needed: tracked convergence.csv and fixed_manifest_summary.json
Image-integrity notes: source-data plotting only
Reviewer risk: single training seed and partial CQ-LR-256 endpoint must remain visible
