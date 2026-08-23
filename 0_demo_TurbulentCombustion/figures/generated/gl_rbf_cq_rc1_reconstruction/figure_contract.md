Core conclusion: GL_rbf_CQ improves matched sparse-condition reconstruction over F0 and CQ-LR-128, with the remaining error concentrated in U_1.
Figure archetype: image plate + quant
Target journal/output: technical release note; Nature-style double-column SVG/PDF/TIFF/PNG
Backend: Python (matplotlib, Agg)
Final size: 183 mm × 122 mm
Panel map:
  a: representative truth and NFE-4 reconstructions under one shared sensor layout and RF prior
  b: three-snapshot mean field-relative L2 at matched NFE-4
  c: worst-field U_1 relative L2 and snapshot variability
Evidence hierarchy:
  hero evidence: representative matched reconstruction plate
  validation evidence: three deterministic validation snapshots
  controls/robustness: identical sparse condition and RF seed within each snapshot; field-resolved errors
Statistics needed: n=3 validation snapshots; arithmetic mean and full point range; no inferential test
Source data needed: per-snapshot summary.json and evaluator-generated reconstructions
Image-integrity notes: identical colormap/range within each truth/model field column; only global plotting transforms
Reviewer risk: three snapshots are illustrative, not a population confidence interval; NFE dependence must be stated
