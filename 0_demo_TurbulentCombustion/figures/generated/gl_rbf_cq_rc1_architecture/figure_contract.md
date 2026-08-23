Core conclusion: GL_rbf_CQ preserves the validated sensor/latent/RBF conditioning core while replacing the expensive query path with a compact decoder and adding condition-aware training features that do not invalidate persistent geometry reuse.
Figure archetype: schematic-led composite
Target journal/output: technical release note; Nature-style double-column SVG/PDF/TIFF/PNG
Backend: Python (matplotlib, Agg)
Final size: 183 mm × 104 mm
Panel map:
  a: Stage 1–7 architecture timeline and accepted/rejected decisions
  b: frozen GL_rbf_CQ dataflow, dimensions, cache boundary, and reused Top-K path
Evidence hierarchy:
  hero evidence: frozen Stage 7 dataflow and dimensional annotations
  validation evidence: commit-linked architecture evolution
  controls/robustness: explicit rejected structured-concat branch and unchanged legacy defaults
Statistics needed: none; this is a source-grounded schematic
Source data needed: Model.py, model_ema.py, Stage 1–7 reports, frozen YAML
Image-integrity notes: vector primitives only; no raster manipulation
Reviewer risk: avoid implying that rejected variants were trained when they only passed or failed a cost gate
