Core conclusion: GL_rbf_CQ is the selected quality–throughput Pareto point, improving fixed-manifest RF loss while reducing training step time, peak memory, and persistent million-query inference cost versus F0.
Figure archetype: quantitative grid
Target journal/output: technical release note; Nature-style double-column SVG/PDF/TIFF/PNG
Backend: Python (matplotlib, Agg)
Final size: 183 mm × 96 mm
Panel map:
  a: fixed-manifest RF loss versus training-step time
  b: fixed-manifest RF loss versus peak GPU memory
  c: persistent 1M-query Euler NFE-4 latency and parameter count
Evidence hierarchy:
  hero evidence: matched B128/Q4096 quality–cost plane
  validation evidence: persistent 1M-query inference benchmark
  controls/robustness: same RTX 6000 Ada hardware and protocol labels
Statistics needed: benchmark medians/means exactly as recorded by Stage 7 scripts; no cross-hardware interval
Source data needed: final fixed-manifest and cost benchmark JSON artifacts
Image-integrity notes: source-data plotting only
Reviewer risk: CQ-LR-256 quality lacks a matched formal cost benchmark and must not be placed on the cost plane
