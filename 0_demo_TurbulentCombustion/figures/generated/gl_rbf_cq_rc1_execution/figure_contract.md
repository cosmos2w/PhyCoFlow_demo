Core conclusion: the Stage 1–5 execution program made the validated model streamable and million-query capable before the CQ architectural revision.
Figure archetype: quantitative grid
Target journal/output: technical release note; Nature-style double-column SVG/PDF/TIFF/PNG
Backend: Python (matplotlib, Agg)
Final size: 183 mm × 96 mm
Panel map:
  a: cumulative execution changes from data path through cached streaming and query microbatching
  b: query-scaling latency/memory at 4k, 16k, 65k, and 1M where available
  c: equivalence guarantees attached to each optimization stage
Evidence hierarchy:
  hero evidence: measured scaling/cost reductions
  validation evidence: numerical-equivalence tolerances and full-regression status
  controls/robustness: clearly label execution-only changes versus architecture changes
Statistics needed: benchmark aggregation as recorded in Stage 1–5 artifacts; no inferential test
Source data needed: Stage1_5 summary, Stage2/3/4/5 benchmark CSV/JSON and reports
Image-integrity notes: vector/source-data plotting only
Reviewer risk: values came from different stage-specific protocols; do not imply every bar is one matched experiment
