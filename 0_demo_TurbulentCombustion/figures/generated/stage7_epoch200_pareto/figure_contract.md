# Figure contract — Stage-7 epoch-200 selection

- Core conclusion: Stage7-All256 (S7-B) is the only Stage-7 screen that clearly
  improves epoch-200 controlled RF quality while retaining the CQ efficiency
  target; it is therefore the sole candidate continued to epoch 1000.
- Evidence chain: a, controlled fixed-manifest convergence; b, paired epoch-200
  RF differences with 95% confidence intervals; c, deterministic reconstruction
  mean and worst-field errors at Euler NFE1/NFE4; d, epoch-200 RF quality versus
  B128/Q4096 step latency with memory and persistent-inference annotations.
- Archetype: quantitative 2x2 selection grid; the paired RF panel is the primary
  statistical evidence and the quality/latency panel is the decision view.
- Backend: Python/matplotlib exclusively, rendered in the `fig` environment.
- Export: 183 mm x 135 mm; editable SVG text, PDF, 300-dpi PNG, and 300-dpi TIFF.
- Sources: controlled fixed-manifest JSON/CSV artifacts, the deterministic shared
  reconstruction summary, and the definitive Stage-7 pre-training benchmark.
- Review risks: reconstruction is one fixed validation snapshot and should not be
  generalized as a dataset-wide rollout statistic; CQ-LR-256 cost is shown as an
  open marker because it comes from cumulative training diagnostics rather than
  the formal Stage-7 benchmark. Epoch-1000 default selection remains pending.
