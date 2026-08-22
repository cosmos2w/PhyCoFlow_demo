# Figure contract — CQ-Balanced efficiency gate

- Core conclusion: structured-concat CQ-Balanced (192-D primary and 224-D sole
  fallback) does not retain enough of CQ-LR's training efficiency to justify a
  quality-training screen.
- Evidence chain: a, clean batch-128 step latency against the 1.15x gate; b,
  allocated/reserved memory reductions against the 10% gate; c, batch-1
  scaling at 4k/16k/65k queries; d, unchanged persistent 1M-query NFE-4 path.
- Archetype: quantitative grid with the clean-protocol gate as the hero row.
- Backend: Python/matplotlib exclusively, rendered in the `fig` environment.
- Export: 183 mm x 120 mm; editable SVG text, PDF, and 300-dpi PNG preview.
- Sources: the three JSON benchmark artifacts in the Stage 6 evidence package.
- Review risks: random candidate weights are sufficient for architecture cost
  but do not provide CQ-Balanced quality; the 192 and 224 candidates therefore
  must not be placed on a validation-loss Pareto axis.
