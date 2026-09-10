# SI-02: All-field scale-resolved source-prior comparison

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Full RFF versus IID phase-sensitive high-band residuals and canonical shell-integrated high-band power in all five fields, separately for last and best. Intervals are mean block-20 confidence intervals; power and residual ideal values are one and zero. All stochastic variants remain in the tables.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si02a_all_field_highband_l2

```json
{
  "id": "si02a_all_field_highband_l2",
  "role": "all-field phase-sensitive high-band residual",
  "metric": "highband_error_relative_l2",
  "source_keys": [
    "high_frequency_summary"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000 states per method/policy",
  "visual_encoding": "RFF hollow red circles; IID filled amber diamonds; policy facets",
  "axes": {
    "x": "High-band relative L2",
    "ideal": 0,
    "xlim": [
      0,
      2.25
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si02a_all_field_highband_l2](si02a_all_field_highband_l2_companion.md).

## si02b_all_field_canonical_power

```json
{
  "id": "si02b_all_field_canonical_power",
  "role": "canonical shell-mean high-band power ratio",
  "metric": "canonical_shellmean_high_energy_ratio",
  "definition": "per-state shell-mean spectrum integrated with existing trapezoidal weighting over strict retained high band; not mode-sum",
  "source_keys": [
    "high_frequency_summary"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000 states per method/policy",
  "visual_encoding": "RFF hollow red circles; IID filled amber diamonds; truth=1 dashed guide",
  "axes": {
    "x": "High-band power / truth",
    "ideal": 1,
    "xlim": [
      0,
      3.8
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si02b_all_field_canonical_power](si02b_all_field_canonical_power_companion.md).

## si02

```json
{
  "id": "si02",
  "role": "SI entry-point composite si02",
  "linked_subpanels": [
    "si02a_all_field_highband_l2",
    "si02b_all_field_canonical_power"
  ],
  "layout": {
    "rows": 2,
    "columns": 1,
    "height_ratios_in": [
      2.9,
      2.9
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.
