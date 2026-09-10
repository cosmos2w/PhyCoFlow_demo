# SI-04: Coupled distributions and histogram-range sensitivity

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Fixed-range paper JSD, overflow-retaining JSD, and predicted/truth retention for three coupled field pairs under both policies. All stochastic configurations use the same archived edges. Low conditional JSD does not establish fidelity of discarded mass or spatial alignment.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si04_joint_pdf_jsd_base2

```json
{
  "id": "si04_joint_pdf_jsd_base2",
  "role": "Paper-range JSD (base 2)",
  "metric": "joint_pdf_jsd_base2",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000; common paper edges",
  "visual_encoding": "method color/marker; policy facets; JSD base 2",
  "axes": {
    "x": "Paper-range JSD (base 2)",
    "pair_targets": [
      "T-U1",
      "CH4-U1",
      "p-U1"
    ],
    "histogram_definition": "64x64 original edges"
  }
}
```

Individual panel companion with every plotted coordinate: [si04_joint_pdf_jsd_base2](si04_joint_pdf_jsd_base2_companion.md).

## si04_joint_pdf_jsd_with_overflow_base2

```json
{
  "id": "si04_joint_pdf_jsd_with_overflow_base2",
  "role": "Overflow-retaining JSD (base 2)",
  "metric": "joint_pdf_jsd_with_overflow_base2",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000; common paper edges",
  "visual_encoding": "method color/marker; policy facets; JSD base 2",
  "axes": {
    "x": "Overflow-retaining JSD (base 2)",
    "pair_targets": [
      "T-U1",
      "CH4-U1",
      "p-U1"
    ],
    "histogram_definition": "66x66 overflow-retaining extension"
  }
}
```

Individual panel companion with every plotted coordinate: [si04_joint_pdf_jsd_with_overflow_base2](si04_joint_pdf_jsd_with_overflow_base2_companion.md).

## si04c_coupling_retention

```json
{
  "id": "si04c_coupling_retention",
  "role": "retained-pair context for truncated and overflow JSD",
  "metrics": [
    "joint_pdf_reconstruction_retained_fraction",
    "joint_pdf_truth_retained_fraction"
  ],
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000; retention is not calibration",
  "visual_encoding": "predicted points offset below pair row; truth ticks above; method color; 0.99 guide",
  "axes": {
    "x": "retention fraction",
    "xlim": [
      0.42,
      1.01
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si04c_coupling_retention](si04c_coupling_retention_companion.md).

## si04

```json
{
  "id": "si04",
  "role": "SI entry-point composite si04",
  "linked_subpanels": [
    "si04_joint_pdf_jsd_base2",
    "si04_joint_pdf_jsd_with_overflow_base2",
    "si04c_coupling_retention"
  ],
  "layout": {
    "rows": 3,
    "columns": 1,
    "height_ratios_in": [
      2.65,
      2.65,
      2.65
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.
