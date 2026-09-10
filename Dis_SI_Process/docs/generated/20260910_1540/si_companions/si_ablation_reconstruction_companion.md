# SI-01: Complete reconstruction and conditioning-route effects

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Physical relative-$L_2$ and fieldwise effects under both policies. Temperature is observed. Effects use log$_2$ ratios of cohort means with a nonsaturating color range; raw means, intervals, paired effects, pressure-normalization checks and the additional local-only versus no-feedback contrast are supplied in tables and standalone diagnostics.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si01a_fieldwise_physical_l2

```json
{
  "id": "si01a_fieldwise_physical_l2",
  "role": "fieldwise primary reconstruction error under both checkpoint policies",
  "metric": "physical_relative_l2",
  "normalization": "physical field norm per state; fieldwise summary",
  "statistics": "mean with 95% circular moving-block interval, block length 20, 2,000 replicates, n=1,000 states",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "visual_encoding": "method color/marker from shared YAML; horizontal point with block-20 interval; observed T labelled",
  "axes": {
    "x": "Physical relative L2",
    "xlim": [
      0.0,
      0.86
    ],
    "policy_facets": [
      "last.pt",
      "best.pt"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si01a_fieldwise_physical_l2](si01a_fieldwise_physical_l2_companion.md).

## si01b_fieldwise_effect_heatmap

```json
{
  "id": "si01b_fieldwise_effect_heatmap",
  "role": "fieldwise effect relative to the full ablation reference",
  "metric": "physical_relative_l2",
  "definition": "log2(mean variant / mean A0) computed from fieldwise cohort means; not a mean of statewise ratios",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "visual_encoding": "diverging RdBu_r heatmap; zero is the full-model reference; T is identified as observed",
  "axes": {
    "color_limits": [
      -3.0,
      3.0
    ],
    "policies": [
      "last",
      "best"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si01b_fieldwise_effect_heatmap](si01b_fieldwise_effect_heatmap_companion.md).

## si01c_normalization_checks

```json
{
  "id": "si01c_normalization_checks",
  "role": "fieldwise standardized and truth-fluctuation normalizations retained alongside physical L2",
  "metrics": [
    "normalized_relative_l2",
    "truth_fluctuation_normalized_l2"
  ],
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "visual_encoding": "method color/marker; fieldwise horizontal point and block-20 interval; observed T labelled",
  "axes": {
    "x": "fieldwise relative L2",
    "xlim_standardized": [
      0,
      1.0
    ],
    "xlim_truth_fluctuation": [
      0,
      2.45
    ],
    "policy_facets": [
      "last.pt",
      "best.pt"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si01c_normalization_checks](si01c_normalization_checks_companion.md).

## si01d_local_conditioning_ch4_detail

```json
{
  "id": "si01d_local_conditioning_ch4_detail",
  "role": "reported local-conditioning sensitivity detail",
  "metric": "physical_relative_l2",
  "source_keys": [
    "last_summary_metrics"
  ],
  "derived_comparison": {
    "definition": "100*(A3/A0-1) from cohort means",
    "percent": 140.55070970045347,
    "formatted": "+140.6%"
  },
  "visual_encoding": "horizontal point and block-20 interval",
  "axes": {
    "x": "CH4 physical relative L2",
    "xlim": [
      0.035,
      0.16
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si01d_local_conditioning_ch4_detail](si01d_local_conditioning_ch4_detail_companion.md).

## si01

```json
{
  "id": "si01",
  "role": "SI entry-point composite si01",
  "linked_subpanels": [
    "si01a_fieldwise_physical_l2",
    "si01b_fieldwise_effect_heatmap"
  ],
  "layout": {
    "rows": 2,
    "columns": 1,
    "height_ratios_in": [
      2.85,
      2.75
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.
