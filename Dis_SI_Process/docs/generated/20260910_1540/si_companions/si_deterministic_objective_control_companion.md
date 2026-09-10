# SI-06: Deterministic objective control

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Separate deterministic objective control against the full stochastic reference: macro/fieldwise physical error and coupling diagnostics under both policies. Pressure fluctuation error and complete retention/overflow companions are tabulated. Lower deterministic point error does not quantify diversity or calibration.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si06a_deterministic_macro_control

```json
{
  "id": "si06a_deterministic_macro_control",
  "role": "separate deterministic objective control",
  "metric": "physical_relative_l2",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "derived_comparison": {
    "last_deterministic_minus_full_percent": -25.974040046966994,
    "best_deterministic_minus_full_percent": -27.186320349262317
  },
  "visual_encoding": "horizontal point/interval; A0 hollow red, A1 filled slate",
  "axes": {
    "x": "unobserved physical relative L2"
  }
}
```

Individual panel companion with every plotted coordinate: [si06a_deterministic_macro_control](si06a_deterministic_macro_control_companion.md).

## si06b_deterministic_fieldwise_control

```json
{
  "id": "si06b_deterministic_fieldwise_control",
  "role": "fieldwise deterministic-versus-full control",
  "metric": "physical_relative_l2",
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "visual_encoding": "A0 hollow red; deterministic A1 filled slate; policy facets",
  "axes": {
    "x": "physical relative L2"
  }
}
```

Individual panel companion with every plotted coordinate: [si06b_deterministic_fieldwise_control](si06b_deterministic_fieldwise_control_companion.md).

## si06c_deterministic_coupling_control

```json
{
  "id": "si06c_deterministic_coupling_control",
  "role": "deterministic coupling diagnostics with range companions",
  "metrics": [
    "joint_pdf_jsd_base2",
    "joint_pdf_jsd_with_overflow_base2"
  ],
  "source_keys": [
    "last_summary_metrics",
    "best_summary_metrics"
  ],
  "statistics": "mean with 95% block-20 interval; common paper edges and 66x66 overflow edges",
  "visual_encoding": "circle/policy points for paper JSD; x markers for overflow JSD; A0 red/A1 slate",
  "axes": {
    "x": "JSD base 2",
    "pair_targets": [
      "T-U1",
      "CH4-U1",
      "p-U1"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si06c_deterministic_coupling_control](si06c_deterministic_coupling_control_companion.md).

## si06

```json
{
  "id": "si06",
  "role": "SI entry-point composite si06",
  "linked_subpanels": [
    "si06a_deterministic_macro_control",
    "si06b_deterministic_fieldwise_control",
    "si06c_deterministic_coupling_control"
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
