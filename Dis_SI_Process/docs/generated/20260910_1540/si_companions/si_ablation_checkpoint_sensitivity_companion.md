# SI-05: Checkpoint sensitivity and recorded training histories

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Paired best-minus-last sensitivity and temporal block-length diagnostics alongside available raw training/validation loss histories. Longer histories are retained at their actual recorded endpoints with selected checkpoint markers. The comparison is not budget matched; validation and test share a holdout. Direct-field and flow-velocity training objectives are not pooled.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si05a_checkpoint_sensitivity

```json
{
  "id": "si05a_checkpoint_sensitivity",
  "role": "best-minus-last checkpoint-selection sensitivity",
  "source_keys": [
    "checkpoint_sensitivity"
  ],
  "statistics": "paired block-20 intervals from saved policy comparison; policy values are not pooled",
  "visual_encoding": "horizontal signed point/interval; zero guide; fixed stochastic order",
  "axes": {
    "x": "best-minus-last",
    "metrics": [
      "physical_relative_l2",
      "joint_pdf_jsd_with_overflow_base2"
    ]
  }
}
```

Individual panel companion with every plotted coordinate: [si05a_checkpoint_sensitivity](si05a_checkpoint_sensitivity_companion.md).

## si05b_block_length_sensitivity

```json
{
  "id": "si05b_block_length_sensitivity",
  "role": "paired best-minus-last sensitivity across circular block lengths",
  "source_keys": [
    "checkpoint_sensitivity"
  ],
  "statistics": "saved paired best-minus-last mean with source block-5/20/50 intervals; n=1,000",
  "visual_encoding": "method-colored horizontal line across block lengths with block-specific vertical intervals and zero guide",
  "axes": {
    "x": "block length in sorted held-out states",
    "blocks": [
      5,
      20,
      50
    ],
    "metrics": [
      "physical_relative_l2",
      "joint_pdf_jsd_with_overflow_base2"
    ],
    "policy_comparison": "best-minus-last"
  }
}
```

Individual panel companion with every plotted coordinate: [si05b_block_length_sensitivity](si05b_block_length_sensitivity_companion.md).

## si05c_recorded_training_histories

```json
{
  "id": "si05c_recorded_training_histories",
  "role": "actual recorded train/validation loss histories",
  "source_keys": [
    "history_A0",
    "history_A2",
    "history_A3",
    "history_A5",
    "history_A4"
  ],
  "statistics": "raw recorded history points; no smoothing, normalization, truncation or endpoint reconstruction",
  "visual_encoding": "method color; train/validation separate axes; selected checkpoint markers: circle=last, diamond=best",
  "axes": {
    "x": "epoch",
    "y": "loss (log scale)",
    "missing_methods": []
  }
}
```

Individual panel companion with every plotted coordinate: [si05c_recorded_training_histories](si05c_recorded_training_histories_companion.md).

## si05

```json
{
  "id": "si05",
  "role": "SI entry-point composite si05",
  "linked_subpanels": [
    "si05a_checkpoint_sensitivity",
    "si05b_block_length_sensitivity",
    "si05c_recorded_training_histories"
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
