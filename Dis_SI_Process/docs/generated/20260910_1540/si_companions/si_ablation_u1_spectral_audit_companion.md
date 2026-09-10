# SI-03: Velocity spectra, prevalence and leakage sensitivity

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Purpose and interpretation: Median shell-power ratios with IQR state-dispersion shading; empirical distributions of mode-sum high-band power and phase-sensitive residuals; separate Hann-window sensitivity. Both policies are shown. These index-space spectra do not identify a physical-wavenumber inertial range. Canonical, mode-sum and tapered quantities are distinct estimators.

Each standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.

## si03a_u1_population_spectra

```json
{
  "id": "si03a_u1_population_spectra",
  "role": "population shell-power ratio distribution across retained index-space",
  "metric": "population_U1_spectra median/IQR",
  "source_keys": [
    "population_U1_spectra"
  ],
  "statistics": "per-state median and IQR; shading is state dispersion, not a confidence band; n=1,000 per shell/method/policy",
  "visual_encoding": "RFF solid red; IID dashed amber; IQR alpha=0.15; truth=1; strict high band shaded",
  "axes": {
    "x": "wavenumber/kmax",
    "xlim": [
      0,
      1
    ],
    "high_band": ">2/3",
    "y_ideal": 1
  }
}
```

Individual panel companion with every plotted coordinate: [si03a_u1_population_spectra](si03a_u1_population_spectra_companion.md).

## si03b_u1_mode_sum_ecdf

```json
{
  "id": "si03b_u1_mode_sum_ecdf",
  "role": "empirical distributions of mode-sum power and high-band residual",
  "metrics": [
    "reconstruction_to_truth_high_energy_ratio",
    "highband_error_relative_l2"
  ],
  "source_keys": [
    "high_frequency_per_state"
  ],
  "statistics": "ECDF over 1,000 states; this mode-sum power is separate from canonical trapezoidal ratio",
  "visual_encoding": "RFF solid red; IID dashed amber; dotted power guide at truth=1",
  "axes": {
    "power_ideal": 1,
    "residual_ideal": 0,
    "y": "ECDF"
  }
}
```

Individual panel companion with every plotted coordinate: [si03b_u1_mode_sum_ecdf](si03b_u1_mode_sum_ecdf_companion.md).

## si03c_u1_hann_sensitivity

```json
{
  "id": "si03c_u1_hann_sensitivity",
  "role": "Hann-window sensitivity kept separate from primary untapered estimators",
  "metrics": [
    "hann_reconstruction_to_truth_high_energy_ratio",
    "hann_highband_error_relative_l2"
  ],
  "source_keys": [
    "U1_hann_robustness_summary",
    "high_frequency_summary"
  ],
  "statistics": "mean with 95% block-20 interval; n=1,000",
  "prevalence_mode_sum_primary": {
    "last_A0": {
      "fraction_gt_1": 0.042,
      "fraction_gt_2": 0.0
    },
    "last_A4": {
      "fraction_gt_1": 0.983,
      "fraction_gt_2": 0.773
    },
    "best_A0": {
      "fraction_gt_1": 0.037,
      "fraction_gt_2": 0.0
    },
    "best_A4": {
      "fraction_gt_1": 0.988,
      "fraction_gt_2": 0.803
    }
  },
  "visual_encoding": "RFF hollow red; IID filled amber; policy facets",
  "axes": {
    "x": "Hann diagnostic",
    "primary_untapered": false
  }
}
```

Individual panel companion with every plotted coordinate: [si03c_u1_hann_sensitivity](si03c_u1_hann_sensitivity_companion.md).

## si03

```json
{
  "id": "si03",
  "role": "SI entry-point composite si03",
  "linked_subpanels": [
    "si03a_u1_population_spectra",
    "si03b_u1_mode_sum_ecdf",
    "si03c_u1_hann_sensitivity"
  ],
  "layout": {
    "rows": 3,
    "columns": 1,
    "height_ratios_in": [
      2.65,
      3.0,
      2.65
    ],
    "standalone_authoritative": true
  },
  "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
  "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical"
}
```

Full coordinates: `Dis_SI_Process/figures/generated/20260910_1540/si/si_plot_manifest.json`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation pass; final artwork status is recorded in qa.json.
