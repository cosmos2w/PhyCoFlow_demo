# Figure 5 V2 completion report

## Promotion status

Figure 5 V2 is an all-formal six-panel candidate. The strict build resolves `a=formal, b=formal, c=formal, d=formal, e=formal, f=formal`; no panel is blocked and no proxy, qualitative reconstruction, architecture substitute, throughput extension, or legacy field label enters the main figure.

The promoted SVG bundle is `Dis_SI_Process/figures/generated/20260830_1907/`:

- `fig5a_calibration_20260830_1907.svg`
- `fig5b_sharpness_20260830_1907.svg`
- `fig5c_spread_error_20260830_1907.svg`
- `fig5d_accuracy_latency_20260830_1907.svg`
- `fig5e_query_memory_20260830_1907.svg`
- `fig5f_nfe_tradeoff_20260830_1907.svg`
- `fig5_composed_v2_20260830_1907.svg`

Each SVG has a timestamp-matched Markdown companion under `Dis_SI_Process/docs/generated/`. The derived display tables and build manifest are under `Dis_SI_Process/results/derived/20260830_1907/`; the formal inference products remain in place under `_Process_Results/ValidationV2/` and are not copied into this workflow.

## Frozen protocol and formal runs

All jobs use `_ValidationPlans/validation_v1.yaml`, SHA-256 `06af0715d3e45576cd8406741c28fb41b8c2e12b440d388ccd020c5d53f746c2`, the adopted Cond_T checkpoint identities, the frozen 1,000-state evaluation mapping, and formal fields `Y_CH4`, `Y_CO`, `T`, `U1`, and `p`.

- `u1_formal_20260830_v1`: 1,000 states × 16 draws at M=256; 5,000 state–field rows.
- `u2_formal_20260830_v1`: 200 states × 64 draws at M=256; 1,000 state–field rows.
- `u3_formal_20260830_v1`: 200 states × 16 draws at M=192/256/384; 3,000 state–field–sensor rows. This supports SI rather than adding a main panel.
- `formal_cost_20260830_v2`: the exact eight Figure 4 Cond_T checkpoints, native 40,300-point errors, synchronized timing, DMF query/memory scaling, and a fixed-cohort measured-NFE sweep.

The uncertainty jobs have exact row counts, no duplicate keys, finite metrics, zero sensor-clamp error, and non-collapsed ensemble variance. The cost run resolves every requested method, passes cached-versus-legacy equivalence (`5.48e-6 < 2e-5`), and gives every accepted timing row at least 10 seconds and 30 synchronized repeats.

## Main findings

- **Calibration (a):** At 90% nominal central coverage, empirical coverage is 0.667 for `Y_CH4`, 0.471 for `Y_CO`, 0.247 for `U1`, and 0.152 for `p`. The raw ensemble is underdispersed and must not be described as calibrated.
- **Sharpness (b):** At the same nominal level, widths normalized by frozen training standard deviation are 0.0428, 0.1100, 0.1063, and 0.0399 for `Y_CH4`, `Y_CO`, `U1`, and `p`, respectively. These narrow widths are interpreted jointly with the undercoverage.
- **Spread–error association (c):** U1 Spearman rho values are 0.631 [0.584, 0.675] for `Y_CH4`, 0.567 [0.513, 0.613] for `Y_CO`, 0.221 [0.137, 0.307] for `U1`, and 0.659 [0.579, 0.728] for `p`. Spread is associated with error for several fields but is not a calibrated or prospectively validated error predictor.
- **Native accuracy–latency (d):** DMF-Gen has the lowest adopted mean relative L2 (0.1171) at 127.05 ms median warm latency. Senseiver is close in error (0.1430) and much faster (17.67 ms); therefore the measurements do not support unqualified Pareto superiority for DMF-Gen.
- **Query/memory scaling (e):** From 1,024 to 40,300 real-coordinate queries, median DMF latency grows 5.84× (about 20.0 to 116.9 ms) and peak allocated memory grows 2.71× (74.9 to 203.3 MiB).
- **Measured-NFE path (f):** Mean error worsens from 0.1133 at NFE 1 to 0.1175, 0.1267, and 0.1349 at NFE 2, 4, and 8, while median latency increases from 106.29 to 190.51 ms.

The SI U3 sweep also contradicts a simple sensor-density remedy: at 90% nominal coverage, moving from M=256 to M=384 changes coverage only from 0.607 to 0.605 (`Y_CH4`), 0.426 to 0.428 (`Y_CO`), 0.224 to 0.229 (`U1`), and 0.137 to 0.142 (`p`). More sensors do not rescue calibration in this range.

## QA and interpretation limits

- `audit_validation_v2.py` passes U1/U2/U3 cardinality, uniqueness, finiteness, identity, and six-panel formal-source checks.
- `qa_figure5_outputs.py --strict-formal` passes all seven SVGs with editable text, exact timestamped stems, no proxy/legacy labels, and no non-formal panels.
- The composed canvas is 183 mm × 145 mm and was visually inspected at export resolution.
- Absolute latency and memory are hardware-, adapter-, precision-, and chunking-specific. Error bars over states follow the frozen bootstrap definitions; latency bars are repeat IQRs.
- No A0/A2/A3/A1 ablation training was started.

The main narrative should therefore be framed as: repeated generations expose field-dependent error association but are severely undercalibrated; DMF offers the best measured reconstruction accuracy among the eight adopted checkpoints at a substantial latency cost; and the tested solver trajectory favors NFE 1 rather than more evaluations.
