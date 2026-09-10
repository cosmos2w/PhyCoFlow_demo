# Figure 5 V7 completion report

Release: 20260910_1540. Final package status: **pass**.

Produced: one 183 mm composed Figure 5 with dedicated right-hand panels c/e;
six main standalones plus separate power/residual exports for e; six SI
composites and 18 standalone SI diagnostics. There are
33 editable SVG artworks with matching 600-dpi PNGs,
66 LaTeX inserts/table files, individual quantitative companions,
and the quantitative figure-making report.

Main artwork: `Dis_SI_Process/figures/generated/20260910_1540/fig5_composed_v7_20260910_1540.svg`.
SI entry point: `Dis_SI_Process/docs/generated/20260910_1540/latex/si_ablation_package.tex`.
Source ledger: `Dis_SI_Process/results/derived/20260910_1540/source_manifest.json`.
Report: `Dis_SI_Process/docs/generated/20260910_1540/quantitative_figure_making_report.md`.

Source validation, inherited-coordinate comparisons, editable SVG structure,
600-dpi raster metadata, current visual review, and LaTeX compile status are
recorded in `qa.json`. Tests are run with the command below. The LaTeX smoke
test uses a 183 mm text width and 300 mm text height to accommodate the main
caption at full artwork width; temporary PDFs/logs are removed.

Required-but-unavailable outputs: none.
The optional cached-field gallery was not produced; it is not required for
the saved-metric evidence sequence. No training, model inference, GPU timing,
solver change, clipping, smoothing, or ensemble reevaluation was performed.
Heavy data and checkpoints remain in place. Prior release hashes are preserved.

The archived benchmark error remains 0.117. The new full reference is separate;
IID's lower bulk error, the full model's high-band attenuation, and the
deterministic aggregate advantage remain explicit. The available configurations
have unequal training endpoints/capacity, a shared validation/test holdout,
and one stochastic draw per evaluation state. Formal QA does not imply a
matched-budget causal comparison.

Reproduce from the repository root:

```bash
rtk proxy conda run -n fig python Dis_SI_Process/scripts/build_figure5_v7_bundle.py --timestamp 20260910_1540 --strict-formal
rtk proxy conda run -n fig python -m unittest Dis_SI_Process.tests.test_figure5_v7_ablation
```

Changed sources or artwork require renewed visual review; the saved review is
bound to exact SVG hashes.
