# Figure 5 V7R2 final release audit

Release `20260910_1707` is complete. All 31 release gates and five source-regression tests pass; the final two-pass LaTeX compile has no warnings, errors or overflowing boxes. No failed or unresolved release checks remain. Development failures and resolutions are retained in the [QA history](../../../results/derived/20260910_1707/qa_history.json) and [execution record](../../../results/derived/20260910_1707/execution_record.json).

Delivered: one 183 mm composed Figure 5, six standalone panels, five complete candidate-field panel-e alternatives, eight SI exports, five CSV/LaTeX SI tables, thirteen LaTeX inserts, companions and the quantitative report. All 20 figure pairs have editable SVG and 600-dpi PNG exports. U1 remains the requested main field; the author-facing field-selection comparison presents all five candidates.

The saved-field collector processed 7,000 caches, preserving matching truth, coordinates and sensor arrays across methods. Recomputed ablation high-band relative-L2 differs from the accepted output by at most 8.89e-16; canonical power/LSD differences are at most 1.78e-15. All new ablation/SI sources use last.pt. The frozen full-ablation checkpoint is epoch 7520; Senseiver metadata verifies epoch 5000. Training endpoints remain unequal; comparisons condition on saved runs and a shared validation/test holdout.

All 225 protected prior-release files remain byte-identical. Panels a/b/c/f preserve validated scientific geometry; historical panel-f error 0.117 remains distinct from the full-ablation mean 0.106321.

The first coordinated build passed source collection, main rendering, SI rendering and document checks; its LaTeX height failure was corrected by a concise main caption. Downstream stages were resumed without repeating validated field processing. Compiled-page review also refined SI numbering, S4 diagnostic columns and S5 provenance density. The final audit and manifest cover these finished products.

Full reproduction:

```bash
rtk proxy conda run -n fig python Dis_SI_Process/scripts/build_figure5_v7r2_bundle.py --timestamp 20260910_1707 --strict-formal
```

The existing branch policy publishes workflow source, instructions and Markdown reports. Rendered figures, derived data, generated LaTeX and machine-readable QA remain local reproducible products.
