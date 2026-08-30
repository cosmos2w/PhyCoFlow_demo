# Figure 5 post-processing workflow

This directory contains the lightweight Figure 5 post-processing and visualization pipeline. It reads existing products in place from the turbulent-combustion and super-resolution project trees; it does not copy checkpoints, reconstruction caches, or raw datasets.

The current first-pass draft is deliberately evidence-aware. Frozen `ValidationV2` products are preferred automatically. When they are absent, the renderer either uses a clearly labelled real-data engineering proxy or produces an explicit pending-data panel. It never synthesizes predictive uncertainty.

## Layout

- `configs/` — source paths, panel contract, palette, and draft settings.
- `scripts/` — command-line build and QA entry points.
- `utils/` — reusable data adapters, statistics, styling, and panel renderers.
- `figures/generated/` — timestamped SVG outputs; ignored by git.
- `docs/` — figure contract and timestamped panel companions.
- `results/` — lightweight derived CSVs and build manifests; ignored by git.

## Run

Use the project figure environment:

```bash
conda run -n fig python Dis_SI_Process/scripts/build_figure5_draft.py
```

To reproduce an exact name:

```bash
conda run -n fig python Dis_SI_Process/scripts/build_figure5_draft.py \
  --timestamp YYYYMMDD_HHMM
```

The command emits eight standalone SVG panels, one composed SVG, a Markdown companion for every output, lightweight derived tables, and a JSON build manifest. PDF is intentionally unsupported in this testing stage.

## Formal-data handoff

Place/freeze planned UQ and cost products under the `ValidationV2` roots named in `configs/figure5_draft.yaml`. Re-running the build will prefer the newest complete formal files. Use `--strict-formal` to prevent proxy or pending panels once the validation campaign begins.

## QA

```bash
conda run -n fig python Dis_SI_Process/scripts/qa_figure5_outputs.py \
  Dis_SI_Process/figures/generated/<timestamp>
```

The QA script checks filenames, SVG parseability, editable text, fixed format, and placeholder/proxy labelling. Visual inspection at final size is still required.
