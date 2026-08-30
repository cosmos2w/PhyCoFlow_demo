# Mixed-resolution unified-v2 figure contract

- Core claim: lower-resolution training improves high-resolution sparse reconstruction across sensor budgets while preserving the spatial organization and variance allocation of large-, intermediate-, and fine-scale structures.
- Archetype: asymmetric mixed-modality full-page figure with Panel C as the qualitative image-plate anchor.
- Backend: Python/Matplotlib only, using constrained nested `SubFigure` containers and shared panel-drawing functions.
- Source policy: finalized CSV summaries and audited reconstruction caches; no model inference during assembly.
- Panel map: a, native resolutions/compositions; b, high-resolution error; c, qualitative transfer; d, sensor efficiency; e, qualitative wavelet components; f, scale-wise pattern correlation and variance-fraction bias.
- Statistics: means with bootstrap 95% confidence intervals for reconstruction/sensor panels; wavelet summaries export mean, standard deviation, median, quartiles, and bootstrap mean 95% confidence intervals, while panel f displays median and IQR.
- Geometry policy: the master uses compact full-width `SubFigure` rows with synchronized paired heights; every physical field axis retains equal aspect with `adjustable="box"`.
- Image integrity: native cached values only; no smoothing, sharpening, resampling for visual advantage, sensor overlay, or model-specific normalization. Panel e uses one shared zero-centred normalization per scale row.
- Exports: editable SVG and PDF plus high-resolution PNG, timestamped `YYYY-MM-DD_HH-MM`.
