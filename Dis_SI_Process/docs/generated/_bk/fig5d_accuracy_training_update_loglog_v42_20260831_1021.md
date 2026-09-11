# Figure 5 V4.2 panel d

- SVG: `fig5d_accuracy_training_update_loglog_v42_20260831_1021.svg`
- Evidence: **FORMAL**

## Protocol

Canonical training update wall time; original V4 single-stage coordinates unchanged. Geo-FNO: clean two-GPU DDP, global batch 192, synchronized max-rank wall time; log–log axes.

## Quantitative result

DMF-Gen: error=0.1171, 527.509 ms/update; FFM-FNO: error=0.3898, 1023.270 ms/update; FFM-Perceiver: error=0.3479, 483.960 ms/update; SiT: error=0.2103, 658.778 ms/update; MLP-RBF: error=0.3962, 404.005 ms/update; Geo-FNO: error=0.2299, 723.615 ms/update; Senseiver: error=0.1430, 479.401 ms/update
