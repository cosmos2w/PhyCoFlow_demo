# Figure 5a — Probabilistic reconstruction

- Source reused in place: `Dis_SI_Process/results/derived/20260831_1409/figure5_v5_source.csv` (accepted V5 plot source).
- Evidence: 200 paired held-out temporal states per method; 64 shared-seed ensemble draws per state.
- Summary: formal mean normalized CRPS with 95% temporal moving-block-bootstrap interval (2,000 replicates; block length 25).
- Metric: empirical CRPS normalized by frozen training-field standard deviation, spatially averaged and equally macro-averaged over `Y_CH4`, `Y_CO`, `U1`, and `p`; lower is better.

DMF-Gen has the lowest accepted mean normalized CRPS (0.0667); the other accepted means are SiT 0.0999, FFM-Perceiver 0.2596, Latent FM 0.3711, and FFM-FNO 0.3989.

CRPS measures predictive-distribution quality but does not by itself establish calibration; accepted reliability evidence remains underdispersed and is retained in SI.
