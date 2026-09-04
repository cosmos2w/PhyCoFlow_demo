# Figure 5c — Uncertainty supports selective reconstruction

- Source reused in place: `Dis_SI_Process/results/ValidationV51/PanelC/panel_c_exploration_20260902_1129/selective_risk.csv` (accepted V5.1 C1 family).
- Main-text choice: normalized C1b only. States are ranked by ascending macro normalized ensemble spread; the least-uncertain 20–100% are retained.
- Y quantity: retained-set mean reconstruction error divided by the same method's full-cohort error, `R(r)/R(1)`. Lower is better and every method ends at 1.0.
- Statistics: 200 paired states, 64 draws per state, and accepted 95% intervals from 2,000 temporal moving-block-bootstrap replicates (block length 25) with ranking recomputed within each resample.
- Display: linear scale; exact evaluated points joined without smoothing.

| Method | AURC ↓ | Error at 80% retained | Error at 100% |
|---|---:|---:|---:|
| DMF-Gen | 0.741 | 0.954 | 1.000 |
| FFM-FNO | 0.795 | 0.995 | 1.000 |
| FFM-Perceiver | 0.792 | 0.999 | 1.000 |
| Latent FM | 0.801 | 1.004 | 1.000 |
| SiT | 0.781 | 0.991 | 1.000 |

C1b replaces the former spatial error-capture main panel because panels a and d already preserve absolute quality. Normalization isolates how effectively each method's own uncertainty ranks cases for selective retention, making panel c the operational consequence of panel b. C1a remains the absolute-error SI/back-up view.
