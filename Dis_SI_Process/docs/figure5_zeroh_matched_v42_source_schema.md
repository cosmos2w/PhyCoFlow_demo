# Metric-matched Zero-H-balanced Figure 5 backup contract

This additive backup replaces no prior artifact. It mirrors the scientific content of formal Figure 5 panels a–d within the `4_ZeroH_Balanced` super-resolution scenario.

- Panels a/b include DMF-Gen and FFM-Perceiver, the only stochastic generative checkpoints tested in this scenario. MLP-RBF and Senseiver are deterministic and are excluded from ensemble-spread analysis rather than assigned artificial zero-spread values.
- Panels c/d include all four adopted best checkpoints: DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver.
- The task is single-field density reconstruction on the native 128×128 H grid (`N=16,384`) from 256 observations. It is not the five-field Cond_T task.
- UQ uses 200 paired unique-case/time states and 64 draws per state. CRPS is computed in the checkpoint's frozen training normalization. Spearman relates normalized spatial RMS ensemble spread to physical ensemble-mean relative L2. Confidence intervals use 2,000 unique-case bootstrap replicates.
- Accuracy uses the audited 300-case best-checkpoint source. Inference timing is clean warm model-core time. Training cost is the equal-weight mean of L- and M-resolution median update time at canonical batch 512, matching the adopted 1:1:0 training recipe.
- The Zero-H inference runner uses the super-resolution archive's legacy full `sample()` path with synchronized wall timing. That path does not expose the Cond_T portable core's persistent top-k geometry/static-feature cache. Therefore panel c is valid for comparisons among the four Zero-H runs under this common runner, but its absolute DMF-Gen coordinate must not be compared directly with the cached Cond_T Figure 5 coordinate.

No Cond_T UQ, accuracy, latency, or training-time result is reused in this backup.
