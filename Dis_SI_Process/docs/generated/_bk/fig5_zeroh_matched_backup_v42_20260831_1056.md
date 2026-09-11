# Zero-H-matched Figure 5 V4.2 backup

- SVG: `fig5_zeroh_matched_backup_v42_20260831_1056.svg`
- Canvas: `183 mm x 116 mm`
- Evidence: **strict formal**

This backup now mirrors formal Figure 5 panels a-d: normalized CRPS, spread-error Spearman association, native accuracy-latency, and accuracy-canonical-training-update time. Panels a/b use the two stochastic adopted models; panels c/d use all four adopted Zero-H best checkpoints. All values are measured inside the Zero-H-balanced scenario and no Cond_T value is reused.

The Zero-H panel-c runner uses the archive's legacy full sampling path and synchronized wall timing. It does not expose the persistent DMF top-k geometry/static-feature cache used by the Cond_T portable implementation, so the absolute DMF latency coordinates across the two scenarios are not directly comparable.
