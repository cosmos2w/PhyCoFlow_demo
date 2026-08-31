# Figure 5 V4.1 completion report

- Generated: `20260831_0952`
- Main panel status: `a=formal, b=formal, c=formal, d=formal, e=formal`
- SVG QA: **PASS**
- Starting Git commit: `6d4f26a83604db42b29de374fda26a49ab385cbf`

## Requested revisions

- Panels a/b now use boxplots plus scatter. Panel a scatters 200 paired states/method and retains mean + block-bootstrap 95% CI. Panel b scatters a deterministic subset of 2,000 block-bootstrap ρ replicates/method and retains the full-sample ρ marker.
- Panels c/d use logarithmic x and y axes.
- Geo-FNO is restored to panel d using two-GPU DDP at global batch 192. The plotted total simultaneous allocation is 55.29 GiB; maximum per-device peak allocation is 27.64 GiB. Wall timing under the pre-existing GPU processes is explicitly inadmissible and unused.
- Panel e contains only the taller peak-allocated-memory axis. The V4 latency half is preserved as provenance and not redrawn.
- a/b and c/d use independently reduced gutters and moderately larger typography; the shared computational legend is enlarged.

## Zero-H-balanced backup

The backup uses the QA-passing `2026-08-06_11-24` source for four available methods × 300 canonical snapshots. It reports physical, gradient, sensor-excluded, and normalized relative-L2 distributions. No cross-model CRPS, ensemble-spread association, or clean Zero-H cost result exists in this archive, so the backup is explicitly an accuracy-distribution alternative rather than a metric-matched replacement.

## Interpretation changes

The a/b boxes have different statistical units: states in a, block-bootstrap estimates in b. Panel d compares total peak allocated memory at method-specific adopted configurations; Latent FM uses the maximum of its non-concurrent required stages, while Geo-FNO sums simultaneous two-rank peaks. Training configurations remain method-specific and descriptive.
