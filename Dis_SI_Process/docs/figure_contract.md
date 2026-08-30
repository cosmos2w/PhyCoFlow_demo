# Figure 5 validation V2 contract

Core conclusion: repeated DMF-Gen generations measure empirical conditional variability under underdetermination, while formal native-mesh and scaling benchmarks expose the measured accuracy–latency–memory trade-off. Figure 5 does not claim that spread predicts error or that DMF-Gen is necessarily Pareto-superior.

- Schema: `figure5-validation-v2`.
- Archetype: compact six-panel mixed quantitative figure in two conceptual rows.
- Target/output: Nature Machine Intelligence-style, 183 mm × approximately 145 mm, editable SVG only.
- Backend: Python/Matplotlib exclusively.
- Top row: empirical conditional uncertainty (`a` calibration, `b` normalized sharpness, `c` spread–error association).
- Bottom row: computational characteristics (`d` native accuracy–latency, `e` query/memory micro-axes, `f` measured-NFE latency–error path); `d` is the visual anchor.
- Formal fields: `Y_CH4`, `Y_CO`, `T`, `U1`, `p`. For `Cond_T`, use unobserved-field order `Y_CH4`, `Y_CO`, `U1`, `p`; legacy `U_0` is forbidden.
- Formal method order: DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, SiT, MLP-RBF, Geo-FNO, Senseiver.
- Source-data policy: read frozen `ValidationV2` results and matching checkpoints/data in place. Do not copy raw data, checkpoints, caches, or existing figures, and do not regenerate an exactly matching frozen FieldL2 table.
- Statistics: summarize coverage/width within state before across-state aggregation; use temporal block-bootstrap 95% intervals over states; use Spearman association as primary; use synchronized repeated warm latency median/IQR, peak allocated GPU memory, and physical relative L2.
- Strict behavior: `--strict-formal` fails on missing, pending, proxy, identity-mismatched, or protocol-incomplete inputs. It never silently substitutes engineering data. Only strict-formal output is a manuscript candidate.
- Claim boundary: no main qualitative map, solver-sensitivity-as-UQ claim, architecture proxy, throughput-extension point, asymptotic-complexity proof, or forced Pareto claim.

## Panel map

### a — Calibration

Plot empirical state-level coverage against nominal central coverage 50%, 80%, 90%, and 95% for the four unobserved fields. Use U2: fixed 200 held-out states × 64 draws, `Cond_T`, M=256, the adopted temperature sensor plan and exact `DMF_Gen/Cond_T/last.pt` identity. Include the ideal diagonal and restrained temporal block-bootstrap 95% intervals. Exclude mechanically hard-clamped T sensor locations from primary UQ summaries; T is SI QA only.

### b — Normalized sharpness

From the same U2 cohort, plot mean central-interval width divided by the predeclared training-set field standard deviation against 50%, 80%, 90%, and 95% nominal level. Use the same field colors and order as panel a. Raw physical-unit widths and M=192/256/384 sensitivity belong in the SI.

### c — Spread–error association

Use U1: 1,000 test states × 16 common-protocol draws at M=256. For each state and unobserved field, compare normalized spatial RMS ensemble standard deviation with ensemble-mean physical relative-L2 error. Show quantile-binned trends (or a light rasterized density plus trends), not 4,000 opaque points. Report Spearman rho and temporal block-bootstrap 95% interval. Describe spread as *associated with* error, never as predicting it.

### d — Actual eight-method native accuracy–latency

At `Cond_T`, M=256, N=40,300, batch size 1, float32, plot synchronized warm median latency (log x; IQR horizontally) against the frozen 1,000-state mean relative-L2 over the four unobserved fields (temporal block-bootstrap 95% interval vertically). Join every result by exact canonical checkpoint identity. Use all available methods in the exact paper order; record a canonically unbenchmarkable method as `unavailable` in the companion rather than substituting an algorithm. Use the adopted Figure 4 colors and highlight DMF-Gen without hiding competitors.

### e — Combined query/memory micro-axes

For DMF-Gen at M=256 and N=1,024, 4,096, 16,384, and 40,300, place aligned latency-versus-N and peak-allocated-GPU-memory-versus-N micro-axes beneath one panel label. Use real coordinates/matching truth subsets, the adopted checkpoint, log x, latency IQR, and mark 40,300 as `native`. Do not add a synthetic 65,536 point or claim a fitted complexity law.

### f — Latency–error measured-NFE path

For a fixed 50-state cohort with common generation seeds, M=256, native N=40,300, plot median synchronized warm latency against mean unobserved-field relative-L2 and connect points in increasing measured vector-field evaluation count. Annotate measured NFE; use latency IQR horizontally and temporal block-bootstrap 95% interval vertically. If multiple solvers appear, distinguish them by marker and never equate nominal steps with NFE.

## Evidence and provenance gate

Every standalone panel and the composed figure require a Markdown companion recording the scientific question, exact source files, checkpoint/run identity, cohort and sensor count, solver and measured NFE, primary results, interval definition, evidence status, and SI destination. The composed companion must connect Figure 5 to the paper's earlier generalization tests across domain, discretization, and measurement content.

Legacy cross-NFE maps, `F0`/`CQ-LR-128`/`S7-B` architecture comparisons, and throughput-extension scaling may remain available for developer QA with a small grey `draft` tag. They cannot enter the strict-formal composition and cannot support main-text scientific claims. Spatial repeated-draw examples move to the SI only after formal data exist. Ablations remain outside Figure 5.

## Output contract

The only figure format is SVG. Required timestamped names are `fig5a_calibration`, `fig5b_sharpness`, `fig5c_spread_error`, `fig5d_accuracy_latency`, `fig5e_query_memory`, `fig5f_nfe_tradeoff`, and `fig5_composed_v2`, each followed by `_YYYYMMDD_HHMM.svg`.
