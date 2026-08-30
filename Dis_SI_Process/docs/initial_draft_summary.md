# Figure 5 initial-draft summary

## Completed

- Created the root-level `Dis_SI_Process/` workflow with separated configs, scripts, reusable utilities, generated figures, derived summaries, tests, and documentation.
- Implemented eight standalone panel renderers (a–h) and one native Matplotlib composed figure.
- Added formal-result adapters for the planned `ValidationV2/Uncertainty` and `ValidationV2/Cost` schemas, plus a `--strict-formal` gate.
- Generated SVG-only timestamped outputs and one companion Markdown file for every panel and the composed draft.
- Added structural QA for SVG parseability, fixed canvases, editable text, evidence-status badges, and the absence of PDF outputs.

## Existing data reused in place

- A real 40,300-point turbulent-combustion truth/reconstruction triplet at measured NFE 1/2/4 supplies the panel-a reconstruction and the explicitly labelled cross-NFE solver-sensitivity proxy.
- The same real arrays supply the one-state spatial sensitivity–error layout proxy in panel d.
- An existing formal architecture cost/quality table supplies the provisional Pareto grammar and two-point NFE diagnostic in panels e and h.
- Existing cached-streamed real-checkpoint reconstruction-scaling measurements supply the provisional latency and peak-memory trends in panels f and g.
- The adopted 1,000-state FieldL2 summary is inventoried as the future native error join source; it is not forced into a latency plot before matching benchmark identities exist.

No checkpoint, HDF5 dataset, reconstruction array, or existing figure is copied into `Dis_SI_Process/`. Only lightweight display tables are written under the ignored `results/` tree.

## Blocked pending formal validation

- Panels b and c require the U2 200-state × 64-draw calibration output with central 50/80/90/95% state-level coverage and physical-unit interval width.
- Panel a requires a predeclared visual case with truth, ensemble mean, absolute error, and true ensemble standard deviation.
- Panel d requires U1 state-level spatial RMS ensemble standard deviation and ensemble-mean relative L2, with temporal block-bootstrap inference.
- Panel e requires synchronized warm native-mesh latency for the exact eight adopted Cond_T checkpoints joined to the frozen FieldL2 identities.
- Panels f and g require the adopted DMF checkpoint at the planned native-supported query counts rather than throughput-extension proxies.
- Panel h requires a fixed 50-state error sweep indexed by measured vector-field calls, common generation seeds, and matching latency metadata.

Until those products pass the Process Plan V2 gates, proxy badges and pending panels are intentional and must remain visible.
