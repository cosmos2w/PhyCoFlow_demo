# Mixed-resolution super-resolution figure V5.1: technical companion

Reference figure: [PDF](MixedResolution_unified_v5_1_20260926_1359.pdf),
[SVG](MixedResolution_unified_v5_1_20260926_1359.svg), and
[PNG](MixedResolution_unified_v5_1_20260926_1359.png)

Machine-readable provenance:
[FigureSourceManifest_unified_v5_1_20260926_1359.json](FigureSourceManifest_unified_v5_1_20260926_1359.json)

Layout and source review:
[LAYOUT_QA.json](../../../../figures/generated/art_style_review/MixedResolution_unified_v5_1_20260926_1359/LAYOUT_QA.json),
[SCIENTIFIC_STATE_COMPARISON.json](../../../../figures/generated/art_style_review/MixedResolution_unified_v5_1_20260926_1359/SCIENTIFIC_STATE_COMPARISON.json), and
[SOURCE_LOCK.json](../../../../figures/generated/art_style_review/MixedResolution_unified_v5_1_20260926_1359/SOURCE_LOCK.json)

**Status:** the V5.1 artwork and source-continuity checks passed. Scientific release remains
pending the existing A03/A04 author decisions. The V3-7 companion is retained beside this
document as the record for that earlier layout.

## Technical summary

This figure tests whether neural field-reconstruction models can recover a high-resolution
(`H`, 128 × 128) density field from sparse point measurements when training trajectories are
distributed across low-, medium-, and high-resolution simulations. The six panels separate
the evidence into:

1. native L/M/H spatial discretizations and five training-data recipes (**a**);
2. mean H-grid reconstruction error for all five recipes at 512 sensors (**b**);
3. mean error versus 64–512 nested sensors for Mixed-HML, Zero-H-balanced, and
   Zero-H-M-rich, stacked in the right column (**c**);
4. one fixed Zero-H-M-rich full-field, common-ROI, and local-error comparison across all four
   models (**d**);
5. large-, intermediate-, and fine-scale truth components and **absolute residual magnitudes**
   for DMF-Gen and the strongest non-DMF zero-H baseline at 256 sensors (**e**); and
6. population-level multiscale pattern correlation and variance-allocation bias for all four
   models and the three mixed/zero-H recipes (**f**).

The principal result is that **DMF-Gen retains the strongest H-resolution reconstruction when
H-resolution training fields are reduced or removed**. At 512 sensors, its mean physical
relative \(L_2\) error is 0.0119 for Mixed-HML, 0.0334 for Zero-H-balanced, and 0.0288 for
Zero-H-M-rich. The M-rich zero-H recipe lowers DMF-Gen error by 13.7% relative to balanced
zero-H training. Under Zero-H-M-rich training, DMF-Gen also preserves fine-scale spatial
structure most faithfully: median fine-scale pattern correlation is 0.815, compared with
0.368 for FFM-Perceiver, −0.089 for Senseiver, and 0.165 for MLP-RBF. Its median fine-scale
variance-allocation bias is only +0.003 percentage points.

These are descriptive comparisons on a shared held-out cohort, not causal estimates. The
figure controls the evaluated case–time pairs and sensor plans, but it does not isolate
architecture from every difference in optimization trajectory, learned checkpoint, or model
capacity. V5.1 changes composition and selected display labels; the data, estimates, and
intervals come from the validated earlier analyses.

## How to read the overall organization

The visual argument proceeds from experimental design to aggregate performance, then to
local and scale-resolved evidence. The 180 × 176 mm canvas uses a 116 mm left column for
**a/b**, a 60 mm right column for **c**, a shared 114/62.6 mm middle-row split for **d/e**,
and a full-width **f**. Thus the three sensor-screening curves stay visible without their own
full-width row.

| Figure region | Panels | Role in the argument |
|---|---:|---|
| Upper left | a–b | Defines discretizations and training budgets, then compares all recipes at 512 sensors. |
| Upper right | c | Shows three recipe-specific sensor-response curves with one shared model legend and log y scale. |
| Middle left | d | Localizes model errors in one fixed Zero-H-M-rich state. |
| Middle right | e | Shows scale components and absolute residual magnitude for the same state at a different sensor budget. |
| Bottom | f | Tests the multiscale interpretation across the 300-snapshot cohort. |

The panels answer complementary questions:

- **Panel a:** What spatial information is available during training, and at what spatial-DOF
  budget?
- **Panel b:** How does the training recipe affect H-grid error at 512 sensors?
- **Panel c:** How do the three displayed recipes respond to additional sensors?
- **Panel d:** Where are errors located in a shared high-gradient region?
- **Panel e:** Which spatial scales and error-magnitude patterns distinguish DMF-Gen from the
  strongest non-DMF zero-H baseline?
- **Panel f:** Do the scale-specific observations persist across the held-out population?

The complete five-recipe sweeps (including H-only and H-limited), expanded multi-recipe
qualitative gallery, full three-scale qualitative display, and extended multiscale
quantitative evidence remain available in the Supplementary Information.

## Data and experimental scope

### Physical field and resolutions

The processed field is the physical **density** channel from the multiresolution CFD dataset.
Formal errors are evaluated in physical units on the H-resolution grid.

| Resolution | Symbol | Native grid | Grid points |
|---|---:|---:|---:|
| Low | L | 32 × 32 | 1,024 |
| Medium | M | 64 × 64 | 4,096 |
| High | H | 128 × 128 | 16,384 |

Panel a uses the same state on all three native grids: case 9005, time index 15, physical time
0.775. The three maps therefore contain 21,504 native field values in total. Their common ROI
is fixed at \(x\in[-0.6071,-0.1189]\), \(y\in[0.0529,0.5411]\) and was placed on a steep
red–blue transition. The maps are displayed as native cells without interpolation.

### Training recipes

Recipe ratios are ordered as **L:M:H**. “Training cases” are active trajectories, each of
which contributes multiple temporal snapshots.

| Recipe | L:M:H | Training cases (L/M/H) | Active cases | Training snapshots | Spatial DOF budget | Fraction of H-only DOF |
|---|---:|---:|---:|---:|---:|---:|
| H-only | 0:0:1 | 0 / 0 / 9,000 | 9,000 | 144,000 | 147,456,000 | 1.0000 |
| H-limited | 0:0:1 | 0 / 0 / 3,060 | 3,060 | 48,960 | 50,135,040 | 0.3400 |
| Mixed-HML | 1:1:1 | 3,000 / 3,000 / 3,000 | 9,000 | 144,000 | 64,512,000 | 0.4375 |
| Zero-H-balanced | 1:1:0 | 4,500 / 4,500 / 0 | 9,000 | 144,000 | 23,040,000 | 0.1563 |
| Zero-H-M-rich | 1:2:0 | 3,000 / 6,000 / 0 | 9,000 | 144,000 | 27,648,000 | 0.1875 |

The spatial degree-of-freedom budget is

\[
B_{\mathrm{DOF}}=N_L(32^2)+N_M(64^2)+N_H(128^2),
\]

where \(N_L,N_M,N_H\) are the numbers of training trajectories assigned to each native
resolution. This is an exposure measure, not a direct measure of GPU time, memory, energy,
optimization steps, or wall-clock cost.

### Held-out evaluation cohort

- The test split contains 1,000 cases.
- Quantitative summaries use **300 distinct held-out CFD case–time pairs**, selected with the
  `stratified_unique_cases` strategy and seed 42.
- One deterministic time is retained per selected case from the common valid time window.
- The statistical sampling unit is a case–time pair, not a grid point.
- Sensor plans and stochastic generation seeds are paired across models, recipes, and sensor
  counts.
- Each physical relative \(L_2\) error summarizes all 16,384 points of an H-grid field.
- Mean-error confidence intervals use 2,000 case-level bootstrap resamples.

### Sparse-observation budgets

The displayed sensor counts are nested subsets of the canonical H-grid sensor plan:

| Sensors | Fraction of the 128 × 128 H grid |
|---:|---:|
| 64 | 0.390625% |
| 128 | 0.781250% |
| 256 | 1.562500% |
| 384 | 2.343750% |
| 512 | 3.125000% |

The formal default is 256 sensors. Panel **b** uses 512 sensors; panel **c** spans 64–512;
the fixed-state panels **d** and **e** use 512 and 256 sensors, respectively. The nested
design adds observations to one canonical plan rather than comparing unrelated masks.

## Models processed

| Figure label | Saved family | Saved backbone | Cached inference contract |
|---|---|---|---|
| DMF-Gen | `pointcloud_ffm` | `GL_rbf_ENH` | Two Euler function evaluations; hard/default observation consistency |
| FFM-Perceiver | `pointcloud_ffm` | `perceiver` | Two Euler function evaluations; smooth endpoint consistency |
| Senseiver | deterministic | `senseiver` | Native deterministic inference; smooth endpoint consistency |
| MLP-RBF | deterministic | `mlp_rbf` | Native deterministic inference; smooth endpoint consistency |

Panels b, c, d, and f include all four models. Panel e contrasts DMF-Gen with Senseiver.
Senseiver is the validated best non-DMF baseline under Zero-H-M-rich training at 256 sensors:
mean physical relative \(L_2\) is 0.07718 for Senseiver, 0.09490 for FFM-Perceiver, and
0.16542 for MLP-RBF. V5.1 performs no training or inference.

## Metric definitions and interpretation

### Physical relative \(L_2\) error

For ground truth \(u\) and reconstruction \(\hat u\), both in physical density units,

\[
E_{\mathrm{rel}L_2}=\frac{\|\hat u-u\|_2}{\|u\|_2+\epsilon},
\qquad \epsilon=10^{-12}.
\]

Lower is better. This global, amplitude-sensitive measure does not reveal the spatial
location or scale of the error.

### Zoom-region relative \(L_2\) and local absolute error

Panel d reports the relative \(L_2\) within the common ROI and displays

\[
e_{\mathrm{abs}}(x,y)=|\hat u(x,y)-u(x,y)|.
\]

The error scale is shared across the four model maps and spans 0 to 0.012677 density units.
Because these annotations are ROI-specific, they should not be substituted for the full-field
values reported separately in this companion.

### Scale-component residual and its absolute display

The underlying scale-component residual is

\[
r_s(x,y)=\hat u_s(x,y)-u_s(x,y),
\]

for scale group \(s\). **Panel e displays \(|r_s(x,y)|\)**, so the residual maps show error
magnitude and do not encode whether a prediction is above or below truth. The two model maps
in each row share a nonnegative, scale-specific display range; display clipping does not
alter the metrics calculated from the cached component arrays. This absolute display is
recorded in the V5.1 manifest and must be stated explicitly in the caption.

### Scale-component relative \(L_2\)

\[
E_{\mathrm{rel}L_2,s}=\frac{\|\hat u_s-u_s\|_2}{\|u_s\|_2+\epsilon}.
\]

This quantity can exceed one when the true component carries little energy. Fine-scale values
must therefore be interpreted alongside pattern correlation and variance allocation.

### Spatial pattern correlation

\[
\rho_s=\frac{\langle\hat u_s,u_s\rangle}
{\|\hat u_s\|_2\,\|u_s\|_2+\epsilon}.
\]

This implementation does not subtract the spatial mean, so it is an uncentered cosine
similarity rather than Pearson correlation. Higher is better; negative values indicate
oppositely aligned component patterns.

### Variance-allocation bias

The variance fraction assigned to scale \(s\) is

\[
f_s(u)=\frac{\|u_s\|_2^2+\epsilon}
{\sum_k(\|u_k\|_2^2+\epsilon)},
\]

and panel f reports

\[
\Delta f_s^{(\mathrm{pp})}=100[f_s(\hat u)-f_s(u)].
\]

Zero is ideal. A positive value means the reconstruction allocates too much of its total
variance to that scale; a negative value means too little. The three biases sum to
approximately zero for each reconstruction.

## Panel-by-panel explanation and generation

## Panel a — resolution protocol and training-data budget

### What it shows

The upper strip shows the same density state on the L, M, and H grids. Each map contains the
same physical ROI, an enlarged native-cell inset, and connectors that make the change in
spatial discretization explicit. The lower stacked bars show the L/M/H case composition of
each recipe. Labels above the bars give spatial-field exposure relative to H-only:
**1.00×, 0.34×, 0.44×, 0.16×, and 0.19×**, in recipe order. The five recipe positions
align exactly with panel b; their names appear once below b.

The panel distinguishes reduction in the number of H trajectories (H-limited) from changes
in resolution composition (Mixed-HML and the zero-H recipes). Exposure is a count of native
spatial degrees of freedom, not measured training cost.

### How it is generated

1. [`10_export_resolution_protocol.py`](../../_Scripts/10_export_resolution_protocol.py)
   provides the validated native grids, common state, sensor projection, and recipe budgets.
2. The inherited
   [`common/publication_panels_unified_v4_7.py`](../../_Scripts/common/publication_panels_unified_v4_7.py)
   drawer renders the native-cell maps, common ROI, insets, connectors, and stacked bars.
3. The [V5.1 renderer](../../../../figures/scripts/render_mixed_resolution_v5_1.py) moves
   the three maps above the aligned recipe bars and records geometry and collision checks.

### Data volume

- One qualitative case–time pair.
- 21,504 native field values across L, M, and H.
- Five validated recipe-budget records.
- 726 projected sensor records in the protocol table: 220 unique L, 250 unique M, and 256 H
  locations at the formal sensor budget.

## Panel b — five-recipe transfer at 512 sensors

### What it shows

The grouped bar chart compares all five recipes and four models at 512 sensors on a linear
y-axis. Bars are population means and whiskers are case-bootstrap 95% confidence intervals.
The last two recipe groups, which contain no H-resolution training fields, share a subtle
background tint. The five recipe names label the positions shared with panel a.

### How it is generated

1. The validated sensor-sweep table is filtered to 512 sensors and five recipes.
2. Means and case-bootstrap 95% intervals are drawn directly from that accepted summary.
3. The V5.1 renderer moves this bar chart below panel a without changing its estimates or
   intervals. It performs no inference or aggregate-metric recomputation.

Primary table:
[`SensorSweepAllRecipes_summary_20260806_1124.csv`](../../_Process_Results/UnifiedPublicationV2/SensorSweepAllRecipes_summary_20260806_1124.csv)

### Data volume

- 4 models × 5 recipes = **20 aggregate bars**.
- Every bar has `valid_n = 300` and summarizes H-grid errors over 16,384 points per case.
- Twelve of these 512-sensor aggregates also appear at the endpoints of panel c; they are
  the same source records, not independently estimated values.

### Exact 512-sensor means and 95% confidence intervals

| Model | H-only | H-limited | Mixed-HML | Zero-H-balanced | Zero-H-M-rich |
|---|---:|---:|---:|---:|---:|
| DMF-Gen | 0.0121 [0.0113, 0.0129] | 0.0129 [0.0121, 0.0137] | 0.0119 [0.0111, 0.0127] | 0.0334 [0.0319, 0.0350] | 0.0288 [0.0272, 0.0304] |
| FFM-Perceiver | 0.0722 [0.0698, 0.0746] | 0.1455 [0.1353, 0.1566] | 0.0748 [0.0719, 0.0777] | 0.0791 [0.0763, 0.0822] | 0.0767 [0.0738, 0.0797] |
| Senseiver | 0.0134 [0.0127, 0.0141] | 0.0169 [0.0161, 0.0178] | 0.0296 [0.0281, 0.0311] | 0.0818 [0.0797, 0.0838] | 0.0729 [0.0711, 0.0749] |
| MLP-RBF | 0.0855 [0.0832, 0.0879] | 0.0990 [0.0964, 0.1016] | 0.0922 [0.0897, 0.0946] | 0.1563 [0.1524, 0.1604] | 0.1398 [0.1366, 0.1435] |

At 512 sensors, DMF-Gen has the lowest mean error in all five recipes. Relative to H-limited,
Mixed-HML reduces mean error by 8.0% for DMF-Gen, 48.6% for FFM-Perceiver, and 6.9% for
MLP-RBF, while Senseiver error increases by 74.5%. Relative to Zero-H-balanced,
Zero-H-M-rich reduces error by 13.7%, 3.1%, 10.9%, and 10.5% for DMF-Gen,
FFM-Perceiver, Senseiver, and MLP-RBF, respectively. These recipe effects are
architecture-dependent and should not be described as universal.

## Panel c — sensor screening for three recipes

### What it shows and how it is generated

Three vertically stacked log-y plots show Mixed-HML, Zero-H-balanced, and Zero-H-M-rich in
that order. Each contains all four model means at **64, 128, 256, 384, and 512** nested
sensors. The source table is the same validated
[`SensorSweepAllRecipes_summary_20260806_1124.csv`](../../_Process_Results/UnifiedPublicationV2/SensorSweepAllRecipes_summary_20260806_1124.csv)
used in panel b. A single four-model legend serves all three plots. Sensor count and its
fraction of the H grid share the bottom x-axis; the three plots use one physical relative
\(L_2\) log-y range. Curves encode means; the source table carries the case-bootstrap
intervals.

Panel c contains **60 plotted aggregate points** (4 models × 3 recipes × 5 budgets), each
with `valid_n = 300`. Together b and c show 80 marks drawn from **68 distinct summary
records**, because 12 three-recipe, 512-sensor values are shown in both. H-only and
H-limited sweeps remain in SI Figure Sx1 and its LaTeX table.

### Exact three-recipe sweep means

Mixed-HML:

| Model | 64 | 128 | 256 | 384 | 512 | Reduction, 64→512 |
|---|---:|---:|---:|---:|---:|---:|
| DMF-Gen | 0.1183 | 0.0487 | 0.0215 | 0.0147 | 0.0119 | 90.0% |
| FFM-Perceiver | 0.2114 | 0.1407 | 0.0965 | 0.0821 | 0.0748 | 64.6% |
| Senseiver | 0.1333 | 0.0616 | 0.0375 | 0.0321 | 0.0296 | 77.8% |
| MLP-RBF | 0.3600 | 0.2091 | 0.1263 | 0.1030 | 0.0922 | 74.4% |

Zero-H-balanced:

| Model | 64 | 128 | 256 | 384 | 512 | Reduction, 64→512 |
|---|---:|---:|---:|---:|---:|---:|
| DMF-Gen | 0.1130 | 0.0601 | 0.0405 | 0.0354 | 0.0334 | 70.5% |
| FFM-Perceiver | 0.2024 | 0.1405 | 0.1023 | 0.0876 | 0.0791 | 60.9% |
| Senseiver | 0.1503 | 0.1023 | 0.0862 | 0.0831 | 0.0818 | 45.6% |
| MLP-RBF | 0.3831 | 0.2503 | 0.1808 | 0.1638 | 0.1563 | 59.2% |

Zero-H-M-rich:

| Model | 64 | 128 | 256 | 384 | 512 | Reduction, 64→512 |
|---|---:|---:|---:|---:|---:|---:|
| DMF-Gen | 0.1122 | 0.0566 | 0.0359 | 0.0308 | 0.0288 | 74.3% |
| FFM-Perceiver | 0.1912 | 0.1281 | 0.0949 | 0.0834 | 0.0767 | 59.9% |
| Senseiver | 0.1444 | 0.0948 | 0.0772 | 0.0740 | 0.0729 | 49.5% |
| MLP-RBF | 0.3739 | 0.2363 | 0.1654 | 0.1476 | 0.1398 | 62.6% |

DMF-Gen has the lowest mean error at every displayed sensor count in **all three** recipes.
The improvement from additional sensors is recipe- and architecture-dependent; the plotted
means should not be read as training-seed uncertainty. The two zero-H curves show diminishing
returns at the largest budgets, particularly for the deterministic baselines.

## Panel d — fixed Zero-H-M-rich physical example

### What it shows

Five columns compare ground truth, DMF-Gen, FFM-Perceiver, Senseiver, and MLP-RBF for one
shared Zero-H-M-rich reconstruction state. The rows show:

1. the complete H-resolution density field;
2. the same high-gradient zoom for every column; and
3. the 512-sensor layout in the reference column and local absolute error for each model.

All field tiles share one diverging normalization, and all error tiles share one sequential
normalization. Frustum connectors identify the common ROI. Therefore, apparent differences
are not produced by model-specific crops or rescaling.

### Fixed state and validated source identity

- Recipe: Zero-H-M-rich (`5_ZeroH_MRich`).
- Snapshot index: 50; case 9160; time index 18; physical time 0.925.
- Sensor count: 512; 22 sensors fall inside the displayed ROI.
- ROI: \(x\in[0.2756,0.7638]\), \(y\in[-0.1496,0.3386]\), selected by maximum
  integrated ground-truth gradient magnitude.
- Sensor plan: `canonical_h_nested_v1`, SHA-256
  `05fdd36e16ff23d32c199e46d9a355c95f6fa4ab3f9814b831e46266206c6a9c`.
- The exact MLP-RBF reconstruction was verified before plotting:
  `RecCache_s0050_n512_14884cfb124a2963.npz`, SHA-256
  `c7ad016bc2fca95db1bd041dfa4eb3679798ea2234e44a3ca547de44ce12e495`.

### Exact errors

| Model | Full-field physical relative \(L_2\) | Zoom-region relative \(L_2\) |
|---|---:|---:|
| DMF-Gen | 0.01972 | 0.03165 |
| FFM-Perceiver | 0.06039 | 0.09046 |
| Senseiver | 0.06330 | 0.08605 |
| MLP-RBF | 0.12470 | 0.13814 |

DMF-Gen is best on both the full field and the selected local region for this snapshot. Its
zoom-region error is 63.2% lower than Senseiver's and 65.0% lower than FFM-Perceiver's. This
is a qualitative example, not the population estimator; panels b/c provide aggregate
full-field comparisons and panel f provides the aggregate multiscale comparison.

### Data volume

- One held-out case–time pair.
- Four validated 512-sensor reconstructions plus one shared truth.
- 16,384 H-grid values per truth or reconstruction.
- Four local absolute-error maps on a common 1,024-point ROI.

## Panel e — qualitative multiscale residual evidence

### What it shows

The shared Zero-H-M-rich state from panel d is decomposed into three wavelet scale groups.
Each row shows the ground-truth component and the **absolute residual magnitude** for
DMF-Gen and Senseiver:

- **Large:** approximation plus level-4 detail; structures approximately ≥16 H-grid cells.
- **Intermediate:** level-3 detail; structures approximately 8–16 H-grid cells.
- **Fine:** levels 1–2 detail; structures approximately 2–8 H-grid cells.

The reconstruction budget is 256 sensors. Senseiver was chosen before plotting as the best
non-DMF baseline by mean Zero-H-M-rich physical relative \(L_2\) at this sensor count. The
selection did not use this snapshot's scale-specific performance.

### How it is generated

1. A four-level orthogonal `db2` discrete wavelet transform is applied with periodization
   boundary handling.
2. Snapshot 50 (case 9160, time index 18) is fixed by the shared qualitative layout contract.
3. DMF-Gen and Senseiver reconstructions are read from validated 256-sensor caches.
4. Truth and predictions are decomposed identically. The display takes the absolute value
   of the signed prediction-minus-truth component residual; zero maps to the dark endpoint.
5. The two residual tiles within a scale row share a nonnegative range from zero to the
   scale-specific display limit. Visual clipping affects neither the source arrays nor the
   numerical relative \(L_2\) values.

Primary metadata:
[`MultiscaleWavelet_metadata_20260802_1250.json`](../../_Process_Results/MultiscaleWavelet/MultiscaleWavelet_metadata_20260802_1250.json)

### Exact scale-component relative \(L_2\)

| Model | Large | Intermediate | Fine |
|---|---:|---:|---:|
| DMF-Gen | 0.0214 | 0.2129 | 0.6832 |
| Senseiver | 0.0513 | 0.3806 | 4.8057 |

Senseiver's error is approximately 2.4×, 1.8×, and 7.0× DMF-Gen's error at large,
intermediate, and fine scales, respectively. Panel e supports spatial interpretation;
population-level claims should be based on panel f.

### Data volume

- One shared case–time pair.
- Two 256-sensor model reconstructions plus one truth.
- Three truth-component maps and six **absolute-residual** display maps.
- No quantitative line plots appear in panel e; the aggregate multiscale evidence is in
  panel f and SI. The displayed component-relative \(L_2\) annotations come from the
  unchanged cached arrays.

## Panel f — population-level multiscale fidelity

### What it shows

The left three matrices report median spatial pattern correlation; the right three report
median variance-allocation bias in percentage points. Within each metric, separate 4 × 3
sub-axes correspond to Mixed-HML, Zero-H-balanced, and Zero-H-M-rich. Rows are models and
columns are Large, Intermediate, and Fine scales.

The two metrics use separate heatmap normalizations, but panel f intentionally omits numeric
colorbars because every cell is annotated directly. `Spatial pattern correlation` and
`Variance allocation bias [pp]` appear above their respective matrix blocks. Every displayed
cell is a validated median over 300 case–time pairs. The compact x-axis label **Mid** means
**Intermediate**; Large and Fine are unabbreviated.

### How it is generated

1. Every cached truth and reconstruction is decomposed with the same `db2`, level-4,
   periodized wavelet transform used for panel e.
2. Pattern correlation and variance-allocation bias are selected from the validated
   multiscale summary.
3. The panel reads medians directly; it does not recompute metrics.
4. Correlation uses a fixed [−0.1, 1.0] range. Bias uses a symmetric range of
   [−1.3635, +1.3635] percentage points.

Primary table:
[`MultiscaleWavelet_summary_20260802_1250.csv`](../../_Process_Results/MultiscaleWavelet/MultiscaleWavelet_summary_20260802_1250.csv)

### Data volume

- 4 models × 3 recipes × 300 snapshots = **3,600 reconstruction records**.
- Each reconstruction contributes three scale groups.
- The two metrics represent 21,600 underlying scalar observations.
- The main panel contains 72 annotated aggregate cells, all with `valid_n = 300`.

### Median spatial pattern correlation

Each triplet is Large / Intermediate / Fine.

| Model | Mixed-HML | Zero-H-balanced | Zero-H-M-rich |
|---|---:|---:|---:|
| DMF-Gen | 1.000 / 0.985 / 0.930 | 1.000 / 0.951 / 0.460 | 1.000 / 0.969 / 0.815 |
| FFM-Perceiver | 0.997 / 0.755 / 0.372 | 0.996 / 0.738 / 0.324 | 0.997 / 0.761 / 0.368 |
| Senseiver | 1.000 / 0.972 / 0.836 | 0.999 / 0.944 / 0.072 | 0.998 / 0.931 / −0.089 |
| MLP-RBF | 0.994 / 0.614 / 0.252 | 0.993 / 0.518 / 0.143 | 0.993 / 0.530 / 0.165 |

### Median variance-allocation bias (percentage points)

Each triplet is Large / Intermediate / Fine.

| Model | Mixed-HML | Zero-H-balanced | Zero-H-M-rich |
|---|---:|---:|---:|
| DMF-Gen | +0.001 / −0.002 / +0.000 | −0.025 / +0.001 / +0.029 | +0.001 / −0.004 / +0.003 |
| FFM-Perceiver | −0.088 / +0.044 / +0.046 | −0.117 / +0.054 / +0.066 | −0.080 / +0.036 / +0.046 |
| Senseiver | +0.004 / −0.005 / +0.001 | −0.341 / −0.010 / +0.369 | −0.141 / +0.002 / +0.147 |
| MLP-RBF | −0.252 / +0.137 / +0.121 | −1.447 / +0.230 / +1.209 | −1.041 / +0.197 / +0.853 |

Large-scale pattern correlation is high for every condition, so the main discrimination lies
at intermediate and fine scales. Under Zero-H-M-rich training, DMF-Gen's fine-scale
correlation is 0.815, more than twice FFM-Perceiver's 0.368 and well above MLP-RBF's 0.165;
Senseiver is negatively aligned at −0.089. DMF-Gen also avoids the marked transfer of
variance from large to fine scales visible for the deterministic baselines.

## Supplementary evidence retained from the V3-7 scientific release

| SI item | Contents | Paper-writing use |
|---|---|---|
| [Figure Sx1](../UnifiedV3_7/20260914_1158/si/SI_Sx1_complete_sensor_sweeps_20260914_1158.pdf) | Complete five-recipe, four-model sensor sweeps from 64 to 512 sensors. | Inspect H-only and H-limited sweeps omitted from panel c. |
| [Figure Sx2](../UnifiedV3_7/20260914_1158/si/SI_Sx2_expanded_recipe_gallery_20260914_1158.pdf) | All four models across Mixed-HML, Zero-H-balanced, and Zero-H-M-rich at 256 sensors. | Compare panel d's focused state with a broader qualitative gallery. |
| [Figure Sx3](../UnifiedV3_7/20260914_1158/si/SI_Sx3_complete_three_scale_qualitative_20260914_1158.pdf) | Complete Large/Intermediate/Fine truth and residual display with row-specific colorbars. | Inspect the earlier signed-residual view; V5.1 panel e displays absolute residual magnitude. |
| [Figure Sx4](../UnifiedV3_7/20260914_1158/si/SI_Sx4_complete_three_scale_quantitative_20260914_1158.pdf) | Extended three-scale quantitative matrices/distributions. | Support the population-level multiscale interpretation and uncertainty reporting. |

Machine-readable manuscript tables are also supplied:

- [512-sensor accuracy](../UnifiedV3_7/20260914_1158/tables/accuracy_512_20260914_1158.tex)
- [complete 64–512 sensor sweeps](../UnifiedV3_7/20260914_1158/tables/sensor_sweeps_64_512_20260914_1158.tex)
- [pattern correlations at all scales](../UnifiedV3_7/20260914_1158/tables/pattern_correlations_all_scales_20260914_1158.tex)
- [variance-allocation bias at all scales](../UnifiedV3_7/20260914_1158/tables/variance_allocation_bias_all_scales_20260914_1158.tex)

## End-to-end generation pipeline

V5.1 is a native, one-canvas Matplotlib composition rather than a screenshot assembly. It
reuses the validated quantitative summaries and reconstruction caches; the V4.7 scientific
figure is its explicit scientific anchor, and V5.0 is its visual baseline.

1. Existing validated protocol tables, aggregate summaries, shared truth/grid arrays, sensor
   plans, and reconstruction caches are resolved by the local post-processing configuration.
2. The inherited
   [`common/publication_panels_unified_v4_7.py`](../../_Scripts/common/publication_panels_unified_v4_7.py)
   drawers produce the validated quantitative and qualitative artists. Their displayed
   local and scale-component metrics are deterministically rederived from unchanged caches.
3. The [V5.1 renderer](../../../../figures/scripts/render_mixed_resolution_v5_1.py) and
   [layout configuration](../../_Scripts/publication_layout_unified_v5_1.yaml) compose the
   six panels on a **180 × 176 mm** canvas, separate the original grouped-bar and
   sensor-sweep evidence into b/c, and export PDF/SVG/PNG.
4. The renderer compares key quantitative values with the V4.7 scientific manifest and the
   V5.0 panel manifest, verifies frozen source hashes, and writes source snapshots, layout
   checks, and a source lock into the [review bundle](../../../../figures/generated/art_style_review/MixedResolution_unified_v5_1_20260926_1359/figure_contract.md).

Relative to V5.0's 180 × 210 mm canvas, V5.1 removes **34 mm (16.2%)** of height. The
rendered a–b, upper-to-middle, and middle-to-f artist gaps are **1.424, 1.402, and 1.884
mm**, respectively, or **35.1%, 35.0%, and 47.3%** of V5.0. At 162 mm insertion width,
the smallest measured major row gap is **1.262 mm**. The d/e evidence rows share exact
physical boundaries. The 600 dpi PNG is **4,251 × 4,157 px**, and the PDF embeds Arial
font subsets. The V5.1 layout, annotation, typography, model-color, and panel-boundary
gates all passed.

## Numerical validation and provenance checks

- Every V5.1 panel reports status `ok`; all panel-f cells have `valid_n = 300`.
- Panels b/c draw **20 bars and 60 sweep points** from **68 unique summary rows**.
- Six thousand reconstruction-cache entries and 18,000 per-snapshot wavelet rows were
  processed in the earlier validated multiscale run; V5.1 does not repeat that run.
- Maximum truth reconstruction residual after summing wavelet groups:
  \(8.81\times10^{-16}\).
- Maximum prediction reconstruction residual: \(9.40\times10^{-16}\).
- Maximum prediction and truth variance-fraction sum error: \(2.22\times10^{-16}\).
- Truth-versus-truth component relative \(L_2\): 0.
- Maximum truth-versus-truth pattern-correlation error: \(8.75\times10^{-8}\).
- Truth-versus-truth variance-allocation bias: 0 percentage points.
- Wavelet reconstruction tolerance: \(10^{-6}\).
- The exact MLP-RBF representative cache is present and hash-verified.
- The V5.1 renderer verified six formal source-data records and six cache-source records
  against the frozen V4.7 hashes. Its V4.7 scientific and V5.0 continuity comparisons
  both passed, and the validated process-result tree remained unchanged.
- No training, inference, new prediction rows, or new aggregate evaluations were performed.
  The inherited drawer did recompute **display annotations** for the fixed cached fields;
  their values matched the earlier scientific figure.
- V4.7 and V5.0 remain intact as intentional comparison baselines. V5.1 review status is
  **art reviewed, scientific release pending A03/A04**.

Main-output SHA-256 hashes:

| Artifact | SHA-256 |
|---|---|
| PDF | `cdfb02abd97d1e6dc97276cc54a4f39c1373e006f7591c942d7e746c51713e74` |
| SVG | `7753c93a7a587afac915beab2247382ea806589de1fd728cf5006c507510360b` |
| PNG | `992241390cb43bbee070c3037d4be5ab82d83c9f0eebe70637e1290a74ee421c` |
| Assembled source manifest | `ae466be123b3221908a5f51ceb9c9a3aecbfedc31819cb5ade61c665e8a161c1` |

## Limitations and interpretation boundaries

1. **Panels d and e use one fixed example.** Their purpose is spatial interpretation,
   not population estimation. Use b/c and f for aggregate claims.
2. **The bootstrap covers held-out cases, not retraining uncertainty.** Intervals do not
   quantify random initialization, checkpoint-selection, or repeat-training variability.
3. **Pattern correlation is uncentered.** It is cosine alignment of scale components, not
   conventional mean-subtracted Pearson correlation.
4. **Fine-scale relative errors can be large when true fine-scale energy is small.** Read
   panel e together with panel f's pattern correlation and variance-allocation bias.
5. **Wavelet boundary handling assumes periodicity.** Dataset metadata does not explicitly
   state boundary conditions; periodization follows the existing unwindowed spectral
   convention.
6. **Display clipping is not metric clipping.** Robust visual limits do not alter the
   numerical calculations. Panel e displays absolute residual magnitude and therefore
   cannot support claims about the sign of local errors.
7. **Spatial DOF is not computational cost.** Do not translate panel-a exposure ratios
   directly into training time, memory, or energy savings.
8. **Recipe effects are architecture-dependent.** Mixed-HML improves FFM-Perceiver strongly
   relative to H-limited but degrades Senseiver; claims must be method-specific.
9. **The evaluated field is density.** Generalization to other channels, flows, or datasets
   requires separate validation.
10. **No formal hypothesis-test multiplicity analysis is shown.** Confidence intervals
    describe sampling uncertainty of means; avoid language implying statistical significance
    unless a separate preregistered or multiplicity-aware test is added.
11. **V5.1 is a visual revision.** Its passed layout and continuity checks do not replace
    the pending A03/A04 scientific-release decisions.

## R&D-section-ready findings

- DMF-Gen has the lowest mean physical relative \(L_2\) in every 512-sensor recipe, including
  both zero-H regimes.
- Mixed-HML nearly preserves DMF-Gen's H-only accuracy while using 43.75% of the H-only
  spatial-DOF exposure: 0.0119 versus 0.0121 mean error at 512 sensors.
- With no H-resolution training fields, M-rich allocation improves all four models relative
  to the balanced L/M recipe at 512 sensors; the largest reduction is DMF-Gen's 13.7%.
- DMF-Gen has the lowest mean error at all five sensor counts in **each of the three**
  panel-c sweeps. Its 64-to-512-sensor reductions are 90.0% (Mixed-HML), 70.5%
  (Zero-H-balanced), and 74.3% (Zero-H-M-rich).
- The fixed qualitative state supports the aggregate ordering: DMF-Gen has the lowest
  full-field and local ROI errors among all four models.
- Large-scale pattern is reconstructed well by every model; intermediate and fine scales are
  more discriminating.
- Under Zero-H-M-rich training, DMF-Gen combines 0.815 median fine-scale pattern correlation
  with +0.003 pp fine-scale variance bias. The corresponding correlations for
  FFM-Perceiver, Senseiver, and MLP-RBF are 0.368, −0.089, and 0.165.
- The strongest claim supported by the full evidence chain is: **DMF-Gen provides the most
  robust H-resolution density reconstruction under mixed-resolution and zero-H training,
  combining low global error with markedly better preservation of intermediate- and
  fine-scale structure.**

## Suggested manuscript language

### Results paragraph scaffold

> Mixed-resolution training affected reconstruction quality in an architecture-dependent
> manner (Fig. 3a–c). At 512 sensors, DMF-Gen achieved the lowest mean physical relative
> \(L_2\) error under all five training recipes, including Zero-H-balanced (0.0334, 95% CI
> 0.0319–0.0350) and Zero-H-M-rich (0.0288, 95% CI 0.0272–0.0304; Fig. 3b).
> Reallocating zero-H trajectories toward medium resolution reduced DMF-Gen mean error by
> 13.7%; across the three displayed 64–512-sensor sweeps it had the lowest mean at every
> budget (Fig. 3c). A fixed held-out example localized error differences to a common
> high-gradient region (Fig. 3d). At 256 sensors, scale decomposition showed smaller
> DMF-Gen component errors than Senseiver at large, intermediate, and fine scales (Fig. 3e).
> Across 300 held-out case–time pairs, DMF-Gen retained a median fine-scale pattern
> correlation of 0.815 under Zero-H-M-rich training, with a median variance-allocation bias
> of +0.003 percentage points (Fig. 3f). Together, these results support stronger
> H-resolution density-structure preservation by DMF-Gen when H-resolution training fields
> are unavailable.

### Caption scaffold

> **Mixed-resolution training supports H-resolution density reconstruction without
> H-resolution training fields.** **a**, Native L (32 × 32), M (64 × 64), and H (128 × 128)
> density fields with a common ROI, and five L:M:H training recipes; numbers above bars
> indicate spatial-DOF exposure relative to H-only. The recipe names below b also label a.
> **b**, Mean physical relative \(L_2\) with case-bootstrap 95% confidence intervals for
> all five recipes at 512 sensors. **c**, Mean physical relative \(L_2\) across 64–512 nested
> sensors for Mixed-HML, Zero-H-balanced, and Zero-H-M-rich training on a shared log scale;
> model colors and markers follow the single legend (\(n=300\) held-out case–time pairs per
> estimate). **d**, Full field, common high-gradient ROI, and local absolute error for one
> fixed Zero-H-M-rich state at 512 sensors; annotations report ROI-relative \(L_2\).
> **e**, Large-, intermediate-, and fine-scale truth components and **absolute residual
> magnitudes** for DMF-Gen and Senseiver at 256 sensors; annotations report scale-component
> relative \(L_2\). **f**, Median spatial pattern correlation and variance-allocation bias
> across 300 held-out case–time pairs for Mixed-HML, Zero-H-balanced, and Zero-H-M-rich;
> *Mid* abbreviates Intermediate. The source arrays and aggregate estimates match the
> validated earlier analyses; no training or inference was performed during figure assembly.

The scaffolds are intentionally conservative. Journal-specific model definitions, dataset
citations, and Methods cross-references should be inserted from the manuscript's canonical
nomenclature rather than inferred from checkpoint folder names.

## Suggested follow-up analyses

1. Repeat training with multiple random seeds to separate optimization variability from
   held-out-case uncertainty.
2. Report confidence intervals for multiscale medians from the existing SI distributions.
3. Repeat the wavelet analysis with a nonperiodic boundary mode or an interior crop to test
   boundary sensitivity.
4. Add measured training time, peak memory, and inference cost before making compute-efficiency
   claims from spatial-DOF exposure.
5. Apply the same protocol to additional physical channels and datasets before making
   field-independent transfer claims.

## Source inventory

- V5.1 main figure: [PDF](MixedResolution_unified_v5_1_20260926_1359.pdf),
  [SVG](MixedResolution_unified_v5_1_20260926_1359.svg),
  [PNG](MixedResolution_unified_v5_1_20260926_1359.png)
- Source manifest:
  [FigureSourceManifest_unified_v5_1_20260926_1359.json](FigureSourceManifest_unified_v5_1_20260926_1359.json)
- Layout configuration:
  [`publication_layout_unified_v5_1.yaml`](../../_Scripts/publication_layout_unified_v5_1.yaml)
- Reproducible renderer:
  [`render_mixed_resolution_v5_1.py`](../../../../figures/scripts/render_mixed_resolution_v5_1.py)
- Resolution fields:
  [`ResolutionProtocol_fields_formal_20260712.csv`](../../_Process_Results/ResolutionProtocol/ResolutionProtocol_fields_formal_20260712.csv)
- Training budgets:
  [`ResolutionProtocol_budgets_formal_20260712.csv`](../../_Process_Results/ResolutionProtocol/ResolutionProtocol_budgets_formal_20260712.csv)
- Sensor protocol:
  [`ResolutionProtocol_sensors_formal_20260712.csv`](../../_Process_Results/ResolutionProtocol/ResolutionProtocol_sensors_formal_20260712.csv)
- All-recipe accuracy and sensor sweeps:
  [`SensorSweepAllRecipes_summary_20260806_1124.csv`](../../_Process_Results/UnifiedPublicationV2/SensorSweepAllRecipes_summary_20260806_1124.csv)
- Multiscale summary:
  [`MultiscaleWavelet_summary_20260802_1250.csv`](../../_Process_Results/MultiscaleWavelet/MultiscaleWavelet_summary_20260802_1250.csv)
- Multiscale metadata:
  [`MultiscaleWavelet_metadata_20260802_1250.json`](../../_Process_Results/MultiscaleWavelet/MultiscaleWavelet_metadata_20260802_1250.json)
- V5.1 review reports:
  [figure contract](../../../../figures/generated/art_style_review/MixedResolution_unified_v5_1_20260926_1359/figure_contract.md),
  [layout QA](../../../../figures/generated/art_style_review/MixedResolution_unified_v5_1_20260926_1359/LAYOUT_QA.json),
  [scientific continuity](../../../../figures/generated/art_style_review/MixedResolution_unified_v5_1_20260926_1359/SCIENTIFIC_STATE_COMPARISON.json), and
  [source lock](../../../../figures/generated/art_style_review/MixedResolution_unified_v5_1_20260926_1359/SOURCE_LOCK.json)
- Earlier scientific context:
  [V3-7 companion](MixedResolution_RnD_companion_v3_7_20260914_1158.md),
  [V4.7 scientific anchor](MixedResolution_unified_v4_7_20260914_2318.pdf), and
  [V5.0 visual baseline](MixedResolution_unified_v5_0_20260926_1336.pdf)
