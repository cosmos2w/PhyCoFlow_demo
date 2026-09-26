# Main-text synchronization required after the SI finalization pass

The main source was audited at `figures/editorial/main_review_candidate.tex`; it was not edited. Locations below refer to that file as it existed on 26 September 2026. The current main Figures 3 and 5 were checked against the V5.1 and V4.1 final exports, respectively.

## Figure 2 and deterministic FNO identity

### Airfoil target channels

- **Current wording and location:** lines 292–295: “The targets comprise the complete stress field or five flow fields, respectively, together with the unobserved physical-domain mask.” Lines 650–655 repeat: “The airfoil target contains five compressible-flow fields and an airfoil/fluid-domain mask.”
- **Corrected wording:** “The targets comprise the complete stress field or four primitive flow fields, respectively, together with the unobserved physical-domain mask; airfoil Mach number is derived from the reconstructed primitive fields.” In Methods: “The airfoil target contains the four primitive fields \((\rho,u,v,p)\) and an airfoil/fluid-domain mask; Mach number is derived during evaluation.”
- **Evidence:** the populated airfoil configurations and SI normalization constants contain four primitive flow outputs plus the mask; the evaluation path derives Mach after inverse transformation. See Supplementary Methods, “Elasticity and airfoil,” in `CURRENT_Supplementary Info_FINAL.tex`.

### FNO baseline name and construction

- **Current wording and location:** line 281: “The deterministic comparisons comprise supervised FNO reconstruction…”; Results use “FNO” and “FNO-R” at lines 322, 327, 335 and 343; Methods lines 772–775 label the method “FNO” and describe a supervised Fourier baseline; the rendered figures use “Geo-FNO” or “FNO-R.”
- **Corrected wording:** “The deterministic neural-operator comparison combines Geo-FNO-style geometry handling with RecFNO-style sparse value and binary support maps. The evaluated Cartesian representations rasterize measurements and masks on the target grid; the same implementation family also provides scatter/gather geometry mapping for non-tensor grids.” Use **Geo-FNO/RecFNO** in prose, Methods and figure labels.
- **Evidence:** the evaluated Cartesian path rasterizes measured values and binary observation masks before the Fourier layers, and the implementation family includes the Geo-FNO scatter/gather geometry interface. This matches the project implementation and the combined-model identity supplied by the authors. RecFNO is the sparse-reconstruction component described in Zhao et al., arXiv:2302.09808.

## Figure 3: mixed resolution

### Physical field

- **Current wording and location:** lines 361–365: “We selected a scalar field and represented it…”
- **Corrected wording:** “We retained streamwise velocity \(V_x\) and represented it on low- (L, \(32\times32\)), medium- (M, \(64\times64\)) and high-resolution (H, \(128\times128\)) grids.”
- **Evidence:** the raw HDF5 variables are ordered `Vx`, `Vy`, `density`, `pressure`; preprocessing stacks them in that order and records the same order; every evaluated configuration selects index 0. The frozen truth cache agrees with raw \(V_x\) and processed channel 0 to maximum absolute difference \(3.73\times10^{-9}\), but differs from raw density by 2.44 for the same case and frame. The stale downstream field-name mapping is therefore not the numerical source.

### Caption panel map and displayed methods

- **Current wording and location:** caption lines 372–379 assigns representative fields to panel **c** and sensor sweeps to panel **d**; panel c is described as three recipe-specific field blocks with three reconstruction methods.
- **Corrected wording:** “**a**, L, M and H representations and five training recipes; labels above the bars give trajectory-level native-grid spatial-DOF exposure relative to H-only. **b**, H-grid relative-\(L_2\) error at 512 sensors for four methods and five recipes. **c**, H-grid error across 64–512 nested sensors for Mixed-HML, Zero-H-balanced and Zero-H-M-rich. **d**, a Zero-H-M-rich representative H state: truth and reconstructions from DMF-Gen, FFM-Perceiver, Senseiver and MLP-RBF; rows show full fields, matched enlargements and local absolute errors, and labels report local-region relative-\(L_2\). **e**, large-, intermediate- and fine-scale truth components and signed residuals for DMF-Gen and Senseiver. **f**, median scale-wise spatial pattern correlation and signed squared-energy allocation bias over 300 paired held-out case–time selections. Error bars in **b** and **c** are 95% case-bootstrap intervals.”
- **Related prose changes:** line 401, “sensor-count sweeps in … d,” should cite panel **c**; line 407, “representative field in … c,” should cite panel **d** and describe the one displayed Zero-H-M-rich state and four methods.
- **Evidence:** `MixedResolution_unified_v5_1_20260926_1359.pdf`, its `figure_contract.md`, and `source_manifest_v5_1.json` record the final a–f mapping and pass the frozen-source scientific comparison.

### Exposure metric

- **Current wording and location:** lines 388–389: “These ratios quantify the total number of scalar spatial degrees of freedom presented during training…”
- **Corrected wording:** “These values are trajectory-level native-grid spatial-DOF exposure, \((1{,}024n_L+4{,}096n_M+16{,}384n_H)/(9{,}000\times16{,}384)\), relative to H-only. They intentionally omit differences in retained frame counts and do not represent training compute, update-sample count or total information.”
- **Evidence:** the final SI training-composition table reproduces the plotted values 1.00, 0.34, 0.44, 0.16 and 0.19 from the existing recipe counts and states the metric boundary explicitly.

### Wavelet quantity

- **Current wording and location:** caption line 378 and Results lines 414, 420–421 use “variance-allocation bias”; the final panel-f heading also reads “Variance allocation bias [pp].”
- **Corrected wording:** replace with **signed squared-energy allocation bias [percentage points]**. The Results definition should state that each component’s squared-energy fraction is compared with the reference.
- **Evidence:** the analysis computes \(E_s=\sum_j u_{s,j}^2\) and reports 100 times the predicted-minus-reference energy fraction. No variance estimator is used.

## Figure 4: held-out cohort terminology

- **Current wording and location:** caption lines 444–446 uses “1,000 test states” and “25 test states”; Results lines 460–461 use “complete test set” and “Across 1,000 test states.”
- **Corrected wording:** use “1,000 held-out evaluation states,” “25 held-out evaluation states,” “complete held-out evaluation cohort,” and “Across 1,000 held-out evaluation states,” respectively.
- **Evidence:** frames outside the training partition form the held-out evaluation pool, and checkpoint selection and final evaluation share this pool. The reported intervals quantify frame sampling for the selected trained realization rather than an untouched-test or independent-training uncertainty.

## Figure 5: panel map, provenance and benchmark boundary

### Panels b and c

- **Current wording and location:** Results line 492 cites panel **b** for spread–error correlation and line 493 cites panel **c** for selective reconstruction. Caption lines 502–503 assign the same reversed mapping.
- **Corrected wording:** panel **b** is relative retained-set error after ranking by increasing ensemble spread; panel **c** is the statewise spread versus ensemble-mean error association with Spearman \(\rho\). In Results, cite panel **c** for \(\rho=0.654\) and panel **b** for the 4.6% reduction at 80% retention.
- **Evidence:** the final V4.1 export and `figure5_art_v4_1_explanation.md` define the displayed panel order and the 200-state, 64-draw estimators.

### Fixed measurements and estimator

- **Current wording and location:** caption lines 499–506 says all experiments use 256 temperature measurements and gives 64 draws per state, but does not state that measurements remain fixed within each state or that point reconstruction is the ensemble mean.
- **Corrected wording:** append to panel a–c description: “For each state and method, the same 256 temperature measurements and sensor locations remain fixed across all 64 conditional draws; point-reconstruction error is computed from the ensemble mean.”
- **Evidence:** the UQ manifest and saved visual cases record one fixed 256-sensor condition and 64 generation seeds per state.

### Distinct 0.117 and 0.1063 analyses

- **Current wording and location:** Fig. 4 Results line 462 reports about 0.117 for Cond-\(T\); Fig. 5 Results lines 512 and 523 report 0.1063 without stating that the analyses have different provenance.
- **Corrected wording:** after line 512 add: “This A0 ablation value and the approximately 0.117 Cond-\(T\) value in Fig. 4 come from distinct checkpoints and evaluation pipelines and are not interpreted as a direct cross-panel performance comparison.”
- **Evidence:** final SI Supplementary Table “Analysis-specific provenance for turbulent-combustion predictions” distinguishes the Cond-\(T\) run at epoch 6,005 from A0 at epoch 7,520 and separately identifies the Fig. 5f resource benchmark.

### Panel f scope

- **Current wording and location:** caption line 506 lists the five displayed coordinates but gives no workload boundary; Results lines 523–526 give DMF-Gen values and call the timing “model-core.”
- **Corrected wording:** add: “Training measurements use a fixed preloaded batch of 32 with 20 warm-up and 100 synchronized updates. Inference uses batch 1 at 40,300 requested points with 20 warm-up calls and at least 30 synchronized repetitions. Model loading, data/host transfer, metrics and plotting are excluded. Training memory is peak allocated CUDA memory; inference memory separately shows model state and peak allocated memory. Latent FM update time is stage 2 and peak training memory is stage 1. The panel compares adopted checkpoints under fixed workloads and is not a matched-training-budget causal efficiency comparison.”
- **Evidence:** final V4.1 scorecard and the SI timing table. The displayed DMF-Gen coordinates are 0.106321, 112.196 ms/update, 7.931 GiB, 16.690 ms inference, 24.827 MiB model state and 417.642 MiB peak inference allocation. No GPU-hour coordinate is present in the final figure.

## Numeric cross-check

- Figure 3’s plotted exposures and the five-recipe, four-method, 512-sensor and sensor-sweep values are unchanged from the frozen V5.1 sources; the panel-map and terminology corrections above are required.
- Figure 4’s reported 0.117/0.0802/0.0454 condition means, spectral-distance values and joint-PDF values remain associated with the existing Fig. 4 pipeline; only cohort terminology needs correction.
- Figure 5 V4.1 uses 0.106321 for DMF-Gen in panel f and retains the other seven frozen scorecard coordinates. Panels a–c use 200 states and 64 draws; panels d–e use 1,000 saved point reconstructions. These populations must remain separate in the main text and caption.
