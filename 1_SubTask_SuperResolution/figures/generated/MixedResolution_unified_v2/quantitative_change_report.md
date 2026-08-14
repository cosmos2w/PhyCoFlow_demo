# Best-checkpoint refresh: quantitative changes

Baseline: `SensorSweepAllRecipes_summary_20260722_1103.csv` / `MultiscaleWavelet_summary_20260722_1128.csv`.
Refresh: `SensorSweepAllRecipes_summary_20260802_1250.csv` / `MultiscaleWavelet_summary_20260802_1250.csv`.
Negative relative-L2 change is an improvement; positive correlation change is an improvement.

## Fixed 512-sensor mean physical relative L2

| Model | Baseline | Refreshed | Change | Improved recipes |
|---|---:|---:|---:|---:|
| DMF-Gen | 0.0245 | 0.0198 | -19.2% | 4/5 |
| FFM-Perceiver | 0.0891 | 0.0869 | -2.5% | 4/5 |
| MLP-RBF | 0.1378 | 0.1145 | -16.9% | 4/5 |
| Senseiver | 0.0525 | 0.0429 | -18.3% | 4/5 |

## Across the full 5-recipe × 5-sensor grid

| Model | Median cell change | Mean cell change | Improved cells |
|---|---:|---:|---:|
| DMF-Gen | -6.4% | -12.7% | 19/25 |
| FFM-Perceiver | -1.7% | -1.6% | 20/25 |
| MLP-RBF | -2.1% | -8.1% | 22/25 |
| Senseiver | -5.6% | -11.0% | 17/25 |

## Largest 512-sensor improvements

- DMF-Gen, H-limited: -50.5% (0.0260 → 0.0129)
- Senseiver, Mixed-HML: -45.8% (0.0545 → 0.0296)
- MLP-RBF, Zero-H-M-rich: -37.3% (0.2231 → 0.1398)
- DMF-Gen, Mixed-HML: -36.5% (0.0187 → 0.0119)
- DMF-Gen, H-only: -22.5% (0.0156 → 0.0121)

## Largest 512-sensor regressions

- DMF-Gen, Zero-H-M-rich: +6.6% (0.0270 → 0.0288)
- FFM-Perceiver, Mixed-HML: +0.9% (0.0741 → 0.0748)
- MLP-RBF, H-limited: +0.9% (0.0981 → 0.0990)

## Displayed-recipe fine-scale spatial correlation

| Model | Baseline | Refreshed | Absolute change |
|---|---:|---:|---:|
| DMF-Gen | 0.709 | 0.735 | +0.026 |
| FFM-Perceiver | 0.346 | 0.354 | +0.008 |
| MLP-RBF | 0.163 | 0.187 | +0.024 |
| Senseiver | 0.323 | 0.273 | -0.050 |

## Displayed-recipe mean absolute variance-allocation bias

| Model | Baseline (pp) | Refreshed (pp) | Change (pp) |
|---|---:|---:|---:|
| DMF-Gen | 0.009 | 0.007 | -0.002 |
| FFM-Perceiver | 0.066 | 0.064 | -0.002 |
| MLP-RBF | 1.288 | 0.610 | -0.678 |
| Senseiver | 0.144 | 0.113 | -0.030 |

## Qualitative snapshot 50 at 512 sensors

| Model | Recipe | Baseline L2 | Refreshed L2 | Change |
|---|---|---:|---:|---:|
| DMF-Gen | Mixed-HML | 0.0135 | 0.0078 | -42.1% |
| DMF-Gen | Zero-H-balanced | 0.0252 | 0.0243 | -3.6% |
| DMF-Gen | Zero-H-M-rich | 0.0205 | 0.0197 | -3.6% |
| FFM-Perceiver | Mixed-HML | 0.0647 | 0.0628 | -2.9% |
| FFM-Perceiver | Zero-H-balanced | 0.0666 | 0.0652 | -2.0% |
| FFM-Perceiver | Zero-H-M-rich | 0.0663 | 0.0604 | -9.0% |
| Senseiver | Mixed-HML | 0.0337 | 0.0194 | -42.3% |
| Senseiver | Zero-H-balanced | 0.0678 | 0.0706 | +4.0% |
| Senseiver | Zero-H-M-rich | 0.0743 | 0.0633 | -14.8% |
