# California Housing: published-MLP optimizer benchmark

Primary source: the frozen GitHub Actions cohort.

**COMPLETE: 15 runs, five seeds per optimizer.**

All ± entries in our tables are two-sided 95% Student-t confidence half-widths across seeds. The machine-readable tables also provide sample SD and interval endpoints. These are not layer error bars.

## Fixed training budgets

| Updates | Optimizer | Test RMSE ↓ | Test MAE ↓ | Confidence/tail average α |
|---:|---|---:|---:|---:|
| 2,000 | SGD + momentum | 0.5711 ± 0.0035 (n=5) | 0.4146 ± 0.0130 (n=5) | 3.5760 ± 0.6134 (n=5) |
| 2,000 | Native AdamW | 0.5482 ± 0.0007 (n=5) | 0.3850 ± 0.0037 (n=5) | 2.1258 ± 0.0453 (n=5) |
| 2,000 | Modified AdamW | 0.5635 ± 0.0023 (n=5) | 0.3999 ± 0.0050 (n=5) | 4.3850 ± 0.0881 (n=5) |
| 20,000 | SGD + momentum | 0.5097 ± 0.0058 (n=5) | 0.3531 ± 0.0105 (n=5) | 2.2778 ± 0.0095 (n=5) |
| 20,000 | Native AdamW | 0.4808 ± 0.0028 (n=5) | 0.3251 ± 0.0053 (n=5) | 1.6781 ± 0.0238 (n=5) |
| 20,000 | Modified AdamW | 0.4838 ± 0.0021 (n=5) | 0.3287 ± 0.0026 (n=5) | 1.6128 ± 0.0035 (n=5) |

## Primary validation-stopped comparison

Each seed contributes the best validation-RMSE checkpoint before its first 16-bad-epoch stop. Selection can occur at different update counts; the fixed-budget table answers a separate question.

| Optimizer | Test RMSE ↓ | Test MAE ↓ | Confidence/tail average α |
|---|---:|---:|---:|
| SGD + momentum | 0.5441 ± 0.0154 (n=5) | 0.3850 ± 0.0181 (n=5) | 3.4712 ± 0.3737 (n=5) |
| Native AdamW | 0.4860 ± 0.0028 (n=5) | 0.3272 ± 0.0012 (n=5) | 1.6732 ± 0.0133 (n=5) |
| Modified AdamW | 0.5052 ± 0.0472 (n=5) | 0.3457 ± 0.0420 (n=5) | 2.7589 ± 2.7135 (n=5) |

## Published reference

The following ± values are **published seed sample SD**, not confidence half-widths.

| Published optimizer | Parameters | Test RMSE mean ± sample SD | Seeds |
|---|---:|---:|---:|
| sgd | 299,745 | 0.517666 ± 0.006416 | 10 |
| adamw | 623,281 | 0.493586 ± 0.011003 | 10 |
| muon-ema | 1,410,145 | 0.470808 ± 0.002643 | 10 |

Our native AdamW primary mean minus the published AdamW mean is -0.007547 ± 0.008034 (95% Welch interval over independent seed means).

Published results use validation stopping and different training seeds/runtime. The current native AdamW recipe matches the published architecture and settings; CPU results need not be numerically identical to the authors' GPU results. Published SGD and Muon-EMA use separately tuned architectures, so they are context, not same-model optimizer controls. Fixed-update results must not be compared as if they used the paper's checkpoint selection. A confidence interval containing zero does not establish equivalence or reproduction within a prespecified tolerance.

Source: [Benchmarking Optimizers for MLPs in Tabular Deep Learning](https://arxiv.org/html/2604.15297v2), Table 10; [authors' repository](https://github.com/yandex-research/tabular-dl-optimizers) at `c34e33152db9eea8ed4b2d742b9bef5628d71fb5`.

## Paired modified-minus-native effects

Negative RMSE/MAE differences favor modified AdamW. An α difference has no universal better/worse sign.

| Endpoint | Metric | Mean difference ± 95% CI |
|---|---|---:|
| 2,000 updates | test_rmse | 0.0154 ± 0.0020 (n=5) |
| 2,000 updates | test_mae | 0.0150 ± 0.0024 (n=5) |
| 2,000 updates | confidence_tail_alpha | 2.2592 ± 0.1268 (n=5) |
| 20,000 updates | test_rmse | 0.0030 ± 0.0027 (n=5) |
| 20,000 updates | test_mae | 0.0037 ± 0.0068 (n=5) |
| 20,000 updates | confidence_tail_alpha | -0.0653 ± 0.0246 (n=5) |
| Primary validation stop | test_rmse | 0.0191 ± 0.0466 (n=5) |
| Primary validation stop | test_mae | 0.0184 ± 0.0428 (n=5) |
| Primary validation stop | confidence_tail_alpha | 1.0857 ± 2.7088 (n=5) |

## Scope and interpretation

One unique 784x784 hidden matrix passes min(shape)>=64. The confidence/tail average equals that single ordinary fitted alpha. Input/readout are excluded a priori; layer dispersion is undefined. This is not a collective many-layer marginality test.

The 8→784→784→1 MLP has 623,281 parameters. Each raw spectral fit uses ordinary α with weight sqrt(num_pl_spikes)/(D+0.02), with no extra matrix-size factor. WeightWatcher's alpha_weighted is not substituted.

RMSE and MAE use $100,000 units: RMSE 0.50 is $50,000, not 50% error. Lower is better. MSE uses squared units; R² and MAPE are supplementary and are retained in the data tables.

The calibration runs use separate development seeds. Native AdamW uses the published recipe. SGD-momentum and graft calibration use validation data only, and their settings are frozen before confirmation.

Prediction error and spectral behavior answer separate questions. A fitted α near 2 does not by itself establish good held-out prediction. This one-matrix architecture cannot establish collective many-layer marginality.

Trajectory bands are pointwise 95% Student-t intervals across seeds, not simultaneous bands. A gap or missing value means the measurement was unavailable or invalid; it is never replaced by zero.
