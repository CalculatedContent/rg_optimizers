# California Housing: three-optimizer confirmation cohort

Primary source: the frozen GitHub Actions cohort.

**COMPLETE: 15 runs, five seeds per optimizer.**

All ± values below are two-sided 95% Student-t confidence half-widths over independent seed results; they are not layer error bars.
Sample SD and interval endpoints are retained separately in summary.csv and summary.json.

| Updates | Optimizer | Test RMSE ↓ | Test MAE ↓ | Confidence-tail average α |
|---:|---|---:|---:|---:|
| 2,000 | SGD + momentum | 0.5559 ± 0.0068 (n=5) | 0.3848 ± 0.0065 (n=5) | 5.5869 ± 0.3211 (n=5) |
| 2,000 | Native AdamW | 0.5289 ± 0.0173 (n=5) | 0.3665 ± 0.0182 (n=5) | 2.0726 ± 0.0487 (n=5) |
| 2,000 | Modified AdamW | 0.5498 ± 0.0806 (n=5) | 0.3621 ± 0.0189 (n=5) | 3.0611 ± 0.1211 (n=5) |
| 20,000 | SGD + momentum | 0.5064 ± 0.0030 (n=5) | 0.3435 ± 0.0026 (n=5) | 4.3792 ± 0.1511 (n=5) |
| 20,000 | Native AdamW | 0.5616 ± 0.0115 (n=5) | 0.3549 ± 0.0077 (n=5) | 2.6448 ± 0.1390 (n=5) |
| 20,000 | Modified AdamW | 0.5523 ± 0.0110 (n=5) | 0.3510 ± 0.0054 (n=5) | 2.5494 ± 0.1701 (n=5) |

## Paired modified-minus-native effects

Negative RMSE/MAE differences favor modified AdamW; an α difference has no universal better/worse sign.

| Updates | Metric | Mean paired difference ± 95% CI |
|---:|---|---:|
| 2,000 | test_rmse | 0.0209 ± 0.0711 (n=5) |
| 2,000 | test_mae | -0.0045 ± 0.0075 (n=5) |
| 2,000 | confidence_tail_alpha | 0.9886 ± 0.1633 (n=5) |
| 20,000 | test_rmse | -0.0093 ± 0.0106 (n=5) |
| 20,000 | test_mae | -0.0039 ± 0.0062 (n=5) |
| 20,000 | confidence_tail_alpha | -0.0954 ± 0.2461 (n=5) |

## Interpretation and provenance

The spectral average uses three prespecified 128×128 hidden matrices, not all five linear weights. Each fit uses ordinary α with weight sqrt(num_pl_spikes)/(D+0.02), without a matrix-size factor. A seed average requires all three fits. Layer α sample SD is recorded separately and never used as the seed uncertainty.

Prediction error and spectral behavior answer separate questions. Neither α near 2 nor a lower training error alone establishes better held-out performance.

Test RMSE and MAE are primary performance outcomes, in units of $100,000. Validation metrics served calibration and remain secondary results in the machine-readable tables. Test data never select the frozen recipes.

The figures show seed means with pointwise 95% Student-t confidence bands; they are not simultaneous trajectory bands.
