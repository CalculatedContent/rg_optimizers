# MAGIC fixed-gain scale-invariant MLP

Complete: False; grid 14/14; endpoints 13/14; checkpoints 83/84; raw fit failures 0.

All three gains frozen at matched initialization. Same training partition, seeds, learning rates, clipping, no WD, no dropout. No new test access or confirmation runs. Calibration cross-fitted on validation. Two-seed exploratory results.

| Optimizer | LR | Ordinary alpha | Learned alpha | Fixed alpha | Ordinary val acc | Learned val acc | Fixed val acc | Learned raw CE | Fixed raw CE | Learned calibrated CE | Fixed calibrated CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sgdm | 0.003 | 4.4407 | 8.2045 | 3.9926 | 0.8696 | 0.8532 | 0.8624 | 0.6736 | 0.3220 | 0.3632 | 0.3214 |
| sgdm | 0.01 | 2.6151 | 2.9990 | 2.5739 | 0.8608 | 0.8388 | 0.8625 | 1.6927 | 0.3172 | 0.3885 | 0.3142 |
| sgdm | 0.03 | 2.0179 | 1.9310 | 1.8775 | 0.8562 | 0.8466 | 0.8687 | 1.9595 | 0.3180 | 0.3739 | 0.3117 |
| sgdm | 0.1 | 1.7359 | 1.6814 | 1.6451 | 0.8586 | 0.8545 | 0.8732 | 2.2547 | 0.3139 | 0.3568 | 0.3088 |
| adamw | 0.0001 | 2.0272 | 2.0515 | 1.7759 | 0.8569 | 0.8534 | 0.8660 | 0.6409 | 0.3171 | 0.3639 | 0.3125 |
| adamw | 0.0003 | 1.8119 | 1.6874 | 1.4647 | 0.8484 | 0.8463 | 0.8700 | 1.8743 | 0.3180 | 0.3712 | 0.3121 |
| adamw | 0.001 | 1.8454 | 1.7896 | 1.6779 | 0.8581 | 0.8497 | 0.8728 | 2.0613 | 0.3143 | 0.3666 | 0.3102 |

Audit errors: ['grid/a2-3470: AssertionError: ']

[Paired endpoints](paired_endpoints.csv) · [Paired trajectories](paired_trajectories.csv) · [Statistics](statistics.csv) · [Alpha trajectory](alpha_trajectory.png) · [Accuracy](validation_accuracy_trajectory.png) · [Calibrated CE](validation_calibrated_ce_trajectory.png)

Intervals across two seeds have df=1 and are descriptive; not independent test population confidence intervals. Raw alpha fits not selected by performance; short tails and alpha>6 warrant separate sensitivity analysis. Fixed gains constrain effective matrix norms and model capacity; this is not merely a gauge transformation.
