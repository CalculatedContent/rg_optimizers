# MAGIC SGDM continuation: 20k to 40k

Complete: True; audited 2/2; failed fits 0; new updates 40000.

Validation only; two paired development seeds, nominal 95% t intervals (df=1). The 20k weights and momentum/batch RNG state are bound to the original frozen archive. No test reuse or AdamW training.

| Metric | 20k | 40k | Paired change |
|---|---:|---:|---:|
| alpha | 4.44074 ± 1.36988 | 3.40085 ± 0.55561 | -1.03989 ± 1.92548 |
| validation_accuracy | 0.86965 ± 0.02004 | 0.85874 ± 0.01837 | -0.01091 ± 0.03840 |
| validation_ce | 0.34060 ± 0.03121 | 0.42842 ± 0.12594 | 0.08782 ± 0.15715 |
| validation_auc | 0.92307 ± 0.00713 | 0.91495 ± 0.00549 | -0.00812 ± 0.00164 |
| validation_ece15 | 0.03551 ± 0.00218 | 0.05917 ± 0.01057 | 0.02365 ± 0.00839 |
| validation_brier | 0.09889 ± 0.00527 | 0.10942 ± 0.03335 | 0.01053 ± 0.03862 |
| validation_mean_confidence | 0.90329 ± 0.00942 | 0.91589 ± 0.03337 | 0.01260 ± 0.04279 |
| accuracy_gap | 0.05359 ± 0.01113 | 0.10538 ± 0.02450 | 0.05179 ± 0.01338 |
| ce_gap | 0.14934 ± 0.02165 | 0.31701 ± 0.05181 | 0.16766 ± 0.07345 |
| D | 0.06532 ± 0.03654 | 0.04445 ± 0.01330 | -0.02087 ± 0.02324 |
| tail | 93.50000 ± 108.00274 | 131.00000 ± 12.70620 | 37.50000 ± 120.70894 |

[Full trajectories](trajectories.csv) · [Statistics](statistics.csv) · [Paired changes](paired_changes.csv) · [Alpha](alpha_trajectory.png) · [Accuracy](validation_accuracy_trajectory.png) · [CE](validation_ce_trajectory.png) · [Calibration](validation_ece15_trajectory.png)

Audit errors: []
