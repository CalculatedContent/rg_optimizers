# MAGIC optimizer-state step audit
Diagnostic on disposable copies of final resume states. No persistent model updates, test evaluation or new training.

Complete: True; endpoints 48/48; trials 288/288; errors 0.

## Actual clipped baseline step (mean over seeds)

| Architecture | Phase | Case | n | LR | FC2 angle (rad) | Update radial cosine | Gradient radial cosine | Clip factor |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ordinary | grid | s0 | 2 | 0.003 | 0.00030347 | 0.011619 | -0.0021316 | 0.7246 |
| ordinary | grid | s1 | 2 | 0.01 | 0.00051895 | 0.021419 | -0.0019983 | 0.70636 |
| ordinary | grid | s2 | 2 | 0.03 | 0.00033429 | 0.016005 | -0.0031734 | 1 |
| ordinary | grid | s3 | 2 | 0.1 | 0.00018305 | 0.015801 | -0.0098008 | 1 |
| ordinary | grid | a0 | 2 | 0.0001 | 0.00046281 | 0.050752 | -0.0026688 | 0.91692 |
| ordinary | grid | a1 | 2 | 0.0003 | 0.00066941 | 0.023402 | -0.00098264 | 0.77276 |
| ordinary | grid | a2 | 2 | 0.001 | 8.9918e-05 | 0.028241 | -0.0035297 | 1 |
| ordinary | confirm | sgdm_wd0 | 5 | 0.003 | 0.00029336 | 0.01119 | -0.0022855 | 0.83532 |
| ordinary | confirm | adamw_wd0 | 5 | 0.0001 | 0.0004423 | 0.045743 | -0.0021302 | 0.77597 |
| normalized | grid | s0 | 2 | 0.003 | 0.00026877 | 0.0018864 | 2.2209e-09 | 0.10834 |
| normalized | grid | s1 | 2 | 0.01 | 0.00094141 | 0.0080504 | -9.3358e-10 | 0.15551 |
| normalized | grid | s2 | 2 | 0.03 | 0.0011966 | 0.010226 | 9.5892e-09 | 1 |
| normalized | grid | s3 | 2 | 0.1 | 3.1282e-05 | 0.00013403 | 8.1084e-08 | 1 |
| normalized | grid | a0 | 2 | 0.0001 | 0.0008425 | -0.0056719 | 2.3173e-08 | 0.39448 |
| normalized | grid | a1 | 2 | 0.0003 | 0.0011765 | 0.014024 | 9.9964e-10 | 0.42818 |
| normalized | grid | a2 | 2 | 0.001 | 0.0023682 | -0.010201 | -2.3242e-08 | 0.73126 |
| normalized | confirm | sgdm_wd0 | 5 | 0.1 | 0.00053242 | 0.006221 | 3.8472e-08 | 1 |
| normalized | confirm | adamw_wd0 | 5 | 0.0001 | 0.00065698 | 0.0044544 | 7.6154e-09 | 0.25124 |

## FC2 counterfactual rescaling (normalized models only)

Expected compensated laws: SGDM LR × b², Adam LR × b and epsilon ÷ b, with FC2-only consistently rescaled state. Other parameters and gains remain fixed at the pre-update point. Relative error measures ||delta_b/b - delta_1||/||delta_1||. Clipping may break exact layerwise covariance; FP32 update rounding can contribute residual.

| Optimizer | Clip | b | Compensated | n | Median error | Max error |
|---|---|---:|---|---:|---:|---:|
| sgdm | True | 0.25 | False | 13 | 15 | 15.113 |
| sgdm | True | 0.25 | True | 13 | 0 | 0.50033 |
| sgdm | True | 4.0 | False | 13 | 0.9375 | 0.93751 |
| sgdm | True | 4.0 | True | 13 | 0 | 0.7028 |
| sgdm | False | 0.25 | False | 13 | 15 | 15 |
| sgdm | False | 0.25 | True | 13 | 0 | 0 |
| sgdm | False | 4.0 | False | 13 | 0.9375 | 0.93751 |
| sgdm | False | 4.0 | True | 13 | 0 | 0 |
| adamw | True | 0.25 | False | 11 | 2.6832 | 3.0227 |
| adamw | True | 0.25 | True | 11 | 0.26625 | 0.38655 |
| adamw | True | 4.0 | False | 11 | 0.74459 | 0.75988 |
| adamw | True | 4.0 | True | 11 | 0.085361 | 0.12751 |
| adamw | False | 0.25 | False | 11 | 3.0001 | 3.0509 |
| adamw | False | 0.25 | True | 11 | 2.4555e-42 | 2.3757e-36 |
| adamw | False | 4.0 | False | 11 | 0.75003 | 0.75988 |
| adamw | False | 4.0 | True | 11 | 0 | 5.8165e-36 |

[Full per-state diagnostics](step_diagnostics.csv) · [Machine-readable summary](summary.json)

Errors: []

This is a local one-step measurement, not an estimate of a whole training trajectory or an ERG Jacobian.
