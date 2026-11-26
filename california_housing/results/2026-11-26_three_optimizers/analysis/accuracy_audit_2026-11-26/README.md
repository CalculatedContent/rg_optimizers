# California Housing: independent accuracy and units audit

Audited on 2026-11-26 against immutable result commit `5b11a1a4ee37b87113d35ef7b08973b5ecf4e288`, GitHub Actions run [45744352135](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45744352135).

## Outcome

The archived prediction metrics, power-law fits, weighted model averages, and across-seed confidence intervals reproduce. **RMSE 0.50 means $50,000 in this dataset's units, not a 50% error rate.** The neural cohort was not retrained or retuned. All observations and seeds remain included.

This numerical agreement does not establish that every fitted power law is a reliable asymptotic description, or that the chosen recipes are optimal. A real extreme-input prediction failure and limited spectral tail ranges are documented below.

## Prediction units and interpretation

The target is the census-block-group median house value in the 1990 California Housing data, expressed in units of $100,000. The [dataset documentation](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) states those units. RMSE is the square root of mean squared prediction error and has the same units as the target; it is not a percentage. MAE is the mean absolute dollar error. R-squared is `1 - sum((prediction-target)^2) / sum((target-test_mean)^2)`, not classification accuracy.

Every ± in the neural results below is a two-sided 95% Student-t confidence half-width across the five seeds (df=4). The data partition is fixed. These intervals do not include uncertainty from new data samples, alternate partitions, or power-law model selection.

| Updates | Optimizer | Test RMSE, USD | Test MAE, USD | Test R-squared | Actual test MAPE, percent |
|---:|---|---:|---:|---:|---:|
| 2,000 | SGD + momentum | 55,589 ± 679 | 38,480 ± 645 | 0.7675 ± 0.0057 | 21.40 ± 1.00 |
| 2,000 | Native AdamW | 52,890 ± 1,731 | 36,651 ± 1,820 | 0.7894 ± 0.0138 | 20.98 ± 2.54 |
| 2,000 | Modified AdamW | 54,982 ± 8,063 | 36,205 ± 1,887 | 0.7700 ± 0.0717 | 20.59 ± 2.30 |
| 20,000 | SGD + momentum | 50,637 ± 300 | 34,345 ± 257 | 0.8071 ± 0.0023 | 19.06 ± 0.51 |
| 20,000 | Native AdamW | 56,158 ± 1,150 | 35,494 ± 774 | 0.7627 ± 0.0097 | 18.79 ± 0.52 |
| 20,000 | Modified AdamW | 55,225 ± 1,101 | 35,103 ± 536 | 0.7705 ± 0.0091 | 18.50 ± 0.51 |

MAPE is `100 * mean(abs(prediction-target)/abs(target))`; its interval half-width is in percentage points. MAPE and R-squared were calculated after the experiment to clarify interpretation. The frozen primary outcomes remain RMSE and MAE. MAPE gives a dollar error more weight when the target value is lower, so it need not rank optimizers like RMSE. Mean RMSE is computed across seeds; R-squared is computed separately for each seed before averaging.

The mean test target is $208,084. A predictor that always returns the **training-set mean** has test RMSE **$115,288**, MAE **$91,445**, and R-squared approximately 0. A simple **ordinary least-squares regression**, fitted on the same training partition with the eight inputs and an intercept, has test RMSE **$75,415**, MAE **$54,011**, and R-squared **0.5721**. These baselines were added after the frozen comparison for context; they did not select or change its settings.

## Independent numerical checks

- Downloaded all **30 endpoint checkpoints** from the immutable commit and verified seed, optimizer, update count, data-manifest identity, and weight hashes.
- Reconstructed the forward pass directly from checkpoint tensors without importing the trainer's model or evaluation routine. Recomputed **90 evaluations**, covering training, validation, and test for every checkpoint.
- Maximum RMSE discrepancy: **2.87e-8** target units, less than **$0.003**. Maximum MAE discrepancy: **4.05e-9** target units. Float64 instead of float32 inference changes test RMSE by at most **1.64e-8** target units.
- Checked all **19 frozen source hashes**, all **15 run source records**, exact split indices, and the training-only scalers. Every prepared feature and target array agrees with reconstruction from the raw cached dataset, with maximum discrepancy 0.
- Confirmed paired initialization and minibatch hashes for each seed's three optimizers, and identical graft update arithmetic to the Shakespeare implementation.
- Independently recalculated all **615** weighted model-average alpha values: maximum discrepancy **1.78e-15**. Recomputed seed statistics and interval endpoints: maximum discrepancy **2.66e-15**.
- Reconstructed all **1,845** Pareto-MLE/KS-cutoff fits directly from archived eigenvalues. No cutoff or tail-count mismatch; fitted alpha differs by at most **5.0e-11**, consistent with serialized rounding.

## A real early prediction failure

Modified AdamW seed 1341 at 2,000 updates predicts **-$1,691,071** for a test target of **$137,500** at original dataset row 19006. That one observation contributes **36.66%** of that run's total test squared error. The five largest residuals contribute 41.37%.

The row's average-occupancy input is **1,243.33**, or **193.09 training standard deviations** above the training mean. The maximum occupancy in training is 599.71; in validation it is 63.75. This is an observed severe prediction failure on an extreme-input row. The observation alone does not identify the causal contribution of any single input feature. At 20,000 updates the same seed predicts approximately $284,051 for that row.

The unconstrained linear output can produce negative predictions. No predictions were clipped, no test rows or seeds were removed, and no corrected score replaces the original result. This failure helps explain the wide early modified-AdamW RMSE interval and remains part of the assessment.

## Spectral interpretation and design limits

The average uses ordinary fitted alpha with weights `sqrt(num_pl_spikes)/(D+0.02)`, all three prespecified 128×128 hidden matrices, and no extra size factor. The 128×8 input and 1×128 output matrices fail the frozen minimum-dimension rule. WeightWatcher's `alpha_weighted` is not substituted.

SGDM's automatically selected tails contain only **9–20 eigenvalues** at 2,000 updates and **14–22** at 20,000, out of 128. They span **0.35–0.55** and **0.50–0.84 decades**, respectively. Native AdamW at 2,000 has longer tails: **24–55 eigenvalues**, spanning **1.36–1.76 decades**. Reproducible fitted alpha values and small between-seed intervals do not remove this finite-spectrum/model-fit uncertainty. A fitted KS distance is not a calibrated goodness-of-fit p-value.

Native AdamW is closest to alpha 2 at the early endpoint. SGDM achieves the strongest late RMSE but its fitted average remains well above 2. This setup does not reproduce every spectral prediction observed in the earlier experiments.

Settings came from a limited three-candidate grid on development seed 1339 at 2,000 updates. SGDM and native AdamW use peak LR 0.003; modified AdamW uses base LR 0.001 and reference LR 0.03. This compares separately calibrated recipes. Weight decay is zero in every arm; the native AdamW arm therefore has no active decay term. No dropout or early stopping was used. The calibration does not establish optimal long-horizon settings or isolate the graft while holding every Adam setting fixed.

## Reproduce this audit without retraining

Use the experiment's pinned Python packages, then run from the repository root. The commands create local audit outputs under the experiment's ignored cache directory and read the existing weights. The immutable checkpoint commit must be present in the clone. For a shallow clone, fetch that commit first. The normal experiment README gives the environment setup.

```bash
campaign=california_housing/results/2026-11-26_three_optimizers
audit_scripts="$campaign/analysis/accuracy_audit_2026-11-26"
audit_data=california_housing/.cache/accuracy_audit_data
audit_output=california_housing/.cache/accuracy_audit_output
python -m california_housing.data --data-dir "$audit_data"
python "$audit_scripts/checkpoint_audit.py" --repo "$PWD" --data-dir "$audit_data" --source "$campaign" --output "$audit_output"
python "$audit_scripts/baselines_and_units.py" --data-dir "$audit_data" --source "$campaign" --output "$audit_output"
python "$audit_scripts/outlier_audit.py" --data-dir "$audit_data" --output "$audit_output"
python "$audit_scripts/spectral_audit.py" --source "$campaign" --output "$audit_output"
```

`checkpoint_audit.json` and `.csv` retain the independent metrics, checkpoint hashes, per-seed values, sample SDs, and confidence intervals. `spectral_audit.json` records the fit reconstruction and weighting checks. `baselines_and_units.json`, `outlier_audit.json`, and the protocol audit files preserve the additional evidence. The source spectra and original results remain in their original committed locations.
