# Published California Housing MLP reproduction

**Status: complete. All 15 confirmation runs and the final analysis succeeded on GitHub.** Each optimizer has five fresh confirmation seeds, 2340–2344. Both committed analysis records require `complete_five_seeds=true`, `n_completed=5`, and `n_completed_runs=15`; the final audit has no errors or warnings.

- [Completed GitHub Actions run](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45753214421)
- [Protocol and reproduction instructions](../../README.md)
- [Frozen design](../../plan.json)
- [Selected settings and development results](frozen_selection.json)
- [Literature audit](literature_audit.json)
- [Complete tables and plots](analysis/RESULTS.md)
- [Machine-readable analysis](analysis/summary.json)
- [Execution cohort](analysis/execution_cohort.json)
- [Independent audit](analysis/independent_audit.json)

## What this comparison found

**SGD with momentum improves during the long continuation.** Its mean test RMSE decreases from 0.57115 at 2,000 updates to 0.50969 at 20,000 updates. Its fitted α moves from 3.57601 to 2.27782. The latter 95% confidence interval, [2.26830, 2.28733], remains above 2; this is proximity, not exact convergence to 2.

**Native AdamW provides the lowest mean RMSE at both fixed budgets.** Its primary validation-selected test RMSE is 0.48604, compared with the published AdamW mean of 0.49359 on the same architecture and dataset protocol. CPU execution and fresh training seeds need not reproduce the authors' GPU trajectory exactly.

**The modified update does not improve early performance in this experiment.** Modified AdamW has higher test RMSE than native AdamW on every confirmation seed at both 2,000 and 20,000 updates. Its α decreases from 4.38500 at 2,000 updates to 1.61279 at 20,000 updates. These results must be retained even though an early advantage was observed in other tasks.

## Reading the tables

All three comparison tables report **mean ± two-sided 95% Student-t confidence half-width across five seeds**, with equal weight per seed and four degrees of freedom. These are confidence intervals, not sample SD. The [machine-readable analysis](analysis/summary.json) and [independent audit](analysis/independent_audit.json) also retain sample SD and interval endpoints.

RMSE and MAE use **$100,000 units**, not percentages; lower is better. Higher R² is better. MAPE is supplementary, and its ± values are percentage points. The same held-out test set is used throughout.

The frozen architecture is 8 → 784 → 784 → 1, with 623,281 parameters. The prespecified minimum matrix dimension of 64 admits **one unique 784×784 hidden matrix**. The confidence/tail weight is `sqrt(num_pl_spikes)/(D+0.02)`, applied to ordinary fitted α without a matrix-size factor. Because only one matrix qualifies, the weighted average equals that matrix's fitted α. No layer dispersion can be estimated, and this is not a collective many-layer marginality test.

### Primary: validation-selected checkpoints

Each seed selects its best validation checkpoint before its **first 16-epoch patience stop**. Later training never replaces that primary checkpoint. The selected update differs by seed and optimizer.

| Optimizer | Test RMSE | Test MAE | Test R² | Test MAPE (%) | Confidence/tail α |
|---|---:|---:|---:|---:|---:|
| SGD + momentum | 0.54406 ± 0.01540 | 0.38503 ± 0.01811 | 0.77291 ± 0.01289 | 23.472 ± 2.050 | 3.47118 ± 0.37370 |
| Native AdamW | 0.48604 ± 0.00284 | 0.32725 ± 0.00121 | 0.81883 ± 0.00211 | 18.461 ± 0.280 | 1.67320 ± 0.01328 |
| Modified AdamW | 0.50518 ± 0.04719 | 0.34569 ± 0.04201 | 0.80340 ± 0.03838 | 19.797 ± 3.343 | 2.75888 ± 2.71353 |

**The modified-AdamW outlier remains in the primary analysis.** Seed 2343 reached its first patience stop at update 1,612 and selected update 780. That checkpoint has test RMSE 0.5729841 and α = 6.6638902. All 16 consecutive nonimproving validation epochs were checked independently. Its later continuation improves, but replacing the primary checkpoint would change the frozen selection protocol.

### Secondary: 2,000 updates

| Optimizer | Test RMSE | Test MAE | Test R² | Test MAPE (%) | Confidence/tail α |
|---|---:|---:|---:|---:|---:|
| SGD + momentum | 0.57115 ± 0.00348 | 0.41461 ± 0.01295 | 0.74983 ± 0.00304 | 26.345 ± 2.252 | 3.57601 ± 0.61344 |
| Native AdamW | 0.54818 ± 0.00068 | 0.38497 ± 0.00366 | 0.76955 ± 0.00057 | 23.055 ± 0.587 | 2.12582 ± 0.04531 |
| Modified AdamW | 0.56355 ± 0.00227 | 0.39995 ± 0.00500 | 0.75644 ± 0.00197 | 24.341 ± 0.783 | 4.38500 ± 0.08806 |

### Secondary: 20,000 updates

| Optimizer | Test RMSE | Test MAE | Test R² | Test MAPE (%) | Confidence/tail α |
|---|---:|---:|---:|---:|---:|
| SGD + momentum | 0.50969 ± 0.00579 | 0.35306 ± 0.01054 | 0.80076 ± 0.00452 | 21.277 ± 1.416 | 2.27782 ± 0.00951 |
| Native AdamW | 0.48078 ± 0.00280 | 0.32505 ± 0.00531 | 0.82273 ± 0.00206 | 18.734 ± 0.722 | 1.67808 ± 0.02384 |
| Modified AdamW | 0.48377 ± 0.00205 | 0.32872 ± 0.00263 | 0.82052 ± 0.00152 | 18.886 ± 0.501 | 1.61279 ± 0.00348 |

The paired modified-minus-native RMSE differences are +0.0153697 ± 0.0019657 at 2,000 updates and +0.0029864 ± 0.0026605 at 20,000 updates. Both intervals are 95% Student-t intervals over matched seed differences and exclude zero. The primary paired difference is +0.0191371 ± 0.0465506 and remains uncertain because of the retained early-stop outlier.

## Published performance reference

The uncertainty in this literature table is **sample SD over ten published seeds**, unlike the five-seed confidence intervals above. RMSE is again measured in $100,000 units.

| Published model | Test RMSE, mean ± sample SD |
|---|---:|
| Plain MLP with SGD + momentum | 0.5176658 ± 0.0064162 |
| Plain MLP with AdamW | 0.4935855 ± 0.0110033 |
| Plain MLP with Muon + exponential weight averaging | 0.4708079 ± 0.0026432 |

Source: [Gorishniy et al. (2026), Table 10](https://arxiv.org/html/2604.15297v2), with [frozen upstream source and per-seed reports](../../reference_manifest.json). These published rows use separately tuned architectures. Our native AdamW recipe reproduces the published AdamW architecture and settings. Our three-optimizer comparison holds that architecture fixed; its SGDM arm is not a reproduction of the separately tuned published SGDM recipe. The Muon-EMA result provides stronger plain-MLP context, not a claim of universal California Housing state of the art.

## Separate CPU reference run

The author-seed-0 AdamW run selected update **10,088** by validation error and reached its first patience stop at update **10,920**.

| Held-out test metric | Measured value |
|---|---:|
| RMSE | **0.4894727**, or **$48,947** |
| MAE | **0.3304884**, or **$33,049** |
| R² | **0.8162640** |
| Mean absolute percentage error | **18.9186%** |

This single reference run is excluded from the primary five-seed statistics.

## Frozen settings and runtime provenance

- **SGD + momentum:** learning rate 0.03, momentum 0.9, zero dampening, zero weight decay.
- **Native AdamW:** published learning rate 0.0002722214565241805, matrix weight decay 0.4555843429087667, betas (0.9, 0.999), epsilon 1e-8; no bias decay.
- **Modified AdamW:** the same native AdamW settings, with momentum-reference learning rate 0.01 and ratio 36.73479720402471.

Calibration used development seeds **2338–2339** and validation scores only. The confirmation seeds **2340–2344** were evaluated after settings were frozen. No confirmation outcome changed the optimizer settings, checkpoint-selection rule, fitting rule, or averaging method.

All runs use CPU float32 and one numerical thread. The recorded source, data, package, and thread identities agree. Paired initialization hashes and minibatch/training RNG hashes agree at 2,000 and 20,000 updates. GitHub assigned AMD EPYC 9V45, AMD EPYC 9V74, and Intel Xeon 6975P-C processors across the jobs; the cohort therefore does not use identical hardware everywhere. Matching version strings do not guarantee identical arithmetic across hardware or kernels.

The independent implementation recalculated 45 requested group-statistic rows and 15 paired rows, covering primary, 2,000-update, and 20,000-update RMSE, MAE, R², MAPE, and α. All 360 numeric checks agreed with the committed analysis to numerical precision; the largest absolute difference was 7.99e-15.

## Retained results and checkpoints

Each `runs/seed_<seed>/<optimizer>/` directory contains `result.json`, the primary `best_checkpoint.pt`, full `checkpoint_step_2000.pt` and `checkpoint_step_20000.pt` states, `resume.pt`, and the workflow log. All **60 expected checkpoint files** are present in the repository. GitHub Actions artifacts provide an additional retained copy; the repository is the durable experimental archive.

The original exploratory cohort remains intact under `california_housing/results/2026-11-26_three_optimizers`. These results do not combine it with the fresh published-protocol cohort.
