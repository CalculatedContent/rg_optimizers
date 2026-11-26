# California Housing: primary GitHub cohort

**COMPLETE:** all 15 confirmation runs and the final analysis succeeded. Five seed triplets, 41 measurements per run, 30 endpoint checkpoints and 15 resumable final states are committed here.

- [Successful GitHub Actions run](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45744352135)
- [Full results and interpretation](analysis/RESULTS.md)
- [Audited machine-readable summary](analysis/summary.json)
- [Execution cohort and provenance](analysis/execution_cohort.json)
- [Alpha trajectory](analysis/figures/alpha_vs_updates.png) and [test RMSE trajectory](analysis/figures/rmse_vs_updates.png)

## Fixed-budget results

All ± values are **95% Student-t confidence half-widths across five seeds**, not layer error bars. RMSE and MAE are in units of $100,000. Sample SDs, interval endpoints and individual seed values are retained in the analysis tables.

| Updates | Optimizer | Test RMSE ↓ | Test MAE ↓ | Confidence-tail average α |
|---:|---|---:|---:|---:|
| 2,000 | SGD + momentum | 0.5559 ± 0.0068 | 0.3848 ± 0.0065 | 5.587 ± 0.321 |
| 2,000 | Native AdamW | 0.5289 ± 0.0173 | 0.3665 ± 0.0182 | 2.073 ± 0.049 |
| 2,000 | Modified AdamW | 0.5498 ± 0.0806 | 0.3621 ± 0.0189 | 3.061 ± 0.121 |
| 20,000 | SGD + momentum | 0.5064 ± 0.0030 | 0.3435 ± 0.0026 | 4.379 ± 0.151 |
| 20,000 | Native AdamW | 0.5616 ± 0.0115 | 0.3549 ± 0.0077 | 2.645 ± 0.139 |
| 20,000 | Modified AdamW | 0.5523 ± 0.0110 | 0.3510 ± 0.0054 | 2.549 ± 0.170 |

The spectral statistic covers the three prespecified 128×128 hidden matrices. The narrow input and scalar readout matrices are excluded by the fixed dimensional rule. All five weight matrices receive the modified optimizer's magnitude graft.

## What this first comparison shows

SGDM achieved the lowest mean test RMSE at 20,000 updates and improved between the two budgets. Native AdamW had the lowest mean test RMSE at 2,000 updates, then a higher test RMSE at 20,000. Modified AdamW's 20,000-update test RMSE was lower than native AdamW's in four of five paired seeds; the mean paired difference was -0.009329 ± 0.010618 (95% CI), so that interval includes zero. Its early RMSE interval is wide; all five seeds, including seed 1341, remain in the statistics.

The spectral behavior differs from the earlier Shakespeare experiment. SGDM's eligible-hidden-matrix average remains well above 2; native AdamW moves from about 2.07 to 2.64, while modified AdamW moves from about 3.06 to 2.55. These are measurements of this fixed regression setup, not an assumed replication of every earlier pattern. The raw fits retain tail sizes and cutoffs for assessing the finite-matrix interpretation.

## Frozen settings

Validation-only development selected peak LR 0.003 for SGDM and 0.003 for native AdamW. Modified AdamW keeps native peak LR 0.001 and selects momentum-reference peak LR 0.03 (ratio 30). These are separately calibrated recipes; the native Adam learning rates are not identical. No confirmation seed or test metric changed the selected settings.

This directory is the durable archive for the authorized comparison of SGDM,
native AdamW, and modified AdamW on the 50,817-parameter California Housing MLP.

- **Source and instructions:** [experiment README](../../README.md)
- **Design:** [frozen plan](../../plan.json)
- **Primary seeds:** 1340–1344, all three optimizers, 20,000 updates each.
- **Primary budgets:** 2,000 and 20,000 updates.
- **Development only:** seed 1339, nine fixed calibration candidates.

The workflow will create:

| Path | Contents |
|---|---|
| `calibration/` | All development runs, validation choices, exact split indices and data provenance |
| `frozen_selection.json`, `configs/` | Frozen optimizer settings, plan/source/data hashes and launch provenance |
| `runs/seed_<seed>/<optimizer>/` | 41 measurements, raw layer fits, runtime metadata and resumable checkpoints |
| `analysis/RESULTS.md` | Human-readable comparison at 2,000 and 20,000 updates |
| `analysis/summary.json` | Mean, sample SD, 95% Student-t CI, paired effects and audit status |
| `analysis/figures/` | Average-alpha and prediction-error trajectories with seed confidence bands |

Results are complete only when the final analysis reports
`complete_five_seeds=true`, five complete seed triplets and 15 complete runs.
The completed analysis reports `complete_five_seeds=true`, `n_completed=5`, and `n_completed_runs=15`, with no audit errors or warnings.
