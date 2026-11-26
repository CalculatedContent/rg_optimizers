# Published California Housing MLP reproduction

**Status: confirmation training is in progress on GitHub.** The numerical checks, data preparation, reference run, development calibration, and configuration freeze succeeded.

- [Live GitHub Actions run](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45753214421)
- [Protocol and reproduction instructions](../../README.md)
- [Frozen design](../../plan.json)
- [Selected settings and development results](frozen_selection.json)
- [Literature audit](literature_audit.json)

## Published performance reference

RMSE is measured in **$100,000 units**, not percent. The uncertainty below is **sample SD over ten published seeds**.

| Published model | Test RMSE |
|---|---:|
| Plain MLP with SGD + momentum | 0.5176658 ± 0.0064162 |
| Plain MLP with AdamW | 0.4935855 ± 0.0110033 |
| Plain MLP with Muon + exponential weight averaging | 0.4708079 ± 0.0026432 |

Source: [Gorishniy et al. (2026)](https://arxiv.org/html/2604.15297v2). These published rows use separately tuned architectures. The new comparison fixes the published AdamW architecture for all three optimizers.

## Completed CPU reference run

The separate author-seed-0 AdamW run selected update **10,088** by validation error and reached its first patience stop at update **10,920**.

| Held-out test metric | Measured value |
|---|---:|
| RMSE | **0.4894727**, or **$48,947** |
| MAE | **0.3304884**, or **$33,049** |
| R² | **0.8162640** |
| Mean absolute percentage error | **18.9186%** |

This single reference run is excluded from the primary five-seed statistics. Its CPU trajectory need not match the authors' GPU trajectory exactly.

## Frozen three-optimizer comparison

- **SGD + momentum:** learning rate 0.03, momentum 0.9, zero dampening, zero weight decay.
- **Native AdamW:** published learning rate 0.0002722214565241805, matrix weight decay 0.4555843429087667, betas (0.9, 0.999), epsilon 1e-8; no bias decay.
- **Modified AdamW:** the same native AdamW settings, with momentum-reference learning rate 0.01 and ratio 36.73479720402471.

Calibration used only development seeds **2338–2339** and validation scores. Settings are now fixed. Confirmation uses **2340–2344** for each optimizer, with matching initializations and minibatches.

The main comparison uses each seed's first validation-stopped checkpoint. Separate **2,000- and 20,000-update** results show the effect of extended training. Final analysis must require all **15** primary runs, report five-seed Student-t 95% confidence intervals, and audit runtime provenance.

Each run commits its results and retained weights to this directory. Final tables and plots will appear in `analysis/`. The original exploratory cohort remains intact under `california_housing/results/2026-11-26_three_optimizers`.
