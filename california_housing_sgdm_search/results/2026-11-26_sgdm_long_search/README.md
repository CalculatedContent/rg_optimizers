# SGDM full grid and long-horizon study

This directory receives the finite 2026-11-26 study. See [the frozen protocol and reproduction instructions](../../README.md).

The workflow first verifies the parent cohort and source, evaluates all 32 settings on two development seeds, freezes the validation-selected setting, and trains five fresh confirmation seeds. In parallel, it extends five original SGDM checkpoints from 20,000 to 100,000 updates without repeating their earlier updates.

- `cohort_origin.json` records the actual workflow and immutable launch commit.
- `development_analysis/DEVELOPMENT_RESULTS.md` and `development_analysis/figures/` contain the full grid after it completes.
- `analysis/RESULTS.md`, `analysis/summary.json`, and `analysis/figures/` contain the final audit, seed statistics and plots.
- `analysis/summary.json` must report `complete_five_seeds=true`, `n_completed=5`, five continuations, five fresh confirmations and no audit errors before the final study is called complete.

Confidence bands and intervals use variation over independent seeds, not variation over layers or hyperparameter settings. This architecture has one eligible spectral matrix. All plotted performance/alpha pairs are bound to the same checkpoint weights.

Parent cohort: [completed workflow45753214421](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45753214421), archived at `84b0219dc87c7caaefdc7245e455e53c300b8dfc`. Its original measurements and checkpoints are preserved.
