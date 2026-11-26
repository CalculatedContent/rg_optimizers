# California Housing: test performance versus confidence/tail alpha

This figure uses the completed three-optimizer cohort, GitHub [run 45753214421](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45753214421), source `67a0bf203ab0f6b34565e37b4f510f123f669876`, and immutable [result archive 84b0219](https://github.com/CalculatedContent/rg_optimizers/tree/84b0219dc87c7caaefdc7245e455e53c300b8dfc/california_housing_benchmark/results/2026-11-26_published_mlp). It does not include observations from the ongoing larger SGDM grid.

The horizontal axis is the ordinary fitted alpha averaged with weights `sqrt(num_pl_spikes)/(D+0.02)`, with no matrix-size factor and no use of WeightWatcher's `alpha_weighted`. Only the 784×784 hidden matrix qualifies for this model, so the confidence/tail-adjusted model average equals its fitted alpha.

The vertical axis is held-out test RMSE. California Housing is regression, so classification accuracy does not apply. Lower RMSE is better. Target units are $100,000: RMSE 0.50 means $50,000, not 50% error.

The left panel preserves all 41 measured budgets from 0 to 20,000 updates, using a logarithmic horizontal axis. The right panel displays 2,000–20,000 on linear axes. Lines connect equal-weight five-seed means in training-update order. They are not sorted by alpha, smoothed, regressed or extrapolated. Faint points show individual seed/checkpoint measurements in the detailed panel. Diamonds mark 2,000 updates and large circles mark 20,000. Crossbars show separate 95% Student-t confidence intervals across the five seeds for alpha and RMSE at those budgets. They are marginal pointwise intervals, not a joint 95% confidence region or simultaneous confidence band. Repeated checkpoints are correlated and do not count as independent seeds.

| Optimizer | 20k confidence/tail alpha | 20k test RMSE |
|---|---:|---:|
| SGD + momentum | 2.277816 ± 0.009514 | 0.509690 ± 0.005787 |
| Native AdamW | 1.678083 ± 0.023836 | 0.480780 ± 0.002797 |
| Modified AdamW | 1.612786 ± 0.003481 | 0.483766 ± 0.002055 |

All ± entries are 95% Student-t confidence half-widths over seeds 2340–2344. At the matched 20k budget, native AdamW has lower test RMSE than modified AdamW despite its slightly higher alpha. SGDM retains a value above 2 and a higher test RMSE. These descriptive trajectories do not establish a universal optimum or a causal effect of alpha; update count, weights and optimizer dynamics change together. No test minimum is used to choose a new model or alter the ongoing grid.

## Data and verification

`independent_performance_alpha_audit.json` contains all 615 paired seed/step observations, source identifiers, statistical summaries, ranges, and the independent verification report. The audit recomputed raw ordinary-alpha fits and checked test RMSE against the square root of same-row test MSE. All 123 mean/SD/CI summary rows matched the official analysis in 1,230 numerical comparisons, to within 8.9e-16. The plotting script independently recalculates each displayed summary from those 615 observations and writes CSV files; its maximum discrepancy is 0.

Both measurements come from the same archived fixed-step history row. The original trainer's hash-verified `observation(measured_model, ...)` computes test metrics and spectral fits without an intervening optimizer update. The legacy rows do not contain per-row weight hashes; the audit therefore records source/row binding, not a cryptographic same-weight-hash check. The new SGDM extension records those extra hashes. The local source projections omit diagnostic eigenvalue arrays and verbose fitting logs only; original result hashes and Git blob identities remain in the audit and full records remain at the pinned source URLs.

## Reproduce the figure

With the repository's existing pinned benchmark environment, run:

```bash
python make_test_performance_alpha_plot.py
```

Or create an isolated plotting environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy==2.3.1 scipy==1.16.0 matplotlib==3.10.3
python make_test_performance_alpha_plot.py
```

The default input is the adjacent `independent_performance_alpha_audit.json`. Use `--audit PATH --output-dir DIRECTORY` to select another location containing those same audited bytes. The script verifies cohort membership and paired observations, recalculates means, sample SD, and 95% CIs, and writes the PNG, vector PDF, individual-measurement CSV, summary CSV and `plot_audit.json`. It does not train or retune any model.
