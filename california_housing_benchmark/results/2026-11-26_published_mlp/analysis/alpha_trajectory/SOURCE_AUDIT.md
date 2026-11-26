# Completed California Housing alpha trajectories

Source: [GitHub workflow 45753214421](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45753214421), immutable archive `84b0219dc87c7caaefdc7245e455e53c300b8dfc`.

All 15 runs use seeds 2340–2344. Every curve uses all five seed measurements at each of 41 fixed budgets (0, 500, ..., 20,000). The shaded regions are pointwise 95% Student-t intervals across seeds, not layer dispersion or simultaneous confidence bands.

Ordinary fitted alpha is aggregated with weights sqrt(num_pl_spikes)/(D+0.02), without a matrix-size factor. Only the 784×784 hidden matrix meets the size threshold, so the model average equals that matrix's alpha. The initial broad spectrum fit does not establish a power-law regime.

| Optimizer | 2,000 updates: mean ± 95% CI half-width | 20,000 updates: mean ± 95% CI half-width |
|---|---:|---:|
| SGD + momentum | 3.576011 ± 0.613439 | 2.277816 ± 0.009514 |
| Native AdamW | 2.125821 ± 0.045309 | 1.678083 ± 0.023836 |
| Modified AdamW | 4.384999 ± 0.088063 | 1.612786 ± 0.003481 |

The main figure uses a logarithmic alpha axis to retain the high initialization fits while showing later differences. A supplementary linear detail includes only the measured 2,000–20,000 interval.

The audit recomputed all 615 seed/step averages from retained raw fit fields and checked all 123 mean/SD/CI summaries against the committed official analysis. Local input projections omit only eigenvalue arrays and verbose fitting logs; their original Git blob IDs and original SHA-256 hashes are preserved in source_audit.json. Raw original result JSONs are available at the pinned source URLs. No new training was run.
