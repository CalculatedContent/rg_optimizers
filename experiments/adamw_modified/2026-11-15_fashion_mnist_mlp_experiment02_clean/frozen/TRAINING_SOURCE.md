# Frozen training source

This folder republishes completed results. It does not contain a new training launch. The original runner, kernel, protocol, workflow and source hashes remain at their immutable recorded commit.

- [Runner and original protocol](https://github.com/CalculatedContent/rg_optimizers/tree/a8fdde0ddf8674d1904068ef92f43b2f6ae97b69/experiments/adamw_modified/2026-11-15_fashion_mnist_mlp_experiment02/follow_up/2026-11-15_clean_fixed_endpoint)
- [Original optimizer kernel](https://github.com/CalculatedContent/rg_optimizers/blob/a8fdde0ddf8674d1904068ef92f43b2f6ae97b69/experiments/adamw_modified/2026-11-15_fashion_mnist_mlp_experiment02/execution/study.py)
- [Data and split provenance](https://github.com/CalculatedContent/rg_optimizers/blob/a8fdde0ddf8674d1904068ef92f43b2f6ae97b69/experiments/adamw_modified/2026-11-15_fashion_mnist_mlp_experiment02/verification/Data_Provenance.json)

`Protocol.json` here is the identical protocol blob. See `verification/Confirmatory_Freeze.json` for the actual source hashes recorded before confirmation runs. Reproduce the comparison, without training, with `python analysis/compare.py`.
