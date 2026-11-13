# Modified AdamW experiments

This campaign studies a **magnitude-only** modification: retain Adam's adaptive matrix-update direction and moment history, and replace its Frobenius magnitude with the SGDM target magnitude computed from that same run's gradient history. Bias updates remain AdamW. Controls are native AdamW and SGDM with matched decay.

## Experiments

| Folder | Dataset / model | Status |
|---|---|---|
| [2026-11-13_mnist_mlp_experiment01](2026-11-13_mnist_mlp_experiment01/) | MNIST, 784→256→256→10 ReLU MLP | 14/135 complete and verified; one seed so far |

## Organization for the coming days

Keep each experiment in a dated child folder: `YYYY-MM-DD_dataset_architecture_experimentNN/`. Continue the same experiment in its existing folder; do not duplicate completed runs. Store immutable source/config snapshots in `frozen/`, small numerical results in `results/`, and audit evidence in `verification/`. Record large artifact locations in `ARTIFACTS.md`.

Use [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for future experiments over the next five days. Maintain [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) as work actually completes. Future work is planned, not already running.

Run one experiment at a time with a 3–5 hour budget, preserve checkpoints, verify results, and review before starting the next experiment. Use dense layers or transformers; no CNNs. Treat seeds as replicates and retain all preset hyperparameter settings. Report upward alpha shifts separately from distance to alpha=2 and accuracy.
