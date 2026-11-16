# Modified AdamW experiments

This campaign studies a **magnitude-only** modification: retain Adam's adaptive matrix-update direction and moment history, and replace its Frobenius magnitude with the SGDM target magnitude computed from that same run's gradient history. Bias updates remain AdamW. Controls are native AdamW and SGDM with matched decay.

## Experiments

| Folder | Dataset / model | Status |
|---|---|---|
| [2026-11-13_mnist_mlp_experiment01](2026-11-13_mnist_mlp_experiment01/) | MNIST, 784→256→256→10 ReLU MLP | 135 original runs + 3 replication runs verified; 132 usable models and 6 failed models; explicit model labels, paired-effect plots, and separate hyperparameter-failure report |
| [2026-11-15_fashion_mnist_mlp_experiment02](2026-11-15_fashion_mnist_mlp_experiment02/) | Fashion-MNIST, 784→256→256→10 ReLU MLP | Historical interrupted/superseded passes; recovery stopped and available outcomes preserved. Completed clean pass is in the parallel folder below. |
| [2026-11-15_fashion_mnist_mlp_experiment02_clean](2026-11-15_fashion_mnist_mlp_experiment02_clean/) | Fashion-MNIST, 784→256→256→10 ReLU MLP | Complete: 120/120 fixed-epoch-20 runs, five paired seeds; verified results, checkpoint links and comparison with MNIST. No new training. |
| [2026-11-16_fashion_mnist_mlp_experiment02_extension](2026-11-16_fashion_mnist_mlp_experiment02_extension/) | Fashion-MNIST, same 120 MLP models | Failed before continuation: 0/120 extended; parent replay loss equality failed. Parent checkpoints preserved; no rerun. |
| [2026-11-16_fashion_mnist_mlp_experiment02_extension_retry01](2026-11-16_fashion_mnist_mlp_experiment02_extension_retry01/) | Fashion-MNIST, same120 saved models | Complete and verified: 120/120 fixed-epoch60 continuations; all120 functional, five seeds. Graft accuracy +0.41950pp versus AdamW, but higher CE and worse mean alpha distance. Checkpoints, loss-tail diagnostic and final audit saved; awaiting review. |
| [2026-11-15_tiny_shakespeare_gpt_experiment03](2026-11-15_tiny_shakespeare_gpt_experiment03/) | Tiny Shakespeare, two-block GPT | On hold: finish Fashion-MNIST first; pilot controls preserved, graft cancelled,45-run expansion cancelled |

## Organization for the coming days

Keep each experiment in a dated child folder: `YYYY-MM-DD_dataset_architecture_experimentNN/`. Continue the same experiment in its existing folder; do not duplicate completed runs. Store immutable source/config snapshots in `frozen/`, small numerical results in `results/`, and audit evidence in `verification/`. Record large artifact locations in `ARTIFACTS.md`.

Use [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for future experiments over the next five days. Maintain [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) as work actually completes. Future work is planned, not already running.

Run one experiment at a time with a 3–5 hour budget, preserve checkpoints, verify results, and review before starting the next experiment. Use dense layers or transformers; no CNNs. Treat seeds as replicates and preserve all preset hyperparameter outcomes. Present shared stable settings separately from documented failed settings, and label any post-hoc selection. Report upward alpha shifts separately from distance to alpha=2 and accuracy.



[Experiment 01 model status and concise results](2026-11-13_mnist_mlp_experiment01/MODEL_STATUS.md) include the complete 138-model register, setting-level status plot and paired α-alignment/accuracy plot.

[MNIST versus completed Fashion-MNIST analysis](2026-11-15_fashion_mnist_mlp_experiment02_clean/COMPARISON.md) uses the eight common settings and five independent seeds. MNIST used 120 epochs; Fashion used 20 epochs. The dataset and schedule changes are explicitly separated from within-experiment optimizer effects.
