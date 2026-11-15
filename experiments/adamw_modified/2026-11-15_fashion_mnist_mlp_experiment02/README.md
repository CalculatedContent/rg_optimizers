# Experiment 02: Fashion-MNIST magnitude-only AdamW graft

**Prepared plan: 120 runs. Completed: 0. Training has not started.** This is a sibling of Experiment 01; none of the MNIST files, results or follow-up runs are overwritten.

This experiment transfers the cleaned MNIST comparison to Fashion-MNIST with the same ReLU MLP **784→256→256→10**, exact optimizer intervention, five paired seeds and training schedule.

## Planned comparison

| Item | Plan |
|---|---|
| Dataset | Fashion-MNIST: 55,000 train / 5,000 validation / 10,000 test |
| Architecture | 784→256→256→10, ReLU; dense layers only |
| Optimizers | Native AdamW, magnitude-only graft, SGD + momentum |
| Seeds | 31001–31005; shared initialization and per-epoch example order across arms |
| Batch 128 | LR multipliers 0.5 and 1 |
| Batch 256 | LR multipliers 0.5, 1 and 2 |
| Batch 512 | LR multipliers 0.5, 1 and 2 |
| Run count | 5 seeds × 8 settings × 3 optimizers = 120 |
| Epochs | 120; five-epoch warmup, cosine schedule to 10% peak |
| Adam peak layer rates | [0.001, 0.001, 0.001] × multiplier |
| SGDM magnitude layer rates | [0.04, 0.2, 0.8] × multiplier |
| Central comparison | Batch 256 / LR ×1, carried over in advance |
| Budget | At most 4.5 hours training within a 5-hour total pass; no automatic extension |

The graft retains Adam direction, moments, bias corrections and bias updates. Each matrix receives the Frobenius magnitude of SGDM momentum computed along that same graft run's gradient history. Preserve the existing 10⁻³⁰ denominator floor and report its small-update exceptions. All arms retain the matched decay per processed example.

Batch 128 / LR ×2 is **excluded from every arm in this plan**, following the MNIST stability failure. It is not labeled a Fashion-MNIST failure because no Fashion-MNIST models have been trained. Any later stress test belongs in a separately labeled follow-up. The eight included settings are inherited working settings; they are not yet validated on Fashion-MNIST.

## What is prepared

- [Protocol plan](protocol/Protocol_Plan.json) and [all 120 planned runs](configs/run_matrix.csv).
- Byte-identical [optimizer reference](frozen/optimizer_reference.py) and [parent protocol](frozen/Parent_Protocol.json).
- A [setup/data validator](execution/validate_setup.py), official Fashion-MNIST checksums, and the required original split-file hash.
- Separate folders for data, results, plots, checkpoints, verification and [follow-up experiments](follow_up/README.md).
- [Execution status](results/STATUS.json) distinguishes planned runs from completed or usable models.

From this folder, `python execution/validate_setup.py` validates the setup without importing the training code, downloading data or training. After the data are staged, `python execution/validate_setup.py --check-data data` validates their identity and split. The [setup audit](verification/Setup_Audit.json) covers the plan only.

**Before training:** stage the Fashion-MNIST files and the byte-identical original split; implement the isolated execution adapter using the reference functions; complete runtime/update/replay checks; then freeze data and executable source hashes. There is deliberately no training launcher in this setup. Do not execute the historical MNIST reference script directly: its dataset-freezing and launch paths belong to the old experiment.

## Outcomes and reporting

Track FC1 and FC2 α first, their separate distances to 2, mean hidden |α−2|, test accuracy/loss, validation behavior and fit support. FC3 has ten eigenvalues and remains exploratory. Use WeightWatcher 0.7.7 default fitting, with no α-targeted cutoff.

Preserve all planned outcomes, label confirmed collapse or nonfinite training as a hyperparameter failure, and distinguish completed runs from usable models. Five seeds are the independent replicates. Across settings, average within each seed before calculating paired 95% intervals. Report α moving upward separately from moving closer to 2 and from improving accuracy. Do not tune using the test set or silently replace failed runs.

[Results layout](results/README.md) · [Planned plots](plots/README.md) · [Data provenance](data/README.md) · [Checkpoint storage](checkpoints/README.md) · [Artifact status](ARTIFACTS.md)

[Previous MNIST experiment](../2026-11-13_mnist_mlp_experiment01/) · [MNIST model-status report](../2026-11-13_mnist_mlp_experiment01/MODEL_STATUS.md)
