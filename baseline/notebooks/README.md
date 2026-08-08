# Baseline notebooks

These notebooks are the interactive entry points for the audited reference
experiments. Read:

- [`../BASELINE_RECIPE_AUDIT.md`](../BASELINE_RECIPE_AUDIT.md);
- [`../BASELINE_EXECUTION_REVIEW.md`](../BASELINE_EXECUTION_REVIEW.md);
- [`../FINAL_BASELINE_QUALIFICATION.md`](../FINAL_BASELINE_QUALIFICATION.md).

The first two documents establish implementation correctness. The final
document defines the validation-only search and lock file required before a
committed source-backed candidate is described as the best baseline for its
exact architecture and data.

## MNIST / MLP3

Run in order:

1. `MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
2. `MNIST_MLP3_AdamW_Baseline.ipynb`
3. `MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`
4. `MNIST_MLP3_Baseline_Comparison.ipynb`

The three training notebooks use the same deterministic 55k/5k
optimization/validation split, official monitoring-only test set, MLP, seeds,
30-epoch budget, update-level warm-up/cosine schedule implementation,
WeightWatcher cadence, and restart protocol. They differ only in their
optimizer-specific reference recipe.

Every notebook can be rerun safely. Compatible completed seeds are loaded;
compatible incomplete seeds resume from `checkpoint_latest.pt`; incompatible
fingerprints fail visibly. The comparison notebook reports final and
validation-selected checkpoints and never selects on test performance.

## CIFAR-10 / small ViT

Run `CIFAR10_ViT_Optimizer_Baselines.ipynb`.

The notebook executes three optimizers × three seeds under the fixed 45k/5k
optimization/validation split and protected official test set. It imports the
final public runtime from `rg_baselines.vit_final`, which adds LayerNorm epsilon
1e-6, fan-in patch initialization, explicit low-LR warm-up, cosine decay, and a
10-epoch non-zero-LR cooldown to the 300-epoch DeiT-style regularization recipe.
All performance bands use the three complete runs. Layer plots give every
physical matrix its own curve and do not pool transformer blocks as extra
replicates.

## nanochat

Run `NanoChat_D12_Reference_Baseline.ipynb`.

The notebook has two explicit profiles from the pinned upstream implementation:

- `d12`: canonical CUDA/server reference;
- `mac_d4`: separate Apple-MPS development baseline.

`RG_NANOCHAT_PROFILE=auto` selects d12 on CUDA and mac_d4 on MPS/CPU. The
notebook creates the platform-correct environment, uses one process on MPS/CPU,
resumes from complete model/metadata/optimizer-shard checkpoints, keeps profile
caches separate, and performs offline WeightWatcher analysis only on principal
hidden matrices.

## One-head nanoGPT

The one-head FineWeb-Edu notebooks live in `../nanogpt_one_head/notebooks/`.
See [`../nanogpt_one_head/README.md`](../nanogpt_one_head/README.md). Protocol v3
uses an eight-pass horizon, common fixed validation/test probes, 64 BLEU
continuations, update-aligned LR logging, verified corpus identity, and
accelerator-complete restart/diagnostic RNG state.

## WeightWatcher and uncertainty

Strict reference notebooks require direct output from:

```python
watcher.analyze(ERG=True, randomize=True, ...)
```

No fallback alpha, proxy `num_traps`, or fabricated `ERG_gap` is permitted.
The unit of replication for every confidence interval is a complete training
run, never a layer, matrix, checkpoint, or fit point.
