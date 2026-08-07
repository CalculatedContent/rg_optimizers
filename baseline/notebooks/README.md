# Baseline notebooks

The notebook directory contains the interactive entry points for the reference
experiments documented in
[`../BASELINE_RECIPE_AUDIT.md`](../BASELINE_RECIPE_AUDIT.md).

## MNIST / MLP3

Run in this order:

1. `MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
2. `MNIST_MLP3_AdamW_Baseline.ipynb`
3. `MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`
4. `MNIST_MLP3_Baseline_Comparison.ipynb`

The three training notebooks use the same MLP, data, seed set, 30-epoch budget,
evaluation protocol, WeightWatcher calls, and checkpoint cadence. They differ
only in their optimizer-specific reference recipes:

```text
SGD + Nesterov:
  peak LR 0.05
  2-epoch warm-up
  cosine floor 5e-4
  momentum 0.90
  matrix weight decay 1e-4

AdamW:
  peak LR 1e-3
  1-epoch warm-up
  cosine floor 1e-5
  betas (0.90, 0.999)
  matrix weight decay 1e-2

Muon + auxiliary AdamW:
  hidden-matrix LR 0.02 -> 0.002
  auxiliary AdamW LR 3e-4 -> 3e-5
  2-epoch warm-up
  Muon momentum 0.95
  five Newton-Schulz steps
```

The historical result path `sgd_momentum_muon` is retained, but that arm now
uses the reference auxiliary-AdamW parameter partition.

## CIFAR-10 / small ViT

Run `CIFAR10_ViT_Optimizer_Baselines.ipynb`.

The notebook runs three optimizers × three seeds on a fixed 45k/5k
train/validation split. The official test set is monitoring-only. The committed
300-epoch recipe uses RandAugment, color jitter, mixup, CutMix, label smoothing,
random erasing, stochastic depth, gradient clipping, optimizer-specific
warm-up/cosine schedules, validation-selected best checkpoints, and restartable
latest checkpoints.

## nanochat d12

Run `NanoChat_D12_Reference_Baseline.ipynb`.

This notebook pins and runs native upstream nanochat d12. The upstream
initialization, data/tokenizer pipeline, parameter groups, scaling rules, and
learning-rate/momentum/weight-decay schedules remain unchanged. WeightWatcher
runs offline on saved checkpoints.

## One-head nanoGPT

The one-head FineWeb-Edu notebooks live in `../nanogpt_one_head/notebooks/`.
See [`../nanogpt_one_head/README.md`](../nanogpt_one_head/README.md).

## WeightWatcher

Reference notebooks require direct outputs from:

```python
watcher.analyze(ERG=True, randomize=True, ...)
```

No fallback alpha, proxy `num_traps`, or fabricated `ERG_gap` is permitted.
