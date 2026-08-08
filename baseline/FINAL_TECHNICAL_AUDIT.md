# Final technical audit of the baseline suite

This audit re-opened the merged baseline implementation after the recipe and
execution reviews. It focused on four separate questions:

1. Does the code implement the intended model and optimizer?
2. Are initialization and learning-rate schedules internally correct?
3. Can a run restart without changing the trajectory or selection policy?
4. Is the committed hyperparameter point enough to call the run the best
   baseline for the selected architecture and data?

## Bottom line

After the corrections in this audit, no static code defect remains that is known
to prevent the MNIST, final CIFAR-10 ViT, one-head FineWeb-Edu, or pinned
nanochat reference from training normally under its stated hardware policy.

The code can now support a defensible “best baseline” claim, but that claim is
conditional on completing the validation-only qualification in
`FINAL_BASELINE_QUALIFICATION.md`. Static inspection cannot establish the
numerical optimum of LR or weight decay.

## MNIST / MLP3

### Result

Accepted without a recipe change.

The final implementation already has:

- a deterministic 55k/5k optimization/validation split;
- an official test set that is monitoring-only;
- step-level warm-up and cosine decay;
- correct matrix/no-decay parameter groups;
- correct Muon hidden-matrix and auxiliary-AdamW partitioning;
- latest, best-validation, final, and per-epoch checkpoints;
- data-generator and CPU/CUDA/MPS RNG restart state;
- randomized WeightWatcher isolation; and
- complete-run Student-t uncertainty.

The remaining uncertainty is empirical tuning, not a known technical error. The
bounded qualification neighborhood tests the consequential LR, weight-decay,
and warm-up alternatives around each committed center.

## CIFAR-10 / small ViT

### Technical discrepancies found

The earlier custom implementation was strong but not fully aligned with the
intended DeiT-style reference:

1. LayerNorm used PyTorch's default epsilon rather than `1e-6`.
2. The global transformer initializer overwrote the patch Conv2d with a
   0.02-truncated-normal distribution instead of preserving fan-in
   initialization.
3. Warm-up began at a fixed fraction of the peak rather than an explicit small
   warm-up LR.
4. The cosine schedule did not include the intended final cooldown at the
   non-zero floor.
5. If epoch zero remained validation-best, the lower-level runner could have
   already saved epoch one as `checkpoint_best.pt`; the post-run verifier then
   rejected rather than repaired that mismatch.

### Corrections

The public notebook now imports `rg_baselines.vit_final`, which enforces:

```text
LayerNorm epsilon:       1e-6
patch projection init:   Conv2d fan-in reset
warm-up start:           explicit optimizer-specific low LR
schedule:                warm-up -> cosine -> 10-epoch floor cooldown
selection:               minimum validation loss
```

The final runtime reconstructs a true epoch-zero best checkpoint when needed,
uses a versioned fingerprint through the inherited config, retains accelerator
RNG state, and preserves randomized WeightWatcher isolation.

The selected architecture remains exactly six blocks, width 192, three heads,
4x4 patches, MLP ratio four, and 0.10 stochastic depth.

## One-head nanoGPT / FineWeb-Edu

### Technical/statistical weaknesses found

1. Five passes supplied about 50M tokens, shorter than the approximately
   12-tokens-per-scaling-parameter target for the tied vocabulary head plus the
   six hidden matrices.
2. Each training seed previously chose a different random validation/test/BLEU
   probe, adding evaluation-sample noise to optimizer comparisons.
3. Sixteen evaluation batches were too small for stable best-checkpoint
   selection on a 1M-token validation split.
4. Sixteen BLEU continuations made the secondary overlap metric unnecessarily
   noisy.
5. Logged LR values described the LR prepared for the next update rather than
   the LR that produced the current checkpoint.
6. Randomized WeightWatcher analysis restored Python, NumPy, and CPU Torch RNG
   state but not CUDA/MPS RNG state.

### Corrections

Protocol version 3 now uses:

```text
training horizon:        8 passes / approximately 80M tokens
optimizer steps:         9,766
validation probe:        64 common fixed batches
protected test probe:    64 common fixed batches
BLEU probe:              64 common fixed continuations
probe identity:          independent of training seed and optimizer
LR CSV semantics:        LR used by the update that produced the row
WW RNG restoration:      Python, NumPy, CPU Torch, CUDA, and MPS
```

The AdamW, SGD, and Muon schedule implementations remain update-level linear
warm-up followed by cosine decay to a non-zero floor. The optimizer partition
and nanoGPT initialization remain correct.

## nanochat

### Result

The canonical d12 profile remains accepted as a pinned native upstream
reference. The wrapper does not reimplement its architecture or optimizer. The
pinned trainer derives:

- the target token horizon;
- total token batch;
- batch-dependent LR scaling;
- model/data-dependent weight-decay scaling;
- the 40-step warm-up;
- the long linear warmdown;
- Muon momentum warm-up/warmdown; and
- cosine weight-decay decay.

The `mac_d4` profile remains explicitly separate from d12. Its purpose is local
Apple-MPS development using the same pinned implementation, not a claim of d12
quality or scale.

An actual MPS smoke run is still required on the target Mac to establish local
runtime throughput and memory headroom. Linux CPU CI cannot prove Apple GPU
performance, although it does test command construction, one-process policy,
checkpoint-shard completeness, seed patching, and resumed-log handling.

## WeightWatcher contract

Every strict reference continues to require direct output from:

```python
watcher.analyze(ERG=True, randomize=True, ...)
```

The code does not synthesize `alpha`, `ERG_gap`, or `num_traps`. Randomized
analysis is isolated from subsequent training RNG state.

## Meaning of “best possible”

The strongest scientifically supportable statement is:

> After qualification, this is the best validation-selected candidate in the
> preregistered source-backed neighborhood for the exact architecture, dataset,
> split, training budget, optimizer implementation, and precision policy.

It would be inaccurate to claim that static review proves a global optimum over
all continuous hyperparameters. The qualification code and lock file make the
stronger practical claim reproducible and auditable.
