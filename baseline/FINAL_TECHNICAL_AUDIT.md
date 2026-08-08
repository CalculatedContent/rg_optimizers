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

### Technical discrepancy found

The architecture and optimizer implementation were correct, but the model
implicitly inherited `nn.Linear`'s generic default initialization. That default
has substantially lower variance than the fan-in He/Kaiming value designed to
preserve signal through ReLU hidden layers.

### Correction and result

Recipe version 3 now fixes the initialization contract:

```text
fc1, fc2 weights:    Kaiming uniform, fan-in, ReLU gain
fc3 weights:         Xavier uniform
all biases:          zero
```

The final implementation also retains:

- a deterministic 55k/5k optimization/validation split;
- an official test set that is monitoring-only;
- update-level warm-up and cosine decay;
- correct matrix/no-decay parameter groups;
- correct Muon hidden-matrix and auxiliary-AdamW partitioning;
- latest, best-validation, final, and per-epoch checkpoints;
- data-generator and CPU/CUDA/MPS RNG restart state;
- randomized WeightWatcher isolation; and
- complete-run Student-t uncertainty.

The remaining uncertainty is empirical tuning, not a known technical error. The
bounded qualification neighborhood tests the consequential LR, weight-decay,
and warm-up alternatives around each committed center while holding the
qualified initialization fixed.

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

1. The original 50M-token budget was shorter than the approximately
   12-tokens-per-scaling-parameter target for the tied vocabulary head plus the
   six hidden matrices.
2. An intermediate correction reached 80M processed tokens by repeatedly
   sampling a 10M-token corpus. That matched the budget but left the available
   corpus unnecessarily small and increased repeated-data risk.
3. Each training seed previously chose a different random validation/test/BLEU
   probe, adding evaluation-sample noise to optimizer comparisons.
4. Sixteen evaluation batches were too small for stable best-checkpoint
   selection on a 1M-token validation split.
5. Sixteen BLEU continuations made the secondary overlap metric unnecessarily
   noisy.
6. Logged LR values described the LR prepared for the next update rather than
   the LR that produced the current checkpoint.
7. Randomized WeightWatcher analysis restored Python, NumPy, and CPU Torch RNG
   state but not CUDA/MPS RNG state.

### Corrections

Protocol version 3 now uses:

```text
available training corpus:   80M document-disjoint FineWeb-Edu tokens
sampled training budget:     approximately 80M tokens
optimizer steps:             9,766
reporting / WW checkpoints:  0, 0.125, ..., 1.0 corpus-equivalent budget
validation probe:            64 common fixed batches
protected test probe:        64 common fixed batches
BLEU probe:                  64 common fixed continuations
probe identity:              independent of training seed and optimizer
LR CSV semantics:            LR used by the update that produced the row
WW RNG restoration:         Python, NumPy, CPU Torch, CUDA, and MPS
```

Training still uses canonical nanoGPT random contiguous windows, so the budget
is corpus-equivalent rather than an assertion that every token is traversed
exactly once. The AdamW, SGD, and Muon schedule implementations remain
update-level linear warm-up followed by cosine decay to a non-zero floor. The
optimizer partition and nanoGPT initialization remain correct.

## nanochat

### Technical discrepancies found

The pinned trainer correctly falls back from Flash Attention to PyTorch SDPA on
MPS and CPU, but it compiled three execution paths unconditionally:

1. the model;
2. the fused AdamW step kernel; and
3. the fused Muon step kernel.

That is the canonical CUDA/server path, not a dependable portable contract for
Apple MPS or generic CPU execution. The earlier wrapper had addressed only the
model compile call.

### Corrections and result

The final portable wrapper applies exact, pinned-source patches to all three
compile sites:

```text
CUDA d12:     model and fused optimizer kernels remain compiled
MPS / CPU:    identical model and optimizer mathematics run in eager mode
MPS:          native kernels preferred; isolated unsupported ops may fall back
runtime lock: profile, commit, device, process, compile, and fallback policy
```

The canonical d12 profile remains accepted as a pinned native upstream
reference. The wrapper still does not reimplement the architecture or optimizer.
The pinned trainer derives:

- the target token horizon;
- total token batch;
- batch-dependent LR scaling;
- model/data-dependent weight-decay scaling;
- the 40-step warm-up;
- the long linear warmdown;
- Muon momentum warm-up/warmdown; and
- cosine weight-decay decay.

A real pinned-source preflight now builds a tiny upstream GPT, runs one
forward/backward pass, and executes the native combined Muon/AdamW optimizer
step. CI runs the CPU preflight; the target Mac must pass the same preflight on
MPS before a long `mac_d4` campaign.

The `mac_d4` profile remains explicitly separate from d12. Its purpose is local
Apple-MPS development using the same pinned implementation, not a claim of d12
quality or scale.

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
> split, initialization, training budget, optimizer implementation, and
> precision/runtime policy.

It would be inaccurate to claim that static review proves a global optimum over
all continuous hyperparameters. The qualification code and lock file make the
stronger practical claim reproducible and auditable.
