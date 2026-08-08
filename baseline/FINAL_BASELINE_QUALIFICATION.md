# Final baseline qualification protocol

This is the final scientific gate between a **strong source-backed recipe** and
a **frozen best baseline for the selected model and data**.

A code review can establish that the implementation is internally correct. It
cannot prove that one learning rate or weight decay is globally optimal. The
repository therefore uses a bounded, preregistered validation search and freezes
the winner before protected test results are interpreted.

The candidate neighborhoods and validation-only ranking implementation live in:

```text
rg_baselines/qualification.py
```

## Non-negotiable rules

1. The official test split is never used to select a candidate, checkpoint,
   training horizon, augmentation, learning rate, weight decay, or schedule.
2. The unit of replication is one complete training run.
3. Candidate screening may use one seed, but qualification requires the same
   three complete seeds for every finalist.
4. A candidate is selected by mean best validation loss. Higher validation
   accuracy and then deterministic candidate ID are the only tie-breaks.
5. The winning configuration is written to a lock file with the source commit,
   data identity, evidence paths, and full validation leaderboard.
6. Only after the lock file exists may final protected-test comparisons be
   interpreted.
7. A change to architecture, dataset identity, split, tokenizer, initialization,
   optimizer implementation, schedule implementation, or upstream commit creates
   a new baseline version and invalidates the old lock.

## Qualification stages

### Stage A — bounded screening

Use one preregistered seed and the exact optimization/validation split.
WeightWatcher may run only at epoch zero and the final screening checkpoint to
avoid making the search prohibitively expensive; it is restored to the full
reference cadence for final runs.

- **MNIST / MLP3:** run every candidate for the full 30 epochs.
- **CIFAR-10 / small ViT:** successive halving at 100 and 200 epochs; the top two
  candidates per optimizer continue to the full 300 epochs.
- **One-head nanoGPT:** run every candidate through the full eight-pass horizon.
- **nanochat d12:** keep the pinned upstream recipe; it already computes the
  training horizon, batch, LR scaling, and weight-decay scaling from the model.
  Any local alternative is a new profile rather than an informal tweak.

Stage A may eliminate only clearly inferior or unstable candidates. It is not
used to report test performance.

### Stage B — three-seed qualification

Run the top two candidates for each optimizer using all three canonical seeds.
Each candidate must have the same:

```text
architecture
optimization and validation examples
token budget or epoch budget
evaluation probes
checkpoint cadence
metric definitions
hardware precision policy
```

Create one row per candidate and seed at the checkpoint chosen by minimum
validation loss. Pass those rows to:

```python
from rg_baselines.qualification import rank_validation_candidates

leaderboard, selected_rows = rank_validation_candidates(
    history,
    expected_seeds=(...),
)
```

Layers, matrices, epochs, minibatches, and WeightWatcher fits are repeated
measurements inside a run. They are not additional replicates.

### Stage C — freeze the reference

Write the winning configuration before examining final protected-test
comparisons:

```python
from rg_baselines.qualification import freeze_winner

freeze_winner(
    "qualification_locks/<baseline>/<optimizer>.json",
    candidate=winner,
    leaderboard=leaderboard,
    evidence_paths=[...],
    source_commit="<git-sha>",
    data_identity={...},
)
```

The lock explicitly records:

```text
protected_test_used_for_selection: false
```

## Candidate neighborhoods

The committed recipe is always included as the center candidate. The search is
small enough to be feasible and large enough to catch the most consequential
mistuning.

### MNIST / MLP3

- SGD: peak LR, matrix weight decay, and warm-up length.
- AdamW: peak LR, matrix weight decay, and warm-up/no-warm-up.
- Muon: matrix LR, matrix weight decay, and auxiliary AdamW LR.

The architecture, split, 30-epoch budget, gradient clipping, and optimizer
parameter partition remain fixed.

### CIFAR-10 / small ViT

- SGD: peak LR, weight decay, and warm-up start.
- AdamW: peak LR over the range from the linearly scaled DeiT value through the
  unscaled reference value, plus weight decay.
- Muon: matrix LR, matrix decay, and auxiliary AdamW LR.

The selected model remains six blocks, width 192, three heads, 4x4 patches. The
final recipe fixes LayerNorm epsilon at 1e-6, restores fan-in initialization for
the patch projection, begins warm-up from a small explicit LR, performs cosine
decay, and holds a non-zero LR floor for the final ten epochs.

### One-head nanoGPT / FineWeb-Edu

- SGD: peak LR, weight decay, and warm-up fraction.
- AdamW: peak LR and weight decay around the nanoGPT reference.
- Muon: matrix LR, matrix decay, and auxiliary AdamW LR.

The final horizon is eight passes over the 10M-token training split. This is
approximately the 12-tokens-per-scaling-parameter regime for the tied 50,257 x
128 language head plus the six hidden transformer matrices. Validation probes
are common across every optimizer and training seed, use 64 fixed batches, and
BLEU uses 64 common held-out continuations.

### nanochat

The canonical d12 baseline remains the pinned native upstream recipe. Its
upstream trainer determines the target token horizon, batch scaling, LR scaling,
weight-decay scaling, warm-up, warmdown, and Muon momentum schedule. The
`mac_d4` profile is a separately labeled development baseline and must never be
reported as d12 evidence.

## What “best baseline” means here

After qualification, “best baseline” means:

> the best validation-selected configuration in the preregistered, source-backed
> candidate neighborhood for this exact architecture, dataset identity, split,
> training budget, optimizer implementation, and precision policy.

It does not mean that every real-valued hyperparameter in an unbounded search
space has been mathematically optimized. That stronger claim would not be
credible for any practical neural-network training experiment.
