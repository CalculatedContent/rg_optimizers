# Baseline executable review

This document records the code-and-notebook review performed after the recipe
audit in `BASELINE_RECIPE_AUDIT.md`. The recipe audit asks whether the data,
initialization, optimizer, and schedule are defensible. This review asks whether
the committed implementation actually enforces that protocol, restarts safely,
and computes uncertainty with the correct unit of replication.

## Review standard

A reference baseline is accepted only when:

1. optimization, validation, and test identities are explicit;
2. validation—not test data—selects hyperparameters and best checkpoints;
3. warm-up and decay are applied at the intended cadence;
4. optimizer parameter partitions match their reference definitions;
5. checkpoints contain enough state to resume the same trajectory;
6. randomized diagnostics do not perturb training RNG streams;
7. the unit of replication for error bars is a complete training run;
8. direct WeightWatcher `alpha`, `ERG_gap`, and `num_traps` are retained without
   fallback values or proxy counts;
9. notebooks fail on incomplete runs rather than silently comparing partial
   outputs; and
10. executable CI covers source syntax, notebook syntax, optimizer updates,
    schedules, data integrity, checkpoint restoration, and statistical
    invariants.

“Best possible baseline” means the strongest source-backed and internally
consistent committed reference that can be defended before looking at protected
test outcomes. It does not mean that a global hyperparameter optimum has been
proved. Any final tuning must be a bounded validation-only search frozen before
interpreting the test set.

## MNIST / MLP3

### Corrections made

- The official 60,000-example training set is now deterministically split into
  55,000 optimization and 5,000 validation examples.
- The official 10,000-example test set is marked monitoring-only.
- The short warm-up is now applied before every optimizer update. The previous
  epoch-level implementation caused a one-epoch AdamW warm-up to jump directly
  to the peak LR on the first epoch.
- SGD, AdamW, and Muon retain separate source-backed peak/floor schedules.
- The Muon arm uses Muon on `fc1.weight` and `fc2.weight`, with auxiliary AdamW
  on the classifier and biases.
- `checkpoint_latest.pt` and `checkpoint_best.pt` now store model, optimizer,
  data-loader generator, Python/NumPy/Torch/CUDA/MPS RNG state, best-validation
  state, and a protocol fingerprint.
- Randomized WeightWatcher analysis saves and restores every training RNG
  stream.
- Completed compatible runs are loaded; compatible interrupted runs resume; an
  incompatible fingerprint fails visibly.
- The notebooks now plot train/validation/test metrics and primary/auxiliary LR
  trajectories and report final versus validation-selected test results.

### Accepted reference

| Optimizer | Peak LR | Floor | Warm-up | Other settings |
|---|---:|---:|---:|---|
| SGD + Nesterov | 0.05 | 5e-4 | 2 epochs, step-level | momentum 0.90, matrix WD 1e-4 |
| AdamW | 1e-3 | 1e-5 | 1 epoch, step-level | betas (0.90, 0.999), matrix WD 1e-2 |
| Muon matrices | 0.02 | 0.002 | 2 epochs, step-level | momentum 0.95, Nesterov, 5 NS steps, WD 0.01 |
| Auxiliary AdamW | 3e-4 | 3e-5 | 2 epochs, step-level | betas (0.90, 0.95), matrix WD 0.01 |

## CIFAR-10 / small ViT

### Corrections made

- The fixed 45,000/5,000 optimization/validation split and protected official
  test set remain enforced.
- The DeiT-style 300-epoch augmentation and regularization stack remains the
  reference recipe.
- Full restart checkpoints now include accelerator RNG state through the
  hardened runtime wrapper.
- Randomized WeightWatcher analysis is isolated from the training RNG stream.
- Completed compatible jobs are skipped and explicit final versus
  validation-selected test summaries are written.
- The original notebook incorrectly pooled six transformer blocks with three
  seeds when computing some layer uncertainty bands. Blocks are repeated
  measurements, not independent experiments. The corrected analysis gives each
  physical matrix its own curve and computes every interval from exactly the
  three complete runs.
- Regression tests now reject pseudo-replication and test-based checkpoint
  selection.

### Accepted reference

| Optimizer | Peak LR | Floor | Warm-up |
|---|---:|---:|---:|
| SGD + Nesterov | 0.10 | 0.001 | 5 epochs |
| AdamW | 1.25e-4 | 1e-5 | 5 epochs |
| Muon matrices | 0.02 | 0.002 | 5 epochs |
| Auxiliary AdamW | 3e-4 | 3e-5 | 5 epochs |

The ViT schedule remains epoch-based because that is the audited DeiT-style
scheduler convention. The notebook reports train, validation, test, LR, alpha,
ERG gap, and trap trajectories with run-level intervals.

## One-head nanoGPT / FineWeb-Edu

### Corrections made

- Cached FineWeb-Edu token files are no longer trusted from metadata alone.
  Before reuse, exact byte counts and SHA-256 hashes are verified for train,
  validation, and test files.
- Dataset name, configuration, pinned revision, tokenizer, dtype, split sizes,
  and document-disjoint status are all checked.
- MPS RNG state is now included in restart checkpoints when the installed
  PyTorch exposes `torch.mps.get_rng_state` and `set_rng_state`.
- The existing step-level warm-up/cosine schedules, validation-selected best
  checkpoint, protected test policy, BLEU probe, and six-matrix WeightWatcher
  analysis remain accepted.
- The comparison notebook keeps matrix identity fixed and computes uncertainty
  across the three seeds, not across matrices.

The GPT-2 embedding remains large relative to a one-block model. Parameter-scale
claims must separately report or exclude embedding parameters, but the matched
optimizer comparison remains valid because every arm shares that embedding and
Muon acts only on the six hidden matrices.

## nanochat

### Canonical d12 profile

The pinned d12 profile remains the canonical CUDA/server reference. It preserves
upstream depth 12, width 768, context 2048, initialization, data/tokenizer
pipeline, Muon/AdamW groups, scaling rules, 40-step warm-up, long warmdown,
momentum schedule, and cautious weight-decay schedule.

### Apple-MPS profile

A separate `mac_d4` profile was added because silently attempting d12 with eight
CUDA processes on a MacBook was not an executable local baseline. The MPS
profile uses the same pinned upstream implementation and optimizer logic at
explicitly reduced depth, context, batch, shard, and tokenizer-preparation
sizes. It is never labeled as a d12 result.

The notebook selects d12 automatically on CUDA and mac_d4 on MPS/CPU, with an
environment override. It now:

- creates the correct CUDA, Linux-CPU, or native macOS uv environment;
- forces one process on MPS/CPU;
- discovers the latest checkpoint containing model, metadata, and every
  optimizer shard;
- resumes with upstream `--resume-from-step` and appends to the log;
- fingerprints profile, commit, seed, device, and process count;
- keeps d12 and mac_d4 caches/results separate;
- analyzes only the six principal hidden matrices in each block;
- requires direct WeightWatcher `alpha`, `ERG_gap`, and `num_traps`; and
- plots training/validation/native evaluation and layerwise spectral metrics
  with complete-run Student-t intervals.

Changing the upstream commit or profile creates a new baseline version.

## Automated validation

The repository now contains a dedicated executable workflow:

```text
.github/workflows/baseline-tests.yml
```

It runs the core baseline test suite and the isolated one-head nanoGPT suite on
CPU. The tests cover:

- Python and notebook syntax;
- finite SGD, AdamW, and Muon updates;
- exact Muon/auxiliary-AdamW parameter assignments;
- step-level warm-up/cosine schedules;
- ViT regularization and schedule invariants;
- ViT checkpoint state restoration;
- rejection of layer pseudo-replication;
- validation-only checkpoint selection;
- one-head corpus hash validation and corruption detection;
- one-head checkpoint restoration;
- nanochat profile/command construction, process policy, resume-shard
  completeness, seed patch, and resumed-log deduplication.

The full long-horizon three-seed campaigns are not part of CI and were not
fabricated during this review. Their first real execution on the target
hardware remains the final empirical validation step.
