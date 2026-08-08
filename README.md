# rg_optimizers

Experimental optimizer extensions motivated by the WeightWatcher spectral RG
program.

## Reproducible baselines

[`baseline/`](baseline) contains the unmodified reference experiments used to
evaluate RG optimizer variants. Each reference has been reviewed at two levels:

- [`baseline/BASELINE_RECIPE_AUDIT.md`](baseline/BASELINE_RECIPE_AUDIT.md):
  data, model, initialization, optimizer, and schedule choices;
- [`baseline/BASELINE_EXECUTION_REVIEW.md`](baseline/BASELINE_EXECUTION_REVIEW.md):
  notebook execution, restart state, RNG isolation, checkpoint selection,
  uncertainty, and automated tests.

| Baseline | Model / data | Reference optimizers | Primary purpose |
| --- | --- | --- | --- |
| **MLP3 / MNIST** | `784 -> 512 -> 512 -> 10` MLP; fixed 55k/5k optimization/validation split; official test monitoring-only | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | Cheap, tightly controlled optimizer and spectral debugging |
| **Small ViT / CIFAR-10** | 6-block, width-192 ViT with 4x4 patches; fixed 45k/5k optimization/validation split; official test monitoring-only | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | Transformer optimization on vision data with strong DeiT-style regularization |
| **One-head nanoGPT / FineWeb-Edu** | 1 block, 1 head, width 128, context 256; pinned document-disjoint FineWeb-Edu corpus | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | Smallest realistic language-model optimizer control with MPS restart support |
| **nanochat d12** | 12 layers, width 768, context 2048; native pinned nanochat data/tokenizer pipeline | Native nanochat Muon + AdamW recipe | Canonical modern small-LLM reference for CUDA/server hardware |
| **nanochat mac_d4** | 4 layers, width 256, context 512; separately cached reduced upstream preparation | Same pinned upstream Muon + AdamW logic | Explicit Apple-MPS development reference; never reported as d12 |

The unit of replication is always a complete training run. Validation data
select hyperparameters and best checkpoints. Protected test measurements never
change optimization, schedules, stopping, or checkpoint selection. Three-seed
intervals are two-sided 95% Student-t intervals across runs; layers, matrices,
checkpoints, and fit points are not counted as extra replicates.

Strict spectral measurements preserve direct output from:

```python
watcher.analyze(ERG=True, randomize=True, ...)
```

No fallback alpha, proxy correlation-trap count, or fabricated ERG gap is
accepted.

### MLP3 / MNIST

The three optimizer notebooks use the identical deterministic 55k/5k split,
three seeds, 30-epoch budget, gradient clipping, step-level linear warm-up and
cosine decay, validation-selected best checkpoints, and epoch-zero/per-epoch
WeightWatcher diagnostics. Latest, best, final, and per-epoch checkpoints save
model, optimizer, data-loader generator, Python/NumPy/Torch/CUDA/MPS RNG state,
and a protocol fingerprint. Randomized diagnostics preserve the training RNG
stream.

Notebooks:

- `baseline/notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
- `baseline/notebooks/MNIST_MLP3_AdamW_Baseline.ipynb`
- `baseline/notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`
- `baseline/notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`

The historical result key `sgd_momentum_muon` is retained for compatibility;
the implementation is **Muon + auxiliary AdamW**.

### Small ViT / CIFAR-10

`baseline/notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb` trains the same small
Vision Transformer with SGD + Nesterov, AdamW, and Muon + auxiliary AdamW. The
reference uses 300 epochs, stochastic depth, RandAugment, color jitter, random
erasing, mixup, CutMix, label smoothing, gradient clipping, optimizer-specific
warm-up/cosine schedules, validation-selected checkpoints, and restartable
state.

The analysis gives every physical transformer matrix its own trajectory and
computes each uncertainty interval from exactly the three complete runs.
Transformer blocks are repeated measurements, not additional replicates.

### One-head nanoGPT / FineWeb-Edu

[`baseline/nanogpt_one_head/`](baseline/nanogpt_one_head) trains a one-block,
one-head nanoGPT on a pinned FineWeb-Edu `sample-10BT` stream rather than Tiny
Shakespeare. Exact document-disjoint 10M/1M/1M-token train/validation/test
splits are shared across all optimizers. Dataset identity, revision, byte count,
and SHA-256 are verified before cached data are reused.

The suite records train/validation/test next-token loss, accuracy, perplexity,
held-out continuation BLEU, learning-rate trajectories, and direct per-matrix
`alpha`, `ERG_gap`, and `num_traps`. Full restart checkpoints include MPS RNG
state where supported.

MacBook quick start:

```bash
cd baseline/nanogpt_one_head
bash scripts/setup_mac.sh
bash scripts/prepare_data.sh
bash scripts/smoke_test.sh

export RG_NANOGPT_ONE_HEAD_ROOT="$HOME/rg-nanogpt-one-head"
caffeinate -dimsu bash scripts/run_all_baselines.sh \
  2>&1 | tee "$RG_NANOGPT_ONE_HEAD_ROOT/run_all.log"
```

Then open `baseline/nanogpt_one_head/notebooks/04_compare_baselines.ipynb`.

### nanochat reference profiles

`baseline/notebooks/NanoChat_D12_Reference_Baseline.ipynb` pins upstream
nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` and preserves its native
initialization, Muon/AdamW groups, separate learning rates, scaling rules,
40-step warm-up, long warmdown, momentum schedule, and weight-decay schedule.

`RG_NANOCHAT_PROFILE=auto` selects canonical d12 on CUDA and separately labeled
mac_d4 on MPS/CPU. The wrapper creates the platform-correct environment, forces
one process on MPS/CPU, resumes only from checkpoints containing model,
metadata, and every optimizer shard, appends/deduplicates logs, fingerprints the
profile and process count, keeps profile caches separate, and analyzes only the
six principal hidden matrices in each block.

## Automated validation

`.github/workflows/baseline-tests.yml` runs executable core-baseline and
one-head nanoGPT tests on CPU. The existing source workflow compiles Python and
parses notebook code cells. These bounded tests cover optimizer updates,
schedules, parameter partitions, data integrity, restart state, validation-only
selection, and statistical invariants. They do not replace the full
long-horizon target-hardware campaigns.

## Optimizer variants

**One-page map (dual-label + dose notes):** [`OPTIMIZER_VARIANTS.md`](OPTIMIZER_VARIANTS.md)

- [`optimizers/trace_log_tracker`](optimizers/trace_log_tracker): the first
  implementation. It removes or tracks the trace-log-normal component of a
  completed AdamW/SGD matrix step using a WeightWatcher-selected midpoint ECS.

- [`optimizers/adaptive_spectral_guard`](optimizers/adaptive_spectral_guard):
  the second implementation. It adds layer-specific cadence and caps,
  WeightWatcher-driven hysteresis, ECS-confidence gating, a trace-log volume
  channel, a trace-log-preserving shell-beta shape channel, and a first-order
  task-loss safeguard. It includes 30-epoch MNIST experiments, matched-
  convergence plots, and FC1-only/FC2-only ablation presets.

- [`optimizers/self_consistent_trace_log_tracker`](optimizers/self_consistent_trace_log_tracker):
  a version of the original one-sided trace-log branch protector. It gets the
  ESD, alpha, and PL boundary from WeightWatcher, but recomputes the ECS with the
  bulk-effective self-consistent normalization instead of using the full-`M`
  `detX_num`.

- [`optimizers/spectral_rg_flow_projector`](optimizers/spectral_rg_flow_projector):
  an experiment that acts in centered log-spectrum shape space rather than
  along the trace-log normal. It estimates a local participation-ratio collapse
  vector and subtracts only the completed optimizer displacement aligned with
  that vector.

- [`optimizers/ecs_probe_loss_trace_wall`](optimizers/ecs_probe_loss_trace_wall):
  a task-directed TraceWall variant. It recomputes the self-consistent ECS,
  truncates selected matrices, measures cross-entropy on a rotating training
  subset, projects the probe gradient into the ECS, and adds a line-searched
  loss-decreasing component. The official test set is evaluation-only.

Each optimizer is kept in its own folder so implementations, notebooks, and
tests can evolve independently.
