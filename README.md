# rg_optimizers

Experimental optimizer extensions motivated by the WeightWatcher spectral RG
program.

## Reproducible baselines

[`baseline/`](baseline) contains the unmodified reference experiments used to
evaluate RG optimizer variants. Each reference has been reviewed at four
levels:

- [`baseline/BASELINE_RECIPE_AUDIT.md`](baseline/BASELINE_RECIPE_AUDIT.md):
  data, model, initialization, optimizer, and schedule choices;
- [`baseline/BASELINE_EXECUTION_REVIEW.md`](baseline/BASELINE_EXECUTION_REVIEW.md):
  notebook execution, restart state, RNG isolation, checkpoint selection,
  uncertainty, and automated tests;
- [`baseline/FINAL_TECHNICAL_AUDIT.md`](baseline/FINAL_TECHNICAL_AUDIT.md):
  remaining defects found in the second pass and their corrections;
- [`baseline/FINAL_BASELINE_QUALIFICATION.md`](baseline/FINAL_BASELINE_QUALIFICATION.md):
  bounded validation-only hyperparameter search and configuration freezing.

| Baseline | Model / data | Reference optimizers | Primary purpose |
| --- | --- | --- | --- |
| **MLP3 / MNIST** | `784 -> 512 -> 512 -> 10` ReLU MLP; fixed 55k/5k optimization/validation split; official test monitoring-only | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | Cheap, tightly controlled optimizer and spectral debugging |
| **Small ViT / CIFAR-10** | 6-block, width-192 ViT with 4x4 patches; fixed 45k/5k optimization/validation split; official test monitoring-only | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | Transformer optimization on vision data with strong DeiT-style regularization |
| **One-head nanoGPT / FineWeb-Edu** | 1 block, 1 head, width 128, context 256; pinned document-disjoint 80M/1M/1M FineWeb-Edu corpus | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | Smallest realistic language-model optimizer control with MPS restart support |
| **nanochat d12** | 12 layers, width 768, context 2048; native pinned nanochat data/tokenizer pipeline | Native nanochat Muon + AdamW recipe | Canonical modern small-LLM reference for CUDA/server hardware |
| **nanochat mac_d4** | 4 layers, width 256, context 512; separately cached reduced upstream preparation | Same pinned upstream Muon + AdamW logic | Explicit Apple-MPS development reference; never reported as d12 |

The unit of replication is always a complete training run. Validation data
select hyperparameters and best checkpoints. Protected test measurements never
change optimization, schedules, stopping, or checkpoint selection. Three-seed
intervals are two-sided 95% Student-t intervals across runs; layers, matrices,
checkpoints, and fit points are not counted as extra replicates.

A committed hyperparameter point is the source-backed center of a bounded
candidate neighborhood. It is called the **best baseline for the selected
architecture and data** only after `rg_baselines.qualification` ranks complete
runs by validation loss and writes a lock file before protected test comparison.

Strict spectral measurements preserve direct output from:

```python
watcher.analyze(ERG=True, randomize=True, ...)
```

No fallback alpha, proxy correlation-trap count, or fabricated ERG gap is
accepted.

### MLP3 / MNIST

The three optimizer notebooks use the identical deterministic 55k/5k split,
three seeds, 30-epoch budget, gradient clipping, update-level linear warm-up and
cosine decay, validation-selected best checkpoints, and epoch-zero/per-epoch
WeightWatcher diagnostics. The hidden ReLU layers use fan-in Kaiming-uniform
initialization, the classifier uses Xavier-uniform initialization, and all
biases start at zero. Latest, best, final, and per-epoch checkpoints save model,
optimizer, data-loader generator, Python/NumPy/Torch/CUDA/MPS RNG state, and a
protocol fingerprint. Randomized diagnostics preserve the training RNG stream.

Notebooks:

- `baseline/notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
- `baseline/notebooks/MNIST_MLP3_AdamW_Baseline.ipynb`
- `baseline/notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`
- `baseline/notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`

The historical result key `sgd_momentum_muon` is retained for compatibility;
the implementation is **Muon + auxiliary AdamW**.

### Small ViT / CIFAR-10

`baseline/notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb` imports the final
reference runtime from `rg_baselines.vit_final`. In addition to the 300-epoch
augmentation and regularization stack, it uses LayerNorm epsilon `1e-6`, fan-in
patch-projection initialization, an explicit low-LR warm-up, cosine decay, and a
10-epoch cooldown at the non-zero LR floor.

The analysis gives every physical transformer matrix its own trajectory and
computes each uncertainty interval from exactly the three complete runs.
Transformer blocks are repeated measurements, not additional replicates.

### One-head nanoGPT / FineWeb-Edu

[`baseline/nanogpt_one_head/`](baseline/nanogpt_one_head) trains a one-block,
one-head nanoGPT on a pinned FineWeb-Edu `sample-10BT` stream rather than Tiny
Shakespeare. Exact document-disjoint 80M/1M/1M-token train/validation/test
splits are shared across all optimizers. Dataset identity, revision, byte count,
and SHA-256 are verified before cached data are reused.

Protocol v3 uses approximately 80M sampled training tokens, close to a
12-tokens-per-scaling-parameter budget, with eight evenly spaced reporting and
WeightWatcher checkpoints. It uses 64 common fixed validation/test batches, 64
common held-out BLEU continuations, update-level warm-up/cosine schedules, and
LR logging aligned with the update that produced each checkpoint. The suite
records train/validation/test loss, accuracy, perplexity, BLEU, and direct
per-matrix `alpha`, `ERG_gap`, and `num_traps`. Full restart checkpoints and
randomized diagnostics preserve MPS RNG state where supported.

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
mac_d4 on MPS/CPU. CUDA keeps upstream compilation for both the model and fused
optimizer kernels; MPS/CPU uses the same pinned model and optimizer mathematics
in eager mode and enables CPU fallback for isolated unsupported MPS operations.
The wrapper creates the platform-correct environment, forces one process on
MPS/CPU, resumes only from checkpoints containing model, metadata, and every
optimizer shard, appends/deduplicates logs, fingerprints
profile/process/device/compile/fallback policy, keeps profile caches separate,
and analyzes only the six principal hidden matrices in each block. A pinned
one-step model/optimizer preflight runs before a long target-device campaign.

## Automated validation

`.github/workflows/baseline-tests.yml` runs executable core-baseline and
one-head nanoGPT tests on CPU. The source workflow compiles Python and parses
notebook code cells. CI also clones the pinned nanochat commit and executes a
real model forward/backward plus native combined Muon/AdamW optimizer step.
These bounded tests cover optimizer updates, schedules, parameter partitions,
data integrity, restart state, validation-only selection, qualification locks,
platform runtime policy, and statistical invariants. They do not replace the
full long-horizon target-hardware campaigns.

## Optimizer variants

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
