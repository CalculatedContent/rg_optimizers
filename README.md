# rg_optimizers

Research implementations of optimizer extensions motivated by the WeightWatcher
spectral renormalization-group program. The repository keeps the unmodified
reference baselines separate from every RG intervention so optimizer claims can
be tested against strong, restartable, statistically controlled experiments.

## Baseline status

The baseline suite has completed a recipe audit, an executable audit, and a
second technical qualification pass.

| Status | Meaning |
|---|---|
| **Implementation-qualified** | Source, notebooks, checkpoint recovery, RNG isolation, validation selection, WeightWatcher integration, and bounded CPU preflights are covered by automated tests. |
| **Source-backed center settings committed** | Every optimizer has a defensible architecture-specific learning rate, warm-up, decay, weight decay, initialization, and parameter partition. |
| **Final empirical qualification still required** | A committed center is promoted to the frozen “best baseline” only after the bounded validation-only search in [`baseline/FINAL_BASELINE_QUALIFICATION.md`](baseline/FINAL_BASELINE_QUALIFICATION.md). |
| **Protected tests remain protected** | Test metrics are monitoring-only and never select hyperparameters, schedules, stopping points, or best checkpoints. |

The full audit trail is in:

- [`baseline/BASELINE_RECIPE_AUDIT.md`](baseline/BASELINE_RECIPE_AUDIT.md)
- [`baseline/BASELINE_EXECUTION_REVIEW.md`](baseline/BASELINE_EXECUTION_REVIEW.md)
- [`baseline/FINAL_TECHNICAL_AUDIT.md`](baseline/FINAL_TECHNICAL_AUDIT.md)
- [`baseline/FINAL_BASELINE_QUALIFICATION.md`](baseline/FINAL_BASELINE_QUALIFICATION.md)

## Reference baseline suite

| Baseline | Data | Model | Optimizer controls | Main entry point |
|---|---|---|---|---|
| **MNIST / MLP3** | Fixed 55k optimization / 5k validation split; official 10k test monitoring-only | `784 → 512 → 512 → 10`, ReLU | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`baseline/notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`](baseline/notebooks/MNIST_MLP3_Baseline_Comparison.ipynb) |
| **CIFAR-10 / small ViT** | Fixed 45k optimization / 5k validation split; official 10k test monitoring-only | 4×4 patches, width 192, 6 blocks, 3 heads | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`baseline/notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb`](baseline/notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb) |
| **One-head nanoGPT / FineWeb-Edu** | Pinned FineWeb-Edu `sample-10BT`; exact document-disjoint 80M / 1M / 1M GPT-2-BPE splits | 1 block, 1 head, width 128, context 256 | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`baseline/nanogpt_one_head/README.md`](baseline/nanogpt_one_head/README.md) |
| **nanochat d12** | Native pinned nanochat corpus and tokenizer pipeline | 12 layers, width 768, context 2048 | Native nanochat Muon + AdamW recipe | [`baseline/notebooks/NanoChat_D12_Reference_Baseline.ipynb`](baseline/notebooks/NanoChat_D12_Reference_Baseline.ipynb) |
| **nanochat mac_d4** | Separately cached reduced nanochat preparation | 4 layers, width 256, context 512 | Same pinned upstream Muon + AdamW mathematics | Same notebook; selected automatically on MPS/CPU |

`nanochat d12` and `nanochat mac_d4` are different baseline versions. A Mac
`mac_d4` result must never be reported as a d12 result.

## Quick start

From a fresh clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './baseline[experiment]'
jupyter lab baseline/notebooks
```

Long-running outputs should live under `$HOME`, not `/tmp`:

```bash
export RG_BASELINE_DATA_DIR="$HOME/rg-optimizer-data"
export RG_BASELINE_RUN_ROOT="$HOME/rg-optimizer-runs"
```

### Recommended notebook order

MNIST:

```text
baseline/notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb
baseline/notebooks/MNIST_MLP3_AdamW_Baseline.ipynb
baseline/notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb
baseline/notebooks/MNIST_MLP3_Baseline_Comparison.ipynb
```

CIFAR-10 ViT:

```text
baseline/notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb
```

One-head nanoGPT:

```bash
cd baseline/nanogpt_one_head
bash scripts/setup_mac.sh
bash scripts/prepare_data.sh
bash scripts/smoke_test.sh

export RG_NANOGPT_ONE_HEAD_ROOT="$HOME/rg-nanogpt-one-head"
caffeinate -dimsu bash scripts/run_all_baselines.sh \
  2>&1 | tee "$RG_NANOGPT_ONE_HEAD_ROOT/run_all.log"
```

nanochat:

```text
baseline/notebooks/NanoChat_D12_Reference_Baseline.ipynb
```

The nanochat notebook runs a real pinned model/optimizer preflight before the
large data preparation or training campaign.

## Final reference recipes

### MNIST / MLP3

The two hidden ReLU layers use fan-in Kaiming-uniform initialization, the
classifier uses Xavier-uniform initialization, and all biases start at zero.
Warm-up and cosine decay are applied before every optimizer update.

| Optimizer | Peak LR | LR floor | Warm-up | Other settings |
|---|---:|---:|---:|---|
| SGD + Nesterov | 0.05 | 5e-4 | 2 epochs | momentum 0.90, matrix WD 1e-4 |
| AdamW | 1e-3 | 1e-5 | 1 epoch | betas (0.90, 0.999), matrix WD 1e-2 |
| Muon matrices | 0.02 | 0.002 | 2 epochs | momentum 0.95, Nesterov, 5 Newton–Schulz steps, WD 0.01 |
| Auxiliary AdamW | 3e-4 | 3e-5 | 2 epochs | betas (0.90, 0.95), matrix WD 0.01 |

Muon acts on `fc1.weight` and `fc2.weight`; the classifier and biases use
auxiliary AdamW. The historical result-directory key `sgd_momentum_muon` is
retained only for compatibility.

### CIFAR-10 / small ViT

The final public runtime is `rg_baselines.vit_final`. The reference uses 300
epochs, LayerNorm epsilon `1e-6`, fan-in patch-projection initialization,
stochastic depth 0.10, RandAugment, color jitter, random erasing, mixup, CutMix,
label smoothing, and gradient clipping.

| Optimizer | Warm-up start | Peak LR | LR floor | Warm-up |
|---|---:|---:|---:|---:|
| SGD + Nesterov | 1e-3 | 0.10 | 0.001 | 5 epochs |
| AdamW | 1e-6 | 1.25e-4 | 1e-5 | 5 epochs |
| Muon matrices | 2e-4 | 0.02 | 0.002 | 5 epochs |
| Auxiliary AdamW | 3e-6 | 3e-4 | 3e-5 | 5 epochs |

The schedule is explicit warm-up → cosine decay → ten-epoch cooldown at the
nonzero LR floor. Validation loss selects `checkpoint_best.pt`.

### One-head nanoGPT / FineWeb-Edu

Protocol v3 uses an 80M-token training corpus, 1M-token validation and test
corpora, 9,766 optimizer steps, and eight evenly spaced reporting and
WeightWatcher checkpoints. All optimizers and seeds use the same 64 fixed
validation batches, 64 fixed test batches, and 64 held-out BLEU continuations.

| Optimizer | Peak LR | LR floor | Warm-up |
|---|---:|---:|---:|
| SGD + Nesterov | 0.05 | 0.005 | 10% of updates |
| AdamW | 6e-4 | 6e-5 | 1% of updates |
| Muon matrices | 0.02 | 0.002 | 5% of updates |
| Auxiliary AdamW | 3e-4 | 3e-5 | 5% of updates |

The prepared corpus is accepted only when dataset identity, pinned revision,
exact token counts, byte counts, and SHA-256 hashes match the configuration.
The LR logged at a checkpoint is the LR used by the update that produced it.

### nanochat

The canonical d12 baseline is pinned to upstream commit:

```text
92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
```

The d12 run keeps upstream initialization, model-size-derived token horizon and
batch size, Muon/AdamW parameter groups, separate learning rates, 40-step
warm-up, long warmdown, momentum schedule, and weight-decay schedule. CUDA keeps
model and fused-optimizer compilation. MPS/CPU uses identical optimizer
mathematics in eager mode and enables fallback for isolated unsupported MPS
operations. Runtime policy, device, process count, profile, and compile policy
are fingerprinted to prevent incompatible result reuse.

## Measurements and statistical contract

Every applicable baseline records:

- train, validation, and monitoring-only test loss and accuracy;
- perplexity and fixed-continuation BLEU for language models;
- optimizer learning-rate trajectories and update diagnostics;
- per-layer or per-matrix WeightWatcher metrics;
- latest, validation-best, final, and periodic restart checkpoints;
- three independent complete runs with two-sided 95% Student-t intervals.

The unit of replication is a complete training run. Layers, blocks, matrices,
checkpoints, minibatches, and fit points are repeated measurements—not extra
replicates.

Strict spectral analysis uses the pinned dependency `weightwatcher==0.7.7` and
requires direct finite output from:

```python
watcher.analyze(
    ERG=True,
    randomize=True,
    ...
)
```

Required outputs include `alpha`, `ERG_gap`, and `num_traps`. No fallback alpha,
proxy trap count, or fabricated ERG gap is accepted.

## Calling a baseline “best”

The committed values above are strong source-backed centers. They become frozen
best baselines only after the bounded qualification protocol:

1. screen the preregistered neighborhood using validation data only;
2. run the finalists with the complete three-seed protocol;
3. select the lowest mean best-validation loss;
4. write the full winning configuration and evidence to a lock file;
5. inspect protected-test comparisons only after the lock exists.

The qualification implementation is
[`baseline/rg_baselines/qualification.py`](baseline/rg_baselines/qualification.py).
A lock records:

```text
protected_test_used_for_selection: false
```

## Automated validation

```bash
cd baseline
PYTHONPATH=. python -m unittest discover -s tests -v

cd nanogpt_one_head
PYTHONPATH=src pytest -q
```

GitHub Actions also compiles all baseline Python sources, parses every notebook
code cell, exercises optimizer update paths and checkpoint recovery, runs a real
WeightWatcher integration test, and executes a pinned nanochat CPU
model/optimizer preflight. These bounded tests do not replace the full
long-horizon target-hardware campaigns.

## Optimizer variants

- [`optimizers/trace_log_tracker`](optimizers/trace_log_tracker): removes or
  tracks the trace-log-normal component of a completed optimizer matrix step.
- [`optimizers/adaptive_spectral_guard`](optimizers/adaptive_spectral_guard):
  adds cadence, hysteresis, confidence gating, trace-log volume and shape
  channels, and a first-order task-loss safeguard.
- [`optimizers/self_consistent_trace_log_tracker`](optimizers/self_consistent_trace_log_tracker):
  recomputes the ECS with bulk-effective self-consistent normalization.
- [`optimizers/spectral_rg_flow_projector`](optimizers/spectral_rg_flow_projector):
  subtracts only the optimizer displacement aligned with a local spectral
  collapse direction.
- [`optimizers/ecs_probe_loss_trace_wall`](optimizers/ecs_probe_loss_trace_wall):
  adds a line-searched, task-directed ECS probe-loss component.

Each optimizer implementation lives in its own folder so it can be evaluated
against the same frozen reference suite.
