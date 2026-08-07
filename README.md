# rg_optimizers

Experimental optimizer extensions motivated by the WeightWatcher spectral RG
program.

## Reproducible baselines

[`baseline/`](baseline) contains the reference experiments used to evaluate the
RG optimizer variants. The goal is to test optimizer behavior across several
architectures and modalities rather than against a single toy model.

| Baseline | Model / data | Reference optimizers | Primary purpose |
| --- | --- | --- | --- |
| **MLP3 / MNIST** | `784 -> 512 -> 512 -> 10` MLP on MNIST | SGD + momentum, AdamW, SGD + momentum + Muon | Cheap, tightly controlled optimizer and spectral debugging |
| **Small ViT / CIFAR-10** | 6-block, 192-wide Vision Transformer with 4x4 patches | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | Transformer optimization on vision data with residual/attention structure |
| **One-head nanoGPT / FineWeb-Edu** | 1 block, 1 attention head, width 128, context 256 on a pinned document-disjoint FineWeb-Edu corpus | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | Smallest realistic language-model optimizer baseline with MPS restart support and per-epoch spectral diagnostics |
| **nanochat d12** | 12-layer, 768-wide, 2048-context nanochat language model | Native nanochat Muon + AdamW recipe | Modern small-LLM reference baseline with tuned initialization, parameter groups, scaling rules, and schedules |

### MLP3 / MNIST

The MNIST suite runs three independent seeds for each optimizer and records full
train/test loss and accuracy, checkpoints, and WeightWatcher diagnostics at
epoch zero and every training epoch. The comparison notebook reports
run-level two-sided 95% Student-t confidence intervals.

Notebooks:

- `baseline/notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
- `baseline/notebooks/MNIST_MLP3_AdamW_Baseline.ipynb`
- `baseline/notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`
- `baseline/notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`

### Small ViT / CIFAR-10

`baseline/notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb` trains the same
small Vision Transformer from scratch with SGD + Nesterov, AdamW, and Muon +
auxiliary AdamW. It uses three seeds, optimizer-specific tuned hyperparameters,
warmup/cosine schedules, CIFAR-10 augmentation, checkpoint persistence, and
WeightWatcher spectral diagnostics.

### One-head nanoGPT / FineWeb-Edu

[`baseline/nanogpt_one_head/`](baseline/nanogpt_one_head) contains the smallest
realistic language-model control. It trains a one-block, one-attention-head
nanoGPT on a pinned FineWeb-Edu `sample-10BT` stream rather than Tiny
Shakespeare. Exact document-disjoint 10M/1M/1M-token train/validation/test
splits are shared across SGD + Nesterov, AdamW, and Muon + auxiliary AdamW.

The suite uses optimizer-specific warmup/cosine schedules, three independent
seeds, restartable full checkpoints, Apple-MPS execution, per-epoch
train/validation/test loss, next-token accuracy and perplexity, fixed held-out
continuation BLEU, and WeightWatcher calls with `ERG=True` and
`randomize=True`. Raw per-matrix `alpha`, `ERG_gap`, and `num_traps` are
retained without fallbacks or proxy counts. Four notebooks produce run-level
95% Student-t confidence intervals and a fixed color map for the six
transformer matrices.

### nanochat d12

`baseline/notebooks/NanoChat_D12_Reference_Baseline.ipynb` is the modern
language-model reference baseline. It pins upstream nanochat commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` and deliberately keeps nanochat's
training recipe intact rather than replacing it with a generic GPT/AdamW
configuration.

The d12 reference uses 12 transformer layers, width 768, and context length
2048. It preserves nanochat's native initialization, Muon/AdamW parameter
partitioning, separate embedding/unembedding/matrix/scalar learning rates,
depth- and batch-aware scaling rules, 40-step warmup, long linear warmdown,
Muon momentum schedule, and cautious cosine-decayed weight decay. The wrapper
adds reproducible independent seeds, periodic checkpoints and validation,
final CORE evaluation, tidy CSV logs, and offline WeightWatcher analysis so
spectral diagnostics do not perturb timed training.

The reusable runner lives at:

- `baseline/rg_baselines/nanochat_reference.py`

See [`baseline/README.md`](baseline/README.md) for the shared baseline
conventions and the experiment-specific READMEs for exact run instructions.

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
  a new version of the original one-sided trace-log branch protector. It gets
  the ESD, alpha, and PL boundary from WeightWatcher, but recomputes the ECS
  with the bulk-effective self-consistent normalization instead of using the
  full-`M` `detX_num`. The optimizer then removes contracting flow along the
  resulting adaptive trace-log normal.

- [`optimizers/spectral_rg_flow_projector`](optimizers/spectral_rg_flow_projector):
  a separate experiment that acts in centered log-spectrum shape space rather
  than along the trace-log normal. On the adaptive self-consistent ECS, it
  estimates a local participation-ratio collapse vector toward the
  no-extensive-ECS/trivial branch and subtracts only the completed optimizer
  displacement aligned with that vector. Its matched MNIST suite tests the
  same projector on AdamW, Adam, and ordinary SGD with classical momentum.

- [`optimizers/ecs_probe_loss_trace_wall`](optimizers/ecs_probe_loss_trace_wall):
  a task-directed TraceWall variant. At each correction it recomputes the
  self-consistent ECS, truncates all selected matrices to that support, measures
  cross-entropy on a rotating random subset of the training set, projects the
  probe gradient back into the ECS, and adds a line-searched loss-decreasing
  component to the completed AdamW or SGD-momentum update. Its paired notebooks
  include a clean baseline in the same run, matched warmup/cosine schedules,
  three-seed error bars, WeightWatcher diagnostics, and complete checkpoints.
  The official test set is used only for evaluation, never for optimization.

Each optimizer is kept in its own folder so implementations, notebooks, and
tests can evolve independently.
