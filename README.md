# rg_optimizers

Experimental optimizer extensions motivated by the WeightWatcher spectral RG
program.

## Reproducible baselines

[`baseline/`](baseline) contains matched MLP3/MNIST baselines for:

- SGD with momentum;
- AdamW;
- SGD with momentum plus Muon.

Each notebook runs three independent seeds and measures full train/test loss
and accuracy plus original WeightWatcher full-`M` diagnostics at epoch zero and
every training epoch. Plots use a fixed color-blind-safe palette and two-sided
95% Student-t confidence intervals across complete training runs.

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
