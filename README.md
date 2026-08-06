# rg_optimizers

Experimental optimizer extensions motivated by the WeightWatcher spectral RG
program.

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

Each optimizer is kept in its own folder so implementations, notebooks, and
tests can evolve independently.
