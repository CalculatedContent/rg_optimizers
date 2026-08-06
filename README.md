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

- [`optimizers/self_consistent_trace_log_tracker`](optimizers/self_consistent_trace_log_tracker):
  a new version of the original one-sided trace-log branch protector. It gets
  the ESD, alpha, and PL boundary from WeightWatcher, but recomputes the ECS
  with the bulk-effective self-consistent normalization instead of using the
  full-`M` `detX_num`. The optimizer then removes contracting flow along the
  resulting adaptive trace-log normal.

- [`optimizers/wwpgd_local_delta`](optimizers/wwpgd_local_delta): an update-space
  WW-PGD variant. At epoch boundaries it forms the completed AdamW or
  SGD+momentum epoch displacement, computes the bulk-effective self-consistent
  ECS of the proposed endpoint, and fractionally damps only the component of
  that displacement outside the retained ECS. It uses the same tall-matrix
  orientation as TraceLogRG, so the mapped-back projection acts on the right
  for tall/square layers and on the left for originally wide layers. This
  modifies the realized update, not the full weight-matrix spectrum.

Each optimizer is kept in its own folder so implementations, notebooks, and
tests can evolve independently.
