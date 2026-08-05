# rg_optimizers

Experimental optimizer extensions motivated by the WeightWatcher spectral RG program.

## Optimizer variants

- [`optimizers/trace_log_tracker`](optimizers/trace_log_tracker): removes or tracks the trace-log-normal component of a completed AdamW/SGD matrix step, using a WeightWatcher-selected midpoint ECS. Includes an AdamW baseline and an MNIST MLP3 notebook that tracks WeightWatcher alpha, ERG gap, and logarithmic-shell beta.

Each optimizer is kept in its own folder so implementations, notebooks, and tests can evolve independently.
