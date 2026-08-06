# Local-delta ECS WW-PGD notebooks

- `MNIST_MLP3_AdamW_LocalDeltaECS_5Runs.ipynb`: five AdamW baseline runs and five AdamW+local-delta ECS runs.
- `MNIST_MLP3_SGD_Momentum_LocalDeltaECS_5Runs.ipynb`: five SGD+momentum baseline runs and five SGD+momentum+local-delta ECS runs.

Both notebooks use the standard MLP3-MNIST architecture `784 -> 512 -> 512 -> 10`, run ten epochs by default, apply the local-delta update correction at the end of every epoch, and plot training/test accuracy, training/test cross-entropy loss, WeightWatcher alpha/ERG diagnostics when available, local ECS diagnostics, and correction-size metrics.
