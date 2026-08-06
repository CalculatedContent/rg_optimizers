# Local-delta ECS WW-PGD notebooks

- `MNIST_MLP3_AdamW_LocalDeltaECS_5Runs.ipynb`: five paired AdamW baseline runs and five AdamW plus local-delta ECS runs.
- `MNIST_MLP3_SGD_Momentum_LocalDeltaECS_5Runs.ipynb`: five paired SGD-with-momentum baseline runs and five SGD-with-momentum plus local-delta ECS runs.

Both notebooks use the standard MLP3-MNIST architecture
`784 -> 512 -> 512 -> 10`, run ten epochs, and apply the fractional local-delta
correction at the end of every epoch. The default reference is the epoch-end
matrix, so the completed optimizer displacement is decomposed using the new ECS
of the proposed endpoint.

The notebooks fail fast unless WeightWatcher runs successfully. They validate
paired initialization, five seeds per arm, the exact fractional damping identity,
orientation-aware left/right projection, and the requested WeightWatcher output
before plotting task, spectral, and correction diagnostics.
