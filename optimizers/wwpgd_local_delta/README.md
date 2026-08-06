# WWPGD Local-Delta ECS

`wwpgd_local_delta` is an update-space variant of WW-PGD for the
`rg_optimizers` repository.

The original [`CalculatedContent/WW_PGD`](https://github.com/CalculatedContent/WW_PGD)
implementation is a **state-space spectral retraction**: AdamW first updates the
full matrix, then the current weight matrix is SVD-shaped and blended back.

This implementation keeps the WW-PGD outer-loop style but changes the actuator:

1. Save the layer matrix at the start of an epoch, `W_start`.
2. Let the base optimizer, AdamW or SGD+momentum, run normally for the epoch.
3. At the epoch boundary form the completed epoch displacement
   `Delta = W_end - W_start`.
4. Compute a local ECS from a reference weight matrix using the finite
   self-consistent trace-log scan.
5. Dampen only the component of the completed displacement that lies outside
   the right-ECS support:

\[
\Delta_{new}
=
\Delta - \eta\Delta(I-P_R)
=
\Delta P_R + (1-\eta)\Delta(I-P_R),
\qquad
P_R = V_RV_R^T.
\]

For `eta=0`, the method is exactly the base optimizer. For `0<eta<1`, this is a
soft local-delta correction. For `eta=1`, it becomes a hard projection of the
completed epoch displacement into the current ECS. The notebooks use fractional
correction by default.

The method modifies the **optimizer displacement**, not the full matrix spectrum.
It is therefore closer to the trace-log optimizer family in `rg_optimizers`, but
uses a subspace damping rule rather than a trace-log normal component.

## Contents

```text
wwpgd_local_delta/
  config.py              user-facing config dataclasses
  ecs.py                 self-consistent ECS scan and projection math
  optimizer.py           AdamW / SGD-momentum compatible wrapper
  weightwatcher.py       WeightWatcher and fallback spectral diagnostics
  mnist_experiment.py    paired MLP3-MNIST experiment harness
notebooks/
  MNIST_MLP3_AdamW_LocalDeltaECS_5Runs.ipynb
  MNIST_MLP3_SGD_Momentum_LocalDeltaECS_5Runs.ipynb
tests/
```

## Minimal use

```python
import torch
from wwpgd_local_delta import LocalDeltaECSConfig, LocalDeltaECSOptimizer

base = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
opt = LocalDeltaECSOptimizer(
    base,
    model.named_parameters(),
    config=LocalDeltaECSConfig(correction_fraction=0.25),
)

for epoch in range(10):
    opt.begin_epoch()
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    opt.apply_epoch_delta_correction(epoch=epoch)
    correction_stats = opt.pop_epoch_stats()
```

## Notebooks

The notebooks run the standard MLP3-MNIST experiment:

\[
784 \rightarrow 512 \rightarrow 512 \rightarrow 10
\]

Each notebook runs five baseline seeds and five local-delta extension seeds,
then plots:

- train and test accuracy;
- train and test cross-entropy loss;
- WeightWatcher alpha per layer;
- WeightWatcher ERG gap, when available;
- local ECS rank and trace-log residual;
- orthogonal epoch-displacement fraction;
- removed correction fraction.

The notebooks default to ten epochs and epoch-boundary correction.

## Tests

```bash
cd optimizers/wwpgd_local_delta
PYTHONPATH=. python -m unittest discover -s tests -v
```
