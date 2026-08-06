# WWPGD Local-Delta ECS

`wwpgd_local_delta` is an update-space variant of WW-PGD for the
`rg_optimizers` repository.

The original [`CalculatedContent/WW_PGD`](https://github.com/CalculatedContent/WW_PGD)
implementation is a **state-space spectral retraction**: the base optimizer first
updates the full layer matrix, then the current weight spectrum is SVD-shaped and
blended back.

This implementation changes the actuator while retaining the epoch-boundary
outer-loop structure:

1. Save the selected layer matrices at the start of an epoch, `W_start`.
2. Let AdamW or SGD with momentum run normally for the entire epoch.
3. Form the completed epoch displacement
   `Delta = W_end - W_start`.
4. Orient the proposed endpoint as `N x M` with `N >= M`, exactly as in the
   TraceLogRG geometry, and compute the bulk-effective self-consistent ECS of
   the proposed endpoint `W_end`.
5. Decompose the completed displacement into retained and orthogonal pieces and
   damp only a fraction of the orthogonal piece:

\[
\Delta_{\mathrm{new}}
=
\Delta-\eta\Delta_{\perp}
=
\Delta_{\mathrm{ECS}}+(1-\eta)\Delta_{\perp},
\qquad 0\leq\eta\leq1.
\]

For an originally tall or square layer, the ECS acts on the right:

\[
\Delta_{\mathrm{ECS}}=\Delta P_R.
\]

For an originally wide layer, the layer is transposed before constructing
`X = W^T W/N`. Mapping the oriented projection back to the original matrix
therefore gives a left action:

\[
\Delta_{\mathrm{ECS}}=P_R\Delta.
\]

This orientation rule is essential for FC1 in the standard MLP3 experiment,
whose PyTorch weight has shape `512 x 784`.

For `eta=0`, the extension is exactly the base optimizer. For `0<eta<1`, it is
a soft local-delta correction. For `eta=1`, it becomes a hard projection of the
completed epoch displacement into the current ECS. The notebooks use
`eta=0.25` and the **epoch-end ECS** by default.

The implementation modifies the realized weight displacement only. AdamW and
SGD momentum state are deliberately left unchanged in this first causal
experiment; the correction logs this fact explicitly. State-consistent momentum
filtering should be treated as a separate ablation.

## Contents

```text
wwpgd_local_delta/
  config.py              user-facing configuration
  ecs.py                 oriented ECS scan and local-delta decomposition
  optimizer.py           AdamW / SGD-momentum compatible wrapper
  weightwatcher.py       required WW diagnostics plus direct-SVD audits
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
optimizer = LocalDeltaECSOptimizer(
    base,
    model.named_parameters(),
    config=LocalDeltaECSConfig(
        correction_fraction=0.25,
        reference="epoch_end",
    ),
)

for epoch in range(10):
    optimizer.begin_epoch()
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    optimizer.apply_epoch_delta_correction(epoch=epoch)
```

## Notebooks

Both notebooks run the standard
`784 -> 512 -> 512 -> 10` MLP3-MNIST experiment for ten epochs. Each uses five
paired baseline seeds and five paired local-delta seeds, with identical initial
weights and minibatch order within each pair.

They record and plot:

- train and test accuracy;
- train and test cross-entropy loss;
- immediate pre/post-correction changes in train/test metrics;
- WeightWatcher alpha, ERG gap, detX count, PL-tail count, and available norm/rank metrics;
- local ECS rank, rank fraction, effective bulk count, and trace-log residual;
- pre/post orthogonal displacement fractions;
- requested versus observed fractional damping;
- Pythagorean decomposition and correction-identity errors.

WeightWatcher is installed by the notebooks if necessary and is required for the
scientific runs. A direct-SVD fallback remains available only for unit/smoke
tests or explicit `ww_required=False` runs.

## Tests

```bash
cd optimizers/wwpgd_local_delta
PYTHONPATH=. python -m unittest discover -s tests -v
```
