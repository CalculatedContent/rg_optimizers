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
3. Form the completed epoch displacement `Delta = W_end - W_start`.
4. Orient the proposed endpoint as `N x M` with `N >= M`, exactly as in the
   TraceLogRG geometry, and compute the bulk-effective self-consistent ECS of
   the proposed endpoint `W_end`.
5. Decompose the completed displacement and fractionally damp only the part
   outside the ECS:

\[
\Delta_{\mathrm{new}}
=
\Delta-\eta\Delta_{\perp}
=
\Delta_{\mathrm{ECS}}+(1-\eta)\Delta_{\perp},
\qquad 0\leq\eta\leq1.
\]

For an originally tall or square layer, the ECS acts on the right,

\[
\Delta_{\mathrm{ECS}}=\Delta P_R,
\]

while an originally wide layer is transposed into the common tall convention,
so mapping back gives a left action,

\[
\Delta_{\mathrm{ECS}}=P_R\Delta.
\]

This orientation rule is essential for FC1 in the standard MLP3 experiment,
whose PyTorch weight has shape `512 x 784`.

For `eta=0`, the extension is exactly the base optimizer. For `0<eta<1`, it is
a soft local-delta correction. For `eta=1`, it becomes a hard projection of the
completed epoch displacement into the current ECS. The notebooks use
`eta=0.25` and the **epoch-end ECS** by default.

## Lifecycle and state behavior

The wrapper now enforces an explicit epoch lifecycle:

```python
optimizer.begin_epoch()
# ordinary minibatch optimizer steps
optimizer.apply_epoch_delta_correction(epoch=epoch)
```

Calling `begin_epoch()` twice, or applying a correction without an active epoch
snapshot, raises by default instead of silently changing the experiment. The
wrapper `state_dict()` includes the active epoch-start snapshot and cached ECS
ranks, so a checkpoint saved in the middle of an epoch can be restored and
corrected correctly.

Parameter filters accept both module names and parameter names. For example,
`parameter_name_filter=("fc1",)` resolves to `fc1.weight`. Unknown or ambiguous
filters fail fast.

By default the correction changes only the realized weight displacement. Set
`synchronize_optimizer_state=True` for a separate state-consistent ablation:

- SGD: apply the same fractional ECS damping to `momentum_buffer`;
- AdamW: apply it to `exp_avg`;
- AdamW `exp_avg_sq` remains unchanged because it is an elementwise second
  moment, not a matrix displacement.

The previously selected ECS rank is used only as a continuity tie-breaker when
multiple finite trace-log crossings are equally good. The new ECS is still
computed from the current endpoint every epoch.

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

Both notebooks run the standard `784 -> 512 -> 512 -> 10` MLP3-MNIST
experiment for ten epochs. Each uses five paired baseline seeds and five paired
local-delta seeds, with identical initial weights and minibatch order within each
pair.

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
