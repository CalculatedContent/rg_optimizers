# MLP3/MNIST optimizer baselines

This folder contains **unmodified optimizer baselines** for the RG-optimizer
experiments. No trace-log projection, self-consistent ECS correction,
WW-PGD retraction, spectral-flow subtraction, or other RG intervention is
applied.

The three notebooks are:

1. `notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
2. `notebooks/MNIST_MLP3_AdamW_Baseline.ipynb`
3. `notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`

All runs use the same architecture and data preprocessing:

```text
784 -> 512 -> 512 -> 10
ReLU after fc1 and fc2
MNIST normalized by mean 0.1307 and std 0.3081
```

## Baseline definitions

### SGD + momentum

```python
torch.optim.SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    dampening=0.0,
    nesterov=False,
    weight_decay=1e-4,
)
```

### AdamW

```python
torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,
)
```

### SGD + momentum + Muon

Muon is applied to `fc1.weight` and `fc2.weight`. It first forms a momentum
update and then replaces that matrix update by its approximate polar factor
using five quintic Newton--Schulz iterations. The final classifier matrix
`fc3.weight` and all biases use ordinary SGD + momentum. This is intentionally
an **SGD-auxiliary** Muon baseline rather than `MuonWithAuxAdam`.

The notebook prints the parameter-to-optimizer assignment before training.

## Every metric is measured per epoch

Epoch zero is evaluated as well; it does not contain `NaN` train metrics.
Every notebook saves:

```text
performance_by_epoch.csv
spectral_metrics_by_epoch_and_layer.csv
weightwatcher_details_by_epoch.csv
optimizer_groups_by_epoch.csv
combined_metrics_by_epoch_and_layer.csv
esd_history.npz
config.json
final_state.pt
plots/
```

`performance_by_epoch.csv` contains full-train and full-test cross-entropy and
accuracy, online training statistics, gradient norms, whole-model parameter
norm, global step, and timing.

`spectral_metrics_by_epoch_and_layer.csv` contains, for every layer and epoch:

- WeightWatcher `alpha`;
- original WeightWatcher `detX_num`;
- original WeightWatcher `num_pl_spikes`;
- original full-`M` `ERG_gap`;
- the original midpoint retained rank

$$
m_{\mathrm{mid}}
=
\left\lfloor
\frac{m_{\mathrm{detX}}+m_{\mathrm{PL}}}{2}
\right\rfloor;
$$

- midpoint trace-log total and trace-log per retained eigenvalue;
- geometric mean of the midpoint rescaled eigenvalues;
- boundary overlap ratio;
- Frobenius norm, spectral norm, stable rank, participation-ratio rank, and
  entropy effective rank;
- top-one, PL, detX, and midpoint energy fractions;
- ESD condition number and normalization audits.

The full unmodified dataframe returned by `watcher.analyze(ERG=True)` is saved
separately in `weightwatcher_details_by_epoch.csv`.

## Required plots

Each notebook creates and saves:

0. full train/test loss and full train/test accuracy;
1. layerwise WeightWatcher `alpha`;
2. original WeightWatcher `detX_num`, `num_pl_spikes`, and full-`M` `ERG_gap`;
3. original midpoint retained rank and midpoint trace-log coordinate;
4. additional effective-rank and retained-energy diagnostics;
5. gradient, parameter-norm, and timing diagnostics.

The run fails loudly if a requested layer or epoch is missing, if `ERG_gap` is
not `detX_num - num_pl_spikes`, or if the midpoint is not the original
WeightWatcher PL/detX midpoint.

## Run

From the repository root, open any notebook in `baseline/notebooks/`. Results
are written to `baseline/runs/<optimizer>/`.

## Tests

```bash
cd baseline
PYTHONPATH=. python -m unittest discover -s tests -v
```
