# MLP3/MNIST optimizer baselines

This folder contains **unmodified optimizer baselines** for the RG-optimizer
experiments. No trace-log projection, self-consistent ECS correction, WW-PGD
retraction, spectral-flow subtraction, or other RG intervention is applied.

The three notebooks are:

1. `notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
2. `notebooks/MNIST_MLP3_AdamW_Baseline.ipynb`
3. `notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`

All runs use the same architecture and preprocessing:

```text
784 -> 512 -> 512 -> 10
ReLU after fc1 and fc2
MNIST normalized by mean 0.1307 and std 0.3081
```

## Independent replicates and error bars

Each notebook now runs three independent complete training trajectories:

```python
SEEDS = (1337, 2027, 31415)
```

The unit of replication is a full model-training run. It is **not** a minibatch,
test example, layer, or WeightWatcher fit point.

Every aggregate curve reports the mean and the two-sided 95% Student-t
confidence interval,

$$
\bar{x}
\pm
t_{0.975,n-1}\frac{s}{\sqrt{n}},
\qquad n=3.
$$

The plots show faint individual-seed trajectories behind the mean, a shaded
confidence band, and capped error bars. Summary CSV files record `n`, sample
standard deviation, standard error, Student-t critical value, interval
half-width, lower/upper bounds, minimum, and maximum.

The color convention is fixed across all three notebooks:

```text
train  blue
test   vermillion
FC1    blue
FC2    orange
FC3    green
```

This prevents colors from changing meaning between optimizers or figures.

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
update and replaces that matrix update by its approximate polar factor using
five quintic Newton--Schulz iterations. The final classifier matrix
`fc3.weight` and all biases use ordinary SGD + momentum. This is intentionally
an **SGD-auxiliary** Muon baseline rather than `MuonWithAuxAdam`.

The Muon notebook prints and asserts the parameter-to-optimizer assignment
before training.

## Every metric is measured per epoch

Epoch zero is evaluated as well; it contains complete train and test metrics.
For every seed and every epoch, the framework records:

```text
train_loss
test_loss
train_accuracy
test_accuracy
online_train_loss
online_train_accuracy
mean_gradient_norm_before_clip
median_gradient_norm_before_clip
max_gradient_norm_before_clip
parameter_l2_norm
global_step
train_time_sec
evaluation_time_sec
weightwatcher_time_sec
epoch_total_time_sec
```

The train and test loss/accuracy values are full-dataset evaluations unless
`train_eval_max_batches` is explicitly changed.

For every layer and epoch, `spectral_metrics_by_epoch_layer_and_seed.csv`
contains:

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
- geometric mean of midpoint rescaled eigenvalues;
- boundary overlap ratio;
- Frobenius norm, spectral norm, stable rank, participation-ratio rank, and
  entropy effective rank;
- top-one, PL, detX, and midpoint energy fractions;
- ESD condition number and normalization audits.

`alpha`, `detX_num`, `num_pl_spikes`, and `ERG_gap` come directly from
`watcher.analyze(ERG=True)`. The code rejects a missing boundary, inconsistent
gap, incomplete epoch/layer, or inconsistent midpoint rather than substituting
a silent fallback.

## Required aggregate plots

Every notebook creates and saves, with individual seed traces and 95% confidence
intervals:

0. full train/test loss and full train/test accuracy;
0b. a dedicated test-accuracy figure;
1. layerwise WeightWatcher `alpha`;
2. original WeightWatcher `detX_num`, `num_pl_spikes`, and full-`M` `ERG_gap`;
3. original midpoint retained rank, midpoint trace-log per eigenvalue, and
   midpoint trace-log total;
4. effective-rank and retained-energy diagnostics;
5. gradient, parameter-norm, and timing diagnostics;
6. spectral scale, midpoint geometric mean, and ESD conditioning.

## Output layout

Each notebook writes to `baseline/runs/<optimizer>/`:

```text
performance_by_epoch_and_seed.csv
spectral_metrics_by_epoch_layer_and_seed.csv
weightwatcher_details_by_epoch_and_seed.csv
optimizer_groups_by_epoch_and_seed.csv
combined_metrics_by_epoch_layer_and_seed.csv
performance_summary_95ci.csv
spectral_summary_95ci.csv
replicate_manifest.json
plots/
seeds/
  seed_1337/
  seed_2027/
  seed_31415/
```

Each seed folder retains its own raw per-epoch CSVs, `esd_history.npz`,
`config.json`, and `final_state.pt`.

## Run

From the repository root, open any notebook in `baseline/notebooks/`.

## Tests

```bash
cd baseline
PYTHONPATH=. python -m unittest discover -s tests -v
```
