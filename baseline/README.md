# MLP3/MNIST optimizer baselines

This folder contains **unmodified optimizer baselines** for the RG-optimizer
experiments. No trace-log projection, self-consistent ECS correction, WW-PGD
retraction, spectral-flow subtraction, or other RG intervention is applied.

The notebooks are:

1. `notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
2. `notebooks/MNIST_MLP3_AdamW_Baseline.ipynb`
3. `notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`
4. `notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`

Run the first three notebooks to produce the persisted optimizer results, then
run the fourth notebook to validate and compare all three experiments.

All training runs use the same architecture and preprocessing:

```text
784 -> 512 -> 512 -> 10
ReLU after fc1 and fc2
MNIST normalized by mean 0.1307 and std 0.3081
```

## Shared persistent run directory

All four notebooks resolve the same run root:

```text
RG_BASELINE_RUN_ROOT, when set
otherwise: baseline/runs/
```

For a clone at `/tmp/rg_optimizers`, the default is therefore
`/tmp/rg_optimizers/baseline/runs`. A different shared local directory can be
selected before starting Jupyter:

```bash
export RG_BASELINE_RUN_ROOT=/tmp/rg_optimizers_baseline_runs
```

The MNIST download/cache directory can likewise be overridden with
`RG_BASELINE_DATA_DIR`; its default remains `baseline/data/`.

The comparison notebook reads only persisted files. It does not depend on a
live `suite` variable or on running all notebooks in one kernel.

## Independent replicates and error bars

Each optimizer notebook runs three independent complete training trajectories:

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

The single-optimizer notebooks keep a fixed train/test and layer color scheme.
The comparison notebook uses a fixed color-blind-safe optimizer mapping:

```text
SGD + momentum          blue
AdamW                   vermillion
SGD + momentum + Muon   bluish green
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

## Checkpoint and result persistence

Every optimizer notebook sets `save_epoch_checkpoints=True`. Each seed folder
therefore contains both a final state and one complete model/optimizer state
after every training epoch:

```text
runs/<optimizer>/
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
      performance_by_epoch.csv
      spectral_metrics_by_epoch_and_layer.csv
      weightwatcher_details_by_epoch.csv
      optimizer_groups_by_epoch.csv
      combined_metrics_by_epoch_and_layer.csv
      esd_history.npz
      config.json
      final_state.pt
      checkpoints/
        epoch_001.pt
        ...
        epoch_020.pt
    seed_2027/
    seed_31415/
```

Each training notebook fails at the end if any aggregate result, final state,
or requested epoch checkpoint is missing.

## Three-optimizer comparison

`MNIST_MLP3_Baseline_Comparison.ipynb` validates that all three optimizers have:

- identical seed tuples;
- identical epoch grids and shared data/evaluation/WeightWatcher settings;
- complete FC1/FC2/FC3 spectral measurements;
- `final_state.pt` for every seed;
- all 20 epoch checkpoints for every seed.

It then produces:

- mean and 95% confidence-interval trajectories for train/test accuracy,
  train/test loss, classification perplexity, and generalization gaps;
- layerwise comparisons of `alpha`, `detX_num`, `num_pl_spikes`, `ERG_gap`,
  midpoint retained rank, midpoint trace-log, and stable rank;
- final-epoch metric tables;
- best-achieved test accuracy and convergence-threshold tables;
- paired seed-level final differences for every optimizer pair;
- a checkpoint inventory and reproducibility manifest.

Classification perplexity is derived from the saved cross-entropy as
`exp(cross_entropy)`. For MNIST this is an effective-class-count transform, not
language-model perplexity.

Comparison outputs are written under:

```text
runs/comparison/
  checkpoint_inventory.csv
  all_optimizers_performance_by_epoch_and_seed.csv
  all_optimizers_spectral_metrics_by_epoch_layer_and_seed.csv
  performance_summary_95ci.csv
  spectral_summary_95ci.csv
  final_epoch_summary_95ci.csv
  convergence_by_seed.csv
  convergence_summary_95ci.csv
  paired_final_differences_95ci.csv
  comparison_manifest.json
  plots/
```

Paired contrasts are always reported as `optimizer_a - optimizer_b`. Positive
is favorable for accuracy; negative is favorable for loss and perplexity. With
only three paired seeds, the notebook reports Student-t intervals and does not
claim high-powered asymptotic significance tests.

## Run order

From the repository root, open the notebooks in `baseline/notebooks/` and run:

```text
1. MNIST_MLP3_SGD_Momentum_Baseline.ipynb
2. MNIST_MLP3_AdamW_Baseline.ipynb
3. MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb
4. MNIST_MLP3_Baseline_Comparison.ipynb
```

The first three may be run in any order; the comparison must be run after all
three have completed under the same run root.

## Tests

```bash
cd baseline
PYTHONPATH=. python -m unittest discover -s tests -v
```
