# MNIST MLP3 baseline notebooks

This folder contains the complete clean-baseline workflow used to compare three
unmodified optimizers on the same MLP3/MNIST experiment:

1. `MNIST_MLP3_SGD_Momentum_Baseline.ipynb`
2. `MNIST_MLP3_AdamW_Baseline.ipynb`
3. `MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`
4. `MNIST_MLP3_Baseline_Comparison.ipynb`

The first three notebooks **train and save** the experiments. The fourth
notebook **loads, validates, analyzes, and compares** the saved results. The
comparison notebook does not retrain the models and does not depend on variables
left in another notebook kernel.

These are true baselines. They contain no trace-log projection, adaptive ECS
correction, WW-PGD retraction, spectral-flow subtraction, or other RG
intervention.

---

## 1. Fixed experiment protocol

Every optimizer uses the same model, data, evaluation protocol, seeds, and
number of epochs.

```text
Dataset:       MNIST
Architecture:  784 -> 512 -> 512 -> 10
Activations:   ReLU after FC1 and FC2
Normalization: mean 0.1307, standard deviation 0.3081
Batch size:    128
Epochs:        20
Seeds:         1337, 2027, 31415
Replicates:    3 complete independent training runs per optimizer
Evaluation:    full training set and full test set at epoch 0 and every epoch
Error bars:    two-sided 95% Student-t confidence interval across the 3 runs
```

The three optimizer definitions are:

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

Muon/Newton--Schulz is applied to `fc1.weight` and `fc2.weight`. The final
classifier matrix `fc3.weight` and all biases use ordinary SGD + momentum. The
Muon notebook prints and asserts the parameter assignment before training.

---

## 2. Environment setup

From the repository root:

```bash
cd /tmp/rg_optimizers
python -m pip install -r baseline/requirements.txt
```

The required packages include PyTorch, torchvision, NumPy, pandas, matplotlib,
WeightWatcher, and Jupyter.

Run the unit tests before launching the full experiments:

```bash
cd /tmp/rg_optimizers/baseline
PYTHONPATH=. python -m unittest discover -s tests -v
cd /tmp/rg_optimizers
```

The notebooks automatically use CUDA when available, otherwise Apple MPS when
available, otherwise CPU. If an unsupported Apple MPS operation is encountered,
launch Jupyter with PyTorch fallback enabled:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## 3. Choose one shared output directory

All four notebooks must use the **same** run root. The default is:

```text
<repository>/baseline/runs/
```

For a checkout at `/tmp/rg_optimizers`, the default is:

```text
/tmp/rg_optimizers/baseline/runs/
```

A separate run directory is recommended when repeating the experiment so that
old and new artifacts cannot be mixed:

```bash
export RG_BASELINE_RUN_ROOT=/tmp/rg_optimizers_baseline_runs_v1
export RG_BASELINE_DATA_DIR=/tmp/rg_optimizers_mnist_data
mkdir -p "$RG_BASELINE_RUN_ROOT" "$RG_BASELINE_DATA_DIR"
```

Use the same environment variables for all four notebooks. The comparison
notebook will reject incomplete or inconsistent optimizer runs.

---

## 4. Run the notebooks interactively

Start Jupyter from the repository root:

```bash
cd /tmp/rg_optimizers
jupyter lab
```

Open `baseline/notebooks/` and run **all cells** in this order:

```text
1. MNIST_MLP3_SGD_Momentum_Baseline.ipynb
2. MNIST_MLP3_AdamW_Baseline.ipynb
3. MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb
4. MNIST_MLP3_Baseline_Comparison.ipynb
```

The first three training notebooks may be run in any order. The comparison
notebook must be run only after all three training notebooks have completed
under the same `RG_BASELINE_RUN_ROOT`.

Each training notebook runs three full 20-epoch experiments. WeightWatcher is
executed at epoch 0 and after every epoch for FC1, FC2, and FC3. WeightWatcher
analysis is therefore a substantial part of the computation.

---

## 5. Run the notebooks non-interactively

The following commands execute the notebooks while writing executed copies
outside the source folder, so the committed notebook files remain clean.

```bash
cd /tmp/rg_optimizers

export RG_BASELINE_RUN_ROOT=/tmp/rg_optimizers_baseline_runs_v1
export RG_BASELINE_DATA_DIR=/tmp/rg_optimizers_mnist_data
mkdir -p "$RG_BASELINE_RUN_ROOT/executed_notebooks" "$RG_BASELINE_DATA_DIR"

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir "$RG_BASELINE_RUN_ROOT/executed_notebooks" \
  --output MNIST_MLP3_SGD_Momentum_Baseline.executed.ipynb \
  baseline/notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir "$RG_BASELINE_RUN_ROOT/executed_notebooks" \
  --output MNIST_MLP3_AdamW_Baseline.executed.ipynb \
  baseline/notebooks/MNIST_MLP3_AdamW_Baseline.ipynb

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir "$RG_BASELINE_RUN_ROOT/executed_notebooks" \
  --output MNIST_MLP3_SGD_Momentum_Muon_Baseline.executed.ipynb \
  baseline/notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir "$RG_BASELINE_RUN_ROOT/executed_notebooks" \
  --output MNIST_MLP3_Baseline_Comparison.executed.ipynb \
  baseline/notebooks/MNIST_MLP3_Baseline_Comparison.ipynb
```

Do not run the comparison command until the first three commands have completed
successfully.

---

## 6. What each training notebook saves

Each optimizer writes its aggregate results beneath:

```text
$RG_BASELINE_RUN_ROOT/<optimizer>/
```

where `<optimizer>` is one of:

```text
sgd_momentum
adamw
sgd_momentum_muon
```

The aggregate files are:

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
```

Each optimizer also has three seed folders:

```text
seeds/
  seed_1337/
  seed_2027/
  seed_31415/
```

Each seed folder contains:

```text
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
  epoch_002.pt
  ...
  epoch_020.pt
```

For each optimizer, the expected checkpoint count is:

```text
3 seeds x 20 epoch checkpoints = 60 epoch checkpoint files
3 seeds x 1 final state         = 3 final_state.pt files
```

Across all three optimizers, the expected total is:

```text
180 epoch checkpoint files
9 final_state.pt files
```

Each training notebook performs a final persistence audit and stops with an
error if any required file is missing.

---

## 7. Metrics recorded at every epoch

The performance tables contain:

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

For every seed, epoch, and layer, the spectral tables contain the original
WeightWatcher outputs and additional diagnostics, including:

```text
alpha
detX_num
num_pl_spikes
ERG_gap
m_midpoint
trace_log_midpoint_per_eval
trace_log_midpoint_total
stable_rank
participation_ratio
entropy_effective_rank
boundary_overlap_ratio
top1_energy_fraction
pl_energy_fraction
detx_energy_fraction
midpoint_energy_fraction
geometric_mean_midpoint
normalized_lambda_max
normalized_lambda_midpoint_cut
eigenvalue_condition_number
```

`alpha`, `detX_num`, `num_pl_spikes`, and `ERG_gap` come from
`watcher.analyze(ERG=True)`. No fallback alpha or replacement ERG boundary is
used.

---

## 8. Hard validation requirements

These conditions are enforced by the code. Failure of any one means the run is
incomplete or invalid rather than merely scientifically surprising.

- Exactly three unique seeds must be present for every optimizer.
- Epochs `0` through `20` must be present in every performance table.
- FC1, FC2, and FC3 must have valid WeightWatcher measurements at every epoch.
- Required train/test metrics and required spectral metrics must be finite.
- `ERG_gap` must equal `detX_num - num_pl_spikes` exactly.
- The midpoint retained rank must equal
  `floor((detX_num + num_pl_spikes) / 2)` exactly.
- Every seed must have `final_state.pt`.
- Every seed must have `epoch_001.pt` through `epoch_020.pt`.
- All three optimizer manifests must use the same seeds and shared experiment
  settings before the comparison is allowed to run.

The comparison notebook intentionally fails before plotting if it detects mixed
run roots, missing files, incomplete seeds, inconsistent epochs, or missing
layers.

---

## 9. What we expect to see in the training curves

The following are **sanity-check expectations**, not hard acceptance thresholds.
Exact values depend on hardware, PyTorch version, numerical kernels, and random
seed.

### Epoch 0

A randomly initialized ten-class classifier should begin near:

```text
train accuracy: approximately 10%
test accuracy:  approximately 10%
cross-entropy:  approximately log(10) = 2.30
perplexity:     approximately exp(2.30) = 10
```

Small deviations are normal. Accuracy remaining near 10% and loss remaining
near 2.30 after several epochs is not normal and indicates that training is not
working.

### Through epoch 20

All three baselines should show:

- rapidly decreasing train and test cross-entropy;
- rapidly increasing train and test accuracy;
- smooth convergence without NaNs or divergent gradient norms;
- train accuracy above test accuracy near the end;
- test loss slightly above train loss near the end;
- relatively small variation across the three seeds in task performance;
- potentially larger seed variation in WeightWatcher fit and spectral metrics.

For this MLP3/MNIST setup, a reasonable broad sanity range after 20 epochs is:

```text
final test accuracy: usually about 97% to 99%+
final train accuracy: usually about 98% to nearly 100%
final test loss:      commonly below about 0.15
final perplexity:     commonly near 1.04 to 1.16
```

These are diagnostic ranges rather than promises. A final test accuracy below
about 95%, a loss curve that does not decrease, or large non-finite values is a
clear reason to inspect the run.

### Expected optimizer tendencies

The experiment does not assume a predeclared winner. Typical qualitative
behavior to look for is:

- **AdamW:** strong early convergence and smooth loss reduction.
- **SGD + momentum:** somewhat more gradual early convergence, often with strong
  final generalization.
- **SGD + momentum + Muon:** competitive task convergence with visibly different
  hidden-layer spectral and effective-rank trajectories because FC1 and FC2
  matrix updates are orthogonalized.

The final ranking may change by seed and metric. Overlapping confidence
intervals are expected with only three independent runs and should not be
interpreted as evidence of a reliable optimizer difference.

---

## 10. What we expect to see in the spectral metrics

These notebooks are baselines, so no spectral quantity is forced toward a
target.

Expected qualitative behavior is:

- WeightWatcher `alpha` should be finite for FC1, FC2, and FC3 at every epoch.
- Alpha trajectories should move away from their random-initialization behavior
  as layer correlations form.
- FC1 and FC2 will generally show clearer structural evolution than the small
  final classifier layer FC3.
- Strong layers may move toward the heavy-tailed region near `alpha = 2`, but
  the baseline does **not** require `alpha = 2`; a layer may remain above 2 or
  temporarily move below 2.
- `detX_num`, `num_pl_spikes`, midpoint rank, stable rank, and effective ranks
  should evolve rather than remain identically constant.
- `ERG_gap` may contract when the detX and power-law boundaries become better
  aligned, but no fixed value or optimizer ordering is required.
- The midpoint trace-log is not constrained to zero. Baseline drift away from
  zero is scientifically useful because later RG optimizers can be compared
  against this unmodified trajectory.
- Muon may produce the largest difference in FC1/FC2 rank, alpha, and trace-log
  trajectories even when final test accuracy is similar.

A missing alpha, missing ERG boundary, inconsistent ERG gap, or missing layer is
not an interesting scientific result; it is treated as a failed measurement.

---

## 11. What the comparison notebook produces

The comparison notebook writes to:

```text
$RG_BASELINE_RUN_ROOT/comparison/
```

It saves:

```text
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

The notebook produces 14 comparison plots covering:

- test accuracy;
- train accuracy;
- test loss;
- train loss;
- test classification perplexity;
- accuracy generalization gap;
- loss generalization gap;
- layerwise WeightWatcher alpha;
- layerwise `detX_num`;
- layerwise `num_pl_spikes`;
- layerwise `ERG_gap`;
- layerwise midpoint retained rank;
- layerwise midpoint trace-log per retained eigenvalue;
- layerwise stable rank.

Optimizer colors are fixed across every comparison figure:

```text
SGD + momentum          blue
AdamW                   vermillion
SGD + momentum + Muon   bluish green
```

The comparison notebook also reports:

- final-epoch means and 95% confidence intervals;
- best test accuracy and the epoch where it occurred;
- first epoch reaching test-accuracy thresholds of 90%, 95%, 97%, and 98%;
- paired final-epoch differences using matched seeds;
- train/test generalization gaps;
- classification perplexity computed as `exp(cross_entropy)`.

For paired tables, the sign convention is always:

```text
optimizer_a - optimizer_b
```

Therefore positive is favorable for accuracy, while negative is favorable for
loss and perplexity. With only three seeds, the Student-t intervals will often
be wide; the tables should be treated as baseline estimates, not high-powered
significance tests.

---

## 12. Verify the completed run from the shell

After all four notebooks finish, these counts should be obtained:

```bash
find "$RG_BASELINE_RUN_ROOT" -path '*/checkpoints/epoch_*.pt' | wc -l
# expected: 180

find "$RG_BASELINE_RUN_ROOT" -name final_state.pt | wc -l
# expected: 9

find "$RG_BASELINE_RUN_ROOT/comparison/plots" -name '*.png' | wc -l
# expected: 14

find "$RG_BASELINE_RUN_ROOT" -name replicate_manifest.json | wc -l
# expected: 3
```

The comparison notebook should end by printing:

```text
Comparison audit passed.
```

---

## 13. Common failure modes

### The comparison notebook reports missing paths

One or more training notebooks did not finish, or the notebooks used different
`RG_BASELINE_RUN_ROOT` values. Rerun the missing optimizer notebook using the
same run root. Do not bypass the audit.

### The comparison reports inconsistent configurations

Artifacts from different experiment versions were mixed. Select a new empty
run root and rerun all three training notebooks.

```bash
export RG_BASELINE_RUN_ROOT=/tmp/rg_optimizers_baseline_runs_v2
```

### Accuracy stays near 10%

Training is not progressing. Inspect the device, optimizer construction,
gradient norms, and notebook output for an earlier exception.

### WeightWatcher metrics are missing or non-finite

Confirm that `weightwatcher>=0.7.7` is installed and that the notebook reached
all WeightWatcher checkpoints. The experiment should not substitute a fallback
fit.

### A training notebook stops before the final audit

The completed seed results remain in the run directory, but the comparison will
correctly reject the incomplete optimizer. Fix the error and rerun that training
notebook before running the comparison.

---

## 14. Interpretation boundary

This workflow establishes the unmodified optimizer trajectories. It is intended
to answer:

- How quickly does each optimizer fit MNIST?
- What final train/test performance does each optimizer reach?
- How large is run-to-run variation?
- How do alpha, ERG gap, midpoint rank, trace-log, and effective rank evolve
  without RG intervention?
- Does Muon change hidden-layer spectral organization even when task performance
  is similar?

It does **not** demonstrate that any optimizer is superior from three runs
alone, and it does not enforce `alpha = 2`, `ERG_gap = 0`, or trace-log equal to
zero. Those are quantities to observe in the baselines and compare against the
RG optimizer extensions.