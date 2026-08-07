# ECS Probe-Loss TraceWall

This folder contains the **ECS Probe-Loss TraceWall** optimizer experiment for the standard MLP3/MNIST problem in `rg_optimizers`.

The purpose of this experiment is simple:

> Instead of trying to suppress motion toward a presumed trivial fixed point, compute the current Effective Correlation Space (ECS), evaluate the task loss on the ECS-truncated network using a rotating random subset of the training data, and add a correction to the optimizer update that lowers that loss inside the ECS.

The experiment is deliberately isolated from the other optimizer implementations in this repository. It does **not** modify the existing trace-log tracker, adaptive spectral guard, spectral RG-flow projector, or local-delta WW-PGD code.

This experiment is intended to be run **locally from the notebooks in this folder**. The full MNIST experiments are not intended to be run as GitHub Actions jobs.

---

## 1. What is in this folder

```text
optimizers/ecs_probe_loss_trace_wall/
|
|-- README.md
|   This file. Describes the method, exact local run procedure, outputs,
|   diagnostics, and what to look for in the results.
|
|-- requirements.txt
|   Python dependencies for the notebooks and experiment package.
|
|-- pyproject.toml
|   Package metadata for installing the local ecs_trace_wall package.
|
|-- ecs_trace_wall/
|   |-- config.py
|   |   Optimizer, TraceWall, and experiment configuration dataclasses.
|   |
|   |-- ecs.py
|   |   Self-consistent ECS calculation and SVD/ECS support logic.
|   |
|   |-- optimizer.py
|   |   The ECS Probe-Loss TraceWall optimizer wrapper and correction logic.
|   |
|   |-- sampler.py
|   |   Rotating random training-subset sampler with checkpointable state.
|   |
|   |-- runtime.py
|   |   MLP3 model, MNIST loading, optimizer construction, device selection,
|   |   learning-rate schedule, deterministic epoch ordering, and utilities.
|   |
|   |-- training.py
|   |   Baseline and TraceWall training loops.
|   |
|   |-- spectral.py
|   |   ECS, WeightWatcher, rank, and spectral diagnostics.
|   |
|   |-- reporting.py
|   |   Performance tables, Student-t confidence intervals, pairing checks,
|   |   and correction summaries.
|   |
|   |-- plotting.py
|   |   Standard baseline-vs-TraceWall plots.
|   |
|   `-- experiment.py
|       Complete paired multi-seed experiment runner and artifact persistence.
|
|-- notebooks/
|   |-- MNIST_MLP3_AdamW_vs_ECS_Probe_Loss_TraceWall.ipynb
|   `-- MNIST_MLP3_SGD_Momentum_vs_ECS_Probe_Loss_TraceWall.ipynb
|
`-- tests/
    Unit and smoke tests for ECS selection, projection, rotating subsets,
    optimizer descent, schedules, paired experiments, plots, and notebooks.
```

---

## 2. The model and data

Both notebooks use exactly the same MLP3 architecture:

```text
784 -> 512 -> 512 -> 10
```

with ReLU after `fc1` and `fc2`, no dropout, and no batch normalization.

The dataset is MNIST. Training and test metrics are evaluated separately.

The **official MNIST test set is never used by the optimizer correction**. It is used only for reporting test loss and test accuracy.

The TraceWall probe is always drawn from the MNIST **training set**.

---

## 3. What the optimizer does

Assume a base optimizer step has produced a proposed matrix

\[
W^{\mathrm{base}}_{t+1}.
\]

For every selected matrix layer, compute its current singular-value decomposition

\[
W = U\Sigma V^\top.
\]

The self-consistent trace-log calculation determines the current ECS rank `m`. The ECS-truncated matrix is

\[
W_{\mathrm{ECS}} = U_m\Sigma_mV_m^\top.
\]

All selected matrices are replaced **simultaneously** by their current ECS-truncated versions while the auxiliary task loss is evaluated.

For a rotating subset `B_t` of training examples,

\[
\mathcal L_{\mathrm{probe}}
=
\frac{1}{|B_t|}
\sum_{(x,y)\in B_t}
\ell\left(f_{W_{\mathrm{ECS}}}(x),y\right).
\]

The gradient of this loss is then projected into the same ECS. In the default `core` projection,

\[
G_{\mathrm{ECS}}
=
(U_mU_m^\top)G(V_mV_m^\top).
\]

The TraceWall proposes a negative-gradient correction

\[
\Delta W_{\mathrm{probe,ECS}} \propto -G_{\mathrm{ECS}}.
\]

The completed update becomes

\[
W_{t+1}
=
W^{\mathrm{base}}_{t+1}
+
a_t\Delta W_{\mathrm{probe,ECS}},
\]

where `a_t` is selected by Armijo backtracking.

The correction is committed only when it lowers the probe loss measured on the ECS-truncated model.

The ECS is recomputed at every correction. Therefore, if the ECS shrinks during training, the correction automatically acts in the smaller space. If it expands, the correction follows the expanded support.

The default projection is `core`. A `rank_m_tangent` projection is included as an ablation but is not the primary experiment.

---

## 4. Rotating training-probe protocol

The default TraceWall probe configuration is:

```text
probe_batch_size              = 256
probe_batches_per_correction  = 2
examples per correction       = 512
corrections per epoch          = 1
```

The rotating sampler uses an independent seeded random permutation of the training set.

It consumes examples without replacement until the permutation is exhausted, then generates a new permutation. A subset that crosses a permutation boundary is still unique within that individual draw.

This is intended to approximate expected task loss over changing random subsets without optimizing against the official test set.

---

## 5. Paired experimental design

Each notebook performs a strict paired comparison:

```text
clean baseline optimizer
versus
same optimizer + ECS Probe-Loss TraceWall
```

For each seed, the baseline and TraceWall arms:

- start from byte-identical initial model weights;
- receive the same MNIST minibatches in the same order;
- use the same gradient clipping;
- use the same base optimizer hyperparameters;
- use the same learning-rate schedule;
- use the same number of epochs;
- use the same train/test evaluation protocol.

The only intended intervention is the post-base-step ECS probe-loss correction in the TraceWall arm.

The default independent seeds are:

```text
1337
2027
31415
```

The primary experiment is 20 epochs per seed.

---

## 6. Optimizer hyperparameters

### AdamW notebook

```text
optimizer       AdamW
peak LR         1e-3
betas           (0.9, 0.999)
epsilon         1e-8
weight decay    1e-2
```

Notebook:

```text
notebooks/MNIST_MLP3_AdamW_vs_ECS_Probe_Loss_TraceWall.ipynb
```

### SGD + momentum notebook

```text
optimizer       SGD
peak LR         5e-2
momentum        0.9
dampening       0.0
Nesterov        False
weight decay    1e-4
```

Notebook:

```text
notebooks/MNIST_MLP3_SGD_Momentum_vs_ECS_Probe_Loss_TraceWall.ipynb
```

### Learning-rate schedule

Both arms of both experiments use the same schedule:

1. one epoch of linear warmup;
2. cosine decay for the remainder of training;
3. final learning rate equal to 5% of the peak learning rate.

---

## 7. TraceWall correction defaults

The primary settings in `TraceWallConfig` are:

```text
selected matrices             fc1.weight, fc2.weight, fc3.weight
projection mode               core
minimum ECS rank              2
normalization gamma           0.0
SVD device                    cpu
correction/base-step ratio    0.25
minimum weight fraction       1e-5
maximum weight fraction       2.5e-3
backtracking                   enabled
backtracking factor           0.5
maximum backtracking steps    7
Armijo coefficient            1e-4
```

The experiment runner converts `corrections_per_epoch=1` into the appropriate optimizer-step interval after it knows the number of minibatches per epoch.

The first correction begins no earlier than the end of the warmup interval.

---

# 8. Exact local run instructions

The intended workflow is to run these experiments locally.

## Step 1: clone or update the repository

```bash
git clone https://github.com/CalculatedContent/rg_optimizers.git
cd rg_optimizers
git pull origin main
```

If the repository is already cloned:

```bash
cd /path/to/rg_optimizers
git checkout main
git pull origin main
```

## Step 2: enter this experiment folder

```bash
cd optimizers/ecs_probe_loss_trace_wall
```

## Step 3: create a Python environment

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Step 4: install dependencies

```bash
pip install -r requirements.txt
pip install -e .
pip install jupyterlab
```

The important scientific dependencies include:

```text
torch
torchvision
numpy
pandas
scipy
matplotlib
weightwatcher
nbformat
```

MNIST will be downloaded automatically by `torchvision` if it is not already present in the configured data directory.

## Step 5: optional output locations

By default, the notebooks write experiment artifacts beneath this experiment's local `runs/` directory and use a local data cache.

To explicitly place everything under `/tmp`, for example:

```bash
export RG_TRACE_WALL_RUN_ROOT=/tmp/rg_trace_wall_runs
export RG_TRACE_WALL_DATA_DIR=/tmp/rg_trace_wall_data
```

Or leave those variables unset to use the notebook defaults.

## Step 6: launch Jupyter

From `optimizers/ecs_probe_loss_trace_wall/`:

```bash
jupyter lab
```

Then open one of the two notebooks.

## Step 7: run AdamW experiment

Open:

```text
notebooks/MNIST_MLP3_AdamW_vs_ECS_Probe_Loss_TraceWall.ipynb
```

Use:

```text
Run -> Run All Cells
```

The notebook runs all three seeds for both arms:

```text
AdamW baseline
AdamW + ECS Probe-Loss TraceWall
```

## Step 8: run SGD-momentum experiment

Open:

```text
notebooks/MNIST_MLP3_SGD_Momentum_vs_ECS_Probe_Loss_TraceWall.ipynb
```

Again use:

```text
Run -> Run All Cells
```

This runs:

```text
SGD + momentum baseline
SGD + momentum + ECS Probe-Loss TraceWall
```

The two notebooks are independent. They can be run in either order.

---

## 9. Optional command-line notebook execution

If you prefer to execute without interacting with JupyterLab, from this folder you can use:

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/MNIST_MLP3_AdamW_vs_ECS_Probe_Loss_TraceWall.ipynb \
  --output MNIST_MLP3_AdamW_vs_ECS_Probe_Loss_TraceWall.executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

and

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/MNIST_MLP3_SGD_Momentum_vs_ECS_Probe_Loss_TraceWall.ipynb \
  --output MNIST_MLP3_SGD_Momentum_vs_ECS_Probe_Loss_TraceWall.executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

These are still **local notebook executions**; they are not GitHub Actions runs.

---

# 10. What gets measured

At epoch zero and after every epoch, the paired experiment records full-dataset performance for both arms.

### Task metrics

```text
train cross-entropy loss
test cross-entropy loss
train accuracy
test accuracy
train classification perplexity
test classification perplexity
accuracy generalization gap
loss generalization gap
```

Classification perplexity here is simply `exp(cross_entropy)`. It is a useful transformed loss metric but should not be interpreted as language-model perplexity.

### Optimization metrics

```text
learning rate
parameter L2 norm
epoch training time
number of correction attempts
number of accepted corrections
```

### ECS and spectral metrics

For FC1, FC2, and FC3:

```text
self-consistent ECS rank
fractional ECS rank
ECS rank fraction
adaptive normalization dimension
bulk effective count
trace-log total
trace-log per retained eigenvalue
retained spectral-energy fraction
stable rank
participation ratio
```

### WeightWatcher metrics

The experiment also records:

```text
alpha
detX_num
num_pl_spikes
ERG_gap
```

WeightWatcher is required in the primary notebook configuration.

### TraceWall-specific diagnostics

Each correction records information such as:

```text
probe loss before correction
probe loss after correction
whether the correction was accepted
Armijo/backtracking scale
ECS rank used for each layer
base-step norm
raw correction norm
committed correction norm
correction/base-step ratio
projection numerical audits
```

---

# 11. Output files

For each optimizer notebook, the experiment output directory contains aggregate CSV files such as:

```text
performance_by_epoch_and_seed.csv
spectral_metrics_by_epoch_layer_and_seed.csv
trace_wall_corrections_by_step_layer_and_seed.csv
performance_summary_95ci.csv
spectral_summary_95ci.csv
trace_wall_correction_summary.csv
config.json
paired_manifest.json
```

There is also a per-seed tree:

```text
seeds/
|-- seed_1337/
|   |-- baseline_final_state.pt
|   |-- trace_wall_final_state.pt
|   `-- checkpoints/
|       |-- baseline_epoch_001.pt
|       |-- trace_wall_epoch_001.pt
|       |-- ...
|       |-- baseline_epoch_020.pt
|       `-- trace_wall_epoch_020.pt
|
|-- seed_2027/
`-- seed_31415/
```

Each TraceWall checkpoint also stores the rotating-probe sampler state so the random-subset sequence can be resumed consistently.

The `paired_manifest.json` records the experiment's effective runtime settings, including steps per epoch, warmup steps, correction interval, probe size, device, initial-state checksums, final-state checksums, and sampler position.

---

# 12. Error bars

The reported curves aggregate the three complete independent runs.

For each metric, the code reports the mean and a two-sided 95% Student-t confidence interval:

\[
\bar{x}
\pm
 t_{0.975,n-1}\frac{s}{\sqrt n},
\qquad n=3.
\]

The independent unit is the complete training run/seed, **not** a minibatch, layer, test example, or individual correction.

---

# 13. What we expect to see

This is an experimental optimizer, so improvement is a hypothesis rather than a guaranteed outcome. The plots should answer the following questions.

## A. Does the correction actually minimize the ECS probe objective?

The first sanity check is mechanical:

```text
probe_loss_after <= probe_loss_before
```

for accepted corrections.

If this is not true, the correction mechanism is not behaving as designed.

Also inspect the correction acceptance fraction. If almost every correction is rejected, the direction, scale, cadence, or ECS restriction may be too aggressive or uninformative.

## B. Does the ECS evolve differently?

Compare baseline and TraceWall for each layer:

```text
ECS rank
alpha
ERG_gap
retained energy
stable rank
participation ratio
```

The central question is whether minimizing task loss specifically within the current ECS produces a measurably different spectral trajectory from the clean optimizer.

FC1 is especially important because it is the largest and usually most spectrally informative MLP3 layer.

## C. Does TraceWall improve generalization?

The main performance comparison is:

```text
baseline test loss      vs TraceWall test loss
baseline test accuracy  vs TraceWall test accuracy
```

Do not look only at the final epoch. Also inspect:

```text
best test accuracy
minimum test loss
convergence speed
late-epoch test-loss rebound
generalization gap
```

A useful outcome would be lower test loss and/or higher test accuracy at comparable training loss.

## D. Does it merely accelerate fitting?

If TraceWall lowers both training and test loss by approximately the same amount while leaving the spectral trajectory essentially unchanged, then it may simply be acting as an additional task-gradient step rather than providing an ECS-specific regularization effect.

That is still informative, but it is a different mechanism.

## E. Does it overfit the rotating probe?

The probe consists of training data. Therefore we must distinguish improved optimization from improved generalization.

A warning sign would be:

```text
probe loss improves strongly
training loss improves
but test loss worsens or test accuracy declines
```

That would mean the ECS-constrained task channel is still capable of increasing overfitting.

## F. Does the shrinking ECS behave as intended?

As the ECS rank changes, verify that the logged correction ranks change with it.

The correction should not continue operating in stale singular directions after the self-consistent ECS has contracted.

---

# 14. Primary comparison tables to inspect

After a notebook finishes, the first files to inspect are:

```text
performance_summary_95ci.csv
spectral_summary_95ci.csv
trace_wall_correction_summary.csv
```

For performance, focus first on:

```text
test_accuracy
test_loss
train_accuracy
train_loss
accuracy_generalization_gap
loss_generalization_gap
```

For FC1 spectral behavior, focus first on:

```text
alpha
ecs_rank
ecs_rank_fraction
ecs_trace_log_per_eval
ERG_gap
retained_energy_fraction
participation_ratio
```

For correction behavior, focus first on:

```text
acceptance fraction
probe loss decrease
correction/base-step ratio
backtracking scale
```

---

# 15. How to interpret possible outcomes

### Outcome 1: better test performance and meaningful ECS changes

This is the strongest positive result. It would suggest that task-directed optimization restricted to the self-consistent ECS changes the learning trajectory in a useful way.

### Outcome 2: better test performance with little spectral change

The extra projected task-gradient step may be helping optimization, but the evidence for an RG/ECS-specific mechanism would be weaker.

### Outcome 3: probe/train loss improves but test performance worsens

The correction is optimizing the intended local objective but increasing overfitting. The next experiments should reduce correction strength, cadence, or probe reuse, or activate the channel only in selected spectral regimes.

### Outcome 4: corrections are mostly rejected

The projected task gradient is not a reliable descent direction after restoring the full model. Examine the ECS projection, correction scale, line search, and whether all three layers should be corrected simultaneously.

### Outcome 5: little difference from baseline

The correction may be too small, too infrequent, or may point mostly in directions already supplied by the base optimizer. Compare correction norms and cosine relationships before increasing its strength.

---

# 16. Running the tests locally

The full MNIST experiment is run from the notebooks, but the package can be checked first with:

```bash
python -m unittest discover -s tests -v
```

You can also compile the package and tests:

```bash
python -m compileall -q ecs_trace_wall tests
```

The tests cover:

- scale-invariant ECS selection;
- SVD truncation;
- ECS gradient projection;
- rotating-subset uniqueness;
- rotating-sampler checkpoint restoration;
- probe-loss descent;
- warmup/cosine schedule behavior;
- paired baseline/TraceWall execution;
- artifact generation;
- plotting;
- notebook JSON validity and code-cell compilation.

Passing the tests does **not** mean the scientific experiment has succeeded. It only verifies that the implementation behaves according to its defined mechanics.

---

# 17. Recommended run order

For a clean first experiment:

```text
1. Update the repository to current main.
2. Create/activate the Python environment.
3. Install requirements and the local package.
4. Run the local test suite.
5. Run the AdamW notebook completely.
6. Inspect its performance, spectral, and correction summaries.
7. Run the SGD-momentum notebook completely.
8. Compare whether the TraceWall effect is optimizer-specific or appears in both.
```

Do not modify TraceWall strength, probe size, cadence, or ECS definition until the initial paired runs have been saved. Those initial notebooks define the baseline experiment for this method.

---

# 18. Short version

The experiment asks:

> After the ordinary optimizer update, if we keep only the current ECS, measure loss on a fresh rotating subset of the training set, and add only an ECS-supported component that lowers that loss, does the network generalize better and does its spectral trajectory improve?

Run the two notebooks locally, compare each TraceWall arm directly against the paired clean baseline, and use the saved performance, WeightWatcher, ECS, and correction diagnostics to determine whether the effect is real.