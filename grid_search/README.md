# Simple MNIST / FashionMNIST grid search

One training loop, two entry points, JSON parameter lists, and a pandas DataFrame
saved to CSV after every run. The model is **784–256–256–10, ReLU, SGD momentum**.
These are ordinary SGD experiments in the RG optimizers repository; there is no
alpha-feedback optimizer or automatic change of hyperparameters during a run.

## Run

From the repository root, using Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r grid_search/requirements.txt

python grid_search/mnist.py --dry-run
python grid_search/mnist.py
python grid_search/fashion_mnist.py
```

Each default grid has **180 runs**: 5 dropout values × 6 FC1 decay values ×
2 bias choices × 3 seeds. Each run trains for all 120 epochs. `auto` chooses CUDA,
then Apple MPS, then CPU. Data are downloaded once and checked against the official
file hashes. `--data-dir PATH` reuses an existing cache.

For a quick full-dataset plumbing check (not a scientific result):

```bash
python grid_search/mnist.py --grid grid_search/configs/smoke.json --epochs 1 --seeds 401
python grid_search/fashion_mnist.py --grid grid_search/configs/smoke.json --epochs 1 --seeds 401
```

## Change the grid

Every JSON key is a parameter; every value is a list. The runner takes their
Cartesian product and runs all seeds. **A supplied grid replaces the default grid**;
omitted parameters use the single defaults below. Edit a file to combine searches.

| Parameter | Default | Meaning |
|---|---:|---|
| `bias` | `true` | Biases on all three linear layers |
| `dropout` | `0` | Dropout after FC1 ReLU only |
| `fc1_weight_decay` | `0` | Coupled L2 decay on FC1 weights only; biases excluded |
| `lr1_factor`, `lr2_factor`, `lr3_factor` | `1` each | Independent multipliers of the dataset's base peak rates |
| `momentum` | `0.9` | SGD momentum |
| `batch_size` | `256` | Training minibatch size |
| `warmup_epochs` | `5` | Linear warm-up from 1% to 100% peak |
| `lr_floor` | `0.1` | Cosine endpoint as a fraction of peak |
| `nesterov` | `false` | Enable Nesterov momentum |
| `init_scale` | `1` | Multiply initial weight matrices (not biases) by this factor |

| Dataset | FC1 peak LR | FC2 peak LR | FC3 peak LR |
|---|---:|---:|---:|
| MNIST | 0.056864187 | 0.379835549 | 1.724122529 |
| FashionMNIST | 0.018476187 | 0.179621415 | 0.547559449 |

Included grids:

| File in `configs/` | Runs per dataset with default seeds | Purpose |
|---|---:|---|
| `mnist.json`, `fashion_mnist.json` | 180 each | Original dropout/decay grid |
| `mnist_high_dropout.json` | 72 | MNIST 40%/50% dropout extension |
| `layer_rates.json` | 162 | All 27 layer-rate combinations, both bias choices, unregularized control |
| `optimizer_sweep.json` | 162 | Momentum × batch size × cooldown, both bias choices |
| `control.json` | 3 | Original unregularized biased control |

```bash
python grid_search/mnist.py --grid grid_search/configs/mnist_high_dropout.json
python grid_search/mnist.py --grid grid_search/configs/layer_rates.json
python grid_search/fashion_mnist.py --grid grid_search/configs/optimizer_sweep.json
```

For a regularized layer-rate sweep, change `dropout` and `fc1_weight_decay` in a
copy of `layer_rates.json`. For a focused optimizer follow-up, put the chosen
dropout/decay/rate factors into a copy of `optimizer_sweep.json`. Learning rates
should be retuned with momentum/batch size. The supplied optimizer grid is an
example follow-up, not a claim that the old rates are optimal for each new setting.

Other options: `--seeds 401 402 403`, `--epochs 120`, `--threads 1`,
`--limit 2`, `--output my_results.csv`. `--limit` takes the first N config/seed
pairs; it is not random sampling. `--dry-run` counts runs without loading data.

## Read the results

```python
import pandas as pd

df = pd.read_csv("grid_search/results/MNIST_mnist.csv")
print(df[["dropout", "fc1_weight_decay", "seed", "validation_accuracy",
          "train_errors", "sustained_interpolation", "alpha_fc1", "alpha_fc2", "alpha_fc3"]])
```

Accuracy columns are fractions, e.g. `0.985` means 98.5%. Each row contains all
parameters, actual peak rates, seed, dataset, epoch, update count, elapsed time,
train/validation loss and errors, three alphas, mean alpha, maximum distance from
2, and whether all three are within 0.10 of 2. Package versions, backend and source
hash are recorded. Nonfinite runs and spectral errors remain visible in the CSV.

`sustained_interpolation` means zero errors on all 55,000 training examples for
the final **10 consecutive epochs**, with dropout off. An average alpha near 2
does not establish that each layer is near 2; inspect the three individual values.

Rerunning the same command skips rows already saved for the same settings, seed,
backend, source and package versions. The current incomplete run restarts from its
seed. This intentionally small runner does **not** save model/optimizer checkpoints
or epoch histories. CSV replacement is atomic. Use one process per output CSV;
different processes/devices need different output files.

## MacBook, GPU, TPU

```bash
# MacBook (CPU also works on Intel Macs)
python grid_search/mnist.py --device mps
python grid_search/mnist.py --device cpu

# NVIDIA GPU: first install a CUDA PyTorch build appropriate to the machine
python grid_search/fashion_mnist.py --device cuda
```

On a **TPU VM**, install a matched PyTorch/PyTorch-XLA pair in a separate
environment. For example, the documented 2.8 release pair is:

```bash
pip install torch==2.8.0 'torch_xla[tpu]==2.8.0'
pip install -r grid_search/requirements.txt
PJRT_DEVICE=TPU python grid_search/mnist.py --device xla
PJRT_DEVICE=TPU python grid_search/fashion_mnist.py --device xla
```

This uses **one XLA device** and `torch_xla.sync()` after optimizer steps. It does
not provision a TPU or distribute one model across a pod. The simple per-step
Python learning-rate schedule may cause XLA recompilation overhead; this is a
portable reference runner, not a TPU throughput benchmark. WeightWatcher runs on
a CPU copy of the final model for every backend.

Hardware paths use float32. CPU uses deterministic algorithms; accelerator
results are statistical replications and are not promised bit-for-bit equal
across devices or package versions. CPU is the reference for exact comparisons.
The original campaign used torch 2.14.1+cpu, NumPy 2.3.5, pandas 2.2.3,
WeightWatcher 0.7.7, powerlaw 2.0.0 and SciPy 1.17.0.

Backend references: [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html),
[PyTorch/XLA single-device training](https://docs.pytorch.org/xla/release/r2.8/learn/pytorch-on-xla-devices.html),
[PyTorch/XLA installation](https://pypi.org/project/torch-xla/2.8.0/).

## Experimental conventions and held-out test data

Pixels are divided by 255 with no augmentation or other normalization. A fixed
stratified split, seed 20261015, holds out 500 examples per class from the official
60,000 training examples. Bias/no-bias models use paired initial weight draws;
the minibatch permutation uses seed + 10000. The schedule is updated each step.
Default WeightWatcher **0.7.7 `analyze()` with no arguments** measures the raw
matrices, including FC3. All reported accuracies and alphas refer to the same final
model. The search does not read or download the official test files.

After choosing a configuration using validation only, write it as singleton
lists in `selected.json`. Confirm it on fresh seeds and evaluate test:

```bash
python grid_search/mnist.py --grid selected.json --test --max-epochs 240 --output confirmed_mnist.csv
```

`--test` requires exactly one configuration and defaults to seeds 501–505.
With `--max-epochs 240`, train at least 120 epochs, then stop once the zero-error
streak reaches 10, or at 240. The learning rate stays at its floor after epoch120.
Without `--max-epochs`, all runs have a fixed 120-epoch budget. Freeze all selections
before using `--test`; do not tune on the resulting test numbers.

These grids reproduce the original primary sweep and provide simple, explicit
follow-up sweeps. They do not recreate the ongoing campaign's adaptive boundary
selection, two-anchor selection, sampled 3,600-run expansion, archive service or
automatic reports. No new full sweep is launched by adding this folder.

## Checks

```bash
python -m unittest discover -s grid_search -p 'test_*.py'
```

See `VALIDATION.md` for what was actually executed on available hardware.
