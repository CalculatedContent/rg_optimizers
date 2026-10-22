# Validation — October 18, 2026

Five offline `unittest` checks passed: paired bias/no-bias initialization,
parameter-group learning rates and FC1-only decay, schedule endpoints and the
stratified split, preset sizes and misspelled-key rejection, and the full-epoch
budget / sustained-interpolation rule. The interpolation test uses mocked
evaluation metrics; it is a logic check, not an empirical result.

The CLI rejects test evaluation of a multi-configuration grid and rejects a
warm-up at least as long as training. Python compilation and `git diff --check`
passed. The CPU baseline below uses actual full datasets and the unmodified
120-epoch training budget, not mock data or shortened training.

## Full CPU reproduction

Settings: seed401, biases enabled, dropout0, FC1 decay0, original layer rates,
momentum0.9, batch256, five-epoch warm-up, cosine floor0.1. The reference is the
saved result from the separate overnight experiment. Only training and validation
data were used; no held-out test evaluation was performed.

| Dataset | Epoch | Validation accuracy | Training errors | Final zero-error streak | FC1 alpha | FC2 alpha | FC3 alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 120 | 98.22% | 0 | 81 | 1.7342237342 | 2.0105711748 | 2.6720028959 |
| FashionMNIST | 120 | 90.16% | 0 | 32 | 1.8387093144 | 2.5386932086 | 2.1805347573 |

Both datasets' accuracy, error counts, and zero-error streak match the saved references.
Losses agree within floating-point CSV round-trip precision; all three alphas
agree within 1e-9 (the original JSON stored 10 decimal places). Running the same
command again skipped the completed row and left one row in the CSV.

Commands (the data cache location is machine-specific):

```bash
python grid_search/mnist.py --grid grid_search/configs/control.json --device cpu --seeds 401
python grid_search/fashion_mnist.py --grid grid_search/configs/control.json --device cpu --seeds 401
```

Environment: Python3.12, torch2.14.1+cpu, NumPy2.3.5, pandas2.2.3,
SciPy1.17.0, scikit-learn1.8.0, WeightWatcher0.7.7, powerlaw2.0.0; one CPU
thread per process. Package versions are also recorded in each output row.

## Hardware qualification

CPU is executed and checked. **MPS, CUDA and XLA/TPU were not available in this
environment and have not been executed here.** Their device paths follow the
official PyTorch interfaces linked in the README. In particular, no claim is made
about TPU throughput or cross-device bitwise reproducibility.
