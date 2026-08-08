# Full Matrix-Log RG optimizer

This folder implements a state-aware optimizer extension that removes completed SGD motion back toward the trivial retained covariance solution

\[
\widetilde X_R = I.
\]

For the retained ECS/PL midpoint support, define

\[
\Phi_R(W)=\frac{1}{2m}\left\|\log \widetilde X_R(W)\right\|_F^2.
\]

Two correction modes are implemented:

- **`radial`** removes only a net first-order decrease of \(\Phi_R\).
- **`modewise`** removes every retained log-eigenvalue motion directed toward zero. This is the stronger full-matrix condition and is the default.

The correction is applied to the **completed optimizer displacement**, after SGD momentum, Nesterov acceleration, learning-rate scheduling, clipping, and weight decay. It does not project or whiten the accumulated weight matrix.

## Mac-oriented implementation

The expensive full-layer spectral analysis is an outer loop:

1. WeightWatcher runs at epoch zero and once per epoch.
2. It supplies direct `alpha`, `detX_num`, `num_pl_spikes`, `ERG_gap`, and `num_traps` measurements.
3. The retained rank is the midpoint of the Trace-Log and power-law ranks.
4. The retained right singular basis is cached until the next checkpoint.

The inner optimizer does **not** recompute a full SVD at every step. At each configured correction step it forms the retained covariance and diagonalizes only an \(m\times m\) matrix. Linear-algebra operations automatically fall back to CPU when an Apple MPS backend does not support them.

The default correction cadence is every 100 optimizer steps and the notebook applies the extension only to `fc1.weight` and `fc2.weight`. These defaults keep the first Mac experiment tractable while still intervening several times per epoch.

## Install on a Mac

From the repository root:

```bash
git checkout agent/full-matrix-log-rg

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e './baseline[experiment]'
python -m pip install -e './optimizers/full_matrix_log_rg[experiment]'
```

The baseline package pins the tested WeightWatcher version (`0.7.7`) and supplies the exact MNIST model, data split, optimizer schedule, diagnostics, and Student-t aggregation used by the reference notebook.

Store datasets and long-running outputs outside `/tmp`:

```bash
export RG_BASELINE_DATA_DIR="$HOME/rg-optimizer-data"
export RG_FML_RUN_ROOT="$HOME/rg-optimizer-runs/full_matrix_log_rg"
```

MPS is selected automatically when available. The DataLoader uses `num_workers=0` on MPS to avoid the common macOS multiprocessing failure.

## Validate before training

```bash
cd optimizers/full_matrix_log_rg
python -m unittest discover -s tests -v
cd ../..
```

The tests cover:

- the isotropic trivial solution;
- anisotropy hidden by a zero scalar Trace-Log;
- the analytic full matrix-log gradient against finite differences;
- radial and modewise one-sided corrections;
- rectangular layers;
- cached-basis equivalence;
- correction cadence and optimizer-state restart;
- a restartable end-to-end runner smoke test.

## Run the notebook

```bash
jupyter lab optimizers/full_matrix_log_rg/notebooks/MNIST_MLP3_SGD_Momentum_vs_FullMatrixLogRG.ipynb
```

The notebook is matched to the qualified baseline:

- MNIST with the fixed 55,000/5,000 optimization/validation split;
- `784 -> 512 -> 512 -> 10` ReLU MLP;
- SGD + Nesterov, peak LR `0.05`, floor `5e-4`;
- two-epoch linear warm-up followed by cosine decay;
- momentum `0.90`, matrix weight decay `1e-4`, gradient clipping `1.0`;
- three complete seeds and run-level 95% Student-t intervals.

### Validation-only grid

The Mac-default grid has eight points:

```text
mode                 radial, modewise
projection_strength  0.5, 1.0
apply_every_steps    25, 100
max_correction_ratio 0.10 (fixed)
```

It uses one preregistered seed and five epochs. The test set is not evaluated during selection. The candidates and their checkpoints are restartable under:

```text
$RG_FML_RUN_ROOT/validation_grid/
```

Useful controls:

```bash
export RG_FML_GRID_EPOCHS=5
export RG_FML_SKIP_GRID=1   # reuse an existing selected_config.json
```

### Final campaign

The selected configuration is compared with unmodified SGD + Nesterov for the same three baseline seeds. Every run writes:

- `checkpoint_latest.pt`;
- `checkpoint_best.pt`, selected only by validation loss;
- optional per-epoch checkpoints;
- `final_state.pt` and `run_complete.json`;
- performance, WeightWatcher, ESD, and correction histories.

Interrupted runs resume from `checkpoint_latest.pt`. Aggregate CSVs, plots, and 95% confidence intervals are written under:

```text
$RG_FML_RUN_ROOT/final/aggregate/
```

## Package API

```python
import torch

from full_matrix_log_rg import FullMatrixLogConfig, FullMatrixLogRG, analyze_supports

base = torch.optim.SGD(...)
optimizer = FullMatrixLogRG(
    base,
    model.named_parameters(),
    FullMatrixLogConfig(
        mode="modewise",
        projection_strength=1.0,
        max_correction_ratio=0.10,
        apply_every_steps=100,
        parameter_names=("fc1.weight", "fc2.weight"),
    ),
)

checkpoint = analyze_supports(
    model,
    epoch=0,
    parameter_names=("fc1.weight", "fc2.weight"),
)
optimizer.set_supports(checkpoint.supports)
```

Refresh the cached supports after each slower WeightWatcher checkpoint.

## Validation status

The pure-PyTorch geometry, projection, rectangular-matrix, cadence, state-restart, packaging, and restartable-runner tests are executable without downloading MNIST. The complete MNIST/WeightWatcher grid and three-seed campaign must be run locally; they are intentionally not represented as completed results in this PR.
