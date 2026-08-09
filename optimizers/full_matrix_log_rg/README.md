# Full Matrix-Log RG optimizer

This folder implements a state-aware optimizer extension that removes SGD flow back toward the retained trivial covariance condition

\[
\widetilde X_R = I.
\]

For the retained ECS/PL midpoint support,

\[
\Phi_R(W)=\frac{1}{2m}\left\|\log \widetilde X_R(W)\right\|_F^2.
\]

## Scientific hierarchy

The three implementations are deliberately separated:

1. **`cone` — corrected primary method.** It solves the active-set minimum-norm quadratic program

   \[
   \min_C \frac12\|C\|_F^2
   \quad\text{subject to}\quad
   \operatorname{sign}(\ell_i)\,[\dot\ell_i + D\ell_i(C)]\ge 0,
   \]

   where \(\ell_i=\log\widetilde\lambda_i\). The accepted correction is also written back into the SGD/Nesterov momentum buffer, so rejected flow is not proposed again on the next step.

2. **`radial` — conservative reference.** It removes only net first-order decrease of \(\Phi_R\).

3. **`modewise` — legacy prototype.** It retains the original regularized full Gram-system equality solve for reproducibility. It is not the primary scientific claim.

The momentum-state ablation is explicit:

- `momentum_projection="projected_state"` is the corrected implementation;
- `momentum_projection="post_step"` reproduces the original weight-only wrapper.

## Normalization ablation

Every cached support stores both:

- `normalization="full_m"`: \(D_R=M\), the original WeightWatcher/full-dimension normalization;
- `normalization="self_consistent"`:

  \[
  D_R=m+r_{\mathrm{bulk}},
  \qquad
  r_{\mathrm{bulk}}
  =\frac{(\sum_{i\in B}\lambda_i)^2}{\sum_{i\in B}\lambda_i^2}.
  \]

The retained basis is computed once per WeightWatcher checkpoint, so switching the normalization does not add an inner-loop full SVD.

## Computational structure

1. WeightWatcher runs at epoch zero and once per epoch.
2. It supplies `alpha`, `detX_num`, `num_pl_spikes`, `ERG_gap`, and `num_traps`.
3. The working rank is the midpoint of the Trace-Log and power-law ranks.
4. The retained right singular basis and both normalization dimensions are cached.
5. At each configured correction step, only the retained \(m\times m\) covariance is diagonalized.

The extension never whitens or projects the accumulated weight matrix. It filters the completed optimizer displacement.

## Install on a Mac

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e './baseline[experiment]'
python -m pip install -e './optimizers/full_matrix_log_rg[experiment]'
```

Store data and restartable results outside `/tmp`:

```bash
export RG_BASELINE_DATA_DIR="$HOME/rg-optimizer-data"
export RG_FML_RUN_ROOT="$HOME/rg-optimizer-runs/full_matrix_log_rg"
```

MPS is selected automatically when available. Unsupported decomposition operations fall back to CPU.

## Validate

```bash
cd optimizers/full_matrix_log_rg
python -m unittest discover -s tests -v
cd ../..
```

The tests cover the matrix-log gradient, active-set KKT projection, mixed inward/outward modes, normalization ablation, Nesterov momentum-state projection, rectangular layers, and checkpoint state round trips.

## Corrected notebook

Open:

```bash
jupyter lab optimizers/full_matrix_log_rg/notebooks/MNIST_MLP3_SGD_Momentum_vs_FullMatrixLogRG.ipynb
```

The validation-only grid compares the corrected cone optimizer under:

```text
normalization       full_m, self_consistent
projection_strength 0.5, 1.0
apply_every_steps   25, 100
```

The official test set is not evaluated during hyperparameter selection. The selected configuration is then compared with the unmodified SGD+Nesterov baseline over the same three final seeds with run-level 95% Student-t intervals.

The notebook now uses the corrected hierarchy; `modewise` and `post_step` remain available only as explicit legacy ablations.

## Minimal API

```python
import torch
from full_matrix_log_rg import FullMatrixLogConfig, FullMatrixLogRG, analyze_supports

base = torch.optim.SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    nesterov=True,
)
optimizer = FullMatrixLogRG(
    base,
    model.named_parameters(),
    FullMatrixLogConfig(
        mode="cone",
        momentum_projection="projected_state",
        normalization="self_consistent",
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

Refresh the supports after each slower WeightWatcher checkpoint.
