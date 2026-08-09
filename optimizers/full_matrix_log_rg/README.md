# Full Matrix-Log RG optimizer

This folder implements optimizer extensions that remove SGD flow back toward the retained trivial covariance condition

\[
\widetilde X_R=I.
\]

For a frozen retained support of rank \(m\),

\[
\Phi_R(W)=\frac{1}{2m}\left\|\log \widetilde X_R(W)\right\|_F^2,
\qquad
\ell_i=\log\widetilde\lambda_i.
\]

The accumulated weight matrix is never whitened or projected onto the trivial state. The intervention acts on the optimizer flow.

## Scientific hierarchy

### 1. Primary: active-set cone projection with projected momentum

The proposed additive step \(\Delta W\) has retained log-eigenvalue drifts

\[
b_i=D\ell_i[\Delta W].
\]

The corrected implementation solves the minimum-norm cone projection

\[
\min_C\frac12\|C\|_F^2
\quad\text{subject to}\quad
\operatorname{sign}(\ell_i)
\left[b_i+D\ell_i[C]\right]\ge0.
\]

Only constraints that are actually violated enter the active set. Previously outward modes are allowed to change as long as they remain outward. This is the default `mode="cone"`.

With `momentum_projection="projected_state"`, the wrapper executes the SGD/Nesterov step itself. After projecting the proposed displacement, it rewrites the momentum buffer so rejected flow is not carried into later iterations. If

\[
\Delta W_{\rm raw}=-\eta(g+\mu v_{\rm raw}),
\qquad
\Delta W_{\rm accepted}=\Delta W_{\rm raw}+C,
\]

then the stored Nesterov buffer is

\[
v_{\rm accepted}=v_{\rm raw}-\frac{C}{\eta\mu}.
\]

Thus the applied step and stored optimizer state remain exactly consistent.

### 2. Conservative baseline: radial potential projection

`mode="radial"` removes only the net first-order decrease of \(\Phi_R\). It is basis invariant and remains the conservative scientific control.

### 3. Legacy ablations

- `mode="modewise"` retains the earlier exactly-targeted modewise linear solve. It is not the primary full-matrix projector because it overconstrains outward modes.
- `momentum_projection="post_step"` retains the original generic wrapper, which changes the completed weights but leaves the base momentum state unchanged.

## Normalization ablation

The same retained basis and rank can be evaluated with two normalization dimensions.

### Full-`M`

\[
D_R=M.
\]

This reproduces the original WeightWatcher/Frobenius convention.

### Bulk-effective self-consistent \(D_R\)

For the discarded bulk \(B_m\), let \(r_{\rm bulk}\) be its effective contributor count. The default uses participation ratio:

\[
r_{\rm bulk}
=
\frac{\left(\sum_{i\in B_m}\lambda_i\right)^2}
     {\sum_{i\in B_m}\lambda_i^2}.
\]

Then

\[
D_R
=
m+r_{\rm bulk}
+\gamma\left[(M-m)-r_{\rm bulk}\right].
\]

The primary setting is `normalization="self_consistent"`, participation ratio, and `normalization_gamma=0`. `normalization="full_m"` is the required ablation.

The slower epoch-level support object caches the complete singular-value spectrum, so both normalizations can be compared without another layer SVD.

## Mac-oriented execution

1. WeightWatcher runs at epoch zero and once per epoch.
2. The ECS/PL midpoint rank and right singular basis are cached.
3. Inner correction steps diagonalize only the retained \(m\times m\) covariance.
4. Unsupported MPS SVD, eigensolve, and linear-solve operations fall back to CPU.
5. The default correction cadence is every 100 optimizer steps and targets `fc1.weight` and `fc2.weight`.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e './baseline[experiment]'
python -m pip install -e './optimizers/full_matrix_log_rg[experiment]'
```

Store datasets and restartable runs outside `/tmp`:

```bash
export RG_BASELINE_DATA_DIR="$HOME/rg-optimizer-data"
export RG_FML_RUN_ROOT="$HOME/rg-optimizer-runs/full_matrix_log_rg"
```

## Tests

```bash
cd optimizers/full_matrix_log_rg
python -m unittest discover -s tests -v
cd ../..
```

The tests cover finite-difference geometry, mixed inward/outward active sets, KKT feasibility, full-`M` versus self-consistent normalization, exact SGD equivalence when no correction is due, projected Nesterov-state consistency, rectangular matrices, checkpoint restart, packaging, and a restartable experiment smoke test.

## Notebook

```bash
jupyter lab optimizers/full_matrix_log_rg/notebooks/MNIST_MLP3_SGD_Momentum_vs_FullMatrixLogRG.ipynb
```

The primary validation grid compares:

```text
projection mode        cone, radial
normalization          self_consistent, full_m
apply_every_steps      25, 100
projection_strength    1.0
max_correction_ratio   0.10
momentum projection    projected_state
```

The legacy `modewise` and `post_step` paths remain available for explicit ablations but are excluded from the primary grid.

The grid is validation-only. The selected configuration is then compared with unmodified SGD + Nesterov over the same three baseline seeds, with run-level 95% Student-t intervals and direct WeightWatcher `alpha`, `ERG_gap`, and `num_traps` measurements.

## API

```python
import torch

from full_matrix_log_rg import (
    FullMatrixLogConfig,
    FullMatrixLogProjectedSGD,
    analyze_supports,
)

base = torch.optim.SGD(...)
optimizer = FullMatrixLogProjectedSGD(
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

Refresh the cached supports after each slower WeightWatcher checkpoint.
