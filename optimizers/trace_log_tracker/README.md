# Trace-log RG component removal

This folder contains the first optimizer variant in `rg_optimizers`.

The extension wraps an ordinary PyTorch optimizer and modifies the **completed matrix displacement** after AdamW/SGD has applied momentum, adaptive preconditioning, clipping, learning rate, and weight decay. It does not truncate the layer and does not project the weights onto an ECS matrix.

## Working ECS

A slow WeightWatcher outer loop selects a working retained rank

\[
m_R = \left\lfloor \frac{m_{\rm PL}+m_{\rm TL}}{2}\right\rfloor,
\]

where `m_PL = num_pl_spikes` and `m_TL = detX_num`. The working ECS may be larger than the eventual fixed-point ECS. Only its retained **rank** is cached between checkpoints; the current retained singular subspace is recomputed for each local correction and frozen during that correction.

## Trace-log normal

For an oriented layer matrix \(W\in\mathbb R^{N\times M}\), \(N\ge M\), and retained right singular vectors \(V_R\), define

\[
X_R(W)=V_R^\top\left(\frac{1}{N}W^\top W\right)V_R.
\]

The raw pre-gauge coordinate is

\[
T_R(W)=\frac{1}{m_R}\operatorname{Tr}\log X_R(W).
\]

Its local normal is

\[
G_T=\nabla_W T_R
   =\frac{2}{m_RN}WV_RX_R^{-1}V_R^\top.
\]

The implementation also supports the WeightWatcher/Frobenius-normalized coordinate

\[
\widetilde\lambda_i=
\frac{M\lambda_i}{\sum_j\lambda_j},
\qquad
\widetilde T_R=\frac1{m_R}\sum_{i\in R}\log\widetilde\lambda_i,
\]

whose gradient includes the radial subtraction

\[
\nabla_W\widetilde T_R
=
\frac{2}{m_RN}WV_RX_R^{-1}V_R^\top
-
\frac{2}{\lVert W\rVert_F^2}W.
\]

This is the default because it uses the same scale convention as WeightWatcher's ERG boundary.

## Three correction modes

Let the base optimizer propose the actual displacement \(\Delta W\), and define

\[
d=\langle G_T,\Delta W\rangle_F.
\]

### `tangent`

Remove all first-order trace-log drift:

\[
\Delta W_{\perp}
=
\Delta W-
\frac{d}{\lVert G_T\rVert_F^2}G_T.
\]

### `one_sided` — default experiment

Remove only contracting drift:

\[
a^- = \min\left(\frac{d}{\lVert G_T\rVert_F^2},0\right),
\qquad
\Delta W_{\not\to F_0}=\Delta W-a^-G_T.
\]

This is the conservative branch-protection test: it prevents first-order return flow toward the weak-tail/trivial branch without assuming that the approximate ECS has converged.

### `tracking`

Actively contract the current trace-log residual:

\[
\Delta W_{\rm track}
=
\Delta W-
\frac{d+\gamma T_R(W)}{\lVert G_T\rVert_F^2}G_T.
\]

To first order, \(T_R\mapsto(1-\gamma)T_R\).

Every mode includes a relative correction-norm cap, configurable correction cadence, and singular-value regularization.

## Experiment notebook

Open:

```text
notebooks/MNIST_MLP3_AdamW_vs_TraceLogRG.ipynb
```

It trains identical `784 -> 512 -> 512 -> 10` ReLU MLPs on the same MNIST batches:

1. AdamW baseline.
2. AdamW plus the trace-log RG wrapper.

At initialization and every epoch, it installs/runs WeightWatcher and records:

- WeightWatcher power-law exponent `alpha`;
- `detX_num`;
- `num_pl_spikes`;
- `ERG_gap = detX_num - num_pl_spikes`;
- midpoint retained rank;
- midpoint trace-log residual;
- logarithmic-shell scale-balance slope `beta_E`;
- full shell-energy RMS and adjacent-shell RG residual;
- RG correction size and predicted trace-log drift.

The WeightWatcher analysis updates the RG model's cached midpoint ranks for the next epoch.

## Run tests

```bash
cd optimizers/trace_log_tracker
PYTHONPATH=. python -m unittest discover -s tests -v
```

## Provenance fields on step stats

`pop_step_stats()` rows include logging-only provenance fields aligned with the
local-delta package grammar: `actuator_id`, `ecs_backend`, `dose_definition`,
`dose_value` (null when no correction applied), and schedule-aware
`is_first_apply`. These fields do **not** change correction mathematics.
