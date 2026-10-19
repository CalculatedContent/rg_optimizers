# Self-consistent ECS trace-log RG tracker

This folder contains a new version of the original trace-log optimizer.  It
keeps the original purpose and intervention rule—remove redundant optimizer
flow back toward the weak-tail/trivial branch—but replaces the old
WeightWatcher full-`M` ECS boundary with the **bulk-effective,
self-consistent ECS normalization** developed in the accompanying diagnostic
notebook.

The implementation is deliberately separate from
`optimizers/trace_log_tracker` so the old and new estimators can be compared
without changing the original experiment.

## What changed

The old tracker reads `detX_num` from `watcher.analyze(ERG=True)`.  That value
uses the full spectral dimension `M` in the Frobenius normalization.  The new
checkpoint analyzer still uses WeightWatcher as the sole source of:

- the ESD through `watcher.get_ESD()`;
- the fitted exponent `alpha`;
- the PL retained count `num_pl_spikes`.

It then recomputes the ECS itself.

For a candidate retained rank `m`, let `B_m` be the discarded small-eigenvalue
bulk.  The default effective contributor count is its participation ratio,

$$
r_{\mathrm{bulk}}(m)
=
\frac{\left(\sum_{i\in B_m}\lambda_i\right)^2}
     {\sum_{i\in B_m}\lambda_i^2}.
$$

The adaptive normalization dimension is

$$
D(m;\gamma)
=
m+r_{\mathrm{bulk}}(m)
+\gamma\left[(M-m)-r_{\mathrm{bulk}}(m)\right],
\qquad 0\leq\gamma\leq1.
$$

The default is `gamma=0`; `gamma=1` exactly restores the old full-`M`
normalization.  Every integer candidate is scanned using

$$
F(m)
=
\frac{1}{m}
\sum_{i\in R_m}
\log\left[
\frac{D(m;\gamma)}{\sum_j\lambda_j}\lambda_i
\right].
$$

The selected self-consistent ECS is the integer adjacent to a zero crossing
of `F(m)`, or the minimum-absolute-residual candidate when the finite spectrum
has no crossing.  The new diagnostic gap is

$$
\mathrm{ERG\_gap}_{\mathrm{SC}}
=
m_{\mathrm{ECS}}^{\mathrm{SC}}-m_{\mathrm{PL}}.
$$

`detX_num_WW` and `ERG_gap_WW` are retained only as audit columns.

## Optimizer action

The base AdamW/SGD optimizer first proposes its complete matrix displacement
`delta_W`, after momentum, adaptive preconditioning, learning rate, clipping,
and weight decay.

As in the original tracker, the default working support is the midpoint

$$
m_R=\left\lfloor\frac{m_{\mathrm{PL}}+m_{\mathrm{ECS}}^{\mathrm{SC}}}{2}\right\rfloor.
$$

On that cached working support, define the mean trace-log coordinate

$$
T_R(W)
=
\frac{1}{m_R}
\sum_{i\in R}
\log\left[
\frac{D_R\,s_i^2}{\lVert W\rVert_F^2}
\right].
$$

With the checkpoint-estimated `D_R` held fixed during a local correction, its
normal is

$$
G_T
=
\frac{2}{m_R}U_R\,\mathrm{diag}(s_i^{-1})V_R^\top
-
\frac{2}{\lVert W\rVert_F^2}W.
$$

Let

$$
d=\langle G_T,\Delta W\rangle_F.
$$

The default `one_sided` mode removes only contraction:

$$
a^-
=
\min\left(\frac{d}{\lVert G_T\rVert_F^2},0\right),
\qquad
\Delta W_{\not\to F_0}
=
\Delta W-a^-G_T.
$$

Expansion is unchanged.  This is branch protection, not a regularizer and not
an exact projection of every checkpoint onto `T_R=0`.

## Conservative default architecture

The default path mirrors the original tracker:

1. An outer WeightWatcher checkpoint estimates `m_ECS_SC`, `D_SC`, and `m_PL`.
2. It forms the original conservative midpoint from the new ECS and PL ranks.
3. The wrapper caches that adaptive support state for the next epoch.
4. At each configured correction step, it recomputes the current retained
   singular subspace but holds the cached rank and normalization dimension
   fixed during the local differential.
5. It removes only the contracting trace-log-normal component.

Optional live ECS refresh and differentiation of `D(W)` are implemented for
experiments, but are **not** the defaults.  They change more than the ECS
estimator and should be treated as separate ablations.

## Package layout

```text
rg_sc_trace_log/
  ecs.py              adaptive ECS scan and normalization
  geometry.py         trace-log residual, gradient, and one-sided correction
  wrapper.py          post-AdamW/SGD optimizer wrapper
  weightwatcher.py    WeightWatcher ESD/alpha + recomputed SC gap
  mnist_experiment.py paired AdamW versus SC-TraceLogRG experiment
notebooks/
  MNIST_MLP3_AdamW_vs_SelfConsistentTraceLogRG.ipynb
tests/
```

## Minimal use

```python
import torch
from rg_sc_trace_log import (
    SelfConsistentTraceLogConfig,
    SelfConsistentTraceLogRGWrapper,
    analyze_weightwatcher_checkpoint,
)

base = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = SelfConsistentTraceLogRGWrapper(
    base,
    model.named_parameters(),
    config=SelfConsistentTraceLogConfig(
        mode="one_sided",
        support_policy="midpoint",
        effective_rank_method="participation_ratio",
        normalization_gamma=0.0,
        refresh_ecs_every_steps=0,
    ),
)

checkpoint = analyze_weightwatcher_checkpoint(
    model,
    run_label="SC-TraceLogRG",
    epoch=0,
)
optimizer.set_support_states(checkpoint.supports, replace=True)
```

After each later outer checkpoint, call `set_support_states(...)` again.

## Provenance logging (step stats)

`pop_step_stats()` rows include logging-only provenance fields aligned with the
local-delta package grammar and the midpoint `trace_log_tracker` extension:

- `actuator_id = self_consistent_trace_log_tracker`
- `ecs_backend = self_consistent_F_m` (bulk-effective participation-ratio / `F(m)` lineage — **not** a free-fit MP edge label)
- `dose_definition = correction_frobenius_over_base_step_delta_frobenius`
- `dose_value` (null when no correction applied)
- `is_first_apply` (first **successful** correction per parameter)
- `is_first_due` (first schedule-due clock step; may have null dose)

These fields do **not** change correction mathematics. See
[`docs/PROVENANCE_INVENTORY.md`](../../docs/PROVENANCE_INVENTORY.md).

## Run the tests

```bash
cd optimizers/self_consistent_trace_log_tracker
PYTHONPATH=. python -m unittest discover -s tests -v
```
