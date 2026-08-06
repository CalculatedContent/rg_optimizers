# Spectral RG-flow projector

This folder contains a new optimizer experiment for the RG-optimizers
repository.  It is deliberately separate from both:

- `trace_log_tracker`, which projects the completed optimizer displacement
  against a trace-log normal; and
- `self_consistent_trace_log_tracker`, which uses the improved adaptive ECS but
  still removes a trace-log-normal component.

The motivation is the observed failure of the adaptive trace-log tracker to
prevent FC1 from crossing into the `alpha < 2` regime.  The trace-log condition
fixes one retained-volume coordinate, but a large physical **shape flow can
remain tangent to that gauge slice**.  This experiment therefore works in
spectral-shape coordinates rather than projecting against the trace-log normal.

## What is source-derived and what is experimental

The RG draft identifies the weak-tail/trivial branch as the branch on which no
extensive retained ECS survives, while the correlated candidate fixed point has
nonzero retained spectral gain, `alpha ~= 2`, and ECS/PL alignment.  The draft
does not provide a unique differentiable local vector pointing toward the
trivial branch.

The new ingredient below is therefore an **operational experimental surrogate**:
within the current adaptive ECS, motion toward a rank-collapsed spectrum is
used as a continuous local proxy for motion toward the no-extensive-ECS branch.
The experiment tests this proxy; it does not assume it is the final RG beta
vector.

## Adaptive ECS

WeightWatcher remains the source of:

- the ESD through `watcher.get_ESD()`;
- the fitted power-law exponent `alpha`;
- the fitted PL retained count `num_pl_spikes`.

The ECS is recomputed with the bulk-effective self-consistent normalization
introduced in the preceding experiment.  For candidate retained rank `m`, the
discarded bulk participation-ratio count is

$$
r_{\mathrm{bulk}}(m)
=
\frac{\left(\sum_{i\in B_m}\lambda_i\right)^2}
     {\sum_{i\in B_m}\lambda_i^2},
$$

and

$$
D(m;\gamma)
=
m+r_{\mathrm{bulk}}(m)
+\gamma\left[(M-m)-r_{\mathrm{bulk}}(m)\right].
$$

The default is `gamma=0`.  The self-consistent ECS is selected from a zero of

$$
F(m)
=
\frac{1}{m}
\sum_{i\in R_m}
\log\left[
\frac{D(m;\gamma)}{\sum_j\lambda_j}\lambda_i
\right].
$$

The default optimizer support is still the conservative midpoint of the new
ECS and the WeightWatcher PL rank.

## Spectral coordinate

On the largest `m` singular values of a layer, define centered log-eigenvalue
coordinates

$$
z_i
=
\log s_i^2
-
\frac{1}{m}\sum_{j=1}^{m}\log s_j^2.
$$

Thus

$$
\sum_i z_i=0.
$$

These coordinates contain spectral shape but not global scale or the uniform
trace-log direction.

Let

$$
p_i
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
=
\frac{s_i^2}{\sum_{j=1}^{m}s_j^2}.
$$

The retained participation-ratio effective rank is

$$
r_{\mathrm{PR}}
=
\frac{1}{\sum_i p_i^2}.
$$

A rank-one collapse has `r_PR = 1`.  The default collapse potential is

$$
C_{F_0}(z)
=
\log\left(\sum_i p_i^2\right)
=
-\log r_{\mathrm{PR}}.
$$

Its local spectral vector is

$$
v_{F_0}
=
\nabla_z C_{F_0}
=
2\left(
\frac{p_i^2}{\sum_j p_j^2}-p_i
\right).
$$

This vector also sums to zero.  It is therefore a **shape direction tangent to
the trace-log gauge**, not the trace-log normal used by the preceding
optimizers.

## One-sided spectral-flow subtraction

The base optimizer first produces its complete matrix proposal.  From the
spectra before and after that proposal, compute

$$
\Delta z=z_{\mathrm{base}}-z_{\mathrm{before}}.
$$

The component toward the collapse/trivial surrogate is

$$
a_0^+
=
\max\left(
\frac{\langle\Delta z,v_{F_0}\rangle}
     {\lVert v_{F_0}\rVert^2},
0
\right).
$$

The corrected spectral displacement is

$$
\Delta z_{\mathrm{RG}}
=
\Delta z-a_0^+v_{F_0}.
$$

Only positive alignment is removed.  A base step that increases retained
effective rank or moves orthogonally to this vector is unchanged.

The correction is implemented as a finite multiplicative change to the base
proposal's retained singular values:

$$
s_i^{\mathrm{corr}}
=
s_i^{\mathrm{base}}
\exp\left[-\frac12 a_0^+v_{F_0,i}\right].
$$

The base proposal's singular vectors are retained.  A common rescaling restores
its Frobenius norm without changing the centered log-spectrum.  A correction
norm cap keeps the intervention subordinate to AdamW/SGD.

## Why this differs from the previous optimizer

The previous adaptive trace-log tracker removes a component normal to one
scalar retained-volume coordinate.  The new optimizer instead removes a
component in the `(m-1)`-dimensional spectral-shape tangent space.  It is not a
trace-log regularizer, not an alpha penalty, and not the WW-PGD rank-order
retraction.

## Package layout

```text
rg_spectral_flow/
  ecs.py              adaptive self-consistent ECS
  flow.py             spectral coordinate, F0 vector, finite projection
  wrapper.py          post-AdamW/SGD optimizer wrapper
  weightwatcher.py    WeightWatcher + adaptive-ECS outer loop
  mnist_experiment.py paired MNIST experiment
notebooks/
  MNIST_MLP3_AdamW_vs_SpectralRGFlowProjector.ipynb
tests/
```

## Minimal use

```python
import torch
from rg_spectral_flow import (
    SpectralRGFlowConfig,
    SpectralRGFlowProjector,
    analyze_weightwatcher_checkpoint,
)

base = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = SpectralRGFlowProjector(
    base,
    model.named_parameters(),
    config=SpectralRGFlowConfig(
        collapse_potential="participation_ratio",
        projection_strength=1.0,
        max_correction_ratio=0.10,
        apply_every_steps=25,
    ),
)

checkpoint = analyze_weightwatcher_checkpoint(
    model,
    run_label="SpectralRGFlow",
    epoch=0,
)
optimizer.set_support_states(checkpoint.supports, replace=True)
```

Refresh the support states after each slower WeightWatcher checkpoint.

## Primary falsification test

The experiment should be judged primarily on FC1:

1. Does the new optimizer keep FC1 closer to `alpha = 2` than matched AdamW?
2. Does `ERG_gap_SC` remain coherent?
3. When a correction is applied, does
   `corrected_flow_component` move toward zero?
4. Does the intervention preserve test accuracy and remain small relative to
   the completed AdamW step?

If FC1 still falls below two while the measured F0 component is removed, this
participation-ratio collapse vector is not the missing RG direction.

## Tests

```bash
cd optimizers/spectral_rg_flow_projector
PYTHONPATH=. python -m unittest discover -s tests -v
```
