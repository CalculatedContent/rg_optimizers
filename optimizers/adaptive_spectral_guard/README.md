# AdaptiveSpectralGuard

`AdaptiveSpectralGuard` is the second optimizer variant in `rg_optimizers`.
It is independent of `trace_log_tracker` and combines a slow WeightWatcher
controller with a fast, completed-step correction around an ordinary PyTorch
optimizer such as AdamW.

The design addresses the failure mode seen in the first experiment: FC1
benefited from stronger spectral intervention, while applying the same
correction to FC2 on every step slowed or destabilized its convergence.

## What is new

### 1. Layer-specific policies

Every matrix layer has its own:

- enable/disable switch;
- correction cadence;
- weak and strong gain;
- trace-log correction cap;
- beta-shape correction cap;
- combined correction cap;
- minimum retained rank;
- loss-neutral safeguard.

The default MLP3 preset uses:

| Layer | Cadence | Role |
|---|---:|---|
| FC1 | every 2 steps | stronger branch protection plus a small shape channel |
| FC2 | every 10 steps | conservative intervention |
| FC3 | disabled | only ten singular values; no reliable shell-beta geometry |

Manual presets are included for `fc1_only`, `fc2_only`, and `adaptive`
(FC1+FC2), so the direct and indirect layer effects can be ablated.

### 2. WeightWatcher-driven hysteresis

At each epoch, WeightWatcher supplies `alpha`, `ERG_gap`, `detX_num`,
`num_pl_spikes`, and the retained midpoint rank. The controller has three
states:

- `off`;
- `weak`;
- `strong`.

It can turn on when alpha approaches the boundary or is falling rapidly, but
it turns off only after alpha has remained safely above the boundary for a
patience window. This avoids step-to-step chatter and treats alpha=2 as a
boundary rather than a point target.

`alpha` and `ERG_gap` are read directly from
`WeightWatcher.analyze(..., ERG=True)`; this package does not fit or
reconstruct either quantity.

### 3. ECS-confidence gate

The layer confidence is reduced when:

- the PL and ERG boundaries have little overlap;
- the midpoint retained rank changes sharply between epochs;
- `ERG_gap` is large relative to the retained support.

The intervention gain is

$$
g_{\ell,e}
=
g_{\ell,\mathrm{regime}}\,
C_{\ell,e}\,
Q_{\ell,e},
$$

where $C_{\ell,e}$ is the spectral/ECS confidence and $Q_{\ell,e}$ is a
task-conflict throttle.

### 4. One-sided retained-volume channel

For a WeightWatcher-normalized retained spectrum, the volume coordinate is

$$
T_R(W)
=
\frac{1}{m_R}
\sum_{i\in R}
\log \widetilde{\lambda}_i .
$$

Let $G_T=\nabla_W T_R$ and let $\Delta W_{\rm base}$ be the completed AdamW
step. The volume channel removes only contracting drift:

$$
\Delta W_T
=
-
g_{\ell,e}
\min\!\left(
\frac{\langle G_T,\Delta W_{\rm base}\rangle_F}
     {\|G_T\|_F^2},
0
\right)G_T .
$$

This channel is norm-capped independently for each layer.

### 5. Trace-log-preserving beta-E shape channel

The retained spectrum is divided into equal-width logarithmic shells. For
shell energies $E_k$ and shell centers $\Lambda_k$, the local shell slope is

$$
\beta_E
=
\frac{d\log E_k}{d\log \Lambda_k}.
$$

The implementation freezes the shell assignments for one local correction,
computes the analytic gradient $G_\beta=\nabla_W\beta_E$, and removes its
trace-log component:

$$
G_{\beta,\perp}
=
G_\beta
-
\frac{\langle G_\beta,G_T\rangle_F}
     {\|G_T\|_F^2}
G_T .
$$

When the WeightWatcher gate indicates the alpha<2 side and the shell geometry
is reliable, the shape channel moves toward beta-E=0:

$$
\Delta W_\beta
=
-
g_{\ell,e}\,\eta_{\beta,\ell}
\frac{\beta_E}{\|G_{\beta,\perp}\|_F^2}
G_{\beta,\perp}.
$$

Thus the new optimizer can alter spectral shape while preserving retained
trace-log volume to first order.

### 6. First-order task-loss safeguard

The task gradient is captured before AdamW applies its step. Let

$$
C_\ell=\Delta W_T+\Delta W_\beta.
$$

If $C_\ell$ would increase the current minibatch loss to first order, its
harmful gradient component is removed:

$$
C_\ell'
=
C_\ell
-
\frac{
\max\!\left(\langle\nabla_\ell L,C_\ell\rangle_F-a_\ell,0\right)
}{
\|\nabla_\ell L\|_F^2+\varepsilon
}
\nabla_\ell L,
$$

where $a_\ell$ is the optional allowed conflict budget (zero by default).

The optimizer logs:

- correction/AdamW angle;
- attempted and post-safeguard task-conflict ratios;
- fraction of the correction removed by the safeguard;
- per-layer correction magnitude and firing rate.

Persistent attempted task conflict reduces that layer's gain at the next
WeightWatcher checkpoint. This is intended to turn FC2 down automatically
when its spectral correction repeatedly opposes task descent.

## Experiment notebook

Open:

```text
notebooks/MNIST_MLP3_AdamW_vs_AdaptiveSpectralGuard_30Epochs.ipynb
```

The notebook runs a paired AdamW baseline and AdaptiveSpectralGuard experiment
for 30 epochs and includes:

- live per-epoch WeightWatcher and controller tables;
- train/test loss and accuracy;
- all-layer alpha, ERG-gap, and beta-E plots;
- layer-specific correction and task-conflict plots;
- test loss versus train loss;
- test accuracy versus train accuracy;
- layer alpha versus train loss;
- optional FC1-only and FC2-only ablation presets.

## Install and test

```bash
cd optimizers/adaptive_spectral_guard
python -m pip install -r requirements.txt
PYTHONPATH=. python -m unittest discover -s tests -v
```
