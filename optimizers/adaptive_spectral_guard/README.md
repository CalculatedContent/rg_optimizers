# AdaptiveSpectralGuard

`AdaptiveSpectralGuard` is the second optimizer variant in `rg_optimizers`.
It is independent of `trace_log_tracker` and combines a slow WeightWatcher
controller with a fast, completed-step correction around an ordinary PyTorch
optimizer such as AdamW.

## Which notebook to run

Use the fail-fast Stabilized V2 notebook for the current experiment:

```text
notebooks/MNIST_MLP3_AdamW_vs_AdaptiveSpectralGuard_StabilizedV2_30Epochs.ipynb
```

The notebook clears stale package modules from the Jupyter kernel, imports the
local repository copy, requires `STABILIZED_V2_API >= 2`, and runs a synthetic
low-confidence, alpha-below-two preflight before training. It stops with an
error rather than silently running the original all-or-nothing confidence
controller.

The older notebook is retained only as the V1 experimental record:

```text
notebooks/MNIST_MLP3_AdamW_vs_AdaptiveSpectralGuard_30Epochs.ipynb
```

## Design

The optimizer addresses the failure mode seen in the first trace-log
experiment: FC1 benefited from stronger spectral intervention, while applying
the same correction to FC2 on every step slowed or destabilized its
convergence.

### Layer-specific policies

Every matrix layer has its own:

- enable/disable switch;
- correction cadence;
- weak and strong gain;
- trace-log correction cap;
- beta-shape correction cap;
- combined correction cap;
- minimum retained rank;
- loss-neutral safeguard.

The Stabilized V2 MLP3 preset uses:

| Layer | Cadence | Shape cap | Role |
|---|---:|---:|---|
| FC1 | every 2 steps | 2% | stronger branch protection plus a bounded shape channel |
| FC2 | every 10 steps | 0.75% | conservative intervention |
| FC3 | disabled | — | only ten singular values; unreliable shell-beta geometry |

### WeightWatcher-driven hysteresis

At each epoch, WeightWatcher supplies `alpha`, `ERG_gap`, `detX_num`,
`num_pl_spikes`, and the retained midpoint rank. The controller has `off`,
`weak`, and `strong` states and treats alpha=2 as a boundary rather than a point
target.

`alpha` and `ERG_gap` are read directly from
`WeightWatcher.analyze(..., ERG=True)`; the optimizer does not fit or
reconstruct either quantity.

### Separate volume and shape confidence

Stabilized V2 smooths ECS confidence over checkpoints and separates the
trace-log volume confidence from the beta-shape confidence. A poor shape
checkpoint may veto the beta channel without disabling all trace-log branch
protection below the alpha boundary.

The channel gains are

$$
g_{\ell,e}^{(T)}
=
g_{\ell,e}^{\rm base} C_{\ell,e}^{(T)} Q_{\ell,e},
\qquad
g_{\ell,e}^{(\beta)}
=
g_{\ell,e}^{\rm base} C_{\ell,e}^{(\beta)} Q_{\ell,e}.
$$

### One-sided retained-volume channel

For a WeightWatcher-normalized retained spectrum,

$$
T_R(W)
=
\frac{1}{m_R}
\sum_{i\in R}\log \widetilde{\lambda}_i .
$$

Let $G_T=\nabla_W T_R$ and let $\Delta W_{\rm base}$ be the completed AdamW
step. The volume channel removes only contracting drift:

$$
\Delta W_T
=
-
g_{\ell,e}^{(T)}
\min\!\left(
\frac{\langle G_T,\Delta W_{\rm base}\rangle_F}
     {\|G_T\|_F^2},
0
\right)G_T .
$$

### Trace-log-preserving beta-E shape channel

The retained spectrum is divided into equal-width logarithmic shells. For
shell energies $E_k$ and shell centers $\Lambda_k$,

$$
\beta_E
=
\frac{d\log E_k}{d\log \Lambda_k}.
$$

The analytic frozen-shell gradient is orthogonalized against the trace-log
gradient:

$$
G_{\beta,\perp}
=
G_\beta
-
\frac{\langle G_\beta,G_T\rangle_F}
     {\|G_T\|_F^2}G_T .
$$

When the WeightWatcher and confidence gates permit it, the shape channel moves
the beta excess toward zero while preserving retained trace-log volume to first
order. Stabilized V2 includes a beta deadband and lower per-layer shape caps.

### First-order task-loss safeguard

Let

$$
C_\ell=\Delta W_T+\Delta W_\beta.
$$

If the proposed correction would increase the current minibatch loss to first
order, its harmful task-gradient component is removed. Persistent attempted
conflict also reduces that layer's gain at the next WeightWatcher checkpoint.

The optimizer logs correction/AdamW angle, attempted and post-safeguard task
conflict, safeguard removal fraction, channel-specific gains, and per-layer
correction magnitude.

## Install and test

```bash
cd optimizers/adaptive_spectral_guard
python -m pip install -r requirements.txt
PYTHONPATH=. python -m unittest discover -s tests -v
```
