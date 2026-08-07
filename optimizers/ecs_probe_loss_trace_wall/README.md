# ECS Probe-Loss TraceWall

This folder contains an independent optimizer experiment for the standard
MNIST MLP3 model.  It does **not** modify the existing trace-log, adaptive
spectral guard, spectral-flow projector, or local-delta WW-PGD implementations.

The method replaces the previous objective of suppressing flow toward a
putative trivial fixed point with a directly testable task objective measured
on a rotating subset of the **training** data.

## Definition

For every selected matrix after a completed base-optimizer step, compute

\[
W = U\Sigma V^\top,
\qquad
W_{\mathrm{ECS}} = U_m\Sigma_m V_m^\top,
\]

where the retained rank \(m\) is the current bulk-effective,
self-consistent trace-log ECS.  All selected matrices are truncated
simultaneously.  On a rotating training probe subset \(B_t\), the optimizer
measures

\[
\mathcal L_{\mathrm{probe}}(t)
=
\frac{1}{|B_t|}
\sum_{(x,y)\in B_t}
\ell\!\left(f_{W_{\mathrm{ECS}}}(x),y\right).
\]

It differentiates this loss at the truncated model, projects each matrix
gradient into the same retained singular subspaces,

\[
G_{\mathrm{ECS}}
=
(U_mU_m^\top)\,G\,(V_mV_m^\top),
\]

and proposes a negative-gradient component.  The completed update is

\[
W_{t+1}
=
W^{\mathrm{base}}_{t+1}
+
a_t\,\Delta W_{\mathrm{probe,ECS}},
\]

where \(a_t\) is selected by Armijo backtracking on the same ECS-truncated
probe objective.  A correction is committed only when it lowers that objective.
The ECS SVD and rank are recomputed at every correction, so the task-loss
channel follows the ECS if its support contracts or expands during training.

The default projection is the strict ECS core shown above.  A rank-\(m\)
manifold tangent projection is available as an explicit ablation.

## Rotating probe protocol

- The probe is drawn only from the MNIST training set.
- The official MNIST test set is used only for reporting test loss and accuracy.
- Probe selection uses an independent seeded random permutation.
- New slices are consumed without replacement; after a complete pass, a new
  permutation is generated.
- A draw that crosses a permutation boundary is still unique within that draw.
- The primary notebooks use 512 examples per correction (two batches of 256)
  and one correction at each epoch boundary.

This avoids tuning directly on the official test set while approximating the
expected loss of a changing random training probe.

## Paired experiment

Both notebooks train a clean baseline and a TraceWall arm in the same run.
For every seed, the two arms:

- start from byte-identical weights;
- receive the same minibatches in the same order;
- use the same gradient clipping;
- use the same optimizer hyperparameters;
- use the same one-epoch linear warmup and cosine decay to 5% of the peak
  learning rate.

Only the TraceWall arm receives the post-step ECS probe-loss component.

Repository-standard peak settings are retained:

- AdamW: learning rate `1e-3`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight
  decay `1e-2`;
- SGD with classical momentum: learning rate `5e-2`, momentum `0.9`, zero
  dampening, no Nesterov, weight decay `1e-4`.

The MLP is `784 -> 512 -> 512 -> 10` with ReLU activations and no dropout or
batch normalization.  Each notebook runs three independent seeds for 20 epochs
and reports two-sided 95% Student-t confidence intervals across complete runs.

## Notebooks

- `notebooks/MNIST_MLP3_AdamW_vs_ECS_Probe_Loss_TraceWall.ipynb`
- `notebooks/MNIST_MLP3_SGD_Momentum_vs_ECS_Probe_Loss_TraceWall.ipynb`

Each notebook records and saves:

- full train/test cross-entropy, accuracy, and classification perplexity;
- learning rate, parameter norm, and epoch timing;
- self-consistent ECS rank, trace-log residual, adaptive normalization,
  retained energy, stable rank, and participation ratio;
- WeightWatcher alpha, `detX_num`, `num_pl_spikes`, and `ERG_gap`;
- every probe loss before/after correction;
- line-search scale, acceptance, correction norms, ECS ranks, and numerical
  projection audits;
- baseline and TraceWall checkpoints after every epoch.

By default outputs are written beneath `runs/`.  Set
`RG_TRACE_WALL_RUN_ROOT` and `RG_TRACE_WALL_DATA_DIR` to redirect experiment
artifacts and the MNIST cache.

## Tests

From this folder:

```bash
python -m unittest discover -s tests -v
```

The tests cover scale-invariant ECS selection, SVD truncation, ECS projection,
rotating-probe uniqueness and checkpoint restoration, optimizer loss descent,
warmup/cosine scheduling, a paired synthetic end-to-end run, plotting, and
notebook validity.
