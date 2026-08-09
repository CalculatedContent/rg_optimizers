# Optimizer variants map

**Class B documentation:** this is a read-only catalog. It changes no optimizer
mathematics, defaults, hyperparameters, or efficacy claims.

Each directory under `optimizers/` is an **independent experiment package**
with its own implementation, notebooks, tests, and scientific hypothesis. The
columns below deliberately separate:

- the **actuator**: what component of the optimizer flow is changed;
- the **support or geometry**: which retained subspace or spectral coordinate
  defines that change; and
- the **cadence**: when the intervention is applied.

## Complete in-repository map

| Folder | Actuator: what changes | Support / spectral rule | Typical cadence and status |
|---|---|---|---|
| [`trace_log_tracker`](optimizers/trace_log_tracker/README.md) | Default `one_sided`: remove only the contracting component of the completed optimizer matrix displacement along the scalar trace-log normal. `tangent` and `tracking` are explicit alternatives. | WeightWatcher midpoint support, `floor((m_PL + m_TL) / 2)`, from the PL retained count and `detX_num`; the current retained singular subspace is recomputed while the cached rank is held fixed during a local correction. | Configured post-base-step cadence. Original full-`M` trace-log experiment. |
| [`self_consistent_trace_log_tracker`](optimizers/self_consistent_trace_log_tracker/README.md) | Same default one-sided trace-log-normal flow subtraction as the original tracker; expansion is unchanged. | Bulk-effective self-consistent ECS scan. Default support is the midpoint of the self-consistent ECS and PL ranks; `ecs`, `midpoint`, and `power_law` support policies are available. | Configured post-base-step cadence. Replaces the ECS estimator, not the basic actuator. |
| [`adaptive_spectral_guard`](optimizers/adaptive_spectral_guard/README.md) | Layer-specific one-sided retained-volume correction plus an optional trace-log-preserving `beta_E` shape channel, followed by a first-order task-loss safeguard. | WeightWatcher `alpha`, `ERG_gap`, midpoint support, hysteresis, and separate volume/shape confidence gates. `alpha = 2` is treated as a boundary rather than a point target. | Layer-specific. The Stabilized V2 MLP3 preset uses FC1 every 2 steps, FC2 every 10 steps, and disables FC3. |
| [`spectral_rg_flow_projector`](optimizers/spectral_rg_flow_projector/README.md) | Remove only the positive alignment of the completed centered log-spectrum displacement with an experimental participation-ratio collapse direction. | Bulk-effective self-consistent ECS with the ECS/PL midpoint as the default working support; the actuator operates in trace-log-free spectral-shape coordinates. | Configured post-base-step cadence. Experimental surrogate for flow toward a no-extensive-ECS branch. |
| [`ecs_probe_loss_trace_wall`](optimizers/ecs_probe_loss_trace_wall/README.md) | Add an Armijo-backtracked correction that lowers a rotating training-probe loss evaluated on the ECS-truncated network; the probe gradient is projected into the ECS. | Bulk-effective self-consistent ECS recomputed at every correction. Default projection is the ECS `core`; `rank_m_tangent` is an ablation. | Task-directed; default is one correction per epoch, beginning no earlier than the end of warmup. |
| [`wwpgd_local_delta`](optimizers/wwpgd_local_delta/README.md) | Fractionally damp only the component of the **completed epoch displacement** outside the local ECS. The accumulated weight matrix is not spectrally retracted. | Orientation-aware local bulk-effective self-consistent ECS. The epoch-end endpoint is the default reference; epoch-start is available. | Epoch boundary; defaults are every epoch, no warmup, and `correction_fraction = 0.25`. Optimizer-state synchronization is an explicit ablation. |
| [`full_matrix_log_rg`](optimizers/full_matrix_log_rg/README.md) | Primary `cone` mode: solve the minimum-norm active-set projection that removes first-order retained log-eigenvalue drift toward the trivial covariance condition, while rewriting the Nesterov momentum state to match the accepted step. `radial` is the conservative control; `modewise` is legacy. | Cached ECS/PL midpoint rank and retained basis. Self-consistent normalization is primary; full-`M` normalization is the required ablation. | Configured optimizer steps; default is every 100 steps. Primary full retained-covariance-flow experiment. |

The map describes implemented behavior. It does **not** identify a repository-wide
preferred optimizer or claim that any intervention improves generalization.
Within a folder, the package README and code are authoritative for exact defaults.

## Density exponent and rank-order exponent

For an ideal continuous power-law density tail

\[
\rho(\lambda) \propto \lambda^{-\alpha},
\qquad \alpha > 1,
\]

the descending ordered eigenvalues obey

\[
\lambda_{(r)} \propto r^{-\mu_{\mathrm{rank}}},
\]

with

\[
\mu_{\mathrm{rank}}
= \frac{1}{\alpha - 1},
\qquad
\alpha
= 1 + \frac{1}{\mu_{\mathrm{rank}}}.
\]

Thus

\[
\alpha = 2
\quad\Longleftrightarrow\quad
\mu_{\mathrm{rank}} = 1.
\]

This is an **exponent correspondence**, not independent evidence that a fitted
spectrum is at an RG fixed point. Finite spectra, truncated fitting windows,
finite-size corrections, correlated noise, and different fitting conventions
can make the empirical relationship approximate rather than exact.

Do not denote the rank-order exponent by `q` in this repository: `q` or `Q` is
commonly used for the matrix aspect ratio in Marchenko–Pastur analysis. Also do
not assume that every package uses or logs the same `alpha` fitting convention;
inspect the package README and recorded columns before comparing results.

## Dose and intervention-strength semantics

A cross-actuator “dose” is meaningful only when its **numerator, denominator,
and time window** are named. A per-step correction ratio is not directly
comparable to an epoch-level correction ratio.

### `wwpgd_local_delta` realized dose

Successful correction rows report

```text
actuator_id     = wwpgd_local_delta
ecs_backend     = self_consistent_local_geometry
dose_definition = removed_frobenius_over_base_epoch_delta_frobenius
dose_value      = removed_fraction_of_base
```

The realized dose is

\[
\mathrm{dose\_value}
=
\frac{\lVert \Delta_{\mathrm{removed}} \rVert_F}
     {\max(\lVert \Delta_{\mathrm{epoch}} \rVert_F,\epsilon)}
=
\eta\,
\frac{\lVert \Delta_{\perp} \rVert_F}
     {\max(\lVert \Delta_{\mathrm{epoch}} \rVert_F,\epsilon)},
\]

where `correction_fraction = eta` is the **requested fraction of the
outside-ECS component removed**. It is not itself the realized dose relative to
the full epoch displacement.

`dose_value` is `null` on skipped or failed rows where no correction was
committed. `is_first_apply` is evaluated against the first correction epoch
implied jointly by `warmup_epochs` and `apply_every_epochs`.

### Controls are not automatically realized doses

Fields such as `correction_fraction`, `projection_strength`,
`max_correction_ratio`, channel gains, and correction caps are control settings
or bounds. They should not be compared across folders as though they were the
same realized quantity. For cross-actuator studies, record at least:

1. the correction norm actually committed;
2. the reference norm used as the denominator;
3. whether the window is a step, minibatch, checkpoint interval, or epoch; and
4. whether optimizer state was changed together with the realized weight step.

External adapters and repositories should document their own fields where they
are implemented. This in-repository map does not assert undocumented telemetry
for external WW-PGD or nanoGPT adapters.

## Baseline separation

`baseline/` contains **no RG correction**. Optimizer folders are separate
experiments and are not drop-in replacements for the audited baseline recipes
without reading the relevant README, matching the base optimizer, and preserving
the paired experimental protocol.
