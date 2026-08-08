# Optimizer variants map

**Class B docs:** read-only catalog. No default changes.  
Each folder under `optimizers/` is an **independent** experiment package (own notebooks/tests).

| Folder | Actuator (one line) | ECS / spectral rule | Typical cadence |
|---|---|---|---|
| `trace_log_tracker` | Remove one-sided contracting flow along WW **midpoint** ECS | Midpoint of PL / detX-class support | Post base step |
| `self_consistent_trace_log_tracker` | Same action class; ECS via **self-consistent** bulk-effective solve | Self-consistent F(m) / support_policy ∈ {ecs, midpoint, power_law} | Post base step |
| `adaptive_spectral_guard` | Layer-specific cadence/caps; α=2 as **boundary** not point target | WW hysteresis + volume/shape channels | Layer-dependent |
| `spectral_rg_flow_projector` | Project in centered log-spectrum shape space | Experimental “no extensive ECS” proxy | Experimental |
| `ecs_probe_loss_trace_wall` | Line-search loss-decreasing component in ECS direction | Self-consistent ECS + train probe | Task-directed |
| `wwpgd_local_delta` | Damp completed **epoch** displacement outside local ECS | Self-consistent local geometry (`ecs_backend` in logs) | Epoch end |

## Dual-label note (density α vs rank q)

If you compare to density-law α (public HTSR / nanogpt-experiments `target_alpha`):

\[
q = \frac{1}{\alpha - 1}, \qquad \alpha = 1 + \frac{1}{q}
\]

Fixed point: α = 2 ⇔ q = 1.  
Do **not** assume every package uses the same α convention without checking logs.

## Dose note

Different actuators use different “strength” metrics. Prefer named fields:

| Actuator | Dose field (when present) |
|---|---|
| nanogpt stock WW-PGD adapter | `dose_relative_frobenius` / `relative_frobenius_change_applied` |
| `wwpgd_local_delta` | `dose_value` = `removed_fraction_of_base` (outside-ECS fraction of epoch Δ) |

## Baseline separation

`baseline/` has **no** RG correction. Optimizer folders are separate experiments, not drop-in replacements for baseline recipes without reading each README.
