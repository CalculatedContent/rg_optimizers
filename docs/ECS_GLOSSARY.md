# ECS vocabulary glossary

**Class B documentation.** This page names objects already used in this repository.
It does not change optimizer mathematics, defaults, hyperparameters, or efficacy claims.

For **which package uses which actuator and cadence**, see
[`OPTIMIZER_VARIANTS.md`](../OPTIMIZER_VARIANTS.md). This glossary is about **what
“ECS” means** when those packages speak of support, normals, and local geometry.

---

## 1. Why a glossary

Several experiment packages correct optimizer flow using a spectral **effective
correlation subspace** (ECS) idea, but they do **not** all compute the same
object:

| Lineage | Typical package | Short idea |
|---|---|---|
| Midpoint / PL–detX support | `trace_log_tracker` | Support from WeightWatcher PL and detX-style counts (midpoint class) |
| Self-consistent bulk-effective | `self_consistent_trace_log_tracker`, parts of `wwpgd_local_delta` | Recompute a bulk-effective support via a self-consistent scan / \(F(m)\)-class solve |
| Layer hysteresis / caps | `adaptive_spectral_guard` | WW-driven gates; α = 2 treated as a **boundary**, not a point target |
| Task-directed probe | `ecs_probe_loss_trace_wall` | ECS-truncated probe loss for a line-search correction |
| Shape-space experimental | `spectral_rg_flow_projector` | Experimental flow in centered log-spectrum coordinates |

Mixing these under a single spreadsheet column labeled “ECS” recreates the same
class of confound as mixing density α with a rank exponent.

---

## 2. What an ECS is *for* (intent classes)

| Intent | Question | Example use in-repo |
|---|---|---|
| **Support / window** | Which modes count as bulk-effective for this action? | Midpoint rank; self-consistent support policy |
| **Correction direction** | Along which subspace do we remove contraction or damp Δ? | Trace-log normal; local-Δ orthogonal complement |
| **Boundary language** | Where is “too collapsed” relative to α ≈ 2? | AdaptiveSpectralGuard volume/shape gates |
| **Task probe** | Which ECS component decreases a probe loss? | `ecs_probe_loss_trace_wall` |

An ECS label answers **one** of these. It is not automatically a public
training target and not automatically a quality score.

---

## 3. Midpoint lineage (PL / detX class)

**Packages:** primarily `trace_log_tracker` (see package README for the exact
actuator: default one-sided removal of the contracting component of the
completed matrix displacement along a trace-log normal on the retained support).

**Idea (conceptual):** form a working support size from WeightWatcher-style
**power-law retained count** and **detX retained count**, often via a midpoint
of those integers (floor/round details are package-authoritative).

**Log / read tips:**

- Treat midpoint support as a **package-defined rank/support index**, not as
  density-law α.
- When PL and detX disagree, behavior is defined in the package implementation
  and tests — do not invent a cross-package rule in analysis notebooks.

---

## 4. Self-consistent lineage

**Packages:** `self_consistent_trace_log_tracker` (replaces the ECS *estimator*,
not necessarily the basic one-sided actuator class); `wwpgd_local_delta` uses
orientation-aware **local** bulk-effective self-consistent geometry on completed
**epoch** displacements.

**Idea (conceptual):** recompute a bulk-effective support from a self-consistent
scan (often denoted \(F(m)\)-class in notes), optionally under an explicit
`support_policy` (for example ecs / midpoint / power_law where the package
exposes it).

**Log / read tips:**

- Prefer an explicit `ecs_backend` / policy field in joint tables when comparing
  midpoint vs self-consistent runs.
- Local-delta ECS reference (epoch-start vs epoch-end) is a **config choice** in
  that package — record it when comparing runs.

---

## 5. α = 2 on plots

In WeightWatcher baseline notebooks, a horizontal line at **α = 2** is a
**reference boundary** (heavy-tail / condensation-side language in HTSR-style
discussion). In this repository:

- Adaptive spectral guard materials treat α = 2 as a **boundary**, not a unique
  point optimum to hammer every layer onto.
- Staying above or near 2 on a healthy MNIST MLP3 baseline is a **descriptive**
  observation for that suite; it is not a proof that α = 2 is a derived
  universal critical exponent from scale counting alone, and not a claim that
  interventions should force α → 2 on every task.

---

## 6. What ECS is *not*

| Construction | Difference |
|---|---|
| Public density **`target_alpha`** (nanogpt-experiments WW-PGD adapter) | Training target for stock WW-PGD projection — different repo, different operator |
| Rank-order exponent **μ_rank** (ideal map \(1/(α-1)\)) | Exponent correspondence under an ideal continuous PL; not an ECS rank |
| Marchenko–Pastur aspect **q / Q** | Matrix aspect ratio — reserved; do not use for rank-order exponent |
| House τ singular-value masks / PRE-style noise cuts | Different spectral windows used in other projects |

---

## 7. Practical joint-table rule

When comparing optimizers or baselines in a shared CSV:

1. Record **package id** (folder name) and **actuator** (from `OPTIMIZER_VARIANTS.md`).  
2. Record **ECS lineage** (midpoint vs self-consistent vs none).  
3. Never put midpoint rank, density α, and μ_rank in one column named “alpha.”  
4. Prefer package README + tests over this glossary when they disagree — this
   page is orientation, not a second implementation.

---

## 8. Open definition checks (read the code)

These are honest open points for careful readers (not blockers for using the packages):

1. Exact midpoint formula (which WW fields; floor vs round) for the tracker you run.  
2. Whether self-consistent support is the same *object* as midpoint with a better
   estimator, or a different subspace, for your chosen `support_policy`.  
3. Default ECS reference for local-delta (epoch_end vs epoch_start) in the config
   you freeze for a paper table.

---

## Related

- [`OPTIMIZER_VARIANTS.md`](../OPTIMIZER_VARIANTS.md) — actuator / support / cadence map  
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — baseline vs optimizers; PR hygiene  
- Package READMEs under `optimizers/*/` — authoritative for defaults and algebra  
