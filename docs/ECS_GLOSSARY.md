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
| Midpoint / PL–detX support | `trace_log_tracker` | Working rank from WW PL and detX counts (midpoint policy) |
| Self-consistent bulk-effective | `self_consistent_trace_log_tracker`, parts of `wwpgd_local_delta` | Recompute support via participation-ratio / \(F(m)\) scan |
| Layer hysteresis / caps | `adaptive_spectral_guard` | WW-driven gates; α = 2 treated as a **boundary**, not a point target |
| Task-directed probe | `ecs_probe_loss_trace_wall` | ECS-truncated probe loss for a line-search correction |
| Shape-space experimental | `spectral_rg_flow_projector` | Experimental flow in centered log-spectrum coordinates |

Mixing these under a single spreadsheet column labeled “ECS” recreates the same
class of confound as mixing density α with a rank exponent.

**Two layers of meaning:**

1. **Conceptual ECS** (SETOL preprint, arXiv:2507.17912 §5.2.3): low-rank subspace
   of generalizing eigencomponents, \(\tilde A = P_{\mathrm{ecs}} A\), often tied to
   an ERG / detX-style volume condition \(\operatorname{Tr}[\ln\tilde X]\simeq 0\).  
2. **Optimizer support policy** (this repo): concrete rank/edge estimators and
   midpoint rules used by each package README.

Conceptual ECS is **not** automatically the midpoint integer your optimizer caches.

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

**Package SoT:** `optimizers/trace_log_tracker/README.md`.

**Exact working rank (as implemented):**

\[
m_R = \left\lfloor \frac{m_{\mathrm{PL}} + m_{\mathrm{TL}}}{2} \right\rfloor,
\]

where \(m_{\mathrm{PL}} =\) `num_pl_spikes` and \(m_{\mathrm{TL}} =\) `detX_num` from
WeightWatcher-style analysis. Only the **rank** is cached between checkpoints;
the current retained singular subspace is recomputed for each local correction.

**Actuator (default idea):** one-sided removal of the contracting component of the
completed matrix displacement along a trace-log normal on that support — see the
package README for modes (`one_sided` / `tangent` / `tracking`).

**Log / read tips:**

- Midpoint support is a **package-defined rank policy**, not density-law α and not
  an MP bulk edge.  
- When PL and detX disagree, the midpoint is the defined compromise; do not invent
  a cross-package override in analysis notebooks.

---

## 4. Self-consistent lineage (bulk-effective \(F(m)\))

**Package SoT:** `optimizers/self_consistent_trace_log_tracker/README.md`.

**What changed vs midpoint:** the ECS *estimator* is replaced; the basic one-sided
actuator class is the same family.

For candidate retained rank \(m\), let \(B_m\) be the discarded bulk. Default
effective bulk size is the participation ratio

\[
r_{\mathrm{bulk}}(m)
=
\frac{\bigl(\sum_{i\in B_m}\lambda_i\bigr)^2}
     {\sum_{i\in B_m}\lambda_i^2},
\]

\[
D(m;\gamma)
=
m + r_{\mathrm{bulk}}(m)
+ \gamma\bigl[(M-m) - r_{\mathrm{bulk}}(m)\bigr],
\qquad 0\le\gamma\le 1
\]

(default \(\gamma=0\); \(\gamma=1\) restores full-\(M\) normalization). Scan

\[
F(m)
=
\frac{1}{m}
\sum_{i\in R_m}
\log\Biggl(
\frac{D(m;\gamma)}{\sum_j\lambda_j}\,\lambda_i
\Biggr)
\]

and take the integer adjacent to a zero crossing of \(F(m)\), or the minimum
\(|F|\) candidate when there is no crossing. Default working support is again a
midpoint:

\[
m_R = \left\lfloor \frac{m_{\mathrm{PL}} + m_{\mathrm{ECS}}^{\mathrm{SC}}}{2} \right\rfloor.
\]

Optional `support_policy` values (ecs / midpoint / power_law) are package-defined.

### Important non-identifications

| Phrase to avoid | Why |
|---|---|
| “Self-consistent ECS = free-fit Marchenko–Pastur edge (\(\sigma^2\), \(Q\))” | This repo’s SC backend is a **participation-ratio / trace-log \(F(m)\)** construction, not a Silverstein–Pastur fixed-point bulk-edge solve |
| “Self-consistent ECS = BBP spike count” | Different mathematical object |
| “PL cutoff = ECS = MP edge = information boundary” | Distinct estimators; may coincide numerically near α ≈ 2 |

**Log / read tips:**

- Prefer an explicit `ecs_backend` / policy field when comparing midpoint vs SC runs.  
- Local-delta ECS reference (epoch-start vs epoch-end) is a **config choice** —
  record it in joint tables.

---

## 5. α = 2 on plots

In WeightWatcher baseline notebooks, a horizontal line at **α = 2** is a
**reference boundary** (heavy-tail / class-boundary language in HTSR-style
discussion: VHT \(\alpha = 1 + \mu_{\mathrm{entry}}/2\) maps \(\mu_{\mathrm{entry}}=2\) to
α = 2 asymptotically — arXiv:1901.08278 Eq. A.4a). In this repository:

- Adaptive spectral guard materials treat α = 2 as a **boundary**, not a unique
  point optimum to hammer every layer onto.  
- Staying above or near 2 on a healthy MNIST MLP3 baseline is a **descriptive**
  observation for that suite; it is not a proof that α = 2 is a derived
  universal critical exponent from scale counting alone, and not a claim that
  interventions should force α → 2 on every task.  
- SETOL “Ideal Learning / Free Cauchy (α = 2)” language is **semi-empirical
  preprint** framing (arXiv:2507.17912), not classical RMT theorem.

---

## 6. What ECS is *not*

| Construction | Difference |
|---|---|
| Public density **`target_alpha`** (nanogpt-experiments WW-PGD adapter) | Training target for stock WW-PGD projection — different repo, different operator |
| Rank-order exponent **\(\mu_{\mathrm{rank}}\)** (ideal map \(1/(\alpha-1)\)) | Exponent correspondence under an ideal continuous PL; not an ECS rank |
| HTSR **\(\mu_{\mathrm{entry}}\)** (element tail index) | Different object from \(\mu_{\mathrm{rank}}\); see dual-label docs |
| Marchenko–Pastur aspect **\(q\) / \(Q\)** | Matrix aspect ratio — reserved; do not use for rank-order exponent |
| MP bulk-edge noise cut | Light-tailed bulk null; not the same as PL/detX/SC support |
| Porter–Thomas / eigenvector noise–information filter (Staats–Thamm–Rosenow) | Vector-statistics cut; **not** a standard literature synonym for “ECS”. Spell out the method — avoid bare “PRE” as if it were a named SETOL synonym |
| House τ singular-value masks (other projects) | Different spectral window |

### Small singular values

Tail-only ECS language must not silently mean “small singular values are noise.”
Staats, Thamm & Rosenow (arXiv:2410.17770) find RMT departures and activation-
covariance overlap involving **small** singular values as well as large ones;
aggressive truncation can damage task metrics. Use as a **caution** when
describing retained support — not as a claim that this repo’s optimizers keep
small-SV modes.

---

## 7. Practical joint-table rule

When comparing optimizers or baselines in a shared CSV:

1. Record **package id** (folder name) and **actuator** (from `OPTIMIZER_VARIANTS.md`).  
2. Record **ECS lineage** (midpoint PL/detX vs self-consistent \(F(m)\) vs none) and
   `support_policy` / `ecs_backend` when available.  
3. Prefer logging: retained rank, edge if defined, \(N,M\), refresh cadence,
   actuator_id, dose_definition, scheduled vs applied.  
4. Never put midpoint rank, density α, and \(\mu_{\mathrm{rank}}\) in one column named “alpha.”  
5. Prefer package README + tests over this glossary when they disagree — this
   page is orientation, not a second implementation.

---

## 8. Open definition checks (read the code)

These are honest open points for careful readers (not blockers for using the packages):

1. Whether self-consistent support is the same *object* as midpoint with a better
   estimator, or a different subspace, for your chosen `support_policy`.  
2. Default ECS reference for local-delta (epoch_end vs epoch_start) in the config
   you freeze for a paper table.  
3. How live ECS refresh (non-default) changes geometry vs the cached-rank default.

---

## Related

- [`OPTIMIZER_VARIANTS.md`](../OPTIMIZER_VARIANTS.md) — actuator / support / cadence map; density↔rank dual-label  
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — baseline vs optimizers; PR hygiene  
- Package READMEs under `optimizers/*/` — authoritative for defaults and algebra  
- External conceptual: Martin & Hinrichs SETOL arXiv:2507.17912 §5.2.3 (preprint tier)  
- External caution: Staats et al. arXiv:2410.17770 (small singular values)
