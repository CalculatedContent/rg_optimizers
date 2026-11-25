# Stock GPT-2 / FineWeb: modified AdamW with an SGDM magnitude reference

This is the sister experiment to [`stock_gpt2_sgdm_tpu`](../stock_gpt2_sgdm_tpu/README.md).
It uses the **same magnitude-only modification as the earlier Shakespeare work**:
AdamW supplies the adaptive update direction; a same-history SGD-with-momentum
reference supplies the update magnitude for each transformer matrix.

## One command when the TPU is ready

From the repository root on the existing TPU VM:

```bash
python3 baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py start
```

The command creates an isolated pinned environment, prepares and verifies FineWeb,
checks the TPU and GitHub write access, runs the complete hyperparameter campaign,
trains the selected finalist, publishes layer measurements incrementally, and
produces final tables and plots. Candidate launches and promotions are automatic.

The default work directory is
`/mnt/disks/rg-data/gpt2small/stock_gpt2_adamw_sgdm_tpu`. Supply
`--work-dir /absolute/persistent/path` if the mounted volume differs. The existing
single-host eight-chip TPU VM, persistent disk and GitHub contents-write credentials
must be available. No TPU was allocated or production training launched while
preparing this folder. See [LAUNCH.md](LAUNCH.md) and [CAMPAIGN.md](CAMPAIGN.md).

The measured configurations are **search candidates**. Their FineWeb convergence
and optimality have not been established by the earlier Shakespeare results or
by synthetic CPU tests. A missed target remains an explicit experimental outcome.

## Exact optimizer modification

For a selected physical matrix, let the globally clipped gradient be `g_t`.
AdamW maintains its ordinary first and second moments:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

Its ordinary adaptive update, including both bias corrections, is

\[
d_t=-\eta_{A,t}\frac{m_t/(1-\beta_1^t)}{
\sqrt{v_t/(1-\beta_2^t)}+\epsilon}.
\]

With fixed `beta1` and zero initial moments, the unnormalized SGDM reference is

\[
b_t=\frac{m_t}{1-\beta_1}=\beta_1b_{t-1}+g_t,
\qquad \eta_{S,t}=r\,\eta_{A,t}.
\]

The graft replaces the update norm:

\[
\Delta_t=d_t\frac{\eta_{S,t}\lVert b_t\rVert_F}
{\max(\lVert d_t\rVert_F,10^{-30})},
\qquad W_{t+1}=(1-\eta_{A,t}\lambda)W_t+\Delta_t.
\]

The reference uses **this model's own gradients**. It has no separate SGDM model
or trajectory. The SGDM reference uses `m_t/(1-beta1)`, without Adam's first-moment
bias correction. Decoupled AdamW decay is applied separately from both update norms.
Adam's coordinatewise adaptive direction and second-moment history remain active.

### Which parameters are grafted?

Exactly four physical weight matrices in each of 12 transformer blocks:

| Physical parameter | Graft norm |
|---|---|
| `attn.c_attn.weight` | One joint Frobenius norm for packed QKV |
| `attn.c_proj.weight` | One norm for the attention output matrix |
| `mlp.c_fc.weight` | One norm for the MLP input matrix |
| `mlp.c_proj.weight` | One norm for the MLP output matrix |

That is **48 grafted physical matrices**. Token/position embeddings, the tied
output head, LayerNorm parameters and biases retain ordinary AdamW equations.
The tied embedding/head parameter is updated once. If nonzero decay is explicitly
configured, it applies to parameters with at least two dimensions; biases and
LayerNorm vectors have zero decay. The declared search uses **weight decay 0**,
matching the preceding Shakespeare graft convention.

The TPU-compatible implementation uses device tensor step counters, learning
rates and bias corrections. It avoids the original CPU wrapper's per-matrix
`.item()` synchronization. The actual TPU preflight checks changed-LR graph reuse
and host scalar extraction before production training. The canonical rule and
pinned CPU parity reference are documented in [SOURCES.md](SOURCES.md).

## Full FineWeb benchmark protocol

| Item | Protocol |
|---|---|
| Model | Stock GPT-2 Small, 124,439,808 unique parameters; 12 blocks, 12 heads, width 768 |
| Context / vocabulary | 1,024 / 50,257 GPT-2 tokens |
| Data | Pinned original FineWeb `sample-10BT` tokenized shards |
| Gradient | Global mean-token gradient, clipped once at norm 1.0 before moment updates |
| Adam parameters | `beta1=0.9`, `beta2=0.99`, `eps=1e-8` |
| Global batch / microbatch | 64 / 16 sequences; four accumulations |
| Tuning evaluation | Reserved disjoint window, 1,048,576 targets after the official prefix |
| Official evaluation | Exact first 10,485,760 validation targets |
| Quality target | Mean validation NLL ≤ **3.28 nats/token**, perplexity ≤ **26.5757727** |

Perplexity is `exp(validation NLL)`. Only a complete production evaluation on the
official slice can certify the target. The same model, data stream, target and
WeightWatcher definitions as the SGDM sister experiment support comparison;
optimizer scope and selected hyperparameters remain explicit in the manifests.

This is a single-seed TPU experiment targeting the stock NanoGPT/FineWeb quality
threshold. An official speed-record submission additionally requires comparable
hardware timing and repeated-run statistical evidence under the benchmark rules.

## Automatic parameter search

The nine candidates combine native AdamW peak learning rates
**0.0003, 0.001 and 0.003** with SGDM-reference/native-LR ratios
**100, 400 and 1000**. The earlier Shakespeare pair **0.001 × 400 = 0.4** is included
as a transfer candidate, without assuming it is optimal on FineWeb. Both learning
rates follow the same 10% warmup and cosine schedule, preserving their ratio.

| Stage | Candidates | Maximum training tokens per candidate |
|---|---:|---:|
| Screen | 9 | 134,217,728 |
| Refine | Best 3 | 1,073,741,824 |
| Full finalist | Best 1 | 10,255,073,280 |
| Longer finalist, only if needed | One fresh run | 20,510,146,560 |

Screen/refine finish equal-token horizons. Promotions and the longer attempt
start from the same seeded initialization with their own complete schedule.
Full attempts stop at the first qualifying official evaluation. The longer
attempt cycles the verified training corpus when needed.

The default maximum planned total is **35,194,404,864 training tokens** across at
most 14 planned runs. Disabling the longer attempt at the first start with
`--max-full-budget-multiplier 1` sets the maximum to **14,684,258,304 tokens**.
Validation, diagnostic work, preflight and replay after a hard interruption are
additional work. The policy bounds planned training horizons, not billed wall time.
The final report distinguishes target attainment, budget exhaustion and operational
failure. See [TUNING.md](TUNING.md) for the exact selection contract.

## Incremental WeightWatcher measurements

The worker measures **72 transformer matrices** by splitting packed QKV into Q,
K and V for analysis and including attention output plus both MLP matrices.
It also measures the token and positional embeddings separately, for **74 rows**.
This diagnostic split does not change the optimizer's joint-QKV graft norm.

At initialization, each configured interval and the final or paused endpoint,
the saved output includes:

- All native WeightWatcher fields for independent raw and `clip_xmax` fits.
- Arithmetic mean alpha, confidence-weighted alpha, tail-weighted alpha and
  combined confidence/tail-weighted alpha, for both fit variants.
- Per-layer, per-block and per-role results, ERG gap, KS distance, exact fitted
  tail count, fit validity, coverage and effective weighted-layer count.
- Exact optimizer step, tokens seen and validation measured at those same weights.

The canonical model average uses the 72 transformer matrices. Its combined weights
are `sqrt(n_tail)/(D_KS+0.02)`, with the alpha, KS distance and tail count taken from
the same fit. These are fit-quality weights, not confidence probabilities.
Native ERG gap and fitted-tail definitions remain identical to the SGDM experiment.
See [METRICS.md](METRICS.md).

Each completed snapshot is published automatically as full JSON and per-step
layer/group CSV, with an updated summary. The dedicated destination is:

```text
CalculatedContent/rg_optimizers
branch: adamw-sgdm-tpu-results
path: baseline/experiments/stock_gpt2_adamw_sgdm_tpu/results/<campaign_id>/
```

The branch is created and write/read-verified at the first actual launch.
Final aggregate CSV, report JSON and PNG/PDF plots are also saved there. Remote
Git blob identities are verified before publication is marked complete.
A publication failure requests a graceful pause and bounded retry. Final success
requires the benchmark target, all declared measurements, and verified publication.
Binary checkpoints stay on persistent disk, with optional GCS backup.
[PUBLISHING.md](PUBLISHING.md) gives the full archive/recovery contract.

## Status and recovery

The entry point remembers the latest campaign under its own work directory:

```bash
python3 baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py status
python3 baseline/experiments/stock_gpt2_adamw_sgdm_tpu/run_speedrun.py resume
```

Use the same work-directory override if one was supplied. The detached controller
continues after SSH disconnects. Resume restores model weights, both Adam moments,
per-parameter step counters, RNG state, data cursor and the unchanged schedule.
Changed optimizer contracts, configs or source files are rejected. Latest, best,
previous and first-target checkpoints have atomic writes and integrity sidecars.

The normal entry point defaults to a read-only `plan` when no action is given.
Production starts require an explicit `start` and a clean complete Git checkout.
A tiny `--smoke` CPU campaign is available for software verification and is always
labeled ineligible for the FineWeb benchmark.

## Provenance and verification

The execution engine is a self-contained sister copy of the SGDM experiment at
commit `98891bed61a816277b7a92f296e44173704f5910`.
[REUSE_PROVENANCE.json](REUSE_PROVENANCE.json) records inherited file hashes.
The original SGDM folder remains unchanged. New internal imports use a separate
package namespace, and every run has a distinct optimizer/configuration identity.
The model/runtime and pinned data manifest are reused from the existing repository.

[VALIDATION.md](VALIDATION.md) and
[validation_evidence.json](validation_evidence.json) state the checks actually
executed for this new folder. Software tests and synthetic curves establish the
implementation paths; the future TPU experiment supplies the actual FineWeb result.
