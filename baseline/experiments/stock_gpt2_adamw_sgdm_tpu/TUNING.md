# AdamW direction with SGDM magnitude: tuning protocol

This sister experiment applies the previously tested magnitude-only graft to the
full stock GPT-2/FineWeb protocol. It preserves AdamW's direction and moment
history, then scales selected transformer-matrix updates to the norm of a
same-history momentum-SGD reference. It is identified in every scientific
configuration as `optimizer: adamw_sgdm_graft_v1`.

The normal entry point is `run_speedrun.py start`. It launches and audits the
entire finite search, promotes candidates, trains the finalist, saves all layer
measurements and publishes them incrementally. See [CAMPAIGN.md](CAMPAIGN.md).
The lower-level commands here are for inspecting or deliberately managing
individual stages; they are not required for the automatic campaign.

The earlier Shakespeare learning-rate/ratio pair is included in the candidate
set. Its transfer to FineWeb remains a hypothesis. No candidate is described as
optimal, and reaching the benchmark target is not guaranteed by the search plan.

## Exact optimizer contract

The source rule is the repository's
[`short_test_adamw_graft/graft.py`](../../../short_test_adamw_graft/graft.py).
The sister implementation adapts that rule for dynamic tensor learning rates
and TPU execution. Its mathematical contract is:

\[
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,
\]

\[
a_t=-\eta_{A,t}\frac{m_t/(1-\beta_1^t)}
 {\sqrt{v_t/(1-\beta_2^t)}+\epsilon},\qquad
r_t=\frac{\eta_{R,t}}{1-\beta_1}\lVert m_t\rVert_F,
\]

\[
\Delta_t=a_t\frac{r_t}{\max(\lVert a_t\rVert_F,10^{-30})},\qquad
\theta_{t+1}=(1-\eta_{A,t}\lambda)\theta_t+\Delta_t.
\]

Here \(g_t\) is the caller's clipped gradient, \(\eta_A\) is the native Adam
learning rate, and \(\eta_R=\texttt{reference_lr_ratio}\times\eta_A\). The
quantity \(m_t/(1-\beta_1)\) is the zero-initialized, unnormalized SGDM buffer
for the same gradient history. No separate reference optimizer or extra gradient
history is needed. The denominator clamp preserves the original zero/tiny-norm
behavior; the rule does not manufacture a direction for a zero update.

The configuration pins `beta1=0.9`, `beta2=0.99`, `eps=1e-8` and
`norm_epsilon=1e-30`. Global gradient norm clipping is 1.0. The proposed campaign
uses weight decay zero, matching the prior Shakespeare contract. If a separate
configuration enables weight decay, it remains decoupled AdamW decay and stays
outside the grafted update norm. Bias and normalization parameters have no decay.

`graft_scope: transformer_physical_matrices` means the four physical weight
matrices in each transformer block: packed QKV, attention output, MLP expansion
and MLP projection. Packed QKV receives one joint norm scale. In the 12-layer
model this is 48 physical matrices. Token/position embeddings, the tied output
head, normalization vectors and biases retain native AdamW updates and moments.
Their handling is not inferred from a parameter's shape alone; matrix names and
the declared scope determine graft eligibility.

The WeightWatcher inventory is intentionally different from the optimizer's
physical-matrix count. It measures the three Q/K/V slices separately, giving 72
logical transformer matrices plus two separately identified embeddings. Alpha,
ERG gap and other layer diagnostics never change learning rates, graft ratios,
update directions or candidate rankings.

## Fixed model, data and numerical protocol

Production uses the pinned stock GPT-2 small model: 12 layers, 12 heads, width
768, vocabulary 50,257, context 1,024, learned positions and tied token/output
embeddings. The upstream model source SHA256 is
`757d0cea0d48cbc4c7d7d70371f955d49cdf3a7cfb4c87701a720c2fe0905c34`.
Smaller dimensions are permitted only for explicitly synthetic smoke tests.

The runner uses the existing verified GPT-2-tokenized FineWeb cache manifest at
`../../gpt2_small/speedrun30/data_manifest.json`: repository
`kjj0/fineweb10B-gpt2`, revision
`889765ea1f903759787add96995d81171b632d0c`. Each run records exact manifest and
shard hashes. Production consumes a deterministic continuous stream in sorted
shard order, carrying tokens across boundaries and repeating the corpus when a
horizon requires it. Candidate runs at a common budget see the same token prefix.
This token order is not a claim of identical execution to a shuffled CUDA run.

The production global batch is 64 sequences, with microbatches of 16 and four
gradient accumulation steps. The number 16 is the logical global microbatch
distributed over the eight-chip TPU, not a further per-chip multiplier. FP32
weights/moments and BF16 TPU autocast follow the pinned runtime. The preflight
checks the actual attention shape and optimizer graph on the eventual TPU.
CPU mathematical tests do not certify TPU convergence or wall time.

## Nine explicitly proposed settings

The search varies native Adam peak LR and reference-LR ratio independently:

| Native Adam peak LR | Ratio 100: reference peak LR | Ratio 400: reference peak LR | Ratio 1000: reference peak LR |
|---:|---:|---:|---:|
| 0.0003 | 0.03 | 0.12 | 0.3 |
| 0.001 | 0.1 | 0.4 | 1.0 |
| 0.003 | 0.3 | 1.2 | 3.0 |

The `(0.001, 400)` pair is the prior Shakespeare setting. All nine pairs use the
same batch, betas, epsilon, norm floor, zero decay, clipping and warmup/cosine
schedule. Native Adam LR still controls embeddings and other ungrafted
parameters; matching a reference peak LR alone does not make two pairs
equivalent.

Every candidate includes a distinct identifier and a hash over all scientific
settings. Changing the ratio, optimizer identity, scope, schedule or validation
contract changes that hash. Plain-SGDM configuration files are rejected by the
sister loader, and sister configurations are rejected by the SGDM loader.

## Equal-token stages and finite fallback

| Stage | Training tokens per candidate | Candidates entering | Action |
|---|---:|---:|---|
| Screen | 134,217,728 | 9 | Complete every eligible horizon; promote the best 3 by endpoint tuning NLL |
| Refine | 1,073,741,824 | Up to 3 | Fresh runs; promote the best 1 by endpoint tuning NLL |
| Full | Up to 10,255,073,280 | 1 | Fresh run; stop at the first qualifying official validation |
| Extended, if needed | Up to 20,510,146,560 | Same finalist | One fresh longer schedule; stop at target or report exhaustion |

With the fixed global batch, these full horizons are 2,048, 16,384, 156,480 and
312,960 optimizer updates respectively. The pilot uses initialization seed 42.
Each promotion starts again from that seed and data prefix with a new complete
warmup/cosine horizon. Promotion never resumes an earlier shorter schedule.

The default maximum planned training total is **35,194,404,864 tokens**, across
at most 14 runs. Selecting `--max-full-budget-multiplier 1` when starting a new
campaign disables the extended attempt and reduces the planned total to
**14,684,258,304 tokens**. Evaluation, diagnostics, checkpoint I/O and work replayed
after interruptions add compute beyond these training-token totals.

Short-budget rankings can change at the full horizon. The finite promotion
scheme can therefore discard the eventual best long-run setting. A single seed
does not establish between-seed reliability. The workflow makes neither a
global-optimality claim nor a promise that doubling the final horizon reaches
3.28. A completed policy that misses its target is labeled
`budget_exhausted_target_not_met` and retains every run's evidence.

## Validation and qualification

| Purpose | Starting token offset | Evaluated next-token targets | Can qualify for the official target? |
|---|---:|---:|---|
| Screen/refine selection | 10,485,760 | 1,048,576 | No |
| Full/extended benchmark | 0 | 10,485,760 | Yes, for production only |

The evaluated target ranges are disjoint. One boundary input token can overlap
without sharing evaluated targets. Every evaluation needs its final lookahead
token. The selector rejects official-window results as tuning evidence.

The official endpoint is finite validation NLL **at most 3.28 nats/token**, with
perplexity **at most `exp(3.28) ≈ 26.5758`**, over the complete official prefix.
Partial evaluations, training losses, tuning losses and synthetic results cannot
qualify. A target flag from another step cannot replace the endpoint evaluation.

Screen/refine always finish their equal-budget horizons. The automatic campaign
sets `stop_at_target=true` only for full and extended runs. After the first
qualifying official evaluation, it saves the exact model/optimizer checkpoint,
waits for every required layer measurement, and verifies the incremental GitHub
archive before reporting success. Standalone `tune.py plan` creates fixed-horizon
full configurations; the automatic campaign records its explicit target-stop
override in the campaign plan.

## Restart, resume and metric completeness

The LR schedule uses linear warmup over a rounded 10% of the horizon and cosine
decay. Every executed update has positive LR; `schedule_factor(max_steps)` is the
terminal zero. Native and reference LRs follow the same schedule and fixed ratio.

Resume requires the identical configuration hash, source hashes, device kind and
horizon, with saved Adam moments and graft contract. A larger horizon or changed
ratio is a fresh configuration. Do not append a zero-LR tail or silently continue
one candidate as another. The automatic extended attempt is a fresh doubled
cosine horizon, not a continuation of the completed first full run.

Selection requires terminal training, the precise planned step/token budget,
matched validation identity and complete diagnostics. Required spectral steps
are initialization, every configured interval and the terminal endpoint. Clean
pauses may add snapshots, all of which must also be committed after resume.
Missing, failed or pending diagnostics block promotion. Explicit numerical
divergence is reported and excluded; it is never assigned an invented favorable
loss. Ties are resolved deterministically by candidate identifier.

Fit warnings remain recorded results with validity flags and coverage. Complete
worker output does not imply every matrix has a valid power-law fit. Model-level
averages retain the canonical 72 transformer-matrix definition; embeddings stay
separate. A bounded snapshot queue applies backpressure instead of dropping
measurements. See [METRICS.md](METRICS.md) and [PUBLISHING.md](PUBLISHING.md).

## Lower-level audit commands

From this experiment directory, these commands create or inspect plans without
starting training or allocating resources:

```bash
python tune.py plan --output plans/pilot
python tune.py select --stage-dir plans/pilot/screen \
  --results-dir runs/pilot/screen --output plans/pilot/refine
python tune.py select --stage-dir plans/pilot/refine \
  --results-dir runs/pilot/refine --output plans/pilot/full
```

The first command creates nine immutable screen configurations. The exported
`plans/pilot/full/winner.json` names the tuning-selected candidate. Its full
benchmark result remains unmeasured until training and verification finish.
`configs/candidates/` contains full-budget tuning-window templates, not nine
claimed successful benchmark recipes.

For an explicitly separate repeated-seed study:

```bash
python tune.py confirm --config plans/pilot/full/winner.json \
  --seeds 42 43 44 --output plans/confirmation
```

This only plans runs. Report all seed outcomes and their spread rather than
selecting the best seed. Repeated-seed runs are outside the default finite
campaign budget.
