# Plain SGDM: proposed tuning protocol

This folder defines a new experiment. None of the candidate configurations has
been shown here to reach validation NLL ≤ 3.28, and no configuration is described
as optimal. The goal is to choose the best completed member of a declared search,
then measure its full-budget result with the official validation window and all
required spectral measurements. The literature and its transfer limitations are
recorded in [SOURCES.md](SOURCES.md).

The normal entry point is `run_speedrun.py start` from the repository root with
this folder's path. It executes the selection protocol automatically, including
all candidate launches and promotions. Commands later in this document are for
auditing or deliberately managing individual stages. [CAMPAIGN.md](CAMPAIGN.md)
defines the finite automatic policy and its stopping rules.

## Fixed scientific contract

Production uses the pinned stock GPT-2 Small model: 12 layers, 12 heads, width 768,
vocabulary 50,257, context 1,024, learned positional embeddings and tied token/output
embeddings. The upstream model source SHA256 is
`757d0cea0d48cbc4c7d7d70371f955d49cdf3a7cfb4c87701a720c2fe0905c34`.
Model-size overrides are accepted only in an explicitly marked smoke configuration.

The source corpus is FineWeb with GPT-2 tokenization. The runner uses the existing
SHA256-verified pretokenized cache manifest at
`../../gpt2_small/speedrun30/data_manifest.json`: repository
`kjj0/fineweb10B-gpt2`, revision
`889765ea1f903759787add96995d81171b632d0c`. The run manifest must record the exact
cache manifest and data receipts; a configuration name alone does not establish
data identity.

All candidates use ordinary SGD equations, unnormalized momentum accumulation,
`dampening=0`, `nesterov=False`, and global gradient clipping at norm 1. The update
convention after the first-buffer initialization is

\[
v_t=\mu v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta_t v_t.
\]

The implementation in `optim.py` subclasses `torch.optim.SGD` and preserves its
momentum buffers and parameter-group state. It multiplies the update by a tensor
LR rather than converting that LR to the scalar `add_` argument. This is an XLA
execution correction, not a change to the mathematical optimizer. Native SGD
parity, state interchange and absence of optimizer host scalar extraction are
tested. The actual TPU compile-reuse gate runs at launch.

When weight decay is nonzero, it is ordinary coupled PyTorch SGD weight decay on
parameters with at least two dimensions. Biases and normalization vectors have
zero weight decay. This is distinct from the manual decoupled decay in the
`orientino/gap` repository. Its published zero-decay search transfers without
that particular ambiguity. There is no alpha feedback, per-layer LR adaptation,
orthogonalization, or hidden adaptive optimizer in this experiment.

Production training reads a continuous deterministic token stream in sorted shard
order and carries the stream across shard boundaries. Candidate batches therefore
see the same exact training-token prefix at a common token budget. This differs
from the older Python loader that discarded incomplete microbatch tails and from
shuffled C-loader runs; the historical control is a hyperparameter control, not a
claim of byte-for-byte reproduction of every earlier data-loader behavior.
`seed` controls model initialization. `data_seed` controls synthetic smoke data;
production uses the fixed sequential corpus order rather than data shuffling.

## Candidate set

| Role | Momentum | Peak LR | Global batch, sequences | Weight decay | Warmup |
|---|---:|---|---|---:|---|
| Proposed small-batch search | 0.98 | 0.01, 0.03, 0.1, 0.3, 1.0 | 16 and 64 | 0 | 10% of the budget |
| Historical hyperparameter control | 0.9 | 0.3 | 512 | 0.00003 | 700/19,560 of the budget |

There are ten proposed candidates and one control. Momentum 0.98 is a transfer
hypothesis informed by small-batch language-model studies, not a published
stock-GPT-2/FineWeb 3.28 recipe. The released `orientino/gap` grid supports the LR
range and standard SGD convention, but its momentum grid is
`{0.9,0.95,0.99,0.999}`, not 0.98, and its main model/data setting is different.

The default logical microbatch is 16 sequences. Global batch 64 therefore uses four
accumulations; the control uses 32. TPU topology and microbatch divisibility must
pass the launcher/preflight checks. Adjusting microbatch creates a new configuration
hash and requires a new preflight. Accumulation preserves the requested global
batch; it is not an additional factor multiplying it.

The control preserves the earlier peak LR, momentum, weight decay, batch size,
clipping and full-budget 700-update warmup. At shorter stages, its warmup is scaled
by the budget fraction. It does not import the existing alpha-control profile or
its learning-rate floor.

## Equal-token stages and promotion

| Stage | Training tokens per candidate | Candidates entering | Promoted | Validation window |
|---|---:|---:|---:|---|
| screen | 134,217,728 | 11 | Best 4 | Separate tuning window |
| refine | 1,073,741,824 | Up to 4 | Best 1 | Separate tuning window |
| full | Up to 10,255,073,280 | 1 | Stop at target; report confirmation | Official benchmark prefix |
| extended, if needed | Up to 20,510,146,560 | Same finalist, fresh start | Stop at target or report budget exhaustion | Official benchmark prefix |

At the full budget, batches 16, 64 and 512 require 625,920, 156,480 and 19,560 optimizer
updates respectively. Compare their losses at equal processed tokens, not equal
update counts. Smaller batches increase update count and can reduce TPU hardware
utilization; a loss improvement alone does not establish a wall-clock speedup.

The pilot uses initialization seed 42 throughout. All promoted runs start again
from that seed and the same data prefix. They do not continue an earlier
checkpoint. Each stage has its own complete warmup/cosine horizon, and selection
uses the last validation result at that horizon. Short-budget rankings can change
at the full budget; this resource-conscious search can discard the eventual best
candidate. It cannot establish a global optimum or guarantee 3.28.

Screen and refinement finish their equal-token horizons before selection. The
automatic campaign sets `stop_at_target=true` for full and extended stages only.
It accepts an early endpoint only when the complete official validation window
meets the target and all required layer measurements and publication complete.
Partial runs are never eligible for equal-budget tuning promotion.

If the first full horizon misses, the default campaign runs the selected candidate
once more from initialization with a twice-longer warmup/cosine horizon. It repeats
the verified corpus when needed. This is a bounded experimental fallback, not
evidence that doubling tokens guarantees convergence. The maximum planned total
is 36,536,582,144 tokens, or 16,026,435,584 with
`--max-full-budget-multiplier 1` at the first start. Failure to reach the target
within that policy is reported as `budget_exhausted_target_not_met`.

Standalone `tune.py plan` retains its original fixed-horizon full confirmation;
the campaign records its explicit target-stop override in the immutable plan.
Do not compare an early target-stop endpoint as though it completed an equal
full-token optimization comparison.

## Validation separation

Both windows come from the pinned validation shard. Offsets are zero-based token
offsets used to form next-token input/target pairs.

| Purpose | Starting offset | Evaluated next-token targets | Can certify 3.28? |
|---|---:|---:|---|
| Hyperparameter selection | 10,485,760 | 1,048,576 | No |
| Full benchmark report | 0 | 10,485,760 | Yes, for production only |

The predicted target ranges are disjoint. One boundary input/context token can be
shared without sharing evaluated targets. The data reader must have the extra
lookahead token needed to form the final target. Repeated inspection can still
overfit the tuning window; the official prefix is withheld from candidate
selection. The selector rejects benchmark-window results as a tuning criterion.

The loss target is fixed at 3.28 nats/token. Only a finite result from the complete
official prefix can certify it. Neither a smoke loss, a partial evaluation, a
tuning-window loss, nor an alpha statistic qualifies.

## Schedules, restart and resume

The default is linear warmup followed by cosine decay toward zero. Schedule
indices start at 0 and refer to the next optimizer update. Every executed update
has positive LR; `schedule_factor(max_steps)` is terminal zero. Updates beyond
that horizon are rejected. The historical control has exactly 700 warmup updates
at the full budget; proposed candidates use a rounded 10% warmup.

`--resume` is valid only for the identical configuration hash and horizon. A
larger budget, a different LR schedule, a promotion, or a changed validation
protocol is a new run. Do not append a zero-LR tail to a completed cosine run.

An explicit larger-budget proposal can use WSD: 10% linear warmup, 80% stable LR,
and 10% cosine decay. This is an untested fresh-start experiment, not continuation
of a completed checkpoint. For example, after selecting a full candidate:

```bash
python tune.py replan-longer \
  --config plans/pilot/full/winner.json \
  --budget-tokens 20510146560 --schedule wsd \
  --output plans/longer-fresh.json
```

This command creates a new hashed configuration and does not launch training.

## Plan, run, inspect and promote

Run these commands from this experiment directory. Use the training/launch
instructions for actual device execution; this tool never allocates hardware or
starts jobs.

```bash
python tune.py plan --output plans/pilot
```

This creates a plan and 11 immutable JSON configurations under
`plans/pilot/screen/configs/`. Execute each explicit configuration, with a distinct
run directory. Once training and required diagnostic workers have completed:

```bash
python tune.py select --stage-dir plans/pilot/screen \
  --results-dir runs/pilot/screen --output plans/pilot/refine
```

Run the exported refine configurations from initialization. Then:

```bash
python tune.py select --stage-dir plans/pilot/refine \
  --results-dir runs/pilot/refine --output plans/pilot/full
```

The exported `plans/pilot/full/winner.json` is an explicitly selected full-run
configuration. It is the pilot winner; its full-budget benchmark outcome remains
unmeasured until the run finishes. There is no default full "optimal" JSON.
The files under `configs/candidates/` are explicit full-budget candidate templates
using the tuning window, not evidence that any candidate won the pilot.

Every configuration embeds a SHA256 over its canonical scientific settings.
Readers reject edited settings with an old hash. Writers refuse to overwrite a
different existing configuration or selection manifest. Identical creation is
idempotent. Selection outputs record input hashes, rankings, exclusions and the
parent-stage manifest hash. They contain no invented execution timestamps.

## Selection gates and measurement completeness

The selector checks the final `TRAINING_STATUS.json` (or `status.json` alias) and
the live `TRACKING_STATUS.json`. A candidate must have the exact planned config
hash, model/data contract, initialization seed, token count, update count,
validation window and evaluation length, with finite endpoint NLL. A stale
best-so-far validation record cannot substitute for the final checkpoint.

Required scheduled spectral checkpoints are step 0, each spectral token interval,
and the final step, deduplicated. Clean pauses can add an optimizer-boundary
snapshot; these are accepted after resume when every scheduled and declared
snapshot is committed. Training must be marked done and failed/pending
measurement counts must both be zero.
Missing, running, failed infrastructure, mismatched and partially measured runs
block selection. Explicit terminal numerical divergence is recorded as an
excluded candidate; it is never assigned a favorable finite loss or silently
dropped. Ties use candidate identifier as a deterministic tie-breaker.

Spectral fit warnings remain scientific results with their recorded coverage and
validity flags. They do not change the NLL ranking, and alpha is never a promotion
criterion. The measurement implementation determines which invalid fits are
excluded from a weighted alpha aggregate; a successful worker is not a claim
that every matrix has an acceptable power-law fit.

Production enables the full embedding inventory as well as the 72 block matrices:
the tied token-embedding/output-head matrix is measured once, and the positional
embedding is measured separately. These additional matrices do not change the
canonical 72-matrix block aggregate. Up to four unprocessed snapshots can wait
for the diagnostics worker; a full queue applies backpressure instead of dropping
measurements.

## Seed confirmation and reporting

The staged pilot establishes a single-seed result. It does not estimate
between-seed variability. To plan independent confirmation of the chosen settings:

```bash
python tune.py confirm --config plans/pilot/full/winner.json \
  --seeds 42 43 44 --output plans/confirmation
```

This plans three runs and reports `planned_not_run`; it fabricates no result.
Report all seed outcomes, their mean and spread, target attainment, elapsed
training/evaluation/measurement time and token throughput. Do not select the best
seed or relabel a single seed as repeated confirmation. Full and longer-budget
results must retain their distinct budget and schedule labels.

For reproducible contract checks without training:

```bash
python -m unittest discover -s tests -p test_config_tune.py -v
```
