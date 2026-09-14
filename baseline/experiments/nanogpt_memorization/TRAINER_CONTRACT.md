# Training adapter contract and acceptance gates

This is a specification for the remaining integration, not a claim that the
trainer exists. The baseline model, optimizer, and spectral monitor are reused;
only the experimental data/sampling and behavioral-evaluation integration change.
The generated baseline YAML files alone do not implement these interventions.

## 1. Load and verify

Resolve `study.json` with `prepare_plan.py`. Read the source-pinned recipe rather
than copying hyperparameters by hand. Validate the prepared FineWeb cache with
`rg_nanogpt_one_head.data.validate_prepared_data(data_dir, baseline_config)`.
Require its dataset revision, exact token/byte counts, document-disjoint split
flag, and SHA-256 checks. Do not alter the original `train.bin` or its metadata.
Store intervention manifests and outputs under a new explicit experiment root.

Use `GPT(GPTConfig(**cfg['model']))` from the pinned model module. Construct
optimizer handles using `make_optimizer_handles(model, profile)` from the pinned
optimizer module. Preserve the entire parameter partition and initialization.
A dedicated training wrapper must explicitly support `muon`: the old dated
campaign's CLI was configured around AdamW and MuonClip and is not an automatic
launcher for this study.

## 2. Freeze information and sample controls

Create target identities, canary universes, templates, rule examples, and the
reserved natural-document pool once with the study data seed. Save tokens,
source document identities, split membership, masks, and content hashes.

Reject every natural target whose full 16/32/64-token scored suffix is already
present in the allowed background or another split at the corresponding score
length. Match held-out natural controls on token length and source domain; the
paired clean model controls the same target's intrinsic predictability. Disjoint
document IDs alone do not eliminate duplicate text across documents.

For random-token targets, scan for collisions rather than assuming probability
zero. For canaries, audit context–suffix pairs and each template-equivalent
mapping. For associations, rotate held-out entity–template combinations while
ensuring every template is learned on other entities. `probes.py` currently
implements random sequences and finite canary enumeration only; the natural
reservation/decontamination and association/rule generators remain to be built.

Do not infer empirical training frequency from the intended duplication factor.
A full experimental presentation is a complete context plus target in one
training record with all scored target tokens included in the loss.

## 3. Paired training stream and exact dose

Use distinct RNG streams for initialization, background batches, injection
slots, filler, evaluation, and spectral randomization. Pair initialization and
all training-data streams between optimizers. Hash initial model state and a
preflight trace of batch IDs to verify pairing.

For each reference-epoch interval, create a `presentation_schedule` over the
actual available record slots. Its seed is derived deterministically from the
training seed and interval, with the same derivation for both optimizers.
Doses are full presentations per interval. The small rounding difference in the
last interval is handled by its actual slot count, not by changing the dose.

The adapter replaces selected ordinary-background records with intervention
records. It must preserve the effective 8,192 supervised prediction targets per
optimizer update. Pack or fill a 257-token record so a 256-token input has 256
next-token labels; do not leave padding targets unintentionally supervised or
silently mask different numbers of tokens between conditions. Record how much
loss comes from context, target, and filler. Do not train a scored target across
an attention-context boundary.

Log each intervention visit with optimizer update, accumulation index, record
index, target ID, template, and number of supervised target tokens. At every
checkpoint compare actual cumulative visits with the planned schedule prefix.
Zero-dose targets must have zero visits. Equivalent base replacement slots are
used in the clean control, filled with ordinary background records.

## 4. Optimizer step and checkpoint identity

Reuse baseline learning-rate scheduling, gradient accumulation, global clipping,
and optimizer-step helpers. Resolve warmup steps using the actual baseline
trainer's rounding convention, then union these actual warmup endpoints into
`plan['spectral_steps_before_warmup_union']`. Do not guess that convention in a
new implementation or rescale the one-epoch schedule to the four-epoch horizon.

An update's logged LR must be the value used for that update. Every behavioral
and spectral record is keyed to the same post-update model-state hash and
sampled-token counter. Initialization is update zero. Precision, device, compile
policy, and dependency versions belong in the run fingerprint.

Save checkpoint state atomically to a temporary file, fsync, and rename. Keep a
rolling restart checkpoint plus immutable diagnostic states. Reuse the baseline
checkpoint/RNG implementation and retain model, optimizer handles, LR position,
CPU/accelerator RNG states, dedicated data RNG, dose counters, and current
schedule offset. Verify round-trip equality and the next resumed update before
starting long jobs. Never restart a partial result directory silently.

## 5. Behavioral and spectral hooks

Evaluate only fixed probes; never train on evaluation-generated continuations.
Use the functions in `metrics.py` for suffix-only NLL, teacher-forced accuracy,
free-running recall, canary ranking, and fixed-boundary prefix sweeps. Save the
returned candidate score vectors, not merely rounded ranks.

After saving a checkpoint, call:

```python
from monitor import monitor_training_state

summary = monitor_training_state(
    model,
    run_dir,
    step=completed_updates,
    tokens_seen=sampled_target_tokens,
    reference_tokens=80_000_000,
    seed=training_seed,
    fingerprint=run_fingerprint,
    ww_config=baseline_config['weightwatcher'],
)
```

The inherited monitor uses one `fix_fingers='clip_xmax'` analysis on CPU clones,
with ERG and randomization enabled. It preserves RNG state and binds results to
the checkpoint hash. Test that enabling diagnostics does not change the next
training batch, next model update, or optimizer state.

Keep `alpha_raw` and `alpha_clip_xmax` separately. Never substitute a raw fit
for a failed clipped fit, synthesize ERG values, label a failed fit zero, or use
WeightWatcher as the behavioral definition of memorization. If strict monitoring
fails, retain the restart checkpoint and write a failure status before stopping;
resume diagnostics only after resolving the cause. A finite alpha may still be
scientifically invalid for a random-like spectrum; the interpretation screen is
separate from successful software execution.

## 6. Before a pilot may be labeled runnable

The adapter must pass all of the following, beyond the component tests shipped
here:

- Tiny synthetic overfit: increasing exposure increases exact recall on a
  deliberately learnable toy cohort; zero-dose rows never enter optimization.
- Natural/association contamination audit and declared finite-universe audit.
- Paired initial hashes, batch/slot traces, and exact realized dose counts.
- Actual pinned nanoGPT AdamW and Muon forward/backward/update preflight.
- Actual WeightWatcher output schema and RNG/model invariance on target hardware.
- Full checkpoint round-trip, interruption/resume, and next-update equivalence.
- Matching masks/target-token counts, context bounds, and final token horizon.
- Same-state checkpoint agreement between online and offline behavioral metrics.

The test suite in this folder currently checks measurement mathematics, suffix
alignment, a controlled transition model, scheduling counts, plan resolution,
source-drift refusal, and monitoring configuration guards. It does not establish
that these end-to-end integration gates have passed.
