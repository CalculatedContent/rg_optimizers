# nanoGPT memorization: AdamW versus Muon

**Status: experimental design and measurement components; no training results.**
This folder is independent of the existing baseline campaigns. It contains a
source-pinned plan resolver, behavioral metrics, synthetic-probe/schedule
utilities, and a hook into the existing WeightWatcher monitor. The end-to-end
training/data adapter is specified in `TRAINER_CONTRACT.md`, not implemented.
Do not run a generated YAML through the old trainer and describe its output as
this memorization study: that would omit the controlled data interventions.

## Question and distinctions

At matched architecture, initialization, sampled-token budget, and data order,
how do AdamW and Muon differ in acquisition, accessibility, and persistence of
specific training information? Do layer-resolved spectral changes accompany
those differences after accounting for training progress and validation loss?

The measurements in the supplied figure are not five independent mechanisms.
Whole-sequence exact match is a criterion for verbatim extraction; token-level
match is a weaker criterion. Exposure measures rank within a specified candidate
universe. Prompt compression measures accessibility under a prompt class.
Training perplexity measures likelihood, not necessarily extractability.
Accordingly, vary the **information being learned**, and apply multiple
measurements to the same targets.

The mechanisms to distinguish are:

1. Rote recall of statistically uninformative, random token sequences.
2. Recall of particular natural-text passages, including repetition dependence.
3. Associative recall of synthetic key–value facts across surface templates.
4. Persistence versus forgetting after further presentations stop.

A held-out compositional rule task is a positive control for generalization,
not a fifth claim of memorization. All canaries and key–value facts are synthetic;
no real personal information is needed.

## Source settings: inherit, do not retune on memorization

Source commit: `3749c36334382a20e48bfe2473c1dc4a1470a830`.
Source recipe:
`baseline/experiments/nanogpt_one_head_2026_08_21_baseline/configs/baseline.yaml`.
Git blob: `7fd3c592afc7fdfea952b1f1aa0f9b44cdc53a2b`.

The repository calls these **source-backed center settings**, not empirically
frozen optimal settings. Its qualification protocol requires validation-only
selection and a winner lock before making the stronger claim. This experiment
does not invent such a lock. The changed data interventions also define a new
experimental protocol, even though model and optimizer settings are inherited.

| Component | Inherited setting |
|---|---|
| Model | 1 block, 1 attention head, width 128, context 256 |
| Vocabulary | GPT-2 BPE, 50,257 tokens; tied embedding/output head |
| Dropout / biases | 0 / false |
| Background data | FineWeb-Edu `sample-10BT`, revision `593b3a867298afb8ce42625a270ef20ddcad28f9` |
| Splits | 80M train / 1M validation / 1M monitoring-only test; document disjoint |
| Effective batch | 4 sequences × 8 accumulation steps × 256 target tokens = 8,192 |
| Gradient clipping | Global norm 1.0 |
| AdamW | LR 6e-4 → 6e-5; betas (0.90, 0.95); epsilon 1e-8; matrix WD 0.10 |
| AdamW warmup | 1% of the inherited one-reference-epoch schedule |
| Muon hidden matrices | LR 0.02 → 0.002; momentum 0.95; Nesterov; 5 Newton–Schulz steps; WD 0.01 |
| Muon auxiliary AdamW | LR 3e-4 → 3e-5; betas (0.90, 0.95); epsilon 1e-8; matrix WD 0.01 |
| Muon warmup | 5% of the inherited one-reference-epoch schedule |
| Replication | Paired seeds 1337, 2027, 4099, 31415, 271828 |
| Runtime | Inherited deterministic FP32/highest-matmul policy; no TF32; MPS fallback recorded |

Muon applies to Q, K, V, attention output, MLP input, and MLP output matrices.
Embeddings, the tied output head, normalization parameters, and other auxiliary
parameters retain the existing auxiliary-AdamW partition. Use the repository's
optimizer implementation; do not substitute a different package's Muon.

The separate historical `muon_clip` profile uses 2e-4 → 2e-5, RMS scaling 0.20,
and QK threshold 100. It is **not standard Muon** and is not an active arm here.
A later MuonClip comparison must be explicitly labeled and separately configured.

The LR schedule decays over one 80M-token reference epoch and remains at the
nonzero floor through reference epoch four, as in the selected campaign. Do not
stretch the cosine over four epochs, restart it at a retention fork, or equate
numerical learning rates across different optimizer update normalizations.

## Matched campaign

| Condition | Training intervention | Main comparison |
|---|---|---|
| `clean` | No experimental targets; matched ordinary-background replacement slots | Same-target counterfactual reference for the other arms |
| `sequence` | Reserved natural passages, uniform-random token sequences, finite-universe canaries | Whole-sequence recall, dose response, exact exposure, prefix accessibility |
| `association` | Synthetic key–value mappings in varied templates; separate rule-control examples | Fact recall versus literal template recall versus rule generalization |

Each condition uses both optimizers and all five paired seeds: **30 root runs**.
At 39,063 updates, each run sees 320,004,096 prediction targets. The complete
root campaign is approximately 9.60 billion target tokens, before evaluation.
This is a staged design, not a recommendation to launch all jobs on a laptop.

Start with a one-reference-epoch pilot, seed 1337, `clean` and `sequence`, both
optimizers: four runs. A pilot checks feasibility, learning signal, fit validity,
and runtime. It is not evidence for an optimizer advantage. Any pilot-driven
protocol changes must be locked before the five-seed confirmation; exclude the
pilot from confirmatory inference if it informed those changes.

For the main campaign, retain fixed-budget final states, validation-best states,
and the complete trajectory. Do not early-stop when a memorization curve looks
favorable. The protected test split never selects a setting, checkpoint, horizon,
or spectral threshold. Muon-versus-AdamW estimates refer first to these inherited
**recipes**, not to an isolated mathematical update rule: their decay, auxiliary
learning rates, and warmups differ by design. A mechanism claim requires a
separately preregistered matched-decay/auxiliary ablation.

### Sequence and repetition experiment

Use 32 targets per dose per family, with doses **0, 1, 4, 16, 64 complete
presentations per reference epoch**. Each target has 128 context tokens followed
by a 64-token continuation. The natural and random families are separately
reported, never pooled into a single headline extraction rate.

Natural passages come from a reserved document pool removed from ordinary
background sampling. Enforce document boundaries and reject exact target
contamination in the remaining train/validation/test streams. A passage described
as zero-dose must never be presented during training, including accidental
cross-window occurrences. Token-random strings exclude EOT and are not described
as ordinary natural language or random-character strings.

Injection replaces complete 257-token records at predetermined training slots;
it never appends extra tokens to one optimizer's budget. Supply the full context
and continuation in one record. Random-window sampling through an injected
binary file is not sufficient: it can miss the target, split it, or expose it
more often than the stated duplication count. Log actual full-target visits,
partial-target visits, and supervised target-token counts.

The two sequence families require 5,440 full-record presentations per reference
epoch. The canary cohort adds 680. This is approximately 1.96% of the 80M-target
budget when each occupies one 256-target record. Both optimizers receive exactly
the same slot schedule; record-level random filler is not a second repeated
memorization target. Match replacement-slot budgets across all conditions.

### Canary exposure experiment

Use eight independently keyed canaries per dose. A canary suffix is three tokens
from a fixed, unique 16-token alphabet. Its **entire declared universe** therefore
contains 16^3 = 4,096 candidates. Select the planted suffix uniformly and keep
candidate length and prefix fixed when ranking. Distinct nonces separate canary
contexts; a code may recur under another nonce, but the keyed mapping must not.

Compute suffix-only summed NLL for every candidate. Report

`exposure = log2(4096) - log2(rank)`.

The maximum is 12 bits for this deliberately small universe. This is not a claim
of 12-bit privacy loss, a universal extraction probability, or the exposure of a
32-token unrestricted secret. Report rank bounds for tied scores; the primary
value is the conservative exposure bound. All-equal scores yield zero primary
exposure. Compare with zero-dose canaries and with the same target in the paired
clean model to control intrinsic token preferences.

Exhaustive exposure is scheduled at reference epochs 0, 1, 2, and 4, not every
training step. Save all candidate scores and the universe hash for auditability.

### Associative recall and generalization control

Generate arbitrary, synthetic key–value mappings with no predictable relation
between key and value. For each entity, rotate one of three templates into a
held-out role; that template must occur for other entities during training.
Hold out **entity–template combinations**, not an entirely unfamiliar language
format. A one-presentation entity sees one training template. At larger doses,
distribute its total dose across the two available training templates.

Measure value-only greedy exact match and suffix NLL on the original template,
held-out template, unseen keys, and deliberately permuted values. Correct recall
on a held-out template supports accessible associative storage; success on
unseen random keys is not expected and signals leakage or a construction error.

Separately train a simple compositional rule, for example a small synthetic
attribute-to-token mapping with held-out attribute combinations. Improving on
held-out combinations is evidence of rule generalization. It must not be
counted as memorization merely because its target is predictable.

### Retention fork (follow-on, not automatically launched)

At reference epoch one of each `sequence` run, fork the full checkpoint into
continued-exposure and withdrawn-exposure children. The continued child is the
ordinary sequence trajectory; only the withdrawn child adds computation. It
replaces future target slots with matched background records. No reset of model,
optimizer momentum, auxiliary AdamW state, RNG, LR schedule, or sampled-token
counter is allowed. Parent and first child state hashes must agree.

Record retention curves, loss of previously acquired exact matches, exposure
changes, and time since each target's last actual presentation. Report half-life
only when an identifiable decline crosses half the initial excess-over-control
signal; otherwise report a censored or undefined estimate, not an invented
half-life. Non-monotonic forgetting/relearning is allowed.

## Measurements on the same fixed probes

**Primary extraction endpoint:** whole-sequence greedy EM@32 with a 64-token
prefix, evaluated at the fixed final token budget. Report EM@16 and EM@64,
free-running token-match fraction, and longest exact prefix separately.
Teacher-forced token accuracy is not free-running extraction.

**Prefix-constrained compression:** use prefix lengths 8, 16, 32, 64, 128,
all ending at the same target boundary. Report the shortest successful prefix
in that grid and continuation-length/prefix-length. Do not change the target
when changing prefix length, assume success is monotone, or call this optimized
adversarial compression. Failure is censored with respect to this prompt class.

**Likelihood:** report mean suffix NLL and perplexity on matched seen/unseen
cohorts. Preserve token weighting and suffix masks. Low training perplexity by
itself is neither proof of verbatim storage nor proof of a privacy leak.

Keep per-probe rows keyed by run, seed, optimizer, step, family, target ID,
intended dose, actual cumulative presentations, last-presentation step, template,
prefix length, and continuation length. Write counts/denominators with every
aggregate; a failed evaluation is missing, not zero memorization.

## WeightWatcher monitoring

Reuse `rg_nanogpt_one_head.spectral.run_weightwatcher` through `monitor.py`.
The required analysis is the repository's single-call configuration:

```python
watcher.analyze(
    ERG=True,
    randomize=True,
    fix_fingers="clip_xmax",
    min_evals=20,
    max_fingers=10,
    plot=False,
)
```

This clips the **fitted spectral range**, not the model weights, and does not
activate MuonClip. Monitoring must never feed back into optimizer updates.

Retain per-matrix `alpha`/`alpha_clip_xmax`, `raw_alpha`/`alpha_raw`, their
difference, finger count, `ERG_gap`, `num_traps`, `rand_distance`, fit distance
`D`, ranks, spectral norms, and whatever fit-support metadata is actually
returned. Preserve missing fields as missing. Save unmodified per-layer outputs
alongside the behavioral metrics and the exact checkpoint hash.

Analyze initialization, updates 1/10/100/500, every quarter-reference-epoch,
each optimizer's actual inherited warmup endpoint, and final state. Behavioral
probes run every 500 updates; expensive exposure/prefix sweeps use their sparse
cadences in `study.json`. Save a restart checkpoint before expensive diagnostics.

Do not treat a finite fitted alpha as proof that a random/MP-like layer has a
meaningful power-law tail. Inspect spectral support, tail sample count, fit
quality, and randomized-ESD separation. Predeclare a null/fit-validity procedure
before interpreting alpha as a heavy-tail signal. Flag random-like or inconclusive
layers; never average their alphas blindly across the network. The current
one-randomization distance is descriptive, not a calibrated significance test.

Hidden matrices are the primary inherited monitor inventory. The tied embedding
is counted once in a separately labeled, lower-cadence follow-on audit if added;
do not pool it with hidden matrices or imply that the six hidden matrices alone
locate every stored association.

## Analysis and interpretation

Use a complete paired training seed as the replication unit. Layers, probes,
checkpoints, and candidate strings are repeated measurements, not independent
training replicates. Show every seed and paired optimizer difference; use
seed-level intervals, with the low precision of five seeds made explicit.
Predeclare multiplicity handling for the extraction, exposure, and associative
primary endpoint families; treat remaining grids as exploratory.

First compare at equal sampled-token budgets. Secondarily compare at equal
validation NLL, but only within the actual overlapping loss range; never
extrapolate or choose levels from protected-test results. An optimizer that
learns faster can otherwise appear to memorize more at the same step simply
because it has progressed further.

Relate spectral metrics to later behavioral changes using within-run differences
and models that account for step, dose, condition, and validation NLL. Check
whether results survive exclusion of invalid/random-like fits. A spectral
correlation is not a causal explanation, a stand-alone memorization detector,
or evidence that one optimizer is safer. Family-specific attribution requires
separate interventions, not a network-average alpha from a mixed corpus.

## Files and commands

- `study.json`: preregistered design, source identities, doses, seeds, and cadence.
- `prepare_plan.py`: verifies source blobs and emits inherited configs plus manifest.
- `metrics.py`: greedy recall, conditional NLL, exact exposure, prefix compression.
- `probes.py`: random-token probes, finite canary universe, exact slot schedules.
- `monitor.py`: strict adapter to the existing WeightWatcher implementation.
- `TRAINER_CONTRACT.md`: required data/training integration and acceptance gates.
- `tests/test_study.py`, `RESULTS.md`: bounded component tests and honest status.

From the repository root, using its existing experiment environment:

```bash
python -m pytest -q baseline/experiments/nanogpt_memorization/tests
python baseline/experiments/nanogpt_memorization/prepare_plan.py \
  --out /tmp/rg-nanogpt-memorization-plan-20260914
```

The second command creates a plan, not jobs. It refuses an existing output
folder and refuses recipe/model/optimizer/spectral-source drift. Setup for the
existing source package, when needed, remains:

```bash
python -m pip install -e './baseline/nanogpt_one_head[dev]'
```

Do not point results or mutated datasets at an older baseline's run/cache folder.

## Primary methodological references

- Carlini et al., *Quantifying Memorization Across Neural Language Models*,
  arXiv:2202.07646 — extraction, context length, and repetition.
- Carlini et al., *The Secret Sharer*, arXiv:1802.08232 — canaries and exposure.
- Schwarzschild et al., *Rethinking LLM Memorization through the Lens of
  Adversarial Compression*, arXiv:2404.15146 — prompt compression; the prefix
  sweep here is deliberately narrower than that paper's optimized attack.
- Repository `baseline/FINAL_BASELINE_QUALIFICATION.md` — validation-only recipe
  selection; source-backed defaults are not a demonstrated optimum.
