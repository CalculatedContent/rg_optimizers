# Analyze the existing AdamW checkpoints without training again

`audit_checkpoints.py` is a read-only behavioral audit of a **completed AdamW
verbatim run**. It imports the pinned model and sampler, loads saved model
weights, and performs inference only. It does not call either optimizer, modify
training data, overwrite checkpoints, or refit WeightWatcher spectra.

## Commands for the completed September 14 run

Keep the same Python/conda environment used for training. From the existing
checkout, update the code and enter the experiment directory:

```bash
cd /tmp/rg_optimizers_memorization
git fetch origin
git switch main
git pull --ff-only origin main
cd baseline/experiments/nanogpt_memorization
```

Point to the completed run:

```bash
RUN="/tmp/nanogpt_memorization_20260914_222611/full/repository/verbatim/adamw/seed_1337"
```

List the actual saved weight files. This does not evaluate a model:

```bash
python audit_checkpoints.py --run-dir "$RUN" --list
```

Run the audit into a new, separate timestamped directory. Do not pre-create
that directory; the script refuses to overwrite an existing output folder.

```bash
AUDIT="/tmp/nanogpt_checkpoint_audit_$(date +%Y%m%d_%H%M%S)"

caffeinate -dimsu python -u audit_checkpoints.py \
  --run-dir "$RUN" \
  --device mps \
  --output "$AUDIT"
```

Read the result and open its folder:

```bash
cat "$AUDIT/report.md"
open "$AUDIT"
```

To share only the small audit outputs, not model checkpoints:

```bash
cd "$AUDIT"
zip -r ~/Downloads/adamw_checkpoint_audit.zip .
```

No shell options are changed. There is no training launch, resume, or restart.
A Control-C interrupts the audit, not the terminal. It leaves the original
training files unchanged; a later audit must use a new output directory.

## What is tested

### 1. Original canary acquisition and retention

Read all saved behavioral audits and reconstruct each canary's cumulative
presentation count using the saved slot schedule. This preserves the original
exact/partial recall and NLL trajectory, including the withdrawal phase.
Lifetime dose is not assumed to equal presentations already received at an
early checkpoint. New forward passes at saved snapshots also reproduce the
original full-prefix NLL within a declared 0.005-nat numerical tolerance.

### 2. Finite-sample background fitting versus fresh-rule generalization

At every saved model snapshot, compare suffix NLL and teacher-forced accuracy
on 64 fixed early training records, 64 recent training records (after update
zero), and 64 fixed fresh records from the same background generator. Data
replay means regenerating examples, not running optimization.

The script reconstructs the completed training input stream and checks that
fresh inputs, including the input-transformed background controls, never
occurred in training. It checks the data fingerprint, canary inventory,
injection schedule, and saved final exposure counts. The background rule is
`target[j] = 16 + (prefix[-32+j] - 16 + 1) % 256`.

A positive fresh-minus-replay NLL gap is evidence consistent with example-specific
fitting, conditional on this single model and these cohorts. A substantial gap
that widens over training is more informative than poor held-out performance
alone. Poor performance on BOTH replay and fresh examples does not establish
ordinary train/test overfitting. Small gaps need uncertainty assessment; this
audit does not manufacture independent training replications from tokens.

### 3. Partial extraction and assisted continuation

At the final snapshot, generate full continuations and score exact first-1,
first-4, first-8, first-16, and first-32-token segments. The original target
boundary stays fixed; shortening the scored segment does not move the prefix.

Also supply the first 8 or 16 correct suffix tokens and ask for the remaining
24 or 16. Compare 16-token-hint conditions with original, wrong, and absent
prefixes, including never-presented canaries. Assisted recovery is labeled by
the remaining scored length. It is NOT unassisted extraction of all 32 tokens.
These tests diagnose recoverable local/conditional information, not all possible
adversarial or stochastic extraction strategies.

### 4. Dependence on the original canary prefix

At the final snapshot, compare the same target under its original prefix, a
same-dose wrong prefix, a shuffled prefix, shorter 8/16/32-token prefixes, and
an all-zero no-prefix control. Save both first-target-token NLL (no gold suffix
history) and whole-tail teacher-forced NLL. Better likelihood under the correct
prefix is evidence of input-target specificity only after considering zero-dose
controls and finite probe counts. Sensitivity to an unfamiliar prompt alone is
not a memorization diagnosis.

### 5. Background rule sensitivity and positional dependence

On fresh background examples, shuffle the irrelevant first half of the prefix
without changing the target. Separately rotate the rule-relevant last half and
score both its NEW correct target and the deliberately inconsistent OLD target.
This separates response to task-relevant information from unrelated-context
sensitivity. A separate test removes 16 leading padding tokens, changing absolute
positions while preserving the meaningful input-target pair.

Position shifts and no-prefix prompts are distribution-shift probes, not clean
in-distribution train/test gaps. Do not call every sensitivity "overfitting."
The model's original padding remains unchanged except in the explicitly labeled
position-shift condition.

### 6. Confidence and output-vocabulary specialization

Record teacher-forced top-token confidence, multiclass Brier score, probability
mass in the actual 256-token target alphabet, and NLL after conditioning on that
alphabet. The latter is a diagnostic renormalization, not a change to the model.
The primary reported likelihood remains full-vocabulary NLL.

Uniform prediction on the known alphabet gives NLL `log(256) = 5.545177` and
accuracy `1/256`. This is only a marginal reference: the background rule is
deterministic, so its attainable conditional loss can be lower. Good alphabet
specialization alone is neither rule learning nor sequence memorization.

## Outputs

- `report.md`: reading order, final recall table, limitations, and warnings.
- `recorded_canaries.csv`: original per-canary time series at ALL recorded audits.
- `checkpoint_scores.csv`: new teacher-forced measures at saved model snapshots.
- `background_gaps.csv`: replay-versus-fresh differences at each saved snapshot.
- `final_probes.csv`: aggregate final-checkpoint extraction and perturbation tests.
- `paired_prompt_effects.csv`: same-target prompt contrasts; hints are compared
  only at equal remaining target length.
- `teacher_scores_*.jsonl` and `final_probe_details.jsonl`: per-example/per-token
  measurements and generated token IDs for independent checking.
- `protocol.json`, `probe_inventory.json`, `background_cohorts.json`,
  `source_hashes.json`, `complete.json`: provenance and read-only checks.

## Boundaries and validation

There are more spectral/behavioral audit timestamps than permanently retained
model snapshots. The script lists and uses the weights that actually exist. It
cannot recreate a discarded intermediate model from its alpha or loss log.
Original WeightWatcher CSVs remain available for subsequent step-aligned analysis;
this audit makes zero new WeightWatcher calls.

This run did not contain `rule_random`, mixed random-label corruption, or a
conflicting key-value phase. Those manipulations cannot be retroactively claimed
as experiments on these weights. This audit instead tests the actual trained
background distribution and the canaries that were actually presented.

Completed locally: **54 combined tests passed**, including 11 new audit tests.
The integration tests use the repository GPT at reduced dimensions with
untrained fixture checkpoints; no scientific memorization or training result is
claimed. They exercise full source/data/weight validation, real forward passes,
score alignment with the parent evaluator, malformed-data rejection, strict
checkpoint loading, and original-file immutability. Actual evaluation on the
user's trained MPS checkpoint files remains to be run.

Runtime has not been benchmarked on the user's Mac. It is evaluation plus input
reconstruction, not another 39,063-update training run. Each saved snapshot prints
its elapsed evaluation time. The final snapshot runs more prompts and will take
longer than the intermediate teacher-forced audits.
