# nanoGPT memorization: AdamW versus Muon

Prepared September 14, 2026. This is a new, isolated experiment directory.
**Status: protocol and runner implemented; 14 local unit/contract tests pass.
No actual AdamW/Muon training campaign or numerical WeightWatcher integration
run has been executed for this suite.** The WeightWatcher unit test uses a mock.

## Question and scope

Which kinds of memorization occur under AdamW and plain Muon, and which
layerwise spectral changes accompany acquisition, generalization, persistence,
and interference? Memorization is established by behavioral probes and explicit
training exposure, not by assigning a meaning to alpha in advance.

These are controlled synthetic, autoregressive **answer-suffix prediction**
tasks. They retain the existing nanoGPT architecture and optimizer profiles,
but replace FineWeb-Edu with generated token sequences and mask prompt/padding
loss. They are not natural-language pretraining runs, cannot establish real-text
privacy leakage, and their losses must not be compared numerically with the
FineWeb baseline. All tokens and records are synthetic; no private data are used.

## Frozen source settings

`configs/suite.json` pins the inspected upstream commit
`3749c36334382a20e48bfe2473c1dc4a1470a830` and the Git blob hashes of the original
model, optimizer implementation, and configuration. The runner imports these
implementations rather than introducing another Muon implementation. It refuses
source-hash mismatches, preventing silent recipe drift.

Source configuration:
`baseline/experiments/nanogpt_one_head_2026_08_21_baseline/configs/baseline.yaml`.

| Setting | Repository recipe |
|---|---|
| Model | 1 transformer block; 1 attention head; width 128; context 256 |
| Vocabulary / weights | 50,257; tied token embedding and output head |
| Dropout / bias | 0.0 / false |
| Batch | 4 sequences x 8 accumulation steps = 32 sequences/update |
| Input positions / update | 8,192, including synthetic context/padding |
| Gradient norm clipping | 1.0, shared across all parameters |
| AdamW | LR 6e-4 -> 6e-5; betas (0.9, 0.95); epsilon 1e-8; decay 0.1 |
| Plain Muon hidden matrices | LR 0.02 -> 0.002; momentum 0.95; Nesterov on; 5 Newton-Schulz steps; epsilon 1e-7; decay 0.01 |
| Muon auxiliary AdamW | LR 3e-4 -> 3e-5; betas (0.9, 0.95); epsilon 1e-8; decay 0.01 |
| Warmup fractions | AdamW 0.01; Muon 0.05 |
| Schedule | Warmup/cosine over ceil(80M/8,192) = 9,766 updates, then LR floor |
| Primary final budget | 39,063 updates, following the four-corpus-equivalent-epoch baseline geometry |
| Paired seeds | 1337, 2027, 4099, 31415, 271828 |

**These are the repository-backed starting settings, not a claimed optimal
configuration.** The source campaign explicitly says it has no checked-in
nanoGPT qualification lock and should be called a baseline. Its existing active
campaign uses MuonClip; this experiment intentionally selects the separately
provided **plain `muon` profile**, not `muon_clip`.

The repository's `baseline/FINAL_BASELINE_QUALIFICATION.md` also states that a new
dataset/objective invalidates an earlier optimum claim. Any subsequent tuning
must use a preregistered development-only search, a fixed compute budget, and a
new lock before interpreting protected tests. This suite does not fabricate such
a search or a winner.

## Experiment matrix: eight conditions, six related phenomena

| Conditions | Manipulation and controls | Main interpretation |
|---|---|---|
| `verbatim`, `verbatim_absent` | Random 64-token prefixes and independent 32-token suffixes. Sixteen canaries per lifetime dose: 0, 1, 4, 16, 64. Injections occur at fixed slots in the first half; the second half contains only background. The absent twin replaces only injection slots with the exact matched background draws. | Exact sequence memorization, exposure dependence, effect of training inclusion, and persistence after withdrawal. |
| `associations` | 128 independent random 8-token keys -> random 8-token values. Train two prompt templates; audit a third. Include wrong-key and disjoint unseen-key controls. | Arbitrary associative memory versus template-specific recall. Unseen random values are not inferable by a genuine rule. |
| `rule_clean` | Modular addition modulo 31; operand pairs split 50/25/25 into train/validation/test before training. | Rule generalization versus fitting observed pairs; delayed generalization is possible, not assumed. |
| `rule_half_noise` | Same operand split; 50% of training labels replaced once with fixed random labels. Score both the assigned and true labels. | Coexistence of useful rule learning and example-specific noise memorization. |
| `rule_random` | Same split; every training label assigned independently at random, then held fixed. | Negative control for genuine addition-rule learning; direct capacity for arbitrary label fitting. Random replacement can coincide with the true label by chance. |
| `forgetting_disjoint`, `forgetting_conflict` | Identical first-half acquisition of mapping A. Second-half B either uses disjoint keys or reuses A keys with different values. Keep optimizer state and the same LR schedule; do not reset the model. | Generic forgetting under unrelated continued learning versus targeted interference/overwriting. |

The canary doses are **actual total presentations per run**, not “duplicates per
epoch.” `injection_schedule.json` fixes every injection slot; `exposures.json`
tracks realized presentations, including explicit zero counts for audit-only
records. Prefix-length probes of 8, 16, 32, and 64 tokens are run at the final
checkpoint with the same suffix and aligned target positions. Earlier audits
use the full prefix to keep overhead bounded.

Canaries and the background have 32 scored target tokens per sequence;
associations have 8, and modular addition has 1. Thus there are respectively
1,024, 256, and 32 loss-bearing targets per update. **Match budgets within an
experiment, not by pretending the tasks have equal numbers of supervised
labels.** The 8,192-position budget includes masked positions.

## Behavioral measurements

At step zero, early logarithmic checkpoints, regular intervals, 17 evenly spaced
permanent states, the phase boundary, and the final state, record:

- Conditional suffix NLL, teacher-forced token accuracy, greedy continuation
  token accuracy, and whole-suffix exact match, including per-example values.
- Seen/novel-template/wrong-key/unseen-key differences; assigned-label versus
  true-label accuracy on corrupted examples; disjoint-pair generalization.
- A and B recall before and after the switch, plus the complete presentation
  schedule needed to align retention with time since last exposure.

For the canary inclusion effect use
`NLL_absent - NLL_present` and `EM_present - EM_absent` for the same seed, canary,
prefix length, and update. This measures the effect of the specified replacement
intervention; it is not an unconditional privacy or global exposure estimate.

For forgetting, report `EM_A(boundary) - EM_A(t)` together with B acquisition;
contrast disjoint and conflicting B. Do not call ordinary validation deterioration
“memorization” without the appropriate seen/unseen behavioral evidence.

A delayed separation between high training accuracy and later high held-out rule
accuracy is the grokking question. There is no guarantee this small architecture
and split will exhibit grokking. Any change-point or threshold-based event
classifier must be fixed before the confirmatory run, not fitted to a pleasing
spectral curve after inspection.

The modular-addition test split appears only in the final audit. It never
selects a checkpoint, hyperparameter, horizon, or optimizer. Periodic audit
scores are diagnostic; all runs finish their fixed budget. The pilot evaluates
up to 16 records per group; full runs evaluate up to 64, or every member of a
smaller group. These are fixed probes, not exhaustive scans of every split.

## WeightWatcher monitoring

The runner uses exactly one analysis call on a detached CPU copy at each
spectral checkpoint:

```python
watcher.analyze(
    ERG=True,
    randomize=True,
    plot=False,
    min_evals=20,
    fix_fingers="clip_xmax",
    max_fingers=10,
)
```

WeightWatcher is pinned to **0.7.7**, matching the parent package. Persist every
returned column, bind each row to its matrix name, and expose `alpha_clip_xmax`
from `alpha` and `alpha_raw` from `raw_alpha`. Keep `D`, `rand_distance`,
`ERG_gap`, `num_traps`, `num_fingers`, tail sizes, and fit warnings whenever
returned. Required API fields must exist; absent metrics are not replaced by
invented proxies. Nonfinite/unsupported fits are retained and flagged, not
interpreted as meaningful power laws.

The six primary matrices are Q, K, V, attention output, MLP input, and MLP output.
**There is no single all-layer alpha headline.** Inspect each matrix separately,
including randomized-control distance, tail support and fit quality. A finite
alpha or a particular numerical threshold does not prove that a small spectrum
is heavy-tailed or that memorization has occurred. The `tail_support_at_least_20`
flag is only a support flag, not a goodness-of-fit or correlation test.

The much larger token embedding/tied head is deliberately excluded from this
six-matrix summary, as in the source campaign, but remains in saved model
checkpoints. This limits localization claims: memory in those parameters cannot
be excluded by unchanged hidden-matrix spectra.

`fix_fingers` corrects the diagnostic spectral fit; it does **not** clip trained
weights. The monitor works on CPU copies, isolates Python/NumPy/CPU-Torch RNGs,
and checks the model's tensor hash before and after auditing. Step-zero weights
provide an initialization control. Any predictive spectral analysis must compare
against time/update count and training loss, then validate on held-out complete
seeds; repeated checkpoints/layers are not independent training runs.

## Optional stricter optimizer comparison

The primary `--recipe repository` comparison preserves the source profiles,
including their different auxiliary learning rates, warmup fractions and decay.
It compares the **whole training recipes**, not only the matrix-update rule.

The secondary `--recipe shared_aux_decay` leaves AdamW unchanged and gives Muon
exactly the same auxiliary AdamW LR, LR floor, betas, epsilon, decay, and warmup.
It sets Muon's hidden decay coefficient to 0.003, so the per-step shrinkage
`LR(t) * decay` matches AdamW throughout the common-shaped schedule:
`0.02 * 0.003 = 0.0006 * 0.1`. Merely setting both decay coefficients to 0.1
would not match shrinkage. Muon's hidden LR, momentum, and Newton-Schulz update
remain its source settings. This is an explicitly labeled diagnostic control,
not a new “optimal” Muon claim, and is stored in a separate recipe directory.

## Execution

Use an existing checkout containing this directory and the pinned parent files.
The suite never modifies baseline code or outputs. Use the currently activated
research environment. When installation is needed, from repository root:

```bash
export RG_MEM_ROOT=/tmp/rg-nanogpt-memorization-20260914
mkdir -p "$RG_MEM_ROOT/cache/pip"
PIP_CACHE_DIR="$RG_MEM_ROOT/cache/pip" python -m pip install -e baseline/nanogpt_one_head
cd baseline/experiments/nanogpt_memorization
python -m pytest -q tests
```

First run the actual model/optimizer/WeightWatcher smoke checks on the intended
hardware. They are required before trusting the integration:

```bash
python run.py run --stage smoke --condition verbatim --optimizer adamw --seed 1337 --device mps --resume
python run.py run --stage smoke --condition verbatim --optimizer muon --seed 1337 --device mps --resume
```

The smoke is two updates with reduced audit populations, but retains the real
model and optimizer architecture. The WeightWatcher dependency is mandatory;
there is no silent “skip WW” production mode. MPS, CUDA, and CPU are explicit
choices; an unavailable accelerator fails rather than falling back silently.
This new runner does not implement TPU/XLA integration.

Next, run a paired pilot:

```bash
python run.py run --stage pilot --condition verbatim --optimizer adamw --seed 1337 --device mps --resume
python run.py run --stage pilot --condition verbatim --optimizer muon --seed 1337 --device mps --resume
```

`python run.py plan --stage pilot` prints, but does not execute, all 16 pilot
commands: eight conditions x two optimizers x one seed, 2,000 updates each.
`python run.py plan --stage full` prints the complete 80-run, five-seed design
with 39,063 updates per run. It does not launch a large campaign automatically.
A pilot is an engineering/difficulty check and is not proof of asymptotic
behavior; its schedule is a prefix of the full schedule.

For the stricter control append `--recipe shared_aux_decay` to both optimizer
commands. To use CUDA or CPU, explicitly replace `--device mps`. Do not pool
heterogeneous hardware blocks or recipes.

Outputs default to `/tmp/rg-nanogpt-memorization-20260914`:

```text
<stage>/<recipe>/<condition>/<optimizer>/seed_<seed>/
  manifest.json                  # configuration, package inventory, hardware, hashes
  probe_inventory.json           # generated records and audit membership
  injection_schedule.json        # exact canary presentation slots
  metrics.jsonl                  # checkpoint-level and per-example behavior
  exposures.json                 # realized presentation counts
  spectral/step_XXXXXXXX.csv      # all raw WW columns plus explicit alpha names
  checkpoint_latest.pt           # atomic restart: model, optimizers, RNG, counters
  model_step_XXXXXXXX.pt          # permanent model-only states
  complete.json                  # created only after the final audit succeeds
```

The runner prints training loss every 25 updates and a spectral-checkpoint line
at each audit. For a read-only behavioral and per-matrix snapshot:

```bash
python run.py monitor /tmp/rg-nanogpt-memorization-20260914/pilot/repository/verbatim/muon/seed_1337
```

Resume requires an identical run fingerprint and complete package inventory.
The runner checks paired-arm initialization/data/hardware identity when a peer
manifest is present, uses an exclusive per-run file lock, and rolls incomplete
metrics back to the latest atomic checkpoint. Caches and run outputs are placed
under the explicit temporary root, never mixed into this source directory.
Temporary storage is ephemeral: preserve completed results elsewhere before a
host reset. Full-suite model-only snapshots alone are roughly 36 GB, so do not
launch the entire design without storage planning.

## Analysis and decision rules

Compare AdamW and Muon at equal updates, equal realized exposures, and, where
both reach a shared target, matched training loss/accuracy. This last comparison
is a secondary check against merely learning at different speeds; avoid
extrapolation outside their overlapping performance range.

The replication unit is a complete seed. Report paired per-seed differences,
all five raw seed outcomes, and uncertainty across seeds. Do not inflate the
sample size using layers, canaries, or repeated checkpoints. The one-seed pilot
has no across-seed confidence interval. Analyze recipe and hardware blocks
separately. There is no automated hypothesis-testing or spectral forecasting
report in this initial implementation; the raw audit tables support that
preregistered follow-on analysis.

## Sources

- Parent campaign README and `configs/baseline.yaml`: model, training geometry,
  distinct Muon/MuonClip recipes, one-call raw/clipped alpha convention.
- `baseline/FINAL_BASELINE_QUALIFICATION.md`: validation-only baseline selection
  and the conditions under which a lock is invalidated.
- Carlini et al., *Quantifying Memorization Across Neural Language Models*,
  https://arxiv.org/abs/2202.07646: duplication and prompting context motivate
  controlled dose and prefix-length probes.
- Carlini et al., *The Secret Sharer*,
  https://www.usenix.org/conference/usenixsecurity19/presentation/carlini:
  synthetic canary methodology. This implementation does not estimate its
  full-space exposure metric.
- Calculated Content, `clip_xmax` feature description,
  https://calculatedcontent.com/2023/03/21/weightwatcher-advanced-features-fix_fingers/.
