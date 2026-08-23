# One-head nanoGPT AdamW/MuonClip baseline — prepared 2026-08-22

This is the reproducible AdamW / MuonClip campaign for the
one-block, one-head nanoGPT model. It is deliberately separate from the
historical SGD/AdamW/Muon notebook suite.

The campaign runs five paired seeds (`1337`, `2027`, `4099`, `31415`,
`271828`) for each of two optimizers. Every seed uses the same initialization
convention, sampled training windows, and fixed evaluation probes. The unit of
replication is one complete seeded run; layers and checkpoints are repeated
measurements, not extra samples. Before analysis or archive, the launcher hashes
the exact step-zero model tensor inventory and requires equality across both
optimizer arms for each seed.

## What is being run

| Item | Frozen value |
|---|---:|
| FineWeb-Edu revision | `593b3a867298afb8ce42625a270ef20ddcad28f9` |
| Train / validation / test | 80M / 1M / 1M tokens |
| Model | 1 block, 1 head, width 128, context 256 |
| Unique parameters | 6,662,656 |
| Effective batch | 8,192 tokens/update |
| Horizon | 4 corpus-equivalent epochs, about 320M sampled tokens |
| Optimizer steps | 39,063 |
| Permanent states | 17: epoch 0, 0.25, ..., 4.0 |
| Seeds | 1337, 2027, 4099, 31415, 271828 |
| Arms | AdamW, MuonClip |

The learning-rate schedule completes the established one-epoch warm-up/cosine
recipe and then holds the LR floor for three further epochs. That makes this a
combined optimization and spectral-relaxation baseline. It is not presented as
a performance-optimal 320M-token schedule.

The committed hyperparameters are source-backed centers already used by the
repository. There is no checked-in nanoGPT qualification lock, so the results
must be called **baseline results**, not globally optimized winners. A future
validation-only bounded search can qualify replacements without changing this
record.

## WeightWatcher: one call, two alphas

At every permanent state, one CPU copy containing all six transformer matrices
is analyzed once:

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

With pinned WeightWatcher 0.7.7, `alpha` is the finger-corrected exponent and
`raw_alpha` is the exponent before finger removal. The persisted canonical
columns are `alpha_clip_xmax` and `alpha_raw`; `alpha_clip_xmax` is primary and
`alpha_raw` is the required sensitivity curve. WeightWatcher is not run twice.
This interpretation follows the Calculated Content
[clip-Xmax/raw-alpha description](https://calculatedcontent.com/2024/01/29/evaluating-llms-with-weightwatcher-part-iii-the-magic-of-mistral-a-story-of-dragon-kings/).

The six matrices are `W_Q`, `W_K`, `W_V`, `W_O`, `W_MLP_IN`, and
`W_MLP_OUT`. The much larger token embedding / tied language head is reported
separately in the model parameter count and is not silently mixed into the
six-matrix alpha summary.

The raw per-matrix tables also retain WeightWatcher's `ERG_gap`, `num_traps`,
`detX_num`, `detX_val`, and `rand_distance` fields. The report produces a
separate ERG-gap/correlation-trap trajectory plot for every optimizer and
matrix; these values are never reconstructed from a proxy statistic.

## Metrics and their exact meaning

- Loss is mean next-token cross-entropy in nats/token on a fixed probe.
- Perplexity is `exp(loss)` with no hidden clipping.
- “Accuracy” is next-token top-1 token accuracy, not classification accuracy.
- Top-5 next-token accuracy and bits/token are recorded as diagnostics.
- Train and validation probes each use 64 fixed batches, or 65,536 target
  tokens, during training. They are fixed-probe estimates, not exhaustive split
  scans.
- The test probe remains untouched during optimization. After the fixed horizon,
  it is evaluated once for `checkpoint_final.pt` and once for the
  validation-selected `checkpoint_best.pt`, using the same 65,536-target-token
  probe.
- BLEU is a secondary post-training lexical-overlap diagnostic on 64 fixed
  greedy continuations. Continuation token accuracy and exact match are also
  recorded for those two checkpoints. None of these test/generation diagnostics
  select a checkpoint, optimizer, horizon, or hyperparameter.

The report computes a descriptive plateau flag from validation only. A run is
called plateau-like when its validation NLL changes by at most 0.01 nats/token
over each of the last two complete one-epoch intervals. Every arm still runs to
the same fixed four-epoch budget.

## Never HOME: required environment

Use one explicit `/tmp` root. The launcher rejects a missing, relative, home, or
non-`/tmp` root and redirects Hugging Face, tiktoken, Matplotlib, Jupyter,
IPython, Torch, XDG, and the child-process `HOME` beneath it.

```bash
cd baseline/experiments/nanogpt_one_head_2026_08_21_baseline
export RG_NANOGPT_EXPERIMENT_ROOT="/tmp/rg-nanogpt-one-head-20260821"
mkdir -p "$RG_NANOGPT_EXPERIMENT_ROOT"/{cache/{home,pip,xdg/{cache,config,data,state},matplotlib},tmp}
export HOME="$RG_NANOGPT_EXPERIMENT_ROOT/cache/home"
export PIP_CACHE_DIR="$RG_NANOGPT_EXPERIMENT_ROOT/cache/pip"
export XDG_CACHE_HOME="$RG_NANOGPT_EXPERIMENT_ROOT/cache/xdg/cache"
export XDG_CONFIG_HOME="$RG_NANOGPT_EXPERIMENT_ROOT/cache/xdg/config"
export XDG_DATA_HOME="$RG_NANOGPT_EXPERIMENT_ROOT/cache/xdg/data"
export XDG_STATE_HOME="$RG_NANOGPT_EXPERIMENT_ROOT/cache/xdg/state"
export MPLCONFIGDIR="$RG_NANOGPT_EXPERIMENT_ROOT/cache/matplotlib"
export TMPDIR="$RG_NANOGPT_EXPERIMENT_ROOT/tmp"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Budget at least **20 GB free** beneath that `/tmp` root, and more if pip/model
caches are retained. The 17 permanent model-only checkpoints are about 26.7 MB
each, or roughly 9.1 GB across twenty runs, before latest/best/final
optimizer-state checkpoints, the approximately 164 MB token corpus, logs,
caches, and reports.

Use the currently activated conda environment. Install the package dependencies
once into that environment:

```bash
python -m pip install -e ../../nanogpt_one_head
```

Commit the experiment setup and core changes before the production preflight.
The launcher deliberately refuses an untracked config or dirty source tree, so
every real run has an immutable source commit rather than a workspace-only
configuration.

## Preflight and corpus preparation

```bash
python scripts/run_experiment.py doctor --device mps
python scripts/run_experiment.py prepare
```

`doctor` is a real backend gate. On the requested accelerator it runs a tiny
one-head forward/backward/update for AdamW and MuonClip, round-trips each
optimizer through a schema-v5 restart checkpoint, and performs one
`fix_fingers="clip_xmax"` WeightWatcher call over all six matrices. The
production run is not started if any numerical, device, checkpoint, raw-alpha,
or clipped-alpha check fails. `run` revalidates that the successful doctor
artifact belongs to the same source commit, config, complete dependency
closure, campaign root, and hardware block; rerun `doctor` after any of those
change.

`prepare` reuses a cache only after verifying the pinned dataset identity,
split, tokenizer, vocabulary, EOT token, exact byte counts, and SHA-256 for all
three token files. In the sandbox used to prepare this protocol there was no
existing nanoGPT corpus under `/tmp`, so no full training run was fabricated.

## Run commands

Run one overnight replicate on the Mac:

```bash
caffeinate -dimsu python scripts/run_experiment.py run \
  --optimizers adamw,muon_clip \
  --seeds 1337 \
  --device mps
```

Then build a clearly marked provisional report to inspect seed 1337 before
committing the Mac to the other four seeds:

```bash
python scripts/run_experiment.py analyze --allow-incomplete
```

This provisional report has no across-seed uncertainty claim. Use its loss,
raw-alpha, `clip_xmax`-alpha, ERG-gap, and trap trajectories to decide whether
the four-epoch horizon has reached a sufficiently stable late regime. The run
budget remains frozen; do not select a shorter per-optimizer stopping point.

Run or resume the complete 2 × 5 campaign:

```bash
caffeinate -dimsu python scripts/run_experiment.py run --device mps
```

The runner streams every training row and each checkpoint's clipped and raw
median alpha to the terminal while retaining a per-replicate log. From another
terminal, a specific overnight run can be followed without creating files:

```bash
tail -f "$RG_NANOGPT_EXPERIMENT_ROOT/logs/runs/adamw/seed_1337.log"
```

For a live table that explicitly shows both raw and `clip_xmax` alpha for every
matrix at the latest permanent state:

```bash
python scripts/run_experiment.py monitor \
  --optimizer adamw \
  --seed 1337 \
  --interval 30
```

Use `--once --no-clear` for a single snapshot. The underlying direct command is
`python -m rg_nanogpt_one_head.monitor`; the launcher supplies the campaign's
strict `/tmp` results and cache roots.

Run a single arm on an H100:

```bash
python scripts/run_experiment.py doctor --device cuda
python scripts/run_experiment.py prepare
python scripts/run_experiment.py run \
  --optimizers muon_clip \
  --seeds 1337,2027,4099,31415,271828 \
  --device cuda
```

Run on a single TPU/XLA device using explicitly ephemeral `/tmp` storage:

```bash
python -m pip install -e '../../nanogpt_one_head[tpu]'
export RG_NANOGPT_ALLOW_EPHEMERAL_TPU_STORAGE=1
export RG_NANOGPT_HARDWARE_BLOCK_ID='tpu-homogeneous-block-a'
python scripts/run_experiment.py doctor --device tpu
python scripts/run_experiment.py prepare
python scripts/run_experiment.py run --device tpu
```

Replace the example block ID with a stable description of the actual TPU
pool. It is required when the provider does not expose
`TPU_ACCELERATOR_TYPE`. This dated campaign intentionally obeys the strict
`/tmp` output rule even on TPU, so the opt-in above is genuinely ephemeral:
loss of the VM can destroy the corpus and checkpoints. Preserve the campaign
root outside the VM during long runs and run `archive` immediately after the
complete 2 × 5 analysis; the Git run record intentionally excludes the large
raw checkpoints and corpus.

Use `--device auto` only when an intentional CPU fallback is acceptable. The
advertised Mac workflow names `mps` explicitly so a CPU-only or incompatible
Torch build fails at preflight instead of silently starting a multi-day CPU run.

Mac/MPS, H100/CUDA, and TPU/XLA runs are separate hardware blocks. Do not pool
seeds from different accelerator types into one confidence interval.
Runtime provenance includes the CUDA driver, UUID, memory and device
properties; Mac model/SoC/memory; or TPU accelerator type. If a platform cannot
report those fields, `doctor` fails and asks for
`RG_NANOGPT_HARDWARE_BLOCK_ID`. Use one stable, descriptive value only for a
genuinely homogeneous device block; changing it intentionally starts a separate
campaign block.

For parallel homogeneous H100/TPU hosts, set the same explicit block ID before
`doctor` and assign disjoint optimizer/seed subsets. Complete run directories
may then be copied into one aggregation root and revalidated with `status`.
The comparison still requires identical accelerator model/capability, driver,
Torch/dependency closure, config, source, and corpus hashes; it ignores only
host/install-path and physical-device instance fields. Never move a partial
run to another accelerator instance for resume—finish it on the originating
device or start that replicate in a new root.

The launcher fails nonzero if any requested replicate fails. It resumes from
the last finite atomic checkpoint; if a process dies before the first periodic
restart state, it falls back to the immutable step-zero checkpoint and
truncates partial CSV/spectral/QK rows before replay. It never treats partial
success as a complete campaign.

Every full-state checkpoint embeds exact model and optimizer-state digests;
every permanent model checkpoint embeds its model digest. WeightWatcher raw
CSVs are bound to the run fingerprint, seed, diagnostic seed, and exact model
state, with a separate raw-file SHA-256 status record. These bindings are
recomputed before reuse, reporting, or archive.

Supported launcher jobs use adjacent nonblocking file locks. Separate
optimizer/seed jobs can run concurrently on workers that share a campaign root,
but a second process targeting the same replicate fails before it can append
metrics or replace checkpoints. `prepare`, `doctor`, and `analyze` are likewise
single-writer operations.

MuonClip QK diagnostics must cover steps `500, 1000, ..., 39000, 39063`
exactly. Interval counts must sum to 39,063 and, for this one-block/one-head
model, head observations must equal optimizer steps in each interval.

## Status, report, and executed notebook

```bash
python scripts/run_experiment.py status
python scripts/run_experiment.py analyze
```

`analyze` requires all ten runs by default and writes aggregate CSVs,
separate AdamW/MuonClip plots, a documented HTML report, a Markdown summary, a
provenance manifest, and an executed notebook below the same `/tmp` root.
`analyze --allow-incomplete` is diagnostic only; `archive` continues to reject
anything other than the exact complete 2 × 5 campaign.

After inspection, create a small, check-in-ready archive containing manifests,
aggregate tables, figures, HTML, and the executed notebook—but no corpus or
model checkpoints:

```bash
python scripts/run_experiment.py archive
```

The archived run record includes the exact Git commit / describe string, dirty
state, config hash, command, dependency freeze, hardware backend, data hashes,
UTC timestamps, and instructions to reproduce by checking out that commit.
Its requirements lock covers the complete installed campaign dependency
closure and rejects opaque direct/VCS/file origins instead of silently
rewriting them. `verify-lock` must match the recreated Python and package
inventory before training. Large binary packages are not vendored into Git;
preserve the public package-channel configuration or an external wheelhouse
needed to resolve the pinned builds, and replay fails closed if it cannot.

## Repository state and actual results

See [RESULTS.md](RESULTS.md). It starts as an honest “not yet run” ledger. The
archive command creates a dated run folder only after a complete report exists;
executed notebooks are never mistaken for source notebooks.
