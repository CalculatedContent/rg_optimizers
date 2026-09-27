# One-head nanoGPT FineWeb baseline and memorization-load study

This directory contains two related, reproducible experiments for the same
one-block, one-head nanoGPT model:

1. the dated AdamW/MuonClip FineWeb-Edu baseline; and
2. a controlled harmful-memorization load sweep embedded in ordinary FineWeb
   next-token training.

Both experiments use real language modeling on the pinned FineWeb-Edu corpus.
They are separate from the synthetic modular-addition and random-label
experiments elsewhere in the repository.

The unit of replication is one complete seeded run. Layers and checkpoints are
repeated measurements, not independent replicates.

## Frozen language-model baseline

| Item | Frozen value |
|---|---:|
| FineWeb-Edu revision | `593b3a867298afb8ce42625a270ef20ddcad28f9` |
| Train / validation / test | 80M / 1M / 1M tokens |
| Tokenizer | GPT-2 BPE, vocabulary 50,257 |
| Model | 1 block, 1 head, width 128, context 256 |
| Unique parameters | 6,662,656 |
| Effective batch | 8,192 tokens/update |
| Horizon | 4 corpus-equivalent epochs, about 320M sampled tokens |
| Optimizer steps | 39,063 |
| Permanent states | 17: epoch 0, 0.25, ..., 4.0 |
| Canonical seeds | `1337`, `2027`, `4099`, `31415`, `271828` |
| Baseline arms | AdamW, MuonClip |

The learning-rate schedule completes the established one-epoch warm-up/cosine
recipe and then holds the learning-rate floor for three further epochs. These
are source-backed baseline settings, not globally optimized winners.

## Harmful-memorization load sweep

### Question

The sweep asks whether forcing a normally trained FineWeb language model to
absorb additional arbitrary random information changes:

- behavioral memorization of controlled random canaries;
- clean FineWeb validation/test performance; and
- the layer-resolved WeightWatcher spectrum of `W_Q`, `W_K`, `W_V`, `W_O`,
  `W_MLP_IN`, and `W_MLP_OUT`.

Ordinary FineWeb next-token prediction remains the primary objective. Random
canaries are injected into selected training batches by
`RandomCanaryExperiment`; they are not baked into `train.bin` during corpus
preparation.

### Two different controls

Do not describe the 0% harmful-load condition as a "no-memorization" control.
It means only that the additional harmful random bank is absent. A 0%-load model
can still memorize or overfit ordinary FineWeb examples.

The actual behavioral negative controls are the **dose-0 tracked canaries inside
every run**. They are generated like the exposed canaries but are never injected
into training. Each run contains tracked canaries with doses:

```text
0, 1, 4, 16, 64
```

The primary continuous memorization contrast is

```text
DeltaNLL_d(t) = mean NLL(dose 0, t) - mean NLL(dose d, t)
```

A positive value means the exposed arbitrary continuation has become more
likely than an otherwise matched never-exposed control. Token recall and exact
recall are stronger, secondary behavioral endpoints.

### Load conditions

| Harmful load | Config |
|---:|---|
| 0% | `configs/harmful_memorization_0pct.yaml` |
| 0.1% | `configs/harmful_memorization_0p1pct.yaml` |
| 0.5% | `configs/harmful_memorization_0p5pct.yaml` |
| 2% | `configs/harmful_memorization_2pct.yaml` |
| 10% | `configs/harmful_memorization_10pct.yaml` |

The load configs share the same pinned corpus, tokenizer, split sizes,
memorization data seed, tracked canary doses, architecture, horizon, and
optimizer definitions. The intended intervention is `harmful_load_fraction`.

### Replication plan

The current confirmatory MuonClip block is:

```text
5 load conditions x 5 matched seeds = 25 complete runs
```

Use the canonical seeds `1337,2027,4099,31415,271828` at every load. Treat the
seed as the independent experimental unit and use paired, within-seed load
comparisons. Do not inflate the sample size using layers or checkpoints.

Five seeds are the minimum registered campaign. A ten-seed extension should be
a new, versioned protocol with five additional tracked seeds; do not append
ad-hoc seed values to an existing five-seed campaign or pool different hardware
blocks.

MuonClip and AdamW are separate optimizer blocks. The commands below run the
current MuonClip load-response campaign. An AdamW replication may be run with
the same load configs, corpus, and seeds, but its uncertainty and conclusions
must be reported separately.

## WeightWatcher: one call, two alphas

At every permanent state, one detached CPU copy containing all six transformer
matrices is analyzed once:

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

With pinned WeightWatcher 0.7.7, `alpha_clip_xmax` is the finger-corrected
exponent and `alpha_raw` is the uncorrected sensitivity curve. WeightWatcher is
not run twice. Raw per-matrix tables also retain `ERG_gap`, `num_traps`,
`detX_num`, `detX_val`, `rand_distance`, fit-support fields, and exact model-state
bindings.

The six canonical matrix names in `spectral/layers.csv` are:

```text
L00_W_Q
L00_W_K
L00_W_V
L00_W_O
L00_W_MLP_IN
L00_W_MLP_OUT
```

Never infer memorization from alpha alone. Alpha is compared against independent
behavioral canary measurements and clean FineWeb performance.

## Metrics and their exact meaning

- Loss is mean next-token cross-entropy in nats/token on a fixed probe.
- Perplexity is `exp(loss)` with no hidden clipping.
- Accuracy is next-token top-1 token accuracy, not classification accuracy.
- Train and validation probes each use 64 fixed batches, or 65,536 target
  tokens, during training.
- The test probe remains untouched until post-training evaluation of the final
  and validation-selected best checkpoints.
- Canary NLL, teacher-forced token accuracy, free-running token accuracy, and
  exact continuation recall are persisted at every canary evaluation state.
- Clean FineWeb harm is assessed with validation loss/perplexity/accuracy at
  matched steps and protected test outcomes after the fixed horizon.

## Required environment

Use one explicit `/tmp` root. The production launcher rejects a relative,
home-directory, dirty-worktree, or untracked-configuration run.

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

Install the package once in the active environment:

```bash
python -m pip install -e ../../nanogpt_one_head
```

Budget substantial `/tmp` storage. Corpus files are about 156 MB, while
checkpoints and optimizer states dominate long multi-seed campaigns.

## Reproduce the dated baseline

Preflight and prepare the corpus:

```bash
python scripts/run_experiment.py doctor --device mps
python scripts/run_experiment.py prepare
```

Run one paired seed:

```bash
caffeinate -dimsu python scripts/run_experiment.py run \
  --optimizers adamw,muon_clip \
  --seeds 1337 \
  --device mps
```

Run or resume the complete dated 2 x 5 baseline:

```bash
caffeinate -dimsu python scripts/run_experiment.py run --device mps
```

`run` resumes automatically from the last verified finite atomic checkpoint;
there is no `--resume` flag.

## Reproduce the five-load x five-seed MuonClip study

### 1. Prepare the FineWeb corpus once

Choose one timestamped sweep root. The 0%-load root is the canonical corpus
source for the other load conditions.

```bash
CODE="$(git rev-parse --show-toplevel)/baseline/experiments/nanogpt_one_head_2026_08_21_baseline"
BASE="/tmp/rg-harmful-memorization-sweep-$(date +%Y%m%d-%H%M%S)"

cd "$CODE"

export RG_NANOGPT_EXPERIMENT_ROOT="$BASE/load_0pct"
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

python scripts/run_experiment.py doctor \
  --config configs/harmful_memorization_0pct.yaml \
  --device mps

python scripts/run_experiment.py prepare \
  --config configs/harmful_memorization_0pct.yaml
```

The first preparation streams the pinned FineWeb-Edu revision. Later load roots
must reuse the exact verified `train.bin`, `val.bin`, `test.bin`, and `meta.json`.
The random canaries and harmful bank are injected at training time, so separate
FineWeb downloads are neither necessary nor desirable.

### 2. Run all 25 MuonClip replicates

The following block never removes an existing file. Missing corpus files are
copied from the verified 0%-load root; existing files must compare byte-for-byte
or the sweep stops. Completed runs are recognized, and incomplete runs resume
from their last verified finite checkpoint.

```bash
(
set -euo pipefail

CODE="$(git rev-parse --show-toplevel)/baseline/experiments/nanogpt_one_head_2026_08_21_baseline"
BASE="${BASE:?export BASE to the timestamped sweep root created above}"
SOURCE_DATA="$BASE/load_0pct/data"
SEEDS="1337,2027,4099,31415,271828"

cd "$CODE"

if [ -n "$(git status --short)" ]; then
  echo "ERROR: production source worktree is not clean"
  git status --short
  exit 1
fi

export PYTHONPATH="$(git rev-parse --show-toplevel)/baseline/nanogpt_one_head/src:${PYTHONPATH:-}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

for FILE in train.bin val.bin test.bin meta.json; do
  test -f "$SOURCE_DATA/$FILE" || {
    echo "ERROR: missing verified corpus file: $SOURCE_DATA/$FILE"
    exit 1
  }
done

LOAD_SPECS=(
  "0pct|harmful_memorization_0pct.yaml"
  "0.1pct|harmful_memorization_0p1pct.yaml"
  "0.5pct|harmful_memorization_0p5pct.yaml"
  "2pct|harmful_memorization_2pct.yaml"
  "10pct|harmful_memorization_10pct.yaml"
)

for SPEC in "${LOAD_SPECS[@]}"; do
  LOAD="${SPEC%%|*}"
  CONFIG="${SPEC#*|}"
  ROOT="$BASE/load_$LOAD"
  DATA="$ROOT/data"

  mkdir -p "$DATA" "$ROOT"/{cache/{home,pip,xdg/{cache,config,data,state},matplotlib},tmp}

  for FILE in train.bin val.bin test.bin meta.json; do
    SRC="$SOURCE_DATA/$FILE"
    DST="$DATA/$FILE"

    if [ -e "$DST" ]; then
      cmp -s "$SRC" "$DST" || {
        echo "ERROR: existing corpus file differs: $DST"
        exit 1
      }
    else
      cp -p "$SRC" "$DST"
    fi
  done

  export RG_NANOGPT_EXPERIMENT_ROOT="$ROOT"
  export HOME="$ROOT/cache/home"
  export PIP_CACHE_DIR="$ROOT/cache/pip"
  export XDG_CACHE_HOME="$ROOT/cache/xdg/cache"
  export XDG_CONFIG_HOME="$ROOT/cache/xdg/config"
  export XDG_DATA_HOME="$ROOT/cache/xdg/data"
  export XDG_STATE_HOME="$ROOT/cache/xdg/state"
  export MPLCONFIGDIR="$ROOT/cache/matplotlib"
  export TMPDIR="$ROOT/tmp"

  python scripts/run_experiment.py doctor \
    --config "configs/$CONFIG" \
    --device mps

  python scripts/run_experiment.py prepare \
    --config "configs/$CONFIG"

  caffeinate -dimsu python scripts/run_experiment.py run \
    --config "configs/$CONFIG" \
    --optimizers muon_clip \
    --seeds "$SEEDS" \
    --device mps \
    --mps-retries 2

  for SEED in 1337 2027 4099 31415 271828; do
    test -f "$ROOT/results/muon_clip/seed_$SEED/run_complete.json" || {
      echo "ERROR: incomplete run: load=$LOAD seed=$SEED"
      exit 1
    }
  done
done
)
```

At roughly four hours per MuonClip replicate on an M2 Pro, the full sequential
25-run block is a multi-day campaign. Parallel execution is valid only within a
single homogeneous hardware block and with disjoint optimizer/seed jobs.

### 3. Monitor and resume

The launcher writes per-run logs below
`logs/runs/<optimizer>/seed_<seed>.log`. For example:

```bash
tail -f "$RG_NANOGPT_EXPERIMENT_ROOT/logs/runs/muon_clip/seed_1337.log"
```

A live spectral snapshot is available with:

```bash
python scripts/run_experiment.py monitor \
  --optimizer muon_clip \
  --seed 1337 \
  --interval 30
```

Rerun the same `run` command to resume. The launcher validates and restores the
last finite checkpoint and never treats a partial replicate as complete.

## Sweep outputs and analysis rules

Each completed replicate writes the canonical files:

```text
results/<optimizer>/seed_<seed>/metrics.csv
results/<optimizer>/seed_<seed>/random_canary_manifest.json
results/<optimizer>/seed_<seed>/random_canary_metrics.csv
results/<optimizer>/seed_<seed>/spectral/layers.csv
results/<optimizer>/seed_<seed>/run_complete.json
```

For the load sweep:

1. match behavioral and spectral rows by `step`, not by nearest visual position;
2. calculate `DeltaNLL_d` from dose-0 and exposed canaries within the same run;
3. compare loads within each matched seed;
4. report all seed-level outcomes, mean, sample standard deviation, standard
   error, and 95% confidence interval across seeds;
5. retain `alpha_clip_xmax` as the primary spectral curve and `alpha_raw` as the
   required sensitivity analysis;
6. report validation loss/perplexity/accuracy at matched checkpoints and final
   protected test metrics;
7. do not count layers, canaries, or checkpoints as independent replicates; and
8. do not claim that `alpha < 2` is sufficient or necessary for memorization
   without the behavioral contrasts.

The 0%-load condition is the zero-additional-bank endpoint of the load-response
curve. It is not the behavioral negative control and does not guarantee absence
of ordinary FineWeb memorization or overfitting.

The dated `build_report.py`/`archive` workflow is locked to the original frozen
`baseline.yaml` campaign. Until a sweep-specific report builder is versioned,
use the canonical CSV files above for the harmful-load analysis and do not pass
a harmful-load config to the dated baseline report as though it were the
original campaign.

## Other hardware and provenance

Mac/MPS, CUDA, and TPU/XLA runs are separate hardware blocks. Do not pool seeds
from different accelerator types into one confidence interval. Use a stable
`RG_NANOGPT_HARDWARE_BLOCK_ID` where the platform cannot provide a sufficiently
specific hardware identity.

The production launcher requires a clean Git worktree, records source/config/
dependency/hardware provenance, uses adjacent nonblocking locks, and binds
WeightWatcher rows to exact model-state hashes. Preserve the campaign root
outside ephemeral compute before cleanup or VM termination.

## Status and the original dated report

For the original paired AdamW/MuonClip baseline:

```bash
python scripts/run_experiment.py status
python scripts/run_experiment.py analyze
python scripts/run_experiment.py archive
```

`analyze` requires all ten original baseline runs by default;
`analyze --allow-incomplete` is diagnostic only. `archive` excludes the large
corpus and checkpoints while preserving manifests, aggregate tables, figures,
reports, and replay provenance.

## Repository state and actual results

See [RESULTS.md](RESULTS.md). Executed notebooks and partial campaign artifacts
must never be represented as completed multi-seed results.
