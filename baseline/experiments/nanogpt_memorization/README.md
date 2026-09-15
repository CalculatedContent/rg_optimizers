# nanoGPT memorization — start with two runs, not eighty

**Default:** verbatim canaries, seed 1337, AdamW and plain Muon. Each full run
has 39,063 updates. The model is unchanged: one block, one head, width 128,
context 256. WeightWatcher still uses `fix_fingers="clip_xmax"`, `ERG=True`,
and `randomize=True`. Existing `run.py`, optimizer code, and `suite.json` are
unchanged by this workflow update.

## Tonight: one AdamW run

From your existing checkout, with the research environment activated:

```bash
cd baseline/experiments/nanogpt_memorization
caffeinate -dimsu python study.py run --optimizer adamw
```

This creates a new, readable timestamped results directory, for example
`/tmp/nanogpt_memorization_20260914_220000`. It prints the exact path, runs the
two-update AdamW smoke check, then starts **one** full AdamW run from scratch.
It saves terminal output in `logs/` and automatically analyzes a successful
full run. The `caffeinate` prefix is for Mac; on Linux use `python study.py ...`
and select `--device cuda` or `--device cpu` explicitly.

No cloning, shell options, pasted loops, environment reinstall, deletion of
old runs, or automatic resumption is performed. Do not start another copy if
the overnight run is already running.

## Analyze the saved results

```bash
python study.py analyze
```

This uses the last results location saved by this launcher and prints its
exact path. It works with a completed, failed, or still-running study. For
results from the earlier commands, specify the existing directory explicitly:

```bash
python study.py analyze --root /tmp/nanogpt_memorization_20260914_210500
```

The path above is an example: use your actual results directory. Analysis
reads JSON/CSV files only; it does not load checkpoints, alter training,
regenerate probes, or rerun WeightWatcher. Use `--no-plots` for tables only.

## Next: the matched Muon run

```bash
caffeinate -dimsu python study.py run --optimizer muon --latest
```

This uses the same results root as the preceding launcher command, but creates
Muon's separate run. It does not rerun AdamW. Without `--latest` or `--root`,
a new timestamped study is created.

To start both optimizers sequentially in a fresh study instead:

```bash
caffeinate -dimsu python study.py run
```

The default is **two smoke checks and two full runs**, not the old 80-run grid.
An error stops the sequence; it never skips bad gradients or silently retries
with a different seed, optimizer, learning rate, or backend.

## What the analysis produces

Results go in the selected results directory's `analysis/` folder:

| File | Contents |
|---|---|
| `summary.md` | Run status, observed exact recall, first observed recall, peak and latest values, limitations and warnings |
| `behavior.csv` | Dose/prefix groups at every observed checkpoint: exact match, free-running token accuracy, teacher-forced accuracy, suffix NLL |
| `examples.csv` | Individual canaries, suffix lengths, and cumulative presentations reconstructed from the saved schedule |
| `dose_summary.csv`, `canary_summary.csv` | First observed exact match, maximum observed group recall, latest recall; missing onset stays missing |
| `zero_dose_contrasts.csv` | Same-checkpoint, same-prefix differences from never-presented canaries |
| `optimizer_comparison.csv` | Muon-minus-AdamW differences only at common checkpoints and verified matching data, source, seed and hardware |
| `spectral.csv`, `figures/` | Raw per-matrix fit fields and separate behavioral/spectral plots, including clipped/raw alpha, fit D, ERG gap, traps and randomization distance |
| `source_hashes.json` | Hashes of the exact saved input files used for this analysis snapshot |

Dose is the **lifetime** presentation count (0, 1, 4, 16, 64), not the number
already seen at an early checkpoint. Per-canary cumulative counts are
reconstructed from the frozen slot schedule and completed updates; the old
runner did not record an independent exposure counter at every checkpoint.
The second half withdraws the canaries. Analysis preserves the full trajectory,
so later forgetting cannot hide earlier observed exact reproduction.

The shorter-prefix probes exist only at the final checkpoint. Their first
observation is not evidence of earlier onset. An interrupted run is not a final
result. A partial last JSON line is ignored in a live snapshot; a malformed
completed line, duplicate step, or inconsistent probe denominator is an error.

One seed is descriptive. Neither probes, matrices nor checkpoints count as
independent optimizer replications. Spectral changes do not alone establish
memorization or identify the matrix storing a particular canary. Inspect fit
support and randomized controls before interpreting alpha. Historical spectral
files are aligned by step, not by an independently stored model-state hash.

## Existing runs and failures

```bash
python study.py run --optimizer adamw --root /tmp/your_existing_results --resume
```

Resume is explicit and remains subject to the original runner's fingerprint
checks. This update does not change `run.py` or `configs/suite.json`, so it does
not itself invalidate existing training fingerprints. Do not change package
versions, backend or source during a run. Existing results are never deleted.
A successful saved smoke can be reused; it is not reused as a trained model.

The previously reported nonfinite-gradient failure is **not fixed by this
launcher update**, and its root cause has not been established. Failures retain
their logs and checkpoints; the launcher attempts a partial-data report before
returning an error to the terminal. It does not execute `exit` in your shell.

## Additional experiments are opt-in

Use `--seed 2027` (or another configured seed) only when ready to replicate.
Use `--stage pilot` for a shorter engineering check, not a full-run replacement.
`--recipe shared_aux_decay` remains a separately labeled secondary control.

The other conditions and five-seed grid remain available through the unchanged
low-level `run.py`; see `PROTOCOL.md` for the extended protocol. In particular,
`run.py plan --stage full` still lists that explicitly requested extended grid.
It is **not** what `study.py run` starts.

`run_full.sh` now delegates to the two-run workflow, without shell option
changes. It accepts the same named flags as `study.py run`; old positional
arguments are no longer used. `python study.py run --dry-run` prints the exact
commands without creating files or starting jobs.

## Installation and validation

Existing research environments need no reinstall for this update. For a new
environment only, from repository root:

```bash
python -m pip install -e './baseline/nanogpt_one_head[dev]'
python -m pytest -q baseline/experiments/nanogpt_memorization/tests
```

The workflow tests use synthetic saved-data fixtures and subprocess test
programs, not measured training results. The original WeightWatcher contract
test remains mocked. Actual target-hardware numerical integration is checked
by the smoke run, not claimed from these unit tests.

The full model matrix is expensive and `/tmp` is temporary storage. Preserve
valuable outputs outside `/tmp` before cleanup or reboot. A one-run wall-clock
estimate still requires measured throughput on the selected device.
