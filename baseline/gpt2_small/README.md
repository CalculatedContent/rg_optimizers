# GPT-2 Small / FineWeb-Edu validation

This is the GPT-2 Small experiment, under `baseline/gpt2_small`. Its training module
is `rg_gpt2_small.experiment`; configurations and launchers live here. It reuses the
shared GPT implementation, optimizers, SPMD, corpus validation and WeightWatcher
adapter from the older `rg_nanogpt_one_head` infrastructure package. That dependency
does not constrain the head count. Existing one-head and continuous8 workflows stay
in their original directory. Existing disk and bucket paths are retained for data reuse.

The TPU launch scripts expose both source packages through `PYTHONPATH`. For local
development, from the repository root, install both with
`pip install -e baseline/nanogpt_one_head -e baseline/gpt2_small`.

## Continue with MuonClip on the existing allocation

The user authorized a fresh continuous MuonClip run after the diagnostic crash.
Use the current 48-hour TPU, installed environment and preserved FineWeb:

```bash
python3 baseline/gpt2_small/scripts/run_muonclip.py start
python3 baseline/gpt2_small/scripts/run_muonclip.py status
```

This starts one fresh process under `rg-gpt2-muonclip-<timestamp>.service` with
`Restart=no`, in `/mnt/disks/rg-data/gpt2small/muonclip-night-<timestamp>`.
It never invokes the cleanup/reallocation scripts, changes the installed packages,
downloads the corpus, or writes into previous experiment directories/cloud prefixes.
The saved `port-check-20261004-045633` crash evidence remains intact. A shared launch
lock and checks for existing services/trainers prevent simultaneous TPU jobs.

Only the crashing per-tensor gradient diagnostic is disabled. A scalar finite
loss/gradient-norm guard remains before each update. The 124M GPT-2 model, data,
MuonClip/auxiliary AdamW settings, batch size and long-run LR schedule are retained.
This bypass does not fix or explain the earlier numerical failure or establish TPU
stability. It is an explicitly requested experimental run, not a passed validation.

Training stays in one process from initialization until the existing allocation
cutoff (20 minutes before its recorded expiry), manual STOP, token budget or error.
There is no AdamW gate, stop/resume transition or automatic restart. A watchdog stops
a phase with no progress for 30 minutes and saves a failure report. First completed
updates are printed explicitly; a watch heartbeat alone is not completion evidence.

Token error, NLL and full checkpoints are recorded at steps 0, 1, 2, 4 and every 25
updates. All 72 matrix raw/clipped alphas are paired with the same-step token metrics
at step 25 and every 100 updates. Evaluation windows stay fixed. Synchronization,
evaluation, spectra and checkpoint upload overhead contribute to elapsed runtime.

Three rolling local full checkpoints and an initialization milestone limit disk
usage. During training, verified uploads keep three rotating cloud slots plus
initialization; every scalar/spectral JSON record is retained. The cloud prefix is
`gs://tpu-builders-504820-ww-continuous8/gpt2small/<run-name>`.
`muonclip/checkpoints/LATEST_VERIFIED.json` is published only after checkpoint and
metrics uploads, with object generation and CRC32C. It names a cloud slot, not a
local filename; validate generation/checksum when downloading for recovery. A failed
upload stops the trainer, retains local evidence, and does not advance this pointer.
Exit backup additionally archives the remaining local files using object permissions.
Old-run checkpoints and cloud objects are never deleted by this workflow.

## Current validation failure

Observed failures and their attribution are tracked in [TPU_PORT_BUGS.md](TPU_PORT_BUGS.md).
Diagnosing bugs in the PyTorch/TPU port is an explicit experiment objective.

From an updated Cloud Shell checkout, use the existing 48-hour TPU for a bounded
four-update AdamW diagnostic (no allocation or data download):

```bash
python3 baseline/gpt2_small/scripts/retry_adamw.py start
python3 baseline/gpt2_small/scripts/retry_adamw.py status
```

It runs independently of Cloud Shell under a new service and fresh output directory,
pins the checked-in source, and refuses concurrent trainers. Training is limited to
20 minutes with up to 10 more for backup. It records per-layer gradients, clipped
gradients, weights and optimizer moments as finite flags/extrema; only small reduced
tables move to CPU. Full checkpoints are saved at each completed update. Source and
runtime versions, input offsets, XLA metrics, stage times and failures accompany the
run. `TPU_PORT_FAILURE.json` records observed failure without assuming upstream fault;
`PROBE_STATUS.json` reports completion or failure. Existing cloud upload verification
runs on exit. This is diagnostic instrumentation, not a throughput measurement.

The saved 2026-10-04 04:19 UTC stack from commit `4631a3d` is inside the
nonfinite-gradient diagnostic's per-parameter CPU copy. That branch is reached only
after detecting a nonfinite loss or aggregate gradient norm. The latest checkpoint
pointer was step 0. This is a failed numerical validation, not evidence of healthy
training. The origin of the nonfinite result is still unconfirmed.

The failure handler now writes the scalar failure and raises immediately, avoiding
the full gradient-copy loop. Validation synchronizes XLA before reading diagnostic
scalars, prints its pre-update stage, and emits Python stacks every five minutes.
Each validation phase has a 30-minute limit (also bounded by allocation time);
on expiry its child is terminated and the report says validation is incomplete.
These changes improve failure reporting and bound waits; TPU numerical stability
has not yet been demonstrated. They do not change the model, data or learning rates.

## Model and data

124,439,808 trainable parameters: 12 independent blocks, 12 heads, width 768,
head dimension 64, MLP width 3072, context 1024, vocabulary 50257. GELU, LayerNorm,
causal attention and tied embedding/output weights; biases enabled. The six separate
projection matrices per block yield exactly 72 named trajectories. No QKV fusion or
layer sharing is introduced.

The existing corpus is `/mnt/disks/rg-data/continuous8/data`: 5 billion training
GPT-2 tokens and 10 million tokens each for validation/test, stored as three uint16
files with document indexes. Every launch validates metadata, sizes, SHA256 identities
and the document-disjoint split contract. This workflow never prepares/downloads data.
Evaluation uses deterministic windows within each split; train/validation/test probes
are fixed across optimizers and resumes. They are not the speedrun benchmark protocol.

## Configurations

- `configs/gpt2_small_fineweb_adamw_baseline.yaml`: AdamW 6e-4, betas .9/.95,
  weight decay .1, gradient clipping 1.0.
- `configs/gpt2_small_fineweb_muonclip_baseline.yaml`: existing advanced MuonClip,
  matrix LR .02, momentum .95, Nesterov, five NS iterations, QK threshold 100;
  auxiliary AdamW LR 6e-4 independently configurable.
- `configs/gpt2_small_fineweb_muonclip_long_ww.yaml`: 100B token presentations;
  refuses CLI launch without `--allow-long-run`. No loss-based stopping.
- `configs/gpt2_small_cpu_smoke.yaml`: tiny synthetic-corpus CPU integration tests.

**LR caveat:** advanced MuonClip scales the orthogonal update by
`update_rms_scale * sqrt(max(matrix.shape))`, here .2 times that square root.
The requested .02 is retained explicitly, but is not equivalent to the plain Muon
parameterization. Its stability is NOT established by this configuration. Do not
call this a reproduced reference benchmark or launch a long run before validating it.

Global microbatch 8, accumulation 4, context 1024 = 32,768 tokens/update, split over
eight chips. Batch size, accumulation, warmup, cosine horizon, floor, max_steps and
max_tokens are configurable. Stop-after is an interruption point, not an LR-horizon
change. max_tokens is rounded up to the next complete optimizer update.

## Run only the short validation on the existing TPU

From a clean Cloud Shell checkout of this commit:

```bash
python3 baseline/gpt2_small/scripts/cloudshell.py
```

This verifies the existing active queue/node, gracefully stops the old trainer,
archives compact scientific results and verifies the archive, then deletes only
recognized model checkpoint files in the known old run directory. It does not
format/delete disks, remove the corpus/caches/indexes, or allocate a replacement TPU.
Old scalar records are retained as well as archived; ambiguous output is not removed.

The exact commit is checked out under a fresh persistent-disk validation directory.
The sequence is AdamW through step 4, MuonClip through step 4, then fresh-process
resume of each through step 25. The baseline horizon stays fixed at 1000 steps.
MuonClip WW runs at steps 4 and 25. All earlier metric/WW file hashes must survive
resume unchanged. Short validation stops on nonfinite metrics, increasing training
probe NLL, incomplete matrix inventory or null controls. Failed alpha fits are retained
as NaN with status/reason, never replaced with clipped alpha.

The cutoff is read from the previous allocation record; five minutes are reserved
for checkpoint/backup. Setup/compilation/WW may consume the remaining window: an
incomplete validation is reported honestly, and no new machine is requested.
Outputs are copied to `gs://tpu-builders-504820-ww-continuous8/gpt2small/` on exit;
a failed cloud copy is an error and leaves persistent-disk outputs intact.

25 updates prove plumbing and initial learning direction only. They do not establish
convergence or reproduce CE 3.28. A longer matched benchmark is needed before claiming
normal GPT-2 training quality, and before the long scientific experiment.

## Replace an expired allocation while retaining FineWeb

From the updated, clean Cloud Shell checkout:

```bash
python3 baseline/gpt2_small/scripts/reallocate_validation.py launch
python3 baseline/gpt2_small/scripts/reallocate_validation.py status
```

The launcher replaces only `ww-continuous8-24h-20261003-s1337` and its node.
It retains the existing `ww-continuous8-pilot-20261002-s1337-data` disk and all
cloud objects. It waits for disk detachment and attaches that same disk to one
new v5litepod-8 with a server-enforced four-hour allocation limit, including setup.
The startup script mounts the existing ext4 filesystem; it never formats a disk.
The corpus and Python environment are reused without downloading or reinstalling.
The fixed replacement queue name prevents duplicate launches on repeated commands.

Short validation runs independently of Cloud Shell in `rg-gpt2-validation.service`.
The new root is `/mnt/disks/rg-data/gpt2small/ww-gpt2-validation-20261004-s1337`.
Cloud uploads are tested before training using object permissions, CRC32C and
size verification. Exit backup uses this same uploader rather than bucket-metadata
operations. The complete logs and outputs remain on the persistent disk on failure.

This is a fresh validation from initialization, not continuation of the failed run.
Before each optimizer update, the runner checks loss and gradient norm. A failure
writes `nonfinite_diagnostics.json` with the step, loss values and aggregate gradient
norm, then stops before applying the invalid update. The earlier nonfinite
gradient's cause is still unconfirmed; these checks do not claim to fix it.
Only if the short AdamW checks pass does MuonClip validation proceed, followed by
the existing resume checks. No long run starts automatically. The service does
not restart automatically after failures or reboot. A finished service does not
delete its TPU: the allocation limit remains four hours unless stopped earlier.

### Allocate 48 hours instead

```bash
python3 baseline/gpt2_small/scripts/reallocate_validation.py launch --hours 48
python3 baseline/gpt2_small/scripts/reallocate_validation.py status --hours 48
```

The queued-resource API exposes no update operation for extending the requested
lifetime. This replaces only the four-hour validation request with
`ww-gpt2-validation-48h-20261004-s1337`. If its validation service is already
running, it is stopped before deletion; all existing files remain on the same
data disk. The new VM mounts that disk, reuses FineWeb and the environment, and
runs fresh short validation in a separate directory. Repeating this command
does not replace an existing 48-hour request or create a second machine.

Both the server-enforced allocation limit and the startup deadline use 48 hours;
the worker deadline reserves ten minutes, and validation reserves another five.
The queue can wait up to four hours for capacity, independently of the 48-hour
allocation lifetime. At the published $0.60/chip-hour Flex-start rate, eight
chips for 48 hours cost $230.40 before storage and any earlier allocation usage.
Short validation still stops after its checks; it does not automatically start
the long experiment. The allocation remains available until deletion or expiry.

## Records, checkpoints and timing

### Capture a stalled validation

From Cloud Shell on the updated branch:

```bash
python3 baseline/gpt2_small/scripts/capture_stall.py --stop
```

This targets the 48-hour validation run above. It captures process CPU/memory,
logs, saved XLA metrics and live Python/native stacks before stopping only
`rg-gpt2-validation.service`. A bounded, optional py-spy installation goes in a
temporary directory, leaving the training environment and checkout unchanged.
If profiling is unavailable, process/log diagnostics are still retained.
Reports remain under the run's `diagnostics/stall-*` directory on the persistent
disk; a compact summary is printed for sharing. This command does not upload them.
Omit `--stop` for a read-only capture. A blocked update may be lost on stop; no new
checkpoint is promised. Existing checkpoints, FineWeb and the TPU allocation
remain. The allocation continues to incur compute usage until deletion/expiry.
Use `--show-last` to print the saved capture, native stack and current service state
without another profiler installation or stop. Both old and new package paths are
recognized when diagnosing the already allocated machine.

The validator's `WAIT` lines only indicate a live process, not a completed update.
Likewise `before_update: 2` confirms the gradient check before update 2, not its
completion. An extended wait requires inspection, not an assumed compilation ETA.

### Scientific records

Per-step immutable JSON scalar and WW records are written incrementally. NLL,
perplexity, top-1 accuracy, error (fraction), steps, token presentations, wall time,
LR and pre-clipping gradient norm are included. All WW library columns are retained,
including raw/clipped fits, randomized distance, bounds, KS/D, spectral measures and
fit status. `xmin/xmax/D` are library-returned clipped-fit fields, not invented raw-fit
bounds. Raw failure is explicit; alpha is never constrained toward 2.

Three rolling full checkpoints contain model, both optimizer states where applicable,
step-derived cosine scheduler state/config, token count, data sampler RNG, all training
RNG states and run identity. Checkpoint writes are atomic; old checkpoints are pruned
only after publishing the new pointer. A checkpoint includes pending metrics/WW records:
resume completes missing writes, appends later records, and refuses histories ahead of
the checkpoint rather than erasing data. Config/data/software version mismatch fails
closed. Optional milestone copies are independent of the rolling set. A single writer
lock prevents concurrent mutation. CPU exact-resume tests compare model and optimizer
states bit for bit. Numerical equivalence on TPU replacement still requires TPU testing.

The first two update durations include compilation and execution and are reported
separately; they are **not pure compiler timing**. Native XLA CompileTime/ExecuteTime
metrics are also saved in `logs/xla_compile_metrics_after_step_*.txt`. Later synchronized update durations
measure training throughput, excluding evaluation/WW/checkpoint overhead. End-to-end
throughput is reported separately. The validation runner intentionally synchronizes
once per update to measure completed TPU work; the long configuration only synchronizes
at measurement boundaries after the first two updates. This is not a reproduced speedrun.

WW supports fixed `interval`, explicit `steps`, and `logarithmic` 1/2/5-per-decade
schedules. A full pass reports seconds and recommends at least 9x that duration of
training between passes for at most 10% WW-only overhead. Include other overhead when
selecting the eventual long-run schedule. Scalar checkpoints are bounded in count;
no thousands of full checkpoint copies are retained.

## Analysis and tests

```bash
python baseline/gpt2_small/scripts/analyze.py /mnt/disks/rg-data/gpt2small/VALIDATION/muonclip
PYTHONPATH=baseline/gpt2_small/src:baseline/nanogpt_one_head/src python -m pytest baseline/gpt2_small/tests -q
```

Analysis creates CSVs and loss/perplexity/error-vs-token plots; mean/min raw and clipped
alpha trajectories; raw-alpha/error regression plots; all-layer raw/clipped plots
by matrix type; and per-matrix initial/latest/min/delta/recent slope/recent variance.
Correlations along one trajectory do not establish causation or independent-sample
significance. Tokens are presentations, not necessarily unique training tokens.
