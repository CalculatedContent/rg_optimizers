# Preparation validation — 2026-11-25

**161 tests passed, 0 failed, 0 skipped** on the pinned CPU numerical stack.
This verifies the prepared software, the automatic campaign, and real small
synthetic training/WeightWatcher execution paths.
It does **not** establish stock GPT-2/FineWeb convergence, TPU throughput, or
optimal SGDM hyperparameters. No TPU or GCS operation was executed during this
preparation. GitHub result publishing was exercised through an in-memory API with
real Git blob hashing; the future TPU host's live permission probe is still needed.

## Executed checks

| Area | Tests | Evidence covered |
|---|---:|---|
| One-command entry point | 12 | Read-only preview, isolated pinned environment, venv symlink preservation, saved campaign selection, explicit resume conflict rejection, free-space gate and sanitized child Python environment |
| Campaign orchestration | 18 | Automatic 11→4→1 search, verified early target, one fresh bounded 2× fallback, honest missed-target outcome, publication gates/retries, source provenance, immutable settings and restart/process identity |
| Configuration and tuning | 15 | Hashed configs, complete positive-LR horizon, fresh promotions, equal token budgets, separate holdouts, rejection of partial/nonfinite results, complete diagnostic requirements, resumed extra snapshots |
| Launch and backup boundaries | 17 | Read-only planning, no overwrite, resume identity, cloud403 before writes, unique destinations, shell-safe remote paths, checkpoint-pair race handling, first-target backup, venv preservation, interpreter mismatch refusal |
| Spectral measurements | 14 | 72/74-matrix production inventory, separate raw/clipped fits, exact matching tail counts/weights, native ERG formula, coverage, RNG preservation, immutable queue, atomic exports and pruning |
| Training/data/recovery | 9 | Batch-independent token prefix, shard-boundary targets, mean-gradient accumulation, exact validation slice, official-target guard, bit-exact CPU resume, checksum rejection, no overwrite, interrupted alias publication |
| Optimizer and TPU readiness gate | 51 | Native SGD parity in FP32/FP64, changed tensor LRs, momentum/decay, state interchange, no host scalar extraction, stable/recompiling/missing XLA telemetry refusal cases |
| Incremental GitHub publisher | 23 | Full native fields, remote Git hashes, immutable collisions, concurrent branch changes, stale/partial output rejection, final completion, credentials and retry backlog |
| Single-run CPU integration | 1 | Actual launcher → detached supervisor → trainer + separate native WeightWatcher worker → queue drain → JSON/CSV and PNG/PDF report export |
| Automatic CPU campaign integration | 1 | Two real candidates → automatic winner selection → fresh finalist; six native snapshots and byte-identical incremental/final archive through the simulated GitHub transport |

The complete suite finished in **253.877 seconds** in the preparation environment.
This duration is a test-suite observation, not an estimate for a TPU run. The
15 warnings were Matplotlib/Pyparsing deprecations and a pandas deprecation in
a deliberately incomplete DataFrame fixture. They did not affect test outcomes.

## Automatic campaign evidence

The multi-run integration used actual detached trainer and WeightWatcher workers,
with a one-block, width-24, two-head model, context 32 and vocabulary 64. Both
screening candidates trained for 64 synthetic tokens; their selected finalist
started fresh and trained for 128 tokens. Selection and all reports were automatic.

| Run | Training tokens | Measured optimizer steps | Matrix rows per snapshot | Verified final artifacts |
|---|---:|---|---:|---:|
| Screen, LR 0.01 | 64 | 0, 2 | 8 | 27 |
| Screen, LR 0.03 | 64 | 0, 2 | 8 | 27 |
| Selected finalist, LR 0.03 | 128 | 0, 4 | 8 | 27 |

All six snapshots had six canonical block matrices and two separately identified
embeddings, with valid raw and `clip_xmax` fits. Every run published initialization
incrementally before its final report. Native JSON bytes matched the committed
local measurement exactly. Each final archive contained the full per-step JSON
and layer/group CSV, aggregate CSV, reports and PNG/PDF plots. All required
diagnostics completed with zero pending/failed jobs. Re-entering the completed
campaign started no training or publication request.

The GitHub transport in this test was a simulated Git object service; the trainer,
optimizer, WeightWatcher, selection, report generator and publisher were real.
No live result branch was populated by the test. The result explicitly remained
`smoke_completed_not_a_benchmark`, `benchmark_target_met: false`, and
`successful: false`. This demonstrates the operational chain, not FineWeb quality.
The generated finalist plot was visually inspected and clearly labels the
synthetic setting, with readable loss, perplexity, alpha and ERG panels.

## Single-run native integration

The real integration smoke used the explicit `mode="smoke"` configuration:
two blocks, width 32, context 32, vocabulary 256, two optimizer updates and 128
synthetic training tokens. Both snapshots (steps 0 and 2) completed. Each contained
12 canonical transformer matrices and 2 separate embeddings; both fitting passes
were valid for every row. Training and tracking processes both exited 0, with
zero failed/pending snapshots. Processed tensor snapshots were pruned after
committed exports. Target attainment remained false because synthetic smoke
runs are categorically ineligible for the benchmark.

The plot exporter generated PNG and PDF output. The PNG was visually inspected:
four labeled panels were readable, and the figure prominently identified itself
as **SYNTHETIC SMOKE — NOT A BENCHMARK**. These smoke curves are not presented as
FineWeb results.

## Resume evidence

A tiny CPU run interrupted at update 2 and resumed to update 4 matched an
uninterrupted run exactly for every model tensor, every SGD momentum buffer,
the data cursor, Torch RNG state, next LR factor and final validation NLL.
Changing a scientific configuration or corrupting checkpoint bytes was rejected.
This is a numerical recovery test on CPU. TPU compiler/runtime and numerical
behavior require their own hardware verification; bitwise TPU parity is not
claimed.

The first-target/best alias recovery test simulates an interruption after latest
publication and verifies that missing aliases/sidecars are repaired from that
same verified checkpoint. A later checkpoint does not overwrite first-target
weights. A mismatched latest checkpoint/sidecar is refused rather than silently
rolled back.

## Conventional SGD and the XLA execution correction

Pinned PyTorch 2.6 SGD's scalar `add_` argument extracts a tensor learning rate as
a host scalar. `TensorLrSGD` uses tensor multiplication for that final update while
retaining ordinary momentum, coupled decay and the native state-dictionary format.
Tests compare six changing learning rates against native SGD in FP32 and FP64,
across zero, 0.9 and 0.98 momentum, separate decay groups, and continuation through
state dictionaries in both directions. The tested tensor-LR optimizer steps have
zero host scalar extractions.

The TPU preflight performs five disposable full-model microbatch updates, allows
three warmup updates, then requires two changed-LR updates without additional
XLA `CompileTime` samples. It also rejects optimizer scalar extraction. Tests
exercise this gate using explicitly simulated telemetry. **The actual TPU gate
has not been executed**, so it cannot yet establish device compilation reuse,
full accumulation parity, throughput or convergence.

## Environment used

| Dependency | Observed version |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.6.0+cpu |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| pandas | 2.2.3 |
| WeightWatcher | 0.7.7 |
| powerlaw | 1.5 |
| matplotlib | 3.9.4 |
| pytest | 8.3.5 |

The machine-readable [validation_evidence.json](validation_evidence.json)
records test counts, observed package versions, integration completion and source
hashes. [SOURCES.md](SOURCES.md) records the benchmark, optimizer studies, runtime
interfaces and GitHub API sources, distinguishing their evidence from new choices.

## Reproduce the checks

From the repository root, using an isolated environment installed from
`requirements-cpu.txt`:

```bash
python -m pytest -q baseline/experiments/stock_gpt2_sgdm_tpu/tests
```

The integration test keeps its parent process alive while detached children
finish. This matters in environments that terminate all descendants when a
single tool/command sandbox exits. It does not change the launcher's ordinary
detached-process behavior on a persistent TPU VM.

## Existing repository CI

The first publication was reviewed in [PR #184](https://github.com/CalculatedContent/rg_optimizers/pull/184).
GitHub's [source-syntax check](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45618716296)
passed. In the [existing executable-test workflow](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45618716108),
the core-baseline job passed, and the one-head tests passed 218 tests with 1 skipped.
The subsequent legacy GPT-2 integration command passed 86 tests and failed 2 tests
in `baseline/gpt2_small/tests/test_repeated_speedrun.py`.

Both failures are deterministic mismatches already present in base commit
`a5ce682eccba852991b9063211cc9070d170386a`:

1. `test_stock_baseline_schedule_matches_pinned_reference_for_every_update`:
   the old live `muon_speedrun/benchmark_config.py` applies
   `0.1 + 0.9*cosine`, while its pinned reference decays to zero. At the final
   update, the corresponding LR is 0.00006 versus 0. This is a real schedule
   disagreement, not floating-point noise.
2. `test_dedicated_experiment_config_matches_live_recipe`: the saved old
   MuonClip/AdamW configuration declares `cosine_to_zero`; the old live protocol
   declares `cosine_to_floor_0.10` and additional SGDM/alpha-control fields.

The test file, old live configuration, repeated-run planner, saved legacy
configuration and pinned reference were compared at the base and initial PR head
`d19ba134b7bb8b61d77253f4d47c55add4e0774b`; all five are byte-identical. The failed
command collects explicit legacy paths and does not collect this new experiment.
The base-source arithmetic reproduces the observed schedule mismatch; the full
legacy base suite was not rerun. These unrelated files were left unchanged.

This folder has its own `config.py` and does not import the old
`benchmark_config.py` or its alpha controller. Its 161 scoped tests, including its
own cosine-horizon and real launcher/metrics tests, passed. Repository-wide CI
must therefore not be described as entirely green, even though the new
experiment's scoped validation passed.

## Checks reserved for the future TPU session

The supplied launch preflight must verify the actual pinned Torch/XLA/libtpu/JAX
environment, existing node identity, eight global/addressable devices, persistent
mount/free space, all required data hashes, causal attention forward/backward,
and finite full-model SGDM updates with stable changed-LR compilation after
warmup. Native WW/ERG capability is checked in a
separate CPU process. An explicit GCS destination requires its own write/read/delete
probe before training; the preparation tests used mocks for cloud failures and
staging races and made no real cloud-storage requests. Production additionally
requires a live GitHub write/readback proof and verified incremental publication.

Measure actual full-model throughput and spectral-worker cost on that hardware
before estimating the candidate search duration. Backpressure, validation and
checkpointing are recorded separately. Final success requires the exact official
validation target, complete declared measurements and confirmed GitHub publication;
a completed budget alone
does not establish either convergence or global optimality.
