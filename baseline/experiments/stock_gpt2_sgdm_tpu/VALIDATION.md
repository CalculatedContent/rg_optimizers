# Preparation validation — 2026-11-25

**56 tests passed, 0 failed, 0 skipped** on the pinned CPU numerical stack.
This verifies the prepared software and its small synthetic execution path.
It does **not** establish stock GPT-2/FineWeb convergence, TPU throughput, or
optimal SGDM hyperparameters. No TPU or GCS operation was executed during this
preparation.

## Executed checks

| Area | Tests | Evidence covered |
|---|---:|---|
| Configuration and tuning | 15 | Hashed configs, complete positive-LR horizon, fresh promotions, equal token budgets, separate holdouts, rejection of partial/nonfinite results, complete diagnostic requirements, resumed extra snapshots |
| Launch and backup boundaries | 17 | Read-only planning, no overwrite, resume identity, cloud403 before writes, unique destinations, shell-safe remote paths, checkpoint-pair race handling, first-target backup, venv preservation, interpreter mismatch refusal |
| Spectral measurements | 14 | 72/74-matrix production inventory, separate raw/clipped fits, exact matching tail counts/weights, native ERG formula, coverage, RNG preservation, immutable queue, atomic exports and pruning |
| Training/data/recovery | 9 | Batch-independent token prefix, shard-boundary targets, mean-gradient accumulation, exact validation slice, official-target guard, bit-exact CPU resume, checksum rejection, no overwrite, interrupted alias publication |
| Full CPU integration | 1 | Actual launcher → detached supervisor → trainer + separate native WeightWatcher worker → queue drain → JSON/CSV and PNG/PDF report export |

The complete suite finished in **85.639 seconds** in the preparation environment.
This duration is a test-suite observation, not an estimate for a TPU run. The
only suite warning was a pandas deprecation warning in a deliberately incomplete
DataFrame test fixture; it did not affect any test outcome.

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
hashes. All 13 external source links in [SOURCES.md](SOURCES.md) returned HTTP 200
during preparation.

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
`benchmark_config.py` or its alpha controller. Its 56 local tests, including its
own cosine-horizon and real launcher/metrics tests, passed. Repository-wide CI
must therefore not be described as entirely green, even though the new
experiment's scoped validation passed.

## Checks reserved for the future TPU session

The supplied launch preflight must verify the actual pinned Torch/XLA/libtpu/JAX
environment, existing node identity, eight global/addressable devices, persistent
mount/free space, all required data hashes, causal attention forward/backward,
and a finite full-model SGDM update. Native WW/ERG capability is checked in a
separate CPU process. An explicit GCS destination requires its own write/read/delete
probe before training; the preparation tests used mocks for cloud failures and
staging races and made no real cloud-storage requests.

Measure actual full-model throughput and spectral-worker cost on that hardware
before estimating the candidate search duration. Backpressure, validation and
checkpointing are recorded separately. Final success requires the exact official
validation target and complete declared measurements; a completed budget alone
does not establish either convergence or global optimality.
