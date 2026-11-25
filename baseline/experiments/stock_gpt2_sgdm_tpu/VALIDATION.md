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
two blocks, width32, context32, vocabulary256, two optimizer updates and128
synthetic training tokens. Both snapshots (steps0 and2) completed. Each contained
12 canonical transformer matrices and2 separate embeddings; both fitting passes
were valid for every row. Training and tracking processes both exited0, with
zero failed/pending snapshots. Processed tensor snapshots were pruned after
committed exports. Target attainment remained false because synthetic smoke
runs are categorically ineligible for the benchmark.

The plot exporter generated PNG and PDF output. The PNG was visually inspected:
four labeled panels were readable, and the figure prominently identified itself
as **SYNTHETIC SMOKE — NOT A BENCHMARK**. These smoke curves are not presented as
FineWeb results.

## Resume evidence

A tiny CPU run interrupted at update2 and resumed to update4 matched an
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
hashes. All13 external source links in [SOURCES.md](SOURCES.md) returned HTTP200
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
