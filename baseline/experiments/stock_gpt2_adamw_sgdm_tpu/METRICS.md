# Spectral measurements for stock GPT-2 / FineWeb / modified AdamW

`metrics.py` observes immutable CPU weights in a separate CPU process. It does
not change the model, optimizer, learning rate, training data iterator, or TPU
graph. The trainer queues the same checkpoint weights whose validation result is
attached to the snapshot. Each result carries the run ID, optimizer update,
tokens seen, validation metadata, snapshot file SHA-256, and a deterministic
content SHA-256.

Per-snapshot `measured_seconds` in the JSON and canonical summary records CPU
measurement wall time (snapshot loading, hashing, both analyses, and row
normalization). It is output metadata, outside the immutable content identity.
Queue waiting is recorded separately by the trainer; neither value should be
confused with accelerator training throughput.

These diagnostics accompany the validation NLL. An alpha value, ERG gap, or
weighted average does not establish that the run reached the 3.28 benchmark
target.

The automatic campaign publishes each completed snapshot's full native JSON and
per-step layer/group CSV to its dedicated GitHub results namespace. The canonical
summary CSV updates incrementally; final aggregate tables and report JSON are
added after diagnostics drain. Remote verification retains the full raw and
`clip_xmax` fields, not just the average alpha. Publication failures request a
graceful training pause and cannot silently pass the final success gate.
[PUBLISHING.md](PUBLISHING.md) specifies the exact paths, commit receipts,
authentication and retry behavior. Tensor checkpoints remain on persistent disk
or optional GCS backup; GitHub stores the scientific measurement artifacts.

The optimizer graft applies to 48 physical transformer matrices, with packed QKV
normalized jointly. Diagnostics split QKV into Q/K/V, yielding the same 72-matrix
analysis inventory as the SGDM experiment. This analysis split does not modify
the optimizer. Embeddings retain native AdamW updates and are measured separately.

## Matrix inventory

The canonical aggregate contains **72 matrices** for the production 12-block
model, with six observations per block:

| Role | Stock GPT-2 source | Shape at width 768 |
| --- | --- | --- |
| Q | First 768 rows of `attn.c_attn.weight` | 768 × 768 |
| K | Second 768 rows of `attn.c_attn.weight` | 768 × 768 |
| V | Third 768 rows of `attn.c_attn.weight` | 768 × 768 |
| O | `attn.c_proj.weight` | 768 × 768 |
| MLP_IN | `mlp.c_fc.weight` | 3072 × 768 |
| MLP_OUT | `mlp.c_proj.weight` | 768 × 3072 |

Canonical names are `L00_W_Q` through `L11_W_MLP_OUT`. Extraction checks the
packed layout and expected shapes; a missing matrix fails snapshot creation.
Small CPU smoke configurations use the same six-matrix-per-block rule.

When `include_embeddings` is enabled, `EMBED_WTE` and `EMBED_WPE` are added as
separate observations. They receive their own summaries and are excluded from
the 72-matrix mean and role/block summaries. The tied `lm_head.weight` is not
counted a second time. Biases and LayerNorm vectors are not weight matrices and
are excluded. WeightWatcher's `max_N` is explicitly raised to 100,000 so the
50,257-row token embedding is not silently skipped by its default 50,000 limit.

## Two independent fits for every matrix

The pinned package is **`weightwatcher==0.7.7`**. Each snapshot is analyzed twice:

| Pass | `fix_fingers` | Recorded fields |
| --- | --- | --- |
| Raw | `False` | `alpha_raw`, `D_raw`, `n_tail_raw`, `xmin_raw`, `xmax_raw` |
| Clipped | `"clip_xmax"` | `alpha_clip_xmax`, `D_clip_xmax`, `n_tail_clip_xmax`, `xmin_clip_xmax`, `xmax_clip_xmax` |

Both passes use `ERG=True`, `randomize=True`, `mp_fit=True`, `fit="PL"`,
`pl_package="ww"`, `max_fingers=10`, `min_evals=20`, and CPU SciPy SVD
(`svd_method="accurate"`). The effective options are saved in each measurement.
Both passes receive the same deterministic diagnostic seed, making randomized
controls comparable. The raw pass supplies its own alpha and fit statistics;
the clipped frame's convenience `raw_alpha` field is retained as native output
but is never paired with the clipped fit's D, tail count, or bounds.

### Matching tail counts

`n_tail` is the number of eigenvalues in the **actual fitted interval**
`xmin <= eigenvalue <= xmax`, counted from the same eigenvalues used by that
WeightWatcher pass. A small observer subclass records this as
`observer_n_tail`; it does not change the fit or spectrum.

This detail matters for WeightWatcher 0.7.7: native `num_pl_spikes` counts
eigenvalues above the chosen xmin in the original spectrum, including removed
top eigenvalues in a clipped fit. Native `num_pl_spikes` remains unchanged and
is preserved separately. The clipped tail count is not reconstructed from
`num_fingers`, and a missing clipped tail count is not replaced by the raw count.

## Averages and weighting

For a given pass, matrix i has exponent alpha_i, KS fit distance D_i, and matched
tail count n_i. The four reported means are:

| Summary | Matrix weight |
| --- | --- |
| Equal mean | `1` |
| Confidence weighted mean | `1 / (D_i + 0.02)` |
| Tail weighted mean | `sqrt(n_i)` |
| Confidence and tail weighted mean | `sqrt(n_i) / (D_i + 0.02)` |

Each mean is `sum(w_i * alpha_i) / sum(w_i)`. Effective matrix count is
`sum(w_i)^2 / sum(w_i^2)`. Every summary also records valid and invalid counts,
weight sums, minimum/maximum alpha, and the count of alpha values below two.
No valid observations produces a null mean and effective count zero.

The canonical columns are:

- `alpha_raw_mean`
- `alpha_raw_confidence_weighted_mean`
- `alpha_raw_tail_weighted_mean`
- `alpha_raw_confidence_tail_weighted_mean`
- The corresponding four `alpha_clip_xmax_*` columns.

Each mean has corresponding `*_valid_count`, `*_effective_n`, and
`*_weight_sum` fields. Additional summaries use exactly the same formulas within
each matrix role and each transformer block. Embedding summaries are separate.

“Confidence” here is the established KS-based weighting heuristic. It is **not**
a calibrated confidence probability or a confidence interval. It is also
different from WeightWatcher's native `alpha_weighted` statistic, which is
retained only in the native fields.

An observation is eligible for these alpha summaries only if the native fit
status is `success`, alpha is positive and finite, D is finite in [0,1], the
matched tail count is a positive integer, and `0 < xmin <= xmax`. Failed and
missing fits remain in the layer table with an explicit status and reason.
Native numerical values are retained even when the normalized fit is invalid.

## ERG gap and all native metrics

The native WeightWatcher 0.7.7 definition is:

```text
num_ERG_spikes = detX_num
ERG_gap = detX_num - num_pl_spikes
```

It is a **signed difference of spike counts**, using WeightWatcher's native
detX constraint and native power-law spike count. It is not an alpha difference,
an absolute gap, or a normalized ratio. The two independently fitted passes can
have different native `num_pl_spikes`, so raw and clipped ERG fields are stored
separately. The exact fitted `n_tail` used by our alpha weights does not replace
`num_pl_spikes` in the native ERG calculation.

Every returned native field is retained in `native_raw` or `native_clip_xmax`.
These include ERG/detX, randomized controls, native `alpha_weighted`, singular
value/rank/norm metrics, MP fits, `num_fingers`, status, and warnings. Native
`num_traps` aliases `rand_num_spikes` in this pinned version. JSON represents
nonfinite values as null. No missing ERG value is reconstructed: it receives
`ERG_status_*="unavailable"`. Canonical ERG means and available counts are
provided separately from the alpha means.

Sources for the pinned implementation and package:

- [WeightWatcher 0.7.7 on PyPI](https://pypi.org/project/weightwatcher/0.7.7/)
- [Native ERG definition and independent fit fields, source commit bef487d](https://github.com/CalculatedContent/WeightWatcher/blob/bef487d1ab1ed4b4ab27c1e66dde7e55c0e2c605/weightwatcher/weightwatcher.py)
- [Clipped-fit implementation in the same source revision](https://github.com/CalculatedContent/WeightWatcher/blob/bef487d1ab1ed4b4ab27c1e66dde7e55c0e2c605/weightwatcher/RMT_Util.py)
- [Power-law interval bounds in the same source revision](https://github.com/CalculatedContent/WeightWatcher/blob/bef487d1ab1ed4b4ab27c1e66dde7e55c0e2c605/weightwatcher/WW_powerlaw.py)

## Coverage and status

Every declared matrix produces a row, including matrices skipped by
WeightWatcher, invalid fits, and missing snapshot matrices. A production result
therefore has 72 canonical rows even if fewer fits are valid. Coverage records
expected/returned/valid counts separately, unexpected or unassigned native
rows, missing weights, native ERG availability, and pass-level errors. A
non-complete fit inventory has `coverage.status="warning"`; this warning is
never converted into a full 72-matrix result by dropping invalid rows.

`TRACKING_STATUS.json` contains:

- `completed`, `failed`, `pending`, and their corresponding step lists.
- `warning_steps` for committed results with incomplete valid fit coverage.
- `expected_steps` and `missing_expected_steps`, read from the trainer's
  `tracking/EXPECTED_STEPS.json` (`{"steps": [0, ...]}`).
- `training_done`, from the run directory's `TRAINING_DONE` marker.
- `complete`: training is done, with no pending snapshots, operational
  failures, or missing expected measurements.
- `all_fits_valid`: at least one result exists and no result has a fit-coverage
  warning. This is deliberately distinct from queue completion.

Thus a processed but mathematically invalid power-law fit remains auditable as
a completed measurement with a warning. An operational failure, such as an
unreadable snapshot or two failed analyzer passes, retains the snapshot and
records a failure. It requires an explicit retry. A watching worker exits
nonzero on operational failure so the launcher can pause the trainer promptly.

## Queue, atomic commit, and disk retention

```python
receipt = queue_snapshot(run_dir, checkpoint, n_layers=12,
                         include_embeddings=True, max_pending=4)
```

The checkpoint supplies `step`, `tokens_seen`, `run_id`, `validation`, `model`,
`config`, and `manifest`. The checkpoint must already contain CPU model tensors.
The queue returns `queued`, `already_queued`, `already_measured`, or `queue_full`.
The caller pauses and retries on `queue_full`. Reusing a step with different
weights, validation, or manifest raises an immutable-identity error.

The worker writes one atomic JSON measurement per step, then atomically replaces
the layer, summary, and group CSV exports. `EXPORTS.json` records the committed
CSV hashes and included steps. Only after this commit does the worker prune the
snapshot's large weight file. An audit receipt retains its SHA-256, content
identity, and committed measurement hash. Failure or export interruption leaves
the original weight snapshot available for recovery. Process locks serialize
queue writers, workers, and export collectors; OS locks release after crashes.

Default retention bounds unprocessed weight snapshots at the queue limit.
Processed JSON measurements and CSVs are retained. `--keep-snapshots` or the
checkpoint configuration's `keep_snapshots=true` preserves processed weight
files too. Training checkpoint retention is managed by the trainer separately.
When the CPU worker falls behind, queue backpressure can pause training; wall
time therefore includes that pause and is not an accelerator-only benchmark.

## Worker commands and API

```bash
python metrics.py --check --threads 2
python metrics.py --run-dir /path/to/run --watch --threads 2
python metrics.py --run-dir /path/to/run --once --threads 2
python metrics.py --run-dir /path/to/run --once --retry-failed
```

`--check` runs two actual small CPU fits and verifies the pinned package,
required native fields, valid fit statistics, and the native ERG formula. It
fails explicitly if those capabilities are unavailable. The worker also runs
this probe before processing a queue. `--watch` polls at `--poll-seconds 2`,
drains queued snapshots after `TRAINING_DONE`, and then exits. The worker sets
CPU-only environment options and CPU thread limits before numerical imports.

Available Python APIs:

- `extract_matrices(state, n_layers=12, include_embeddings=False)`
- `queue_snapshot(run_dir, checkpoint, *, n_layers=None, include_embeddings=False, max_pending=4)`
- `measure(snapshot_path_or_payload, *, analyzer=None)`
- `collect(run_dir)`
- `process_pending(run_dir, *, analyzer=None, keep_snapshots=False, retry_failed=False)`
- `capability_check()`

`measure` does not seed training generators. It runs in the separate worker;
as an additional safeguard, direct calls save/restore Python, NumPy, and CPU
Torch RNG state, including when an analyzer raises. It never calls
`torch.manual_seed`, which would also reseed accelerator generators.

Output layout:

```text
RUN/TRACKING_STATUS.json
RUN/tracking/snapshots/000000000.pt
RUN/tracking/receipts/000000000.json
RUN/tracking/measurements/000000000.json
RUN/tracking/failures/000000000.json       # only on operational failure
RUN/tracking/layers.csv
RUN/tracking/summary.csv
RUN/tracking/groups.csv
RUN/tracking/EXPORTS.json
RUN/tracking/CAPABILITIES.json
```

Measurement JSON is the authoritative per-step record and keeps nested native
fields without column loss. CSVs encode native dictionaries as JSON cells and
can be rebuilt at any time using `collect(run_dir)`.

## Verification

`tests/test_metrics.py` checks the 72-matrix packed inventory, independent raw
and clipped statistics, all weighting formulas and effective counts, missing
and invalid fit coverage, preservation of native fields, RNG restoration,
immutable/idempotent bounded queues, snapshot tampering, failure recovery,
atomic export/pruning order, expected-step completion, and a real small
WeightWatcher CPU capability probe. These tests do not train a language model
or create a claimed FineWeb benchmark result.
