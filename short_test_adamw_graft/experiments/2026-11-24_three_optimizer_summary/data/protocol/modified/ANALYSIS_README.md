# Five-seed modified AdamW analysis

Run `python analyze_results.py` from this directory after all five current
runs, seeds 1340–1344, finish. Every cohort result must come from
`runs/seed_<seed>/confirmation/rate_0.4/graft/seed_<seed>/result.json`.
The old result in `existing_seed1340/` is retained as a separate historical
record and is never included in the new cohort's mean or uncertainty.
The script reads recorded results only; it does not train or refit spectra.
Use `--root /absolute/experiment/path` when running from another directory.

The default command exits with status 2 if any seed is missing. It still
writes explicitly labeled partial outputs for inspection. Use
`python analyze_results.py --allow-partial` for a successful progress-only
invocation. A final five-seed summary is identified by
`complete_five_seeds: true` and `n_completed: 5` in `analysis/summary.json`.
When no current run has finished, outputs correctly report zero completed
seeds and all five missing. The historical result does not fill that gap.
An exclusive file lock serializes analysis writes by the four-worker
supervisor and the five-seed cohort monitor.

Each model average uses all 18 unique matrices, with the tied token/output
matrix counted once:

`weight = sqrt(num_pl_spikes) / (D + 0.02)`

`average_alpha = sum(weight * alpha) / sum(weight)`

Here `alpha` is the ordinary fitted exponent, not WeightWatcher's separate
`alpha_weighted` field. The script requires 18 successful fits at every
recorded step and the complete 41-point history (0, 500, ..., 20,000).

## Outputs

- `metrics_per_seed.csv`: validation loss, perplexity, accuracy, and weighted
  alpha at 2,000 and 20,000 updates; test metrics at 20,000 only.
- `model_alpha_history.csv`: every seed's weighted average and validation
  metrics at all 41 recorded steps.
- `model_alpha_history_summary.csv`: mean, sample SD, SEM, and 95% Student-t
  interval for the seed mean at each step.
- `layer_alpha_history.csv`: all recorded individual matrix fits, distances,
  tail sizes, and calculated weights (3,690 rows when complete).
- `layer_alpha_endpoints.csv`: the 90 final matrix/seed fits when complete.
- `endpoint_layer_summary.csv`: each final matrix's exponent mean and
  uncertainty across seeds.
- `paired_metrics_per_seed.csv`: current modified AdamW minus historical
  native AdamW, matched by seed label and budget. Negative loss/perplexity
  differences favor the graft; positive accuracy differences, expressed
  in percentage points, favor the graft. These comparisons span recorded
  runtimes and do not isolate optimizer effects.
- `historical_seed1340_metrics.csv`: the old seed's 2,000- and 20,000-step
  measurements, labeled separately by the filename and summary metadata.
- `historical_seed1340_discrepancies.csv`: current seed 1340 minus archived
  seed 1340 at every recorded step, available after the current run finishes.
  These are descriptive differences only, without confidence limits or a
  claim of runtime equivalence.
- `summary.json`: the aggregate measurements, paired differences, explicit
  completion status, runtime/package records, and input/source SHA256 hashes.

Seed-level model averages receive equal weight. SD uses denominator `n-1`;
SEM is `SD/sqrt(n)`; the interval uses the Student-t critical value with
`n-1` degrees of freedom. SD, SEM, and interval limits are null for a single
seed. Layer variation, fit-quality weights, and seed uncertainty remain
separate quantities. Perplexity is computed within each seed before its
mean is taken. Unrecorded 2,000-step test metrics are left blank.

The script checks the frozen architecture, data identity, schedule, source
hashes, and selected graft recipe against archived seed 1340. All available
current cohort runs must share recorded runtime/package metadata. Matching
package version strings alone does not establish identical kernels, numerical
libraries, hardware, or results. The archived native AdamW and SGDM records
remain historical-runtime comparators; they are not assumed to share the
current cohort's exact runtime. Only the five newly executed modified-AdamW
runs determine the current cohort's reported mean and uncertainty.

