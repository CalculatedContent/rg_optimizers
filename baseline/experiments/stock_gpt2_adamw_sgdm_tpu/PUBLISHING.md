# Incremental GitHub archive for modified AdamW layer measurements

The production campaign saves completed WeightWatcher measurements to GitHub
while the modified AdamW optimizer with an SGDM-magnitude graft is running. The
result destination defaults to:

- Repository: [`CalculatedContent/rg_optimizers`](https://github.com/CalculatedContent/rg_optimizers).
- Results branch: [`adamw-sgdm-tpu-results`](https://github.com/CalculatedContent/rg_optimizers/tree/adamw-sgdm-tpu-results).
- Campaign subtree: `baseline/experiments/stock_gpt2_adamw_sgdm_tpu/results/<campaign_id>`.

The branch link becomes available when the first authorized campaign creates it.
Every campaign gets its own immutable identity and source commit. The production
campaign requires publication; there is no production switch that silently
leaves the scientific results only on the TPU VM. CPU smoke tests may explicitly
disable publication.

Use the automatic campaign described in [LAUNCH.md](LAUNCH.md). It performs the
permission check, watches new measurements, retries temporary failures, and
verifies final publication. `publish_metrics.py` also provides standalone commands
for recovery and a Python interface for the campaign supervisor.

## What is saved at each measured checkpoint

A checkpoint is publishable only after the CPU worker has committed its complete
measurement JSON and a coherent CSV export generation. The publisher briefly
shares the collector lock while reading that generation, then releases it before
contacting GitHub. Training and diagnostics can continue during upload.

Under `runs/<stage>/<candidate>/`, each completed optimizer step has:

| Remote artifact | Contents |
|---|---|
| `tracking/measurements/000000000.json` | Complete immutable measurement, exact validation pairing, all matrix rows, summaries, coverage, every available native raw and `clip_xmax` WeightWatcher field |
| `tracking/layers/000000000.csv` | All matrix rows for that step; nested native dictionaries are preserved as JSON-valued CSV cells |
| `tracking/groups/000000000.csv` | Per-block and per-role aggregates for that step |
| `tracking/receipts/000000000.json` | Local measurement fingerprint/hash and, when available, snapshot-pruning receipt |
| `tracking/summary.csv` | Updated compact summary across all published steps |
| `tracking/EXPORTS.json` | Published CSV paths and SHA256 checksums for this remote generation |
| `tracking/LOCAL_EXPORTS.json` | The corresponding local collector receipt, including checksums for local aggregate tables |
| `publication_manifest.json` | Run identity, committed steps, export-generation hash and SHA256 of each staged run artifact |

Filenames use **nine decimal digits** for the optimizer step. The example above
is initialization. Exact run ID, configuration hash, optimizer step, processed
tokens and weight/validation fingerprint accompany every snapshot. A mismatch,
missing declared matrix, changed immutable measurement or mixed CSV generation
stops publication rather than substituting another snapshot.

The archive contains all 72 transformer matrices and the two separately
identified embeddings when embedding tracking is enabled. The canonical model
averages still use the 72 transformer matrices. Alpha, KS statistics, fitted
tail counts, ERG gap, native fit fields, validity flags and coverage are preserved
for both raw and upper-tail-adjusted fits. See [METRICS.md](METRICS.md) for the
definitions; confidence weights are fit-quality heuristics, not confidence
intervals or probabilities.

Each commit can contain one new measurement or a bounded batch that completed
between supervisor polls. The full layer/group table is represented incrementally
by immutable CSV files for each step. Publishing a growing full table every time
would repeatedly store the same rows in Git history. At final publication the
publisher also saves the convenient aggregate `tracking/layers.csv` and
`tracking/groups.csv`.

The same run subtree includes immutable `config.json`, `manifest.json`,
`run_identity.json` and available `data_receipts.json`, plus current
`TRAINING_STATUS.json`, `TRACKING_STATUS.json`, required-step inventory,
`TRAINING_DONE`, and optional `DIAGNOSTICS_RECOVERY.json`. Final report JSON and
PNG/PDF plots are published after they are generated. Report identity must match
the actual terminal training step and token count.

Campaign manifests, search plans, selections, final report and campaign status
are archived under `campaign/`. The original campaign manifest is also pinned
as `campaign_manifest.json` beside the namespace ownership receipt
`PUBLISHING.json`. This provides the settings and decisions needed to interpret
every published run.

## Credentials and the preflight write check

Authenticate GitHub on the machine that starts the campaign. The publisher uses,
in order, `GH_TOKEN`, `GITHUB_TOKEN`, or the existing `gh` authentication for
`github.com`. A fine-grained token needs access to the result repository with
**Contents: read and write**. Credentials are inherited by the detached
supervisor through its environment or obtained from `gh`; they are not stored
in the campaign manifest, result files or command arguments.

Before expensive work, preflight reads the pinned source commit and result
branch, creates the dedicated branch if necessary, and commits a small unique
receipt under `preflight/`. It then reads back and checks the remote Git blob
identities. Resume performs a fresh write check, so an earlier success cannot
hide subsequently revoked access. The check does not change repository
permissions, collaborators or cloud IAM.

Ordinary HTTP 403 permission failures are explicit errors. Network failures,
server errors and rate-limit responses can be retried. GitHub error text is
redacted before it reaches receipts or console output. Credential-like fields
in proposed scientific metadata cause publication to be refused.

## Safe updates and durable retry

Publication operates only inside the assigned campaign subtree. It never writes
to `main` or `master`, deletes remote files or force-updates a reference. Every
new commit is based on the current result-branch tree, preserving other campaigns
and unrelated repository files. If another publisher advances the branch, the
publisher rechecks collisions and rebuilds its commit against the new parent;
conflict retries are bounded.

`PUBLISH_STATUS.json` is the local durable receipt for each run and campaign. On
success it records the remote commit SHA, exact verified Git blob SHAs, measured
steps and the result URL. On failure it records the error, whether retry is
appropriate, the last successful commit, published steps and pending completed
steps. Measurement JSON and local exports remain available for retry. Publication
does not require the large CPU tensor snapshot after its diagnostic result has
been committed.

The campaign checks for new completed steps during supervision. An archive
failure requests a graceful training pause and performs a bounded retry sequence
(default five attempts). Once publication recovers, training can resume from its
saved checkpoint without changing the scientific horizon. An exhausted failure
budget leaves an explicit incomplete campaign; it cannot produce a successful
completion marker. Recovery uses the same campaign identity and reruns the
publication checks before continuing.

A final run receipt has `complete: true` only after terminal training, a complete
required-step inventory, complete diagnostics with no pending/failed jobs, and
verified remote artifacts. An official early `target_reached` endpoint is valid;
it need not consume the full token budget. An explicitly diverged search run
can archive every requested checkpoint and its divergence status even when the
failing optimizer step has no requested spectral snapshot. Such a run remains
excluded from candidate selection. Complete publication does not imply that a
run met the benchmark loss target.

The campaign's success decision separately requires the official loss target,
complete diagnostics and confirmed publication of all required run artifacts
and the final campaign report. Repeating an unchanged final publication still
verifies the remote tree and returns its commit SHA and `complete: true`.

## Standalone inspection and recovery

The automatic campaign supplies these arguments from its immutable manifest.
For a manual recovery, use the **same** repository, branch, prefix, campaign ID,
source commit and run key recorded there:

```bash
python publish_metrics.py preflight \
  --repo CalculatedContent/rg_optimizers \
  --branch adamw-sgdm-tpu-results \
  --prefix baseline/experiments/stock_gpt2_adamw_sgdm_tpu/results/CAMPAIGN_ID \
  --campaign-id CAMPAIGN_ID \
  --source-sha PINNED_40_CHARACTER_SOURCE_COMMIT

python publish_metrics.py once \
  --repo CalculatedContent/rg_optimizers \
  --branch adamw-sgdm-tpu-results \
  --prefix baseline/experiments/stock_gpt2_adamw_sgdm_tpu/results/CAMPAIGN_ID \
  --campaign-id CAMPAIGN_ID \
  --source-sha PINNED_40_CHARACTER_SOURCE_COMMIT \
  --run-dir /persistent/adamw-sgdm/CAMPAIGN_ID/runs/full/CANDIDATE \
  --run-key full/CANDIDATE \
  --final
```

Run these commands from `baseline/experiments/stock_gpt2_adamw_sgdm_tpu` in the
repository checkout. Replace placeholders with recorded values, including the
real persistent mount.
`once` without `--final` publishes committed partial progress. `watch` uses the
same run arguments, polls every 15 seconds by default and automatically requests
final publication after training and diagnostics finish. Normal diagnostic drain
is not counted as a publication failure. Its transient-error budget defaults to
five consecutive failures; a standalone watcher exits on exhaustion, so use the
campaign supervisor when automatic trainer pause/recovery is required.

The `campaign` action takes `--campaign-dir` instead of `--run-dir`; `--final`
requests a verified final campaign metadata sync. This low-level action archives
supplied metadata. The campaign supervisor owns the benchmark-success decision.

The callable interface is:

```python
publisher = Publisher(repo, branch, prefix, source_sha, campaign_id)
publisher.preflight(campaign_manifest=manifest)
publisher.publish_run(run_dir, run_key="full/candidate", final=False)
publisher.publish_run(run_dir, run_key="full/candidate", final=True)
publisher.publish_campaign(campaign_dir, final=True)
```

## Scope and verification performed

GitHub receives scientific JSON/CSV, report plots and selected campaign metadata.
Tensor checkpoints, raw CPU weight snapshots, tokenized datasets, arbitrary logs,
PID files, caches and credentials are excluded. Keep resumable checkpoints on
persistent storage; optional verified GCS backup is a separate facility. GitHub
publication is the metric archive, not a substitute for checkpoint backup.

An individual artifact is limited to 80 MiB, below GitHub's documented 100 MB
blob limit. A truncated remote tree response is rejected rather than allowing
an incomplete namespace/collision check. Cross-origin HTTP redirects are refused
so authorization cannot be forwarded to another host.

This implementation derives from the publisher and report code in
`baseline/experiments/stock_gpt2_sgdm_tpu` at source commit
`98891bed61a816277b7a92f296e44173704f5910`. [REUSE_PROVENANCE.json](REUSE_PROVENANCE.json)
records the original file hashes. The sister experiment keeps the native
WeightWatcher schema and archive checks, uses its own configuration and package
imports, and writes to its own results branch and campaign namespace. Reports
identify the optimizer as modified AdamW with an SGDM-magnitude graft.

The previous SGDM experiment's test runs and measurements are provenance for
the reused implementation; they are not results for this optimizer. Verification
performed on this sister experiment is recorded in [VALIDATION.md](VALIDATION.md).
The focused sister publisher suite passed **23 tests** during preparation.
The publisher test suite uses an in-memory GitHub API with real Git blob hashes
to check native-field retention, incremental/final CSV exports, concurrent branch
changes, idempotence, immutable collisions, credential redaction, path limits,
stale measurements, target completion, divergence and retry backlog. The future
campaign's actual permission probe remains required before training.

The implementation follows GitHub's primary REST documentation for
[Git references](https://docs.github.com/en/rest/git/refs),
[Git trees](https://docs.github.com/en/rest/git/trees), and
[Git blobs](https://docs.github.com/en/rest/git/blobs). The checked documentation
uses API version `2026-03-10`.
