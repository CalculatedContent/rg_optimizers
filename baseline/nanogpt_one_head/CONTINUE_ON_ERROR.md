# Continue after a failed MPS replicate

The Apple-MPS supervisor runs every optimizer/seed replicate in a fresh worker
process. A failed worker is retried from the last verified finite checkpoint.
If all configured attempts fail, the supervisor records the incomplete run and
continues to the next optimizer/seed instead of terminating the entire batch.

This behavior is deliberately explicit rather than silent:

- the incomplete run receives `run_failed.json`;
- the batch receives `_batch_status.json` in the results root;
- the terminal prints a final completed/failed summary;
- the failed run never receives `run_complete.json` and is therefore excluded
  from normal completed-run analysis.

Use `--fail-fast` when debugging to restore immediate termination on the first
replicate that exhausts its retries.

Example:

```bash
rg-onehead-train \
  --config configs/reference.yaml \
  --optimizer muon \
  --seeds 1337,2027,4099,5003,6007,7013,8017,9011 \
  --device auto \
  --mps-retries 1
```

If one seed fails twice, the remaining seeds still run. Inspect the latest batch
record with:

```bash
cat /tmp/rg-nanogpt-one-head/results/_batch_status.json
```

A failed seed may be resumed later by running that seed alone without
`--no-resume`; the runner will use its last verified finite
`checkpoint_latest.pt` when available.
