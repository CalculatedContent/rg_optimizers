# Main nanoGPT speedrun: fixed recipes, repeated seeds

## Scientific question

Do the Muon/AdamW loss and spectral trajectories reproduce across random
initializations at the same token budget? This is a baseline variability study,
not a hyperparameter sweep. Both optimizers use **stock GPT-2 Small** and the
pinned GPT-2-tokenized FineWeb benchmark. The model has 12 blocks, 12 heads,
width 768, MLP width 3072, context 1024 and vocabulary 50257; 124,439,808 parameters.
See [the architecture audit and all matrix dimensions](../STOCK_ARCHITECTURE.md).
Historical modified six-head results do not enter this paired comparison.

## Frozen protocol

- Six sequential fresh runs: Muon/AdamW paired at seeds **1337, 1338, 1339**.
  Optimizer order alternates by seed. Previous single-run results are retained
  separately and are not silently substituted for new replicates.
- **19,560 updates / 10,255,073,280 tokens per run**, 524,288 tokens per update;
  global microbatch 64, eight accumulation passes, verified TPU flash attention.
- Same initialization within each seed pair. The initialization varies between
  pairs; token order remains fixed for all six runs. This does not estimate
  variability from corpus resampling or different training orders.
- Muon hidden-matrix learning rates and mathematics remain fixed. Muon
  uses LR 0.04 on hidden matrices; AdamW uses 0.0006 and decoupled decay 0.1 there.
  Both use AdamW LR 0.0006 on embeddings, biases and LayerNorm; betas 0.9 / 0.95.
  Embedding matrices use decay 0.1, biases/LayerNorm use zero decay. The tied
  token embedding/output is one parameter with one optimizer state. The old
  architecture's separate embedding/head rates cannot be retained with weight tying.
  **Weight decay differs, so this compares recipes, not an isolated optimizer effect.**
- 700-update warmup, cosine decay to zero at update 19,560; global L2 gradient clipping at 1.0.
  [Pinned reference and explicit TPU/Muon differences](BENCHMARK.md).
- No target-based early stop. Record first observed NLL <=3.28, but continue to
  19,560 for equal-budget final statistics. Target times have 250-update resolution.
  This target comes from the original GPT-2/FineWeb reference; TPU convergence remains unverified.
- Every 250 updates: full 10,485,760-token validation loss, perplexity, top-1 token
  error, checkpoint and raw/clipped WeightWatcher alpha for all 72 matrices.
  These are validation metrics, not a separately held-out test score.
- Existing CPU-only spectral worker, immutable snapshots, disk saves and verified
  final cloud backup are retained. No in-training per-tensor diagnostic scans added.

## Runtime and access

Runs use one existing, single-host v5e-8 TPU and the mounted experiment disk.
The launcher checks the live Google Cloud lease under the invoking account and
refuses less than **72 hours 45 minutes remaining**. Each run is capped at twelve
hours including setup and backup; the suite reserves another 15 minutes for
reporting, plus a 30-minute lease margin. These are conservative limits, not ETAs.
Actual six-run duration must be measured. A 48-hour allocation cannot accommodate
the six 12-hour caps; use the single-run launcher for one full-budget experiment. This entry point does not allocate, extend, or delete TPUs.

It requires the existing environment at `/mnt/disks/rg-data/continuous8/venv`,
reuses the verified benchmark cache, and requires 240 GiB free for retained
checkpoints and spectral snapshots. Missing requirements are explicit failures;
it never deletes old data to free space or silently changes microbatch.

## Commands

Use a clean checkout of the pushed commit. From the repository root:

```bash
# Read-only plan, safe on a Mac or TPU:
python3 baseline/gpt2_small/speedrun.py plan

# Mac / Cloud Shell: specify the sufficiently long-lived TPU allocation:
python3 baseline/gpt2_small/speedrun.py start --node YOUR_TPU_NODE

# Already on that TPU (avoids self-SSH; cloud read permissions still required):
python3 baseline/gpt2_small/speedrun.py start --here --node YOUR_TPU_NODE

python3 baseline/gpt2_small/speedrun.py status --node YOUR_TPU_NODE
python3 baseline/gpt2_small/speedrun.py report --node YOUR_TPU_NODE
```

`--zone` defaults to `us-west4-a`; the project is `tpu-builders-504820`.
`--ssh-key-file /path/to/key` selects an existing usable SSH key from the Mac.
Nothing is launched until the lease and idle-machine checks pass. It will not
stop a currently active AdamW or Muon job. After the systemd service starts,
it runs independently of the SSH session. No automatic restart or re-run of a
partially completed job. A failure halts the suite and preserves evidence;
remaining runs require an explicit decision after diagnosis.

## Outputs and statistics

`/mnt/disks/rg-data/gpt2small/NANOGPT_SPEEDRUN_SUITE_LATEST.json` identifies the
suite directory, pinned commit, lease and systemd service. Each child directory
has a unique suite/optimizer/seed name, `manifest.json`, training/validation
metrics, checkpoints, per-layer spectral CSVs and verified cloud backup under
`gs://tpu-builders-504820-ww-continuous8/gpt2small/<child-name>`.

The suite produces:

- `PLAN.json`, `SUITE_STATUS.json`: frozen plan, current job, explicit outcome.
- `COMPARISON.json`: all six completion states, paired final differences
  **AdamW minus Muon**, first-observed target times and sample sizes.
- `observations.csv`: per-seed validation observations with training elapsed and
  end-to-end elapsed time (including that run's setup).
- `curves.csv`: optimizer/update/metric mean and sample SD across available seeds.
- `layers.csv`: optimizer/update/matrix raw/clipped alpha mean and sample SD
  across available seeds, paired to the same validation snapshot.
- `comparison.png`: loss, token error, mean raw alpha and minimum raw alpha
  trajectories with mean ± SD across available seeds.

Reports update after each successful run and can be regenerated with `report`.
Partial trajectories are visible with their available sample counts. Final paired
statistics require completed, fully evaluated, tracked and backed-up pairs.
A single seed has no estimated SD. Unreached targets remain non-successes;
conditional crossing-time means must not be mistaken for overall speed.

Three seeds are exploratory, not high-powered evidence. Checkpoints along a
trajectory are correlated: do not pool them as independent replicates or treat
an alpha/loss time correlation as causal. This protocol reports seed SD, not a
confidence interval and not the spread across different matrix types. Tune or
extend runs only after inspecting these fixed-recipe baselines.
