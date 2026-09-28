# Four-device TPU data parallelism

This is an **additive execution path**. It does not change or replace the
existing single-device runner, configs, checkpoints, or TPU setup script.

## What it does

`rg_nanogpt_one_head.tpu_distributed` launches one process per TPU device with
`torch_xla.launch`. On a `v5litepod-4`, that is four training processes.

The configured global optimizer batch is preserved:

```text
single device:
  batch 4 x context 256 x accumulation 8 = 8,192 tokens/update

four devices:
  4 ranks x batch 4 x context 256 x local accumulation 2
  = 8,192 tokens/update
```

Gradients are averaged across all four ranks **before** clipping and before the
Muon/auxiliary-AdamW optimizer steps. Rank zero alone owns evaluation,
WeightWatcher, metrics, checkpoints, and the held-out test audit.

The distributed protocol is fingerprinted separately through additive runtime
metadata. It cannot be confused with or resumed as a historical single-device
run.

## Quick run

After the normal TPU setup:

```bash
cd /tmp/rg_optimizers/baseline/nanogpt_one_head
chmod +x tpu_fourway_run.sh
./tpu_fourway_run.sh start
```

Monitor without attaching:

```bash
./tpu_fourway_run.sh status
./tpu_fourway_run.sh tail
```

The browser or SSH connection may disconnect; the run remains in detached
`tmux`.

## Direct invocation

Prepare a compatible corpus once:

```bash
rg-onehead-prepare \
  --config configs/tpu_fourway_quick.yaml \
  --output-dir /tmp/rg-nanogpt-fourway-data/4m
```

Then run:

```bash
PYTHONPATH=src python3 -m rg_nanogpt_one_head.tpu_distributed \
  --config configs/tpu_fourway_quick.yaml \
  --optimizer muon \
  --seed 1337 \
  --data-root /tmp/rg-nanogpt-fourway-data/4m \
  --results-root /tmp/rg-nanogpt-fourway-results \
  --world-size 4 \
  --overwrite
```

## Outputs

The canonical result layout remains:

```text
results/muon/seed_1337/
  manifest.json
  distributed_plan.json
  distributed/
    rank_00.json
    rank_01.json
    rank_02.json
    rank_03.json
  metrics.csv
  epoch_metrics.csv
  checkpoint_initial.pt
  checkpoint_latest.pt
  checkpoint_best.pt
  checkpoint_final.pt
  spectral/
    layers.csv
    summary.csv
    raw/
  test_results.json
  run_complete.json
```

`run_complete.json` additionally records the four-device world size, local
gradient accumulation, and preserved global tokens per optimizer step.

## Scope

The new path currently starts a fresh distributed replicate. Existing
single-device resume behavior remains unchanged. Use a new results root for
each four-device run or pass `--overwrite` for that distributed optimizer/seed
directory only.
