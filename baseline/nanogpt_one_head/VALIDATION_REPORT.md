# Validation accuracy by epoch

Install the package and use:

```bash
rg-onehead-validation
```

The command reads `metrics.csv`, detects how far the run has progressed, and
prints the validation measurement nearest each integer epoch. It therefore
works for active or completed one-head baseline runs of any configured length.

The table includes:

```text
TARGET_EPOCH
ACTUAL_EPOCH
EPOCH_ERROR
STEP
VAL_ACC_%
VAL_LOSS
LR
IS_CURRENT
```

`EPOCH_ERROR` is `ACTUAL_EPOCH - TARGET_EPOCH`. The command uses
`metrics.csv` rather than relying on `epoch_metrics.csv`.

## MuonClip + RMS

```bash
rg-onehead-validation \
  --results-root /tmp/rg-nanogpt-long-muonclip-50ep/results \
  --optimizer muon_clip \
  --seed 1337
```

## Ordinary Muon

```bash
rg-onehead-validation \
  --results-root /tmp/rg-nanogpt-long-muon-50ep/results \
  --optimizer muon \
  --seed 1337
```

## Point directly to a run

```bash
rg-onehead-validation \
  --run-dir /path/to/results/adamw/seed_2027
```

## Include the current partial epoch

```bash
rg-onehead-validation \
  --run-dir /path/to/run \
  --include-current
```

## Quarter-epoch measurements

```bash
rg-onehead-validation \
  --run-dir /path/to/run \
  --interval 0.25
```

## Include training metrics

```bash
rg-onehead-validation \
  --run-dir /path/to/run \
  --include-train
```

## Print every validation evaluation

```bash
rg-onehead-validation \
  --run-dir /path/to/run \
  --all-evaluations
```

## Save the selected table

```bash
rg-onehead-validation \
  --run-dir /path/to/run \
  --interval 0.25 \
  --include-train \
  --output-csv /tmp/validation_by_epoch.csv
```

The footer reports the current state, the highest validation accuracy across
all evaluation rows, and the minimum validation loss across all evaluation
rows.
