# Live spectral monitoring

WeightWatcher is invoked with `randomize=True`, so each analyzed transformer
matrix receives the direct `rand_distance` output in addition to `alpha`, fit
`D`, `ERG_gap`, and `num_traps`.

`rand_distance` is WeightWatcher's Jensen-Shannon distance between the empirical
eigenvalue distribution and the eigenvalue distribution obtained after
entry-wise randomization of the same matrix. It is not a replacement for
`alpha` or `D`: it measures how far the observed spectrum is from its randomized
control.

The raw per-layer values are stored in:

```text
results/<optimizer>/seed_<seed>/spectral/layers.csv
```

The checkpoint summaries now include:

```text
rand_distance_n
rand_distance_mean
rand_distance_median
rand_distance_std
rand_distance_min
rand_distance_max
```

in:

```text
results/<optimizer>/seed_<seed>/spectral/summary.csv
```

## Monitor a run

After installing the package from `baseline/nanogpt_one_head`:

```bash
python -m pip install -e .
```

monitor the default Muon seed with:

```bash
rg-onehead-monitor --optimizer muon --seed 1337
```

For the long-Muon run stored under a custom results root:

```bash
rg-onehead-monitor \
  --results-root /tmp/rg-nanogpt-long-muon-v2/results \
  --optimizer muon \
  --seed 1337
```

The display refreshes every 30 seconds and reports, per matrix:

```text
matrix_name
alpha
D
rand_distance
ERG_gap
num_traps
```

It also reports the latest mean/median/range for `alpha` and `rand_distance`,
plus recent checkpoint medians. Stop it with Control-C.

Useful options:

```bash
# Print one snapshot and exit.
rg-onehead-monitor --once

# Refresh every ten seconds.
rg-onehead-monitor --interval 10

# Show the most recent 12 WeightWatcher checkpoints.
rg-onehead-monitor --recent 12

# Point directly to a run directory.
rg-onehead-monitor --run-dir /path/to/results/muon/seed_1337
```

The monitor only reads CSV files. It does not synchronize the training device,
run WeightWatcher, modify checkpoints, or alter the optimizer trajectory.

## Existing active runs

No restart is required merely to display `rand_distance`. The existing
WeightWatcher path already preserves all columns returned by
`watcher.analyze(randomize=True)` in `spectral/layers.csv`. Reinstalling the
updated package provides the monitor command, which can read those values while
the old process continues. New processes additionally aggregate and validate
`rand_distance` as a required first-class spectral metric.
