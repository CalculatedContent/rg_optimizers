# Saved-initial vs final MuonClip angular analysis

`07_muonclip_initial_final_angular_weightwatcher.ipynb` compares two actual
checkpoint files for one NanoGPT baseline run:

```text
checkpoint_initial.pt   # step 0, before evaluation or any optimizer update
checkpoint_final.pt     # completed final step
```

The training runtime now writes `checkpoint_initial.pt` automatically for new
runs. It is immutable: resume does not overwrite it. A legacy run started
before this change may not contain the file; the strict notebook will stop
rather than reconstructing the initialization from a seed.

## Environment variables

The notebook resolves the run in this order:

1. `RUN_DIR`
2. `RESULTS_ROOT/<TARGET_OPTIMIZER>/seed_<TARGET_SEED>`
3. `RUNROOT/results/<TARGET_OPTIMIZER>/seed_<TARGET_SEED>`
4. shallow automatic discovery near the repository and under `/tmp`

Use `RUN_DIR` whenever several run roots contain the same seed. Variables must
be exported before Jupyter is launched so the kernel inherits them.

For the MuonClip seed-4242 run described by the training command:

```bash
cd /path/to/rg_optimizers

export RG_OPTIMIZERS_ROOT="$PWD"
export RUNROOT=/tmp/<the-same-run-root-used-for-training>
export RESULTS_ROOT="$RUNROOT/results"
export TARGET_OPTIMIZER=muon_clip
export TARGET_SEED=4242
export RUN_DIR="$RESULTS_ROOT/$TARGET_OPTIMIZER/seed_$TARGET_SEED"

jupyter lab baseline/nanogpt_one_head/notebooks/angular/07_muonclip_initial_final_angular_weightwatcher.ipynb
```

For a new seed, change `TARGET_SEED`. For a new experiment root, change
`RUNROOT` or `RESULTS_ROOT`. The endpoint paths can also be supplied directly:

```bash
export INITIAL_CHECKPOINT_PATH="$RUN_DIR/checkpoint_initial.pt"
export FINAL_CHECKPOINT_PATH="$RUN_DIR/checkpoint_final.pt"
```

Optional controls:

```bash
export ANGULAR_N_NULL=100
export ANGULAR_N_ENTRY_NULL=24
export ANGULAR_SHOW_PLOTS=1
export ANGULAR_OUTPUT_DIR="$RUN_DIR/diagnostics/angular_initial_vs_final"
```

For publication-quality null intervals, increase `ANGULAR_N_NULL` to 500 or
1000.

## Matching the current training command

The training command uses `--results-root "$RUNROOT/results"`. The notebook
therefore needs the identical exported `RUNROOT`, or the exact derived
`RUN_DIR`:

```bash
export RUNROOT=/tmp/<your-run-root>
export TARGET_SEED=4242
export TARGET_OPTIMIZER=muon_clip
export RUN_DIR="$RUNROOT/results/muon_clip/seed_4242"
```

Then launch Jupyter from the repository root. The notebook prints the resolved
repository, run directory, source environment variable, and both checkpoint
paths before loading either model.

## Output

By default, results are written to:

```text
<RUN_DIR>/diagnostics/angular_saved_step_0000000_vs_final_<step>/
```

The directory contains per-layer radial and angular ESD plots, projective-tail
plots, summary exponent plots, `angular_initial_vs_final_summary.csv`, and
`analysis_manifest.json` recording the exact environment and checkpoint paths
used.
