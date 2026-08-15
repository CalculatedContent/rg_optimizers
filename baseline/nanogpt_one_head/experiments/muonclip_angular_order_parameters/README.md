# Experimental MuonClip angular quotient order parameters

This directory contains one notebook per candidate reconstruction of a hidden
MuonClip order parameter from singular-vector geometry. Every notebook:

- loads the exact saved step-zero, intermediate, best, and final checkpoints;
- analyzes all six one-head nanoGPT transformer matrices;
- compares the raw matrix with one explicit quotient/composite field;
- uses checkpoint-matched Haar, Gaussian, or temporally scrambled controls;
- runs native WeightWatcher with `plot=True`, `savefig=...`,
  `randomize=True`, `ERG=False`, and `fix_fingers="clip_xmax"`;
- shows all flow dashboards and native ESD contact sheets inline;
- applies the same first-moment RG coordinate

$$
y_E=2-\alpha
$$

  to the transformed Gram spectrum.

## Notebooks

1. `00_polar_intensity_composite.ipynb` — connected polar-intensity anchor.
2. `01_stiefel_log_gauge.ipynb` — additive tangent-space gauge field.
3. `02_flag_shell_curvature.ipynb` — multiscale projector commutator.
4. `03_haar_connected_susceptibility.ipynb` — connected four-point field.
5. `04_diffusion_green_susceptibility.ipynb` — slow-mode Green operator.
6. `05_temporal_block_spin_drift.ipynb` — temporal block-spin angular drift.

## Run one experiment

From `baseline/nanogpt_one_head`:

```bash
export RUN_DIR=/tmp/rg-nanogpt-muonclip-3ep-seed4242-20260814_090245/results/muon_clip/seed_4242
export TARGET_SEED=4242
export ANGULAR_QUOTIENT_NULLS=8
export ANGULAR_QUOTIENT_HAAR_SAMPLES=64
export MPLBACKEND=Agg

jupyter lab experiments/muonclip_angular_order_parameters/01_stiefel_log_gauge.ipynb
```

Or use Papermill:

```bash
papermill \
  experiments/muonclip_angular_order_parameters/01_stiefel_log_gauge.ipynb \
  experiments/muonclip_angular_order_parameters/01_stiefel_log_gauge.out.ipynb \
  --log-output
```

Results are cached under
`$RUN_DIR/diagnostics/experimental_muonclip_angular_order_parameters/<experiment>`.
Set `ANGULAR_QUOTIENT_FORCE=1` to recompute.

## Interpretation discipline

None of the nonlinear maps is assumed correct. A candidate is retained only
when its trained ESD separates from its method-matched null, has an accepted
broad finite-window tail, exhibits controlled fit distance, becomes stationary
late in training, and is stable under its explicit coarse-scale parameters.
