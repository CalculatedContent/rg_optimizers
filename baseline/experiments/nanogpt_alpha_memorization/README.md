# nanoGPT memorization study: AdamW vs Muon

This folder runs **10 paired nanoGPT experiments on Apple MPS**: five ordinary AdamW seeds followed by five ordinary Muon seeds. Online WeightWatcher is deliberately **disabled** so spectral analysis does not dominate training time. The same five seeds are used for both optimizers: `1337, 2027, 4099, 31415, 271828`.

The optimizer hyperparameters come directly from the pinned repository baseline recipe. AdamW uses the existing AdamW profile; Muon uses the existing ordinary Muon profile. There is **no spectral guard, rollback, alpha objective, or outcome-based seed selection**.

## Run all 10

With `ww_prod310` activated:

```bash
cd /tmp/rg_optimizers_memorization
git fetch origin
git switch main
git pull --ff-only origin main
cd baseline/experiments/nanogpt_alpha_memorization
caffeinate -dimsu python run_study.py run
```

The launcher creates a timestamped results directory under `/tmp` (canonicalized to `/private/tmp` on macOS), performs a two-update MPS preflight, then runs AdamW seeds `1337, 2027, 4099, 31415, 271828`, followed by the same five Muon seeds. A failed seed is preserved and the queue continues.

Preview with `python run_study.py plan`.

## Behavioral measurements during training

Each run records exact 32-token greedy canary recall; partial token recall and longest correct prefix; teacher-forced accuracy; per-example NLL and perplexity; dose response at 0, 1, 4, 16, and 64 presentations; exact finite-universe exposure for short canaries; prefix/context dependence at 8, 16, 32, and 64 tokens; fixed-random-label fitting; clean held-out rule performance; and withdrawal/retention after canary presentations stop.

Behavioral audits run every 500 optimizer updates. Exposure/compression audits run every 2500 updates and at the end.

## Post-hoc WeightWatcher

No WeightWatcher call occurs in the scientific training loop. Model checkpoints are saved every **500 updates**, including step 0. After training finishes, update the repository and run:

```bash
python run_study.py spectra
```

This command is resumable: completed checkpoint CSVs are reused. It reconstructs every saved AdamW and Muon checkpoint on CPU and runs the pinned WeightWatcher `0.7.7` configuration with `fix_fingers='clip_xmax'`, `ERG=True`, `randomize=True`, `min_evals=20`, and `max_fingers=10`. It verifies checkpoint identity against behavioral audits where available and never modifies training checkpoints.

The output is written under the study results directory in `posthoc_weightwatcher/` and includes:

- `alpha_all.csv`: every matrix/checkpoint/seed measurement;
- `alpha_seed_summary.csv`: minimum/mean alpha and below-2 matrix counts per seed and checkpoint;
- `alpha_summary.csv`: mean and pointwise 95% Student-t intervals across seeds;
- `alpha_matrix_summary.csv`: per-matrix alpha trajectories with seed intervals;
- `alpha_regimes.csv`: whether each seed ever crossed below 2 on the saved checkpoint grid and its first observed crossing;
- `figures/`: PNG/SVG overall and per-matrix alpha plots.

Post-hoc spectra describe only the saved 500-update grid, not unseen intermediate updates. A numerical alpha below 2 should also be interpreted with fit support and the randomized-spectrum diagnostics retained by WeightWatcher.

## Reports and review bundle

```bash
python run_study.py report
python run_study.py spectra
python run_study.py export
```

`report` creates the behavioral CSV tables and PNG/SVG plots. `spectra` creates the post-hoc WeightWatcher analysis. `export` then executes both `01_Memorization_Results.ipynb` and, when spectra exist, `02_Spectral_Boundary.ipynb`, and creates a review bundle containing reports, figures, tables, protocol, manifests, and executed notebooks. Model weights and credentials are excluded.

## Resume training

If training is interrupted, do not delete the result directory. Resume with:

```bash
caffeinate -dimsu python run_study.py run --resume
```

The protocol fingerprint must match. Completed seeds are not retrained.

## Scientific interpretation

The study does **not** assume AdamW will produce alpha below 2 or that Muon will remain above 2. The behavioral study establishes what each optimizer memorizes and how generalization changes. The post-hoc WeightWatcher pass then tests whether those transitions coincide with a valid alpha-below-two regime.


## Four-head large-data pilot

The original synthetic rule study has only 480 training combinations. For a more meaningful generalization test, `protocol_four_head_large.json` changes the model to four attention heads and expands the modular rule universe to modulus 251: 31,500 training combinations, 15,750 validation combinations, and 15,751 test combinations. The effective batch remains 32, so 30,000 updates are about 30.5 passes over the training set.

The pilot runs one matched seed with ordinary AdamW followed by ordinary Muon. It keeps the same pinned optimizer profiles, 25% fixed random-label corruption, and canary memorization probes. Canary counts are increased to 32 long and 8 short examples per dose. Behavioral audits run every 1,000 updates; checkpoints are saved every 1,000; expensive exposure/compression audits run every 5,000. Online WeightWatcher remains disabled.

Run the pilot with:

```bash
caffeinate -dimsu python run_study.py run --protocol protocol_four_head_large.json
```

Preview it with:

```bash
python run_study.py plan --protocol protocol_four_head_large.json
```

This is a pilot, not a five-seed statistical comparison. If both arms train sensibly and held-out accuracy improves, replicate the frozen protocol across the remaining seeds rather than changing hyperparameters after seeing the result.
