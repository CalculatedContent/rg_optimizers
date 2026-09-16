# nanoGPT memorization study: AdamW vs Muon

This folder runs **10 paired nanoGPT experiments on Apple MPS**: all five ordinary AdamW seeds first, followed by all five ordinary Muon seeds. Online WeightWatcher is deliberately **disabled** so spectral analysis does not dominate training time. The same five seeds are used for both optimizers: `1337, 2027, 4099, 31415, 271828`.

The optimizer hyperparameters come directly from the pinned repository baseline recipe. AdamW uses the existing AdamW profile; Muon uses the existing ordinary Muon profile. There is **no spectral guard, rollback, alpha objective, or outcome-based seed selection** in this version.

## Run all 10

With the existing `ww_prod310` environment activated:

```bash
cd /tmp/rg_optimizers_memorization
git fetch origin
git switch main
git pull --ff-only origin main
cd baseline/experiments/nanogpt_alpha_memorization
caffeinate -dimsu python run_study.py run
```

The launcher creates a timestamped results directory under `/tmp` (canonicalized to `/private/tmp` on macOS), performs a two-update MPS preflight for AdamW and Muon, then runs all five AdamW seeds followed by all five Muon seeds. A failed scientific seed is preserved and the queue continues to the remaining independent seeds.

Preview the exact queue with:

```bash
python run_study.py plan
```

The planned order is AdamW seeds `1337, 2027, 4099, 31415, 271828`, then Muon seeds `1337, 2027, 4099, 31415, 271828`.

## What is measured during training

Every run uses the same mixed synthetic task and records the behavioral endpoints needed to distinguish useful rule learning from example-specific memorization:

- exact 32-token greedy canary recall;
- partial token recall and longest correct prefix;
- teacher-forced accuracy;
- per-example NLL and perplexity;
- controlled canary dose response at 0, 1, 4, 16, and 64 presentations;
- exact finite-universe exposure for the short-canary cohort;
- restricted prefix/context dependence using 8, 16, 32, and 64 prefix tokens;
- fixed-random-label fitting;
- clean held-out rule performance;
- withdrawal/retention after canary presentations stop.

Behavioral audits are recorded every 500 optimizer updates. The expensive exposure/compression audits run every 2500 updates and at the end.

## WeightWatcher is OFF during training

No WeightWatcher call occurs in the scientific training loop. No alpha threshold is enforced. This means the ten runs test **ordinary AdamW versus ordinary Muon** at the pinned parameter settings.

Model checkpoints are saved every **500 updates**, including step 0. This is intentional: after the fast training campaign finishes, WeightWatcher can be applied **post hoc** to exactly the saved model states. That lets us measure `alpha_raw` and `alpha_clip_xmax` with `fix_fingers='clip_xmax'` without repeatedly pausing MPS training. Post-hoc spectra therefore describe the saved 500-update grid, not unseen intermediate updates.

## Reports, plots, and error bars

```bash
python run_study.py report
python run_study.py export
```

`report` creates CSV tables plus PNG/SVG behavioral plots. Error bars are pointwise 95% Student-t intervals across independent training seeds after averaging probes within each seed. Paired differences are Muon minus AdamW.

`export` executes `notebooks/01_Memorization_Results.ipynb` and creates a review bundle containing the report, figures, tables, protocol, manifests, and executed notebook. It excludes model weights and credentials.

## Resume

If the process is interrupted, do not delete the result directory. Resume the latest study with:

```bash
caffeinate -dimsu python run_study.py run --resume
```

The protocol fingerprint must match. Completed seeds are not retrained.

## Scientific interpretation

This version does **not** assume AdamW will produce alpha below 2 or that Muon will remain above 2. Those are measurements to make from the saved checkpoints. The behavioral study establishes what each optimizer memorizes and how generalization changes. The post-hoc WeightWatcher pass can then test whether those transitions coincide with a valid alpha-below-two regime.
