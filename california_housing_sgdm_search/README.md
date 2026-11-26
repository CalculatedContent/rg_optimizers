# California Housing: full SGDM grid and longer training

This finite study tests whether limited tuning or a short training horizon explains the initial SGD-with-momentum performance deficit. It preserves the completed three-optimizer benchmark and separates two questions: what happens when the same SGDM checkpoints train longer, and what happens when a broader validation-only search selects new settings?

The experiment directory is `results/2026-11-26_sgdm_long_search`. The workflow is `.github/workflows/california-housing-sgdm-long-search.yml`; the branch is `california-housing-sgdm-long-search-20261126`. `plan.json` fixes every candidate, seed, budget, comparison and retention policy before training. The workflow does not repeat this search automatically.

## Completed parent experiment

The parent is [workflow 45753214421](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45753214421), source `67a0bf203ab0f6b34565e37b4f510f123f669876`, archived at immutable commit `84b0219dc87c7caaefdc7245e455e53c300b8dfc`. Its [results and independent audit](https://github.com/CalculatedContent/rg_optimizers/tree/84b0219dc87c7caaefdc7245e455e53c300b8dfc/california_housing_benchmark/results/2026-11-26_published_mlp) remain authoritative. All 15 confirmation runs completed; 360 independently recalculated numeric values agreed with the committed analysis.

At 20,000 updates, the parent test RMSE was 0.50969 ± 0.00579 for SGDM, 0.48078 ± 0.00280 for native AdamW, and 0.48377 ± 0.00205 for modified AdamW. These intervals are 95% Student-t confidence half-widths across five seeds, conditional on the fixed data split. Modified AdamW had worse RMSE than native AdamW on every seed at both fixed budgets. This new study does not revise that completed comparison.

The parent SGDM configuration reached the top learning rate of its small initial search. Its hidden-matrix alpha moved from a five-seed mean of 3.576 at 2,000 updates to 2.278 at 20,000. Those observations motivate the additional study; they do not establish that the new search must improve performance or reach alpha=2.

## Fixed design

| Component | Seeds | Settings | Budget |
|---|---|---|---|
| Full development grid | 3338, 3339 | All 32 combinations below | 50,000 updates per run |
| Exact parent continuations | 2340–2344 | Original SGDM settings unchanged | Resume 20,000 → 100,000 |
| Fresh confirmation | 3340–3344 | One configuration frozen from development | 100,000 updates per run |

The development grid crosses learning rates `{0.003, 0.01, 0.03, 0.1}`, momentum `{0.9, 0.99}`, weight decay `{0, 0.001}`, and dropout `{0, 0.3690073279296028}`. All runs use dampening 0, Nesterov disabled, and constant learning rate. PyTorch SGD applies coupled weight decay to matrix parameters; biases have no weight decay, following the unchanged parent optimizer construction. No candidate is pruned by an early-stopping rule. Every finite run reaches its full declared budget.

Each candidate receives both development seeds. Selection minimizes the equally weighted mean of their best validation RMSE through 50,000 updates, with a deterministic candidate-ID tie break. The search never evaluates the test split. A numerical failure remains a visible failed trial and cannot win selection. Other failures require execution repair; they are not silently treated as poor hyperparameters. The search freezes settings in `frozen_selection.json` before any fresh confirmation begins. Selection at 50,000 followed by confirmation at 100,000 is an explicit horizon extension, not a claim of optimal tuning at 100,000.

The five continuations isolate the effect of additional training. They restore each parent's complete 20,000-update model, momentum state, dropout and random-number states, minibatch generator, current permutation, offset and digest. Their learning rate is 0.03, momentum 0.9, weight decay 0, and dropout 0.3690073279296028. They execute only the additional 80,000 updates. `PARENT_MANIFEST.json` pins every parent result and checkpoint by repository path, Git blob identity and SHA-256. The paired primary comparison is 20,000 versus 100,000 updates within each original seed.

The new confirmation cohort uses five fresh seeds. Its primary model minimizes validation RMSE through 100,000 updates. Fixed checkpoints at 2,000, 20,000, 50,000 and 100,000 provide budget-specific results. A continuation's optional validation-selected model is selected only within `[20,000, 100,000]`; it does not claim to recover the best unretained checkpoint from the earlier interval. The new best-validation rule keeps searching after long plateaus, whereas the completed parent study retains its original first-patience-stop endpoint.

## Architecture, data and numerical settings

The network remains the published two-block MLP: `8 → 784 → 784 → 1`, with ReLU and dropout after each hidden affine layer. It contains 623,281 trainable parameters. Changing dropout does not change the parameter count. The implementation uses `tabm.MLPBackbone` 0.0.3 and a scalar output layer. The trainer imports unchanged parent model, optimizer, metric and spectral helpers from the pinned source; source verification covers the old and new files before any run starts.

The baseline comes from the [official tabular optimizer benchmark](https://github.com/yandex-research/tabular-dl-optimizers/tree/c34e33152db9eea8ed4b2d742b9bef5628d71fb5) accompanying [the paper](https://arxiv.org/html/2604.15297v2). Its native AdamW result was test RMSE 0.49359 with sample SD 0.01100 across ten seeds. That is a useful reproducible MLP reference, not a claim of the best result across all California Housing architectures. The completed parent repository retains the exact upstream configurations, reports, literature audit and licenses.

Data come from the authors' fixed arrays, not a newly randomized split. The split contains 13,209 training, 3,303 validation and 4,128 test rows. Training-only noisy quantile preprocessing uses 440 quantiles, preprocessing seed 0 and noise scale 1e-5 for fitting. Target standardization uses the training mean 2.0749101638793945 and standard deviation 1.1605902910232544. Predictions are transformed back to the original target units before scoring. Prepared data and all raw arrays have content hashes; the original preprocessing manifest is restored byte-for-byte, not regenerated during this extension.

The prepared NPZ SHA-256 is `d546ac115ae7bd9e49a20adae63c949db6052a225d45294a58dabdd516ee1314`. Its manifest SHA-256 is `3ec27779f784aaf7ac1110b92fb4af792291b68b2daf0c259b5c0539d0537a8e`. `prepare.py` restores those exact archived bytes, verifies them, recovers the individually hashed raw arrays using the unchanged bounded data downloader, and verifies all five parent checkpoints. Full source/data provenance appears in the parent `reference_manifest.json`, its archived `data/manifest.json`, and this study's manifests.

Every run uses CPU float32, deterministic PyTorch operations, batch size 256, a final partial minibatch, and gradient-norm clipping at 1. Python, NumPy and PyTorch seeds follow the unchanged parent seeding procedure. A dedicated seeded generator produces each epoch permutation; checkpoints retain its exact state and the next offset. There are 52 updates per full epoch. Validation runs at each epoch boundary. The extension records full performance at 5,000-update intervals and the additional declared checkpoint budgets; it never stops for patience.

The GitHub runtime is Python 3.12.9 with `torch==2.7.1+cpu`, `numpy==2.3.1`, `scipy==1.16.0`, `scikit-learn==1.7.0`, `tabm==0.0.3`, `delu==0.0.26`, `rtdl_num_embeddings==0.0.12`, `pandas==2.3.1`, `matplotlib==3.10.3`, `weightwatcher==0.7.7`, `powerlaw==2.0.0`, and `pytest==8.4.1`. The requirements file imports the unchanged parent pins. Torch intra-op/inter-op threads and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` are set to 1. Runtime records retain CPU model, OS, package versions, thread settings, source hashes, GitHub run ID, job and attempt. Matching version strings do not guarantee identical kernels or hardware across hosted jobs; exact continuation-state checks do not erase that runtime distinction.

## Performance versus confidence/tail-adjusted alpha

California Housing is regression. The main vertical axis is RMSE, with lower values better; there is no classification accuracy percentage. Targets use units of $100,000, so RMSE 0.50 means $50,000, not 50% error. The archive also retains MAE, R-squared, MAPE and training/validation metrics. Development plots use validation RMSE; final plots use the held-out test RMSE of already fixed models.

The horizontal axis uses ordinary fitted alpha with weights `sqrt(num_pl_spikes)/(D+0.02)`. It does not use WeightWatcher's `alpha_weighted` or an extra matrix-size factor. The unchanged WeightWatcher 0.7.7 `analyze()` fit uses powerlaw 2.0.0. The minimum matrix dimension of 64 admits exactly one matrix, the 784×784 hidden weight. Therefore the confidence/tail-adjusted model average equals that matrix's alpha. This architecture cannot measure collective variation across eligible layers. Fit records retain alpha, D, fitted-tail size, xmin, validity, the complete fit output and diagnostic eigenvalues; invalid fits are reported as unavailable, not replaced with invented values.

Every plotted RMSE/alpha pair comes from the same checkpoint and carries matching weight hashes. Development spectra are measured at 2,000, 20,000 and 50,000 updates and at the validation-selected model. Final cohorts measure spectra at 5,000-update intervals plus the extra fixed checkpoints and selected models. Spectral measurements preserve training random-number states. The analyzer independently checks raw fit aggregation, checkpoint identities, retained model hashes, data/source provenance and seed membership.

The grid plot shows both development seeds for every setting, with learning-rate colors and dropout panels; accompanying tables retain every candidate and failure. Configuration means and two-seed sample SD describe that grid only. The analysis never treats 32 configurations as 32 seed replicates. The final cohorts use equal-weight five-seed means, sample SD, and two-sided 95% Student-t intervals. For five seeds the confidence half-width is `t(0.975,4) × SD/sqrt(5)`. These intervals quantify seed variation on one fixed split, not uncertainty across future datasets or confidence in the selected hyperparameters.

The continuation figure shows update trajectories and RMSE/alpha paths for each of the five original seeds. Paired changes isolate the longer horizon while preserving their initial configuration and saved state. Comparisons of freshly tuned SGDM at 100,000 updates with original AdamW at 20,000 use independent-sample uncertainty and explicitly unequal training/tuning budgets. A dropout change also changes regularization. Such a comparison does not identify an optimizer-only causal effect. Alpha=2 is a displayed theoretical reference, not an imposed optimum or selection criterion.

## Run and resume without a notebook

Use Python 3.12.9 and a checkout of this study's immutable launch commit (recorded in `cohort_origin.json`). From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r california_housing_sgdm_search/requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg PYTHONHASHSEED=0
python -m california_housing_sgdm_search.verify
python -m pytest -q california_housing_sgdm_search/tests
study_root=california_housing_sgdm_search/results/2026-11-26_sgdm_long_search
python -m california_housing_sgdm_search.prepare --root "$study_root"
```

The workflow records actual GitHub execution identity and initializes one cohort before training. Local reproductions should use a separate output directory and preserve their actual local provenance; they must not impersonate GitHub runs or be merged into the primary cohort. The low-level trainer accepts the same fixed plan and a candidate configuration without requiring a notebook:

```bash
python -c 'import json; p=json.load(open("california_housing_sgdm_search/plan.json")); json.dump(p["candidates"][0]["config"],open("candidate.json","w"),indent=2)'
python -m california_housing_sgdm_search.long_train \
  --plan california_housing_sgdm_search/plan.json --config candidate.json \
  --data-dir "$study_root/data" --output local_reproduction/sgdm_00/seed_3338 \
  --seed 3338 --phase development
```

For the authorized GitHub cohort, the workflow executes the following wrappers for every declared candidate/seed. They reject changed identities and reuse verified completed outputs:

```bash
python -m california_housing_sgdm_search.search run --root "$study_root" --candidate sgdm_00 --seed 3338
python -m california_housing_sgdm_search.search freeze --root "$study_root"
python -m california_housing_sgdm_search.analysis --root "$study_root" --development-only
python -m california_housing_sgdm_search.run_one --root "$study_root" --phase continuation --seed 2340
python -m california_housing_sgdm_search.run_one --root "$study_root" --phase confirmation --seed 3340
python -m california_housing_sgdm_search.analysis --root "$study_root"
```

Run `freeze` only after all 64 development trials have terminal records. It will not accept a partial grid. Run final analysis after all five continuations and five fresh confirmations complete. Final success requires `complete_five_seeds=true`, `n_completed=5`, five completed runs in each final cohort, all 64 development records, and no audit errors. The workflow also produces a development-only audited report before fresh confirmation finishes.

Full checkpoints contain model, optimizer, random-number states, permutation, offset, digest and current best-validation state. Repeating the identical wrapper resumes an incomplete run from its retained checkpoint. A completed development run is verified from its compact archived result and retained best/final weights; the wrapper does not construct a new training run merely because full development resume states were intentionally omitted. Checkpoint files are trusted repository artifacts whose pinned hashes are verified before loading.

For an execution failure, inspect the actual failed job and retained artifact. Repair routine execution/archival errors without changing the grid or optimizer settings, and rerun only that failed job. If training completed but publication failed, restore the retained recovery artifact before rerunning publication so training is not duplicated. Do not start another workflow cohort to work around an absent local directory.

## Compute limits and archival policy

The workflow uses standard GitHub-hosted `ubuntu-24.04` CPU runners. It caps development concurrency at 16 and continuation concurrency at 4; no stage combination exceeds 20 training jobs. Five fresh confirmations begin after the grid freezes. Each training job has a 90-minute timeout. The finite training budget is 3.2 million development updates, 400,000 additional continuation updates, and 500,000 fresh-confirmation updates. Observed parent speeds imply approximately 14–23 runner-hours of training, excluding setup/evaluation/archival overhead, and roughly 1–2 hours elapsed under parallel execution; queuing can extend that estimate.

Under [GitHub's current billing rules](https://docs.github.com/en/billing/concepts/product-billing/github-actions), standard runners in public repositories have no Actions compute charge. Storage has separate allowances. The study uses the existing CPU dependency cache and retains lightweight Actions measurement/log artifacts for seven days. It uploads full recovery artifacts only on failure. It does not use a paid larger runner or GPU, change account billing settings, or delete prior files.

All results, source, instructions and selected checkpoints persist in the repository. Each successful development run retains complete JSON/log measurements plus `best_checkpoint.pt` and `final_weights.pt`; routine intermediate/full resume states are not newly archived for completed development trials. Failed or incomplete trials retain full recovery states. All final confirmation and continuation checkpoints are retained. The publisher preserves unrelated files and newer progress, uses non-force branch updates, retries concurrent commits, and cannot write outside this study's result prefix. Its token stays in the normal GitHub transport.

`development_analysis/` contains the complete grid tables and figures. `analysis/RESULTS.md`, `analysis/summary.json`, `analysis/seed_metrics.csv`, `analysis/seed_statistics.csv`, and `analysis/figures/` contain final results. The analysis records explicit incomplete status when any required measurement, seed, checkpoint or provenance check fails; it never fills gaps with synthetic data.
