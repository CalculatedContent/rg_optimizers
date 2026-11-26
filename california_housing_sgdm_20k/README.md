# California Housing SGDM: 20,000-update limit

The user changed the experiment on 2026-11-26: stop training beyond **20,000
updates** and evaluate every SGDM grid setting within that budget. This amendment
supersedes the earlier 50k development, 100k continuation, and 200k continuation
plans for future execution.

Both superseded workflows were canceled:

- [Original grid/long-horizon workflow 45758895218](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45758895218)
- [Queued 200k continuation 45762869296](https://github.com/CalculatedContent/rg_optimizers/actions/runs/45762869296)

The cancellation preserved repository archives. **All 64 development runs had
already recorded 20,000-update measurements**, so this amended study performs
**zero additional development training**. It reuses only the portions of those
records through 20k and then runs the previously authorized five fresh
confirmation seeds, each with a hard 20k limit.

## Reused grid and provenance

The grid remains the original 32 settings, each with development seeds 3338 and
3339:

| Parameter | Values |
|---|---|
| Learning rate | 0.003, 0.01, 0.03, 0.1 |
| Momentum | 0.9, 0.99 |
| Weight decay | 0, 0.001 |
| Dropout | 0, 0.3690073279296028 |

Dampening is zero and Nesterov is disabled. All runs use the original published
623,281-parameter 8→784→784→1 MLP, the same prepared data and split, batch size
256, clipping norm 1, and a constant learning rate. Original model/data details
and pinned dependencies remain in `california_housing_benchmark/README.md`.

The source cohort is workflow `45758895218`, launch commit
`f89e5e00a3b9ad09c906b272fbc1d627ccecd714`. Preparation pins the complete
post-cancellation archive at
`ff1cf0fb3fd4252f34041a2e74c75887db85f80c` and verifies each source file's identity.
It preserves the untouched original records under `inputs/development/`.

Fifty-five original records completed 50k and nine stopped above 20k. The amended
analysis does **not** relabel those trajectories as runs that originally stopped
at 20k. It uses their already measured 20k models and their earlier validation
histories. Nine source runs retain full 20k checkpoints; the original compact
policy did not retain full 20k checkpoints for the 55 completed 50k grid runs.
All 64 retain the 20k measurements and matching performance/spectral weight hashes.
The inventory reports this distinction explicitly. No unavailable checkpoint is
recreated by repeating training.

## Selection and failed spectral fits

Selection uses the lowest equal-weight mean of the two seeds' minimum validation
RMSE. Eligible observations are initialization, every completed epoch through
20k, and the fixed 20k endpoint. Ties use the fixed candidate ID. Observations
after 20k, test values, and alpha do not enter selection. The fixed winner is
archived in `frozen_selection.json` before any confirmation seed starts.

The scatter plot pairs **fixed 20k validation RMSE with fixed 20k alpha from the
same weights**. It does not pair an earlier best-validation error with the final
alpha. All 64 performance observations remain in the grid. Four observations
(`sgdm_30` and `sgdm_31`, both development seeds) have explicit failed spectral
fits after their hidden matrices nearly vanished. They retain their measured
errors and an unavailable alpha; they are not missing training runs and are not
rerun or replaced by another spectral statistic.

Alpha uses ordinary fitted alpha with weights `sqrt(num_pl_spikes)/(D+0.02)` and
no matrix-size factor. Only the 784×784 hidden matrix is eligible, so its fit
equals the model average. Confidence intervals across seeds do not represent
layer dispersion or uncertainty in the fitted power-law region.

## Fresh confirmation: hard cap at 20k

The fresh seeds remain **3340–3344**. The wrapper permits only these five seeds,
rejects a plan above 20k, rejects saved states above 20k, and exposes no
development-training route. It calls the unchanged numerical trainer with
`total_updates=20000`. It saves full checkpoints at 2k and 20k, the final resume
state, and validation-selected/final weight exports.

The primary report compares **fixed 20k endpoints**. Optional validation-selected
models within 0–20k are reported separately. The original three optimizer
controls at 20k use seeds 2340–2344 and are restored from archive commit
`84b0219dc87c7caaefdc7245e455e53c300b8dfc`. They provide independent-seed context;
the new SGDM search may change dropout and has a different tuning budget, so a
performance difference is not an optimizer-only causal comparison.

Reports include equal-weight seed means, sample SDs, and 95% Student-t confidence
intervals over the five confirmation seeds. RMSE and MAE use target units of
$100,000; lower is better. These are regression errors, not classification
accuracy or percentage errors.

## Execution, storage, and recovery

The workflow `.github/workflows/california-housing-sgdm-20k.yml` verifies the cap
and immutable source, stages and analyzes all reused inputs, freezes selection,
runs at most five standard CPU confirmation jobs, then audits and archives the
results. Each training job has a 45-minute limit and one numerical thread.
Python 3.12.9, PyTorch 2.7.1+cpu, and float32 match the prior benchmark. Runtime
metadata record the actual machine; matching versions do not guarantee identical
rounding across CPU kernels.

The original numerical trainer's source manifest remains unchanged. New records
separately identify this orchestration package's source manifest. Results,
measurements, full confirmation checkpoints, plots, and reproduction code are
committed to `california_housing_sgdm_20k/results/2026-11-26_sgdm_20k` in this
repository. Temporary Actions artifacts last seven days. Failed jobs retain full
recovery state; retry only failed jobs in the same cohort. Completed training is
reused, and missing completed checkpoints require archive restoration.

Run the source and cap checks from the repository root:

```bash
python -m pip install -r california_housing_sgdm_20k/requirements.txt
python -m california_housing_sgdm_20k.verify
python -m pytest -q --import-mode=importlib california_housing_sgdm_20k/tests
```

The workflow uses `prepare --root ...`, `analysis --development-only --root ...`,
`run_one --root ... --seed 3340` (and the other four seeds), then `analysis --root
...`. Official cohort commands enforce the actual GitHub execution identity.
Read the committed plan, source inventory, and workflow for the exact input and
archive paths. Do not rerun the superseded long-horizon workflows.
