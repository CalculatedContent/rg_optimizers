# MNIST MLP3 tangent-space RG measurement experiments

This directory is the experiment-facing entry point for a deliberately
falsifiable question:

> Can a heavy-tailed RG trajectory that is difficult to see in the raw weight
> ESD under Muon or MuonClip-RMS be recovered by a precisely defined finite-flow,
> tangent, quotient, or local-response operator?

The raw weight ESD remains the control. None of the transformed operators is
called *the* RG operator in advance. A method is retained only if it is
well-defined, separates from its matched nulls, is stable under its numerical
and finger-removal sensitivity parameters, reproduces across complete training
runs, and becomes stationary near the independently measured late-training
fixed-point regime.

The notebooks are thin, restart-safe front ends to
`rg_baselines.tangent_rg`. Training and analysis mathematics belong in that
tested package, not in notebook-only helper implementations. Rebuild the
versioned notebooks after changing `scripts/build_notebooks.py`:

```bash
python baseline/experiments/mnist_mlp3_tangent_rg/scripts/build_notebooks.py
```

## Quickstart

From the repository root, install the baseline package with the experiment
extras and rebuild the committed notebooks:

```bash
python -m pip install -e './baseline[experiment]'
python baseline/experiments/mnist_mlp3_tangent_rg/scripts/build_notebooks.py
```

Choose a persistent output root, then resolve the pilot schedule and storage
contract without downloading MNIST or training:

```bash
export RG_MNIST_TANGENT_ROOT=/absolute/persistent/path/mnist-mlp3-tangent-rg
export RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT=/tmp/rg-mnist-mlp3-tangent-checkpoints
cd baseline
python -m rg_baselines.tangent_rg.cli train \
  --config experiments/mnist_mlp3_tangent_rg/configs/pilot_1000_epochs.yaml \
  --optimizer muon --seed 1337 \
  --output-root "$RG_MNIST_TANGENT_ROOT" \
  --tail-checkpoint-root "$RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT" --dry-run
```

Pilot/reference notebook launches refuse the default `/tmp` root. Point
`RUN_ROOT` or `RG_MNIST_TANGENT_ROOT` at persistent storage; the explicit
`ALLOW_TEMPORARY_LONG_RUN=True` override is only for deliberately disposable
runs. The two-epoch smoke stage may use `/tmp`.

After the smoke stage passes, launch or resume the complete three-optimizer,
three-seed pilot matrix:

```bash
for optimizer in adamw muon muonclip_rms; do
  for seed in 1337 2027 31415; do
    python -m rg_baselines.tangent_rg.cli train \
      --config experiments/mnist_mlp3_tangent_rg/configs/pilot_1000_epochs.yaml \
      --optimizer "$optimizer" --seed "$seed" \
      --output-root "$RG_MNIST_TANGENT_ROOT" \
      --tail-checkpoint-root "$RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT" --resume
  done
done
```

The 10,000-epoch stage uses the same command with
`long_horizon_10000_epochs.yaml`. Each config supplies a distinct suite name,
so the root contains `mnist_mlp3_tangent_rg_v1_smoke/`,
`mnist_mlp3_tangent_rg_v1_pilot1000/`, or
`mnist_mlp3_tangent_rg_v1_reference10000/`; pilot and reference artifacts
cannot collide. In Jupyter, set `PROFILE` (or `CONFIG_PATH`) to the same stage,
run notebook `00`, then `01`--`03`, `04`, analysis notebooks `10`--`14`, then
`16` and `17`, and run the cross-method comparison `15` last. Notebook `18`
is an additional single-seed MuonClip diagnostic driver. Weight-state quotient
notebooks `19` (one seed) and `20` (three seeds with error bars) run post facto
on the same verified final-100 checkpoint caches. Training notebooks are
launch-safe by default: set
`EXECUTE_TRAINING=True` explicitly, while analysis notebooks require completed
artifacts. The checkpoint-based notebooks `10`, `12`, `13`, `15`, `16`, `19`,
and `20` require
the verified tail cache and never fall back to training or to the sparse
WeightWatcher checkpoint series. Notebooks `11`, `14`, and `17` require already saved
dense captures because model-only checkpoint files do not contain update
sources, minibatches, RNG state, or optimizer state.

## Preregistered baseline

| Item | Contract |
|---|---|
| Dataset | `torchvision.datasets.MNIST`, normalized by mean `0.1307` and standard deviation `0.3081` |
| Split | Fixed 55,000 optimization / 5,000 validation split; official 10,000-example test set is monitoring-only |
| Model | `784 -> 512 -> 512 -> 10`, ReLU after FC1 and FC2 |
| Initialization | Existing recipe-v3 MLP3 contract: Kaiming-uniform ReLU hidden weights, Xavier-uniform classifier, zero biases |
| Independent replicates | Seeds `1337`, `2027`, `31415` |
| Uncertainty | Two-sided 95% Student-t interval across the three complete runs |
| Fixed-point references | Native WeightWatcher ESD/energy `alpha_e = 2`; derived operator amplitude `alpha_b = 3` only when testing that same energy hypothesis; exact energy transform `alpha_e = 2`; independently supported trace-log `0` |
| Selection | Validation loss selects `checkpoint_best.pt`; test metrics cannot select a checkpoint, schedule, method, support, or fit variant |

The unit of replication is a complete seeded training run. Layers, matrices,
checkpoints, minibatches, null draws, finite-difference probes, and fit points
are repeated measurements rather than additional replicates.

## One-command MuonClip-RMS Jacobian experiment

The portable runner
`scripts/run_muonclip_jacobians.sh` performs the complete MuonClip-only pilot:

1. creates or reuses an isolated Python 3.11 Conda environment;
2. installs NumPy, Torch, and TorchVision from one pip-wheel ecosystem to avoid
   duplicate `libomp` runtimes on macOS;
3. trains the 1,000-epoch MuonClip-RMS arm for seeds `1337`, `2027`, and
   `31415`;
4. verifies exactly 100 cached checkpoints per seed, covering epochs
   `901`--`1000`; and
5. executes every notebook that computes a genuine Jacobian for this suite.

From a fresh clone on macOS or Linux (or Linux under WSL), with Conda already
installed and available on `PATH`, run:

```bash
git clone https://github.com/CalculatedContent/rg_optimizers.git /tmp/rg_optimizers
cd /tmp/rg_optimizers
bash baseline/experiments/mnist_mlp3_tangent_rg/scripts/run_muonclip_jacobians.sh
```

Persistent training artifacts default to
`~/rg-mnist-mlp3-tangent-runs`. The final-100 analysis cache defaults to
`/tmp/rg-mnist-mlp3-tangent-checkpoints`. Because the cache is ephemeral and
cannot be reconstructed from the sparse persistent checkpoints, keep it until
all notebooks finish or back it up before the host clears `/tmp`. Three seeds
produce 100 cached states per seed, or 300 checkpoint files total.

The runner executes these Jacobian notebooks:

- `11_Muon_Update_Stiefel_Tangent.ipynb`;
- `13_Single_Checkpoint_Map_Jacobians.ipynb`;
- `14_Calibrated_Local_Training_Map.ipynb`;
- `16_Additional_Weight_Only_ECS_Jacobians.ipynb`; and
- `17_Data_Dependent_ECS_Jacobians.ipynb`.

Notebook `10` is a two-checkpoint finite-flow control, not a Jacobian. Notebook
`12` contains radial/angular quotient controls rather than another Jacobian.
They are intentionally not included in the Jacobian-only runner.
Notebooks `19` and `20` are also excluded because they materialize and fit
weight-state quotient hypotheses rather than Jacobians. Notebook `21` is
excluded because it only reads completed-run metric tables.

The default mode is restart-safe: a new run starts when no artifacts exist and
a compatible interrupted run resumes. Useful invocations are:

```bash
# Deliberately replace the selected MuonClip seed artifacts.
bash baseline/experiments/mnist_mlp3_tangent_rg/scripts/run_muonclip_jacobians.sh \
  --overwrite

# Train and verify checkpoints without running Papermill.
bash baseline/experiments/mnist_mlp3_tangent_rg/scripts/run_muonclip_jacobians.sh \
  --training-only

# Analyze previously completed training artifacts and the verified /tmp cache.
bash baseline/experiments/mnist_mlp3_tangent_rg/scripts/run_muonclip_jacobians.sh \
  --analysis-only --skip-setup

# Override paths, device, environment name, or seed subset.
bash baseline/experiments/mnist_mlp3_tangent_rg/scripts/run_muonclip_jacobians.sh \
  --run-root /persistent/rg-runs \
  --cache-root /tmp/rg-mnist-checkpoints \
  --data-root /persistent/mnist \
  --device auto \
  --env-name rg-muonclip-run \
  --seeds 1337,2027,31415
```

Run `scripts/run_muonclip_jacobians.sh --help` for the complete option list.
The script refuses to run the notebooks unless every selected seed has exactly
100 cached checkpoint files. The package's strict cache loader then validates
the manifest, run identity, epoch/step grid, file sizes, and SHA-256 hashes.

### Single-seed MuonClip Jacobian audit

Notebook `18_Single_Run_MuonClip_Jacobian_Audit.ipynb` is the compact debugging
and sharing workflow for one completed MuonClip seed. It runs notebooks `11`,
`13`, `14`, `16`, and `17` as Papermill children, verifies every child
provenance manifest against the selected run fingerprint, consolidates all
declared Jacobian $J^*J$ energy fits, and creates a readable single-run report.
It intentionally excludes notebook `10` (a two-checkpoint finite-flow
operator) and notebook `12` (quotient/decomposition controls).

The report includes all persisted WeightWatcher `raw` and
`fix_fingers=clip_xmax` rows, independent continuous-MLE `powerlaw.Fit`
analyses of saved weight ESDs, and explicit top-0 through top-5 finger
sensitivities. It never chooses a finger count after inspecting fit quality.
Because there is only one independent training run, it draws no confidence
bands or error bars; checkpoints, probes, examples, and spectral modes are not
treated as replicates.

After the 1,000-epoch MuonClip baseline and its final-100 cache have completed,
run the notebook from the repository root:

```bash
export RUNS="$HOME/rg-mnist-mlp3-tangent-runs"
export CACHE="/tmp/rg-mnist-mlp3-tangent-checkpoints"
mkdir -p "$RUNS/single-seed-notebooks"
python -m ipykernel install --user \
  --name rg-muonclip-run --display-name "Python (rg-muonclip-run)"

papermill \
  baseline/experiments/mnist_mlp3_tangent_rg/notebooks/18_Single_Run_MuonClip_Jacobian_Audit.ipynb \
  "$RUNS/single-seed-notebooks/muonclip_seed_1337.executed.ipynb" \
  -k rg-muonclip-run \
  -p RUN_ROOT "$RUNS" \
  -p CHECKPOINT_CACHE_ROOT "$CACHE" \
  -p PROFILE pilot_1000_epochs \
  -p OPTIMIZER_SLUG muonclip_rms \
  -p SEED 1337 \
  -p RUN_CHILD_NOTEBOOKS true \
  -p SHOW_PLOTS false
```

The default report artifacts are written beneath
`$RUNS/mnist_mlp3_tangent_rg_v1_pilot1000/notebook_outputs/` followed by
`single_run_muonclip_jacobian_audit/muonclip_rms_seed_1337/`. To regenerate
only the consolidated report from already verified child outputs, rerun with
`-p RUN_CHILD_NOTEBOOKS false`. Clean notebooks remain checked in; executed
notebooks and generated plots remain run artifacts.

### Lightweight metrics and WeightWatcher audit

Notebook `21_Single_Run_Metrics_and_WeightWatcher_Audit.ipynb` reads only the
saved performance and WeightWatcher CSV files for one completed seed. It plots
test accuracy and loss, a train/validation/test context check, and the per-layer
`raw` versus `fix_fingers=clip_xmax` alpha trajectories. It also writes explicit
provenance, completeness, bounds, finite-value, fit-availability, and
final-horizon sanity checks. It loads no checkpoints, runs no Jacobians,
launches no child notebooks, and performs no new WeightWatcher fits, so it can
run alongside a heavier analysis notebook.

Execute it from the repository root. `RUN_ROOT` is the persistent root that
contains the protocol directory, not the optimizer or seed directory. No
checkpoint-cache argument is needed:

```bash
export REPO="/private/tmp/rg_optimizers"
export RUNS="$HOME/rg-mnist-mlp3-tangent-runs"
mkdir -p "$RUNS/single-seed-notebooks"
cd "$REPO"

papermill \
  baseline/experiments/mnist_mlp3_tangent_rg/notebooks/21_Single_Run_Metrics_and_WeightWatcher_Audit.ipynb \
  "$RUNS/single-seed-notebooks/muonclip_seed_31415.metrics_ww_audit.ipynb" \
  -k rg-muonclip-run \
  -p RUN_ROOT "$RUNS" \
  -p PROFILE pilot_1000_epochs \
  -p OPTIMIZER_SLUG muonclip_rms \
  -p SEED 31415 \
  -p SHOW_PLOTS false
```

The CSV/JSON audit and PNG figures are written beneath
`$RUNS/mnist_mlp3_tangent_rg_v1_pilot1000/notebook_outputs/` followed by
`single_run_metrics_weightwatcher_audit/muonclip_rms_seed_31415/`.

### Ten-seed MuonClip-RMS versus AdamW comparison

Notebook `22_MuonClip_AdamW_10Seed_Bollinger_Comparison.ipynb` compares the
matched 100-epoch MuonClip-RMS and AdamW arms for seeds `101`, `202`, `303`,
`404`, `505`, `606`, `707`, `808`, `909`, and `1010`. It reads only completed
performance and WeightWatcher CSVs. Training/test accuracy and loss and the
default per-layer `clip_xmax` WeightWatcher alpha are plotted as the cross-seed
mean with Bollinger-style mean plus or minus two sample-standard-deviation
bands. These are run-dispersion bands, not confidence intervals.

From the repository root, run:

```bash
export REPO="/private/tmp/rg_optimizers"
export RUNS="/private/tmp/rg-mnist-mlp3-short100-runs"
mkdir -p "$RUNS/executed_notebooks"
cd "$REPO"

papermill \
  baseline/experiments/mnist_mlp3_tangent_rg/notebooks/22_MuonClip_AdamW_10Seed_Bollinger_Comparison.ipynb \
  "$RUNS/executed_notebooks/22_MuonClip_AdamW_10Seed_Bollinger_Comparison.executed.ipynb" \
  -k rg-muonclip-run \
  -p RUN_ROOT "$RUNS" \
  -p PROTOCOL_SLUG mnist_mlp3_tangent_rg_v1_muonclip_short100_10seed \
  -p SHOW_PLOTS false
```

Figures and summary tables are written beneath
`$RUNS/mnist_mlp3_tangent_rg_v1_muonclip_short100_10seed/notebook_outputs/`
followed by `muonclip_adamw_10seed_bollinger_comparison/`.

### Short-100 ten-seed quotient and weight-only Jacobian rerun

Notebook `23_Short100_10Seed_Weight_Quotients.ipynb` applies the frozen primary
profiles of the five quotient candidates to all 100 cached checkpoints from
both the MuonClip-RMS and AdamW arms and reports 95% Student-t intervals across
the ten independent seeds. It retains the raw-weight, midpoint-ECS, and uniform
singular-translation controls and evaluates every matrix with the same dual
WeightWatcher raw and `fix_fingers=clip_xmax` path.

The post-facto runner executes the complete genuine-Jacobian suite used by the
earlier audit. Notebooks `13` and `16` contain weight-only Jacobians
reconstructible from saved matrices; notebooks `11`, `14`, and `17` consume
the dense optimizer/minibatch captures saved during baseline training. The
runner never resumes training and stops before analysis if any run lacks its
checkpoint cache or dense captures.

```bash
export RG_MNIST_TANGENT_ROOT="/private/tmp/rg-mnist-mlp3-short100-runs"
export RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT="/private/tmp/rg-mnist-mlp3-short100-checkpoints"

bash baseline/experiments/mnist_mlp3_tangent_rg/scripts/run_short100_quotients_jacobians.sh
```

The runner first verifies exactly 100 cached checkpoints for every one of the
20 optimizer/seed runs. It is safely resumable inside the quotient notebook;
executed notebooks are written beneath
`$RG_MNIST_TANGENT_ROOT/executed_notebooks/short100_quotients_jacobians/`.

### Post-facto Muon/MuonClip weight-quotient notebooks

Notebook `19_One_Seed_Muon_MuonClip_Weight_Quotients.ipynb` is the exploratory
one-seed analysis. It defaults to seed `1337`, scans the declared parameter
profiles, and plots all 100 checkpoint trajectories without confidence bands.
Notebook `20_Three_Seed_Muon_MuonClip_Weight_Quotients.ipynb` applies the frozen
primary profile to seeds `1337`, `2027`, and `31415`, then reports two-sided
95% Student-t intervals across the three complete runs. Both notebooks analyze
the `muon` and `muonclip_rms` arms by default; `OPTIMIZER_SLUGS` can select one
arm when only that completed cache is available.

Each notebook first runs the original checkpoint matrix through the existing
dual WeightWatcher path and fixes the midpoint ECS rank from the standardized
`clip_xmax` row. It then reconstructs a full-shape transformed FC1, FC2, and
FC3 matrix for each method and calls the same dual path again:

1. `fix_fingers=False`;
2. `fix_fingers="clip_xmax"`, with the backend `xmax` defining exact tail
   membership.

Every transformed matrix is the rectangular-diagonal canonical representative
of its diagnostic `O(out) x O(in)` singular-spectrum orbit. This removes the
left/right basis coordinates that WeightWatcher's ESD does not observe and
keeps the declared zero modes exactly zero after the float32 model copy. It is
not an exact hidden-unit symmetry of the ReLU MLP, and the transformed models
are never used for forward predictions or accuracy comparisons.
WeightWatcher's entry-randomization fields are still computed to preserve the
audited baseline API, but they are explicitly labelled gauge-dependent and are
not interpreted as invariants of this quotient. The ESD and power-law fit are
the quotient observables.

The five cells are scalar Gram ridge subtraction, blockwise singular shifts,
anchor-frozen Feshbach downfolding, empirical rectangular D-transform
deconvolution, and a discarded-bulk-calibrated monotone MP shrinker. Raw full
weights, midpoint ECS truncation, and the uniform singular-translation rule
are retained as explicit controls. The D-transform method scans the lower-bulk
fraction used for its empirical noise law and is labelled a separated-spike
approximation; it is not presented as unrestricted full-rank rectangular free
deconvolution. FC3 remains a low-rank auxiliary-AdamW control
and may be unavailable when a method requires at least eight discarded modes.

The default clean invocation after the baseline and its cache have completed
is:

```bash
papermill \
  baseline/experiments/mnist_mlp3_tangent_rg/notebooks/19_One_Seed_Muon_MuonClip_Weight_Quotients.ipynb \
  /tmp/weight_quotients_seed_1337.executed.ipynb \
  -p RUN_ROOT "$RG_MNIST_TANGENT_ROOT" \
  -p CHECKPOINT_CACHE_ROOT "$RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT" \
  -p PROFILE pilot_1000_epochs \
  -p SEED 1337
```

Use notebook `20` after all three complete seed caches are available. Its
error bars use seeded runs only; checkpoints, layers, modes, fit variants, and
parameter values are never treated as replicates. It plots both raw and
`clip_xmax` fits and saves incomplete/failed-seed coverage separately. Before
pooling, it runs the suite's device/software/determinism provenance audit.

Both notebooks default to `RESUME_PARTIAL_RESULTS=True`. After every completed
checkpoint/profile dual-WeightWatcher call, the fit and operator tables are
atomically replaced. A rerun reuses a group only when its checkpoint identity,
run fingerprint, parameter JSON, source kind, and analysis-code hash all
match. `method_provenance.json` is marked incomplete at startup and becomes a
completed manifest only after the exact Cartesian result grid passes.

### Optimizer arms

1. **AdamW.** The existing audited MLP3 AdamW recipe and parameter grouping.
2. **Muon.** The existing audited recipe: Muon on `fc1.weight` and
   `fc2.weight`, auxiliary AdamW on `fc3.weight` and all biases, five
   Newton--Schulz steps, decoupled matrix weight decay, and the committed
   warm-up/cosine schedule.
3. **MuonClip-RMS.** The same MLP partition, with the canonical Muon exponential
   moving-average source followed by the orthogonalized update and exact RMS
   matching

   ```text
   M_t = lerp(M_(t-1), G_t, 1 - momentum)
   P_t = NS5(M_t)                         # Nesterov is disabled by default
   O_t = P_t * rms_scale / RMS(P_t)      # RMS(O_t) = rms_scale exactly
   W_t = (1 - lr * weight_decay) * W_(t-1) - lr * O_t
   ```

   and auxiliary AdamW on the classifier and biases. The resolved configuration
   and one-step diagnostics record the exact momentum, learning rates, decay,
   RMS scale, and schedule used. **QK clipping is not applicable to an MLP**:
   there are no query/key matrices or attention logits. This arm is therefore
   named `MuonClip-RMS`, not attention MuonClip.

Optimizer-specific learning rates are allowed because the update geometries
differ. Architecture, data split, initialization, seeds, total update budget,
evaluation states, and measurement schedule remain matched. Every run writes a
protocol fingerprint; incompatible artifacts are rejected rather than reused.
Headline pooling additionally requires identical dataset/split hashes, model
and initialization contract, normalization, analysis plan, device, software
versions, and determinism settings across all nine runs. Checkpoint and capture
payload identities are checked against their filenames and parent manifest.

## Horizon and measurement cadence

Run the stages in this order:

1. **Two-epoch smoke:** one seed first, then all three seeds. Validate the
   optimizer step, checkpoint schema, WeightWatcher calls, transformed-operator
   algebra, and every notebook artifact check.
2. **1,000-epoch pilot:** all three seeds and all three optimizer arms. Do not
   tune a method against the 10,000-epoch result.
3. **10,000-epoch reference:** launch only after the pilot completes and the
   resolved protocol is frozen.

The resolved run configuration is authoritative and is copied into every run
directory. The default long-horizon schedule is:

- update-level optimizer warm-up and cosine decay, followed by the declared
  nonzero floor rather than silently restarting the schedule;
- online training loss/accuracy accumulated every epoch;
- full validation and monitoring-test evaluation every five epochs by default,
  with performance rows also written at the sparse analysis states;
- initial and final states, 96 approximately logarithmically spaced epochs,
  explicit epochs `0, 1, 2, 5, 10, 30`, and any declared tail cadence,
  deduplicated and sorted;
- one restartable latest checkpoint rewritten at the resolved declared cadence
  (every epoch for smoke, every five pilot epochs, and every ten reference
  epochs by default), plus validation-best and final checkpoints; immutable
  epoch-zero and later analysis checkpoints use
  `analysis_epoch_<epoch>_step_<step>.pt`; and
- a separate rolling analysis cache containing the final 100 *trained*
  epoch-boundary model checkpoints (epochs `max(1, E-99), ..., E`) beneath
  `/tmp/rg-mnist-mlp3-tangent-checkpoints` by default, independent of the
  WeightWatcher schedule; and
- eight consecutive-update captures after the default anchor epochs
  `0, 1, 10, 100, 1000` when training updates remain. The resolved config may
  add later anchors before launch but cannot change them after artifacts exist.

Saving every MLP matrix after every update for 10,000 epochs is intentionally
not part of the protocol. The final-100 cache stores one model-only state per
trained epoch. It is ephemeral but required and cannot be reconstructed from
the sparse persistent checkpoints. Keep the same cache through notebooks `10`,
`12`, `13`, `15`, and `16`, or back it up and restore it byte-for-byte before
analysis. If it is lost after a run completes, recovery requires a full
`--overwrite` retraining; `--resume` deliberately refuses a completed run with
a missing cache. Cache and capture schedules and estimated storage are
validated before training. The reference stage must use an explicitly
selected persistent output root; only the dedicated final-100 cache belongs
under `/tmp`.

## Spectral-fit contract

### Native WeightWatcher measurements

Raw weight matrices and any transformed matrices explicitly supported by
WeightWatcher are analyzed with the pinned `weightwatcher==0.7.7` backend. Each
analysis state saves two labelled variants using the same diagnostic RNG seed:

```python
watcher.analyze(
    ERG=True,
    randomize=True,
    plot=False,
    ...,
)

watcher.analyze(
    ERG=True,
    randomize=True,
    plot=False,
    fix_fingers="clip_xmax",
    max_fingers=10,
    ...,
)
```

The preregistered fixed-point alpha is the `clip_xmax` result when the fit is
valid; the unmodified fit remains a required audit. Tables retain `alpha`,
`raw_alpha` where available, `D`, `xmin`, `xmax`, tail size, `num_fingers`,
`ERG_gap`, `num_traps`, status, warning, and complete fit provenance. A required
backend, call, or artifact failure fails the strict analysis state. An
individual invalid layer fit--especially the initial state or rank-10 FC3--is
retained with `status=failed` and disqualifies that measurement; it does not
destroy an otherwise restartable 10,000-epoch campaign. Diagnostic
randomization must not perturb any subsequent training RNG stream.

The pinned WeightWatcher finger count has a known endpoint ambiguity: its
`clip_xmax` loop fits `evals[:-idx]` while reporting `num_fingers = idx - 1`.
When the backend reports a finite `xmax`, the suite reconstructs the primary
window exactly from the saved ESD endpoints (`xmin <= lambda <= backend_xmax`)
rather than guessing membership from either count. The reported count and the
inferred internal-slice `+1` interpretation are both persisted as labelled,
non-certifying sensitivities when the endpoint is available. If `backend_xmax`
is unavailable, the source-audited internal slice removes
`num_fingers + 1` when `num_fingers > 0` and becomes the declared primary
fallback, `weightwatcher_inferred_internal_slice_fallback_tail`; the reported
count remains sensitivity-only. A production claim must be unchanged by the
one-eigenvalue convention, and no sensitivity may be selected after seeing
which is closer to `alpha=2` or trace-log zero.

### Standalone derived-spectrum fits

Spectra that are not passed through WeightWatcher use only
`powerlaw.Fit(values, discrete=False, verbose=False)`. The package chooses
`xmin` using its continuous MLE/KS procedure; no custom regression or homemade
tail-start scan is permitted. The primary fit supplies every finite positive
observation and removes no upper-tail values. Explicit `clip_top_k = 1, ...`
rows are unranked sensitivity analyses and can never be selected after looking
for the smallest KS distance.

Every fit records `operator_kind`, `map_definition`, `spectrum_kind`, fit
backend, fit variant, alpha, sigma, KS distance, package-selected `xmin`,
observed maximum, tail count, tail fraction, tail decades, warnings, and the
checkpoint/layer/seed identity. Amplitude-to-energy conversion is an exact
change of variables on the same fitted sample, not a second fit with a new
support. If `p(b) ~ b^{-alpha_b}` and `e=b^2`, then
`alpha_e=(alpha_b+1)/2`: the native energy reference `alpha_e=2` corresponds
to amplitude `alpha_b=3`. Amplitude plots use reference 3 only when explicitly
testing that same energy hypothesis; exact transformed-energy plots use
reference 2. A generic alpha-two line is never applied to amplitude rows.

### Trace-log support

The claimed fixed-point test is

```text
abs(trace_log_per_eval) <= declared_tolerance
```

only on a support selected independently of that trace-log value--for example,
the preregistered WeightWatcher support
`weightwatcher_backend_xmax_exact_fit_tail`. If the backend does not report a
finite `xmax`, the separately labelled
`weightwatcher_inferred_internal_slice_fallback_tail` is the preregistered
source-audited fallback; `weightwatcher_reported_finger_count_sensitivity_only`
cannot certify the condition.
That certified window excludes the upper fingers reported by the selected
`clip_xmax` fit; it does not silently put clipped fingers back into the logdet.
Searching the same cumulative trace-log curve for the rank nearest zero is
saved as a diagnostic but **cannot certify the trace-log condition**. Every row
therefore records `support_rank_source` and
`support_selected_from_same_trace_log`.

For compatibility with the repository's earlier baseline, notebook `04` also
loads exact `fit_variant=raw`, `support_source=weightwatcher_midpoint` rows from
`metrics/trace_log.csv` when present and plots them separately. That historical
midpoint observable is explicitly non-certifying and never enters the strict
late-state verdict.

Fixed-point qualification is persistent rather than a final-point crossing:
for each optimizer/seed/layer, at least four of the last five scheduled
measurements must jointly satisfy the declared alpha window, KS threshold,
minimum tail count, and independent-support trace-log tolerance. A single
late `alpha=2` crossing cannot qualify a run.

## Operators and identifiability

Every derived row and plot must display both `operator_kind` and
`map_definition`.

- `W_t - W_(t-1)` is a checkpoint displacement.
- `W_t W_(t-1)^+`, or its orientation-correct counterpart, is a discrete
  relative-flow operator.
- For rectangular layers, the exact nonzero support of the minimum-norm
  left/right transfer and a separately labelled Procrustes-aligned rank-
  `r` core are finite checkpoint maps. Ambient structural zeros are counted,
  never included in the tail fit.
- A difference divided by the step interval is a beta surrogate.
- `D Pi(W)` is the Jacobian of a specified projection such as the polar map
  `Pi(W)=UV^T`.
- Notebook `13` treats six maps computable from one checkpoint as candidate
  RG transformations and forms the actual derivative of every one: the polar
  map, normalized smaller-Gram map, centered matrix-log Gram map, centered
  log-singular radial map, and the exact configured finite Muon NS5 map applied
  directly to `W`, plus the wide-FC1 restricted retracted-core ECS Grassmann
  Cartan cover. The latter is the smooth checkpoint-anchored map
  `Phi_W(E)=V_c^T(2 P_row(R_W(K_W(E)))-I)V_k`, with
  `K_W(E)=V_c^T E^T U_k Sigma_k^-1`; its actual derivative at `E=0` is
  `J[E]=2 V_c^T E^T U_k Sigma_k^-1` and has amplitudes `2/sigma_i`
  repeated `q-k` times. Its retained `k` is the finger-aware power-law top-mode
  boundary. The requested primary variant takes outer `q` to be the checkpoint
  numerical row rank; a separately named sensitivity takes `q=detX_num`.
  Both use exact same-checkpoint WeightWatcher/trace states. Metric roles are
  never swapped; missing, coincident, or reversed detX boundaries are recorded
  rather than filled. At fixed `k`, changing `q` only changes the uniform
  multiplicity `q-k`, so detX-`q` is a shell-dimension/multiplicity sensitivity,
  not independent spectral-shape or alpha corroboration. ECS finger
  sensitivities remove complete `(q-k)`-mode core groups, and `fit_ok` requires
  at least `MINIMUM_TAIL` retained-core groups above the package-selected
  `xmin`; repeated modes cannot qualify a tail by themselves. Notebook `15`
  treats the comparison as incomplete and stops unless every optimizer/seed
  has a successful group-qualified primary full-`q` ECS-cover fit. The
  detX-`q` shell remains a sensitivity and is not required to exist when its
  boundary is coincident or reversed.
  Only the nonzero singular spectrum of each Jacobian is fit;
  an undifferentiated Gram translation or checkpoint displacement is not
  admitted as a single-checkpoint Jacobian candidate.
- A calibrated local optimizer response is the derivative of the fully
  specified training step, including its loss batch, optimizer state, and
  calibration perturbation.
- Notebook `16` adds five further weight-only derivatives: the exact gap-aware
  retained projector, a soft logistic ECS projector, the outer trace-free
  log-Gram map, a multiscale trace-free resolvent, and the trace-free log of a
  Feshbach/Schur effective core. The hard-cutoff ranks are joined only at exact
  same-checkpoint WeightWatcher states. In the checkpoint SVD gauge the
  Feshbach coupling block is zero, so its shell correction vanishes at first
  order; that equality is recorded rather than presented as independent shell
  information.
- Notebook `17` adds five captured-minibatch derivatives: input-to-output,
  Grassmann-parameter-to-output, per-example quotient loss, quotient
  generalized Gauss--Newton, and the full replayed Muon/MuonClip one-step
  quotient-stability map. The first four are exact on the selected examples.
  The input-output control uses only the capture; the other quotient maps also
  require an exact same-state ECS boundary and never use a neighboring rank.
  The full step is an explicitly labelled central-difference restriction to a
  preregistered orthonormal set of quotient probes and is replay-qualified
  before fitting.

For notebooks `13`, `16`, and `17`, the fitted energy sample is always the
nonzero spectrum of `J^*J`. Empirical-Fisher
rows already equal a Jacobian Gram. When the quotient GGN itself is treated as
the declared curvature Jacobian, its `J^*J` energies are the squared nonzero
GGN eigenvalues. Mode, minibatch, checkpoint, and probe counts never inflate
the three independent seeded runs used for confidence intervals.

Calibrated-map fitting is gated by an unperturbed replay on the manifest's
original device. The replay must have finite maximum error no greater than the
declared tolerance. An explicit cross-device exploratory override persists
`replay_qualified=false` and cannot enter notebook `15` retention summaries.

None of these objects is automatically the Jacobian of an unknown RG flow.
From a weight matrix alone, the training-map Jacobian is not identifiable
without specifying the map, optimizer state, data/loss probe, and perturbation
protocol. The notebooks preserve this distinction even when two spectra happen
to have similar fitted exponents.

For rectangular matrices, the orientation, supported rank, row/column-space
gauge, forced subspace intersection, and zero-mode policy are part of the map
definition. FC3 is only `10 x 512`, so its Gram spectrum has at most ten
positive modes. Power-law fits there are intrinsically weak and are reported as
low-rank diagnostics rather than evidence equal to FC1 or FC2.

## Nulls and method-retention rule

Candidate methods are compared against controls appropriate to their map:

- identical checkpoints / exact zero displacement;
- known isotropic scaling;
- pure left or right rotation;
- pure rectangular subspace motion;
- entry-shuffled or Gaussian matrices with matched shape and scale;
- Haar/Stiefel angular controls; and
- temporally scrambled checkpoint order for multi-checkpoint operators.

An apparent approach to `alpha=2` is not enough. A method is considered
validated only if all of the following hold without changing the preregistered
fit/support rules:

1. analytic identities and numerical convergence checks pass;
2. fit quality, tail count, and tail extent are credible;
3. the result is stable over finite-difference epsilon, probe count, checkpoint
   spacing, and explicit finger-sensitivity rows;
4. the trained result separates from the matched null distribution;
5. late-training stationarity reproduces over the three complete seeds; and
6. the conclusion is not carried only by rank-10 FC3.

Until those tests pass, notebook results are exploratory hypotheses rather than
validation of a tangent-space RG measurement.

## Artifact layout

Set a persistent root for pilot/reference runs, for example:

```bash
export RG_MNIST_TANGENT_ROOT=/absolute/persistent/path/mnist-mlp3-tangent-rg
```

The notebooks expect the package runtime to write:

```text
<root>/<protocol-slug>/
  <adamw|muon|muonclip_rms>/
    seed_<seed>/
      manifest.json
      resolved_config.json
      run_complete.json
      metrics/
        training_by_epoch.csv
        validation_by_epoch.csv
        performance_by_analysis_epoch.csv
        weightwatcher_fits.csv
        trace_log.csv
        spectral_metrics_by_analysis_epoch.csv
        esd/esd_epoch_<epoch>_step_<step>.npz
      checkpoints/
        checkpoint_latest.pt
        checkpoint_best.pt
        checkpoint_final.pt
        analysis_epoch_<epoch>_step_<step>.pt
      captures/burst_epoch_<epoch>/capture_step_<step>.pt
  notebook_outputs/
    comparison/
    analyses/
      two_checkpoint_finite_flow/
      muon_update_stiefel_tangent/
      radial_angular_quotients/
      single_checkpoint_map_jacobians/
      calibrated_local_training_map/
      additional_weight_only_ecs_jacobians/
      data_dependent_ecs_jacobians/
      method_nulls_stability/
      weight_only_muon_quotients_one_seed/
      weight_only_muon_quotients_three_seed/
      single_run_metrics_weightwatcher_audit/
      # every method directory also contains method_provenance.json
```

The independent, model-only trajectory consumed by notebooks `10`, `12`,
`13`, `15`, `16`, `19`, and `20` is stored separately:

```text
/tmp/rg-mnist-mlp3-tangent-checkpoints/
  <protocol-slug>/<adamw|muon|muonclip_rms>/seed_<seed>/
    manifest.json
    cache_complete.json
    checkpoints/
      analysis_epoch_<epoch>_step_<step>.pt  # final min(100, E) trained epochs
```

`CHECKPOINT_CACHE_ROOT` and
`RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT` may select another safe child of
`/tmp`. The cache is ephemeral but required, so deleting `/tmp` means the final
training window is lost and full `--overwrite` retraining is required; an analysis
notebook never recreates it by training and `--resume` does not backfill it.
The exact FP32 model-tensor payload is about 255 MiB per 100-state run, or
2.25 GiB for the complete nine-run matrix, before Torch serialization and
filesystem overhead. Before any matrix is loaded, the notebooks compare the
cache against the completed persistent run and require exact agreement in
suite, optimizer, seed, protocol fingerprint, expected epoch/step grid, filenames, payload
identity, checkpoint count, and completion marker.

The default checkpoint analyses use all 100 states. Notebooks `10` and `12`
report their exact pair counts before computing. They use all 99 adjacent
pairs as the primary trajectory, plus eight deterministic pairs at each of
strides `2`, `4`, and `8` as bounded spacing sensitivities: 123 pairs per
optimizer/seed by default, rather than the exhaustive 385-pair stride grid.
Checkpoint payloads are held in a bounded 24-entry
LRU so repeated layer extraction does not repeatedly deserialize the same
files or retain the full nine-run cache in memory.

Notebook-generated figures and CSVs are written beneath the selected output
root. Executed notebooks use `.out.ipynb` or `.executed.ipynb` names and are not
versioned here.

## Notebook order

Run `00_Protocol_and_Smoke.ipynb` first. Then run the three training notebooks,
followed by `04_Fixed_Point_Comparison.ipynb`, notebooks `10`--`14`, and the
new Jacobian notebooks `16` and `17`; run notebook `15` last because it verifies
and combines every preceding method. The analysis notebooks
reuse the verified final-100 checkpoint cache or the pre-existing dense
captures; they do not retrain a private notebook-local model. Specifically,
`10`, `12`, `13`, `16`, `19`, `20`, and the checkpoint-derived nulls in `15`
consume the cache;
`11`, `14`, and `17` consume captures because their objects require optimizer and
minibatch state not present in a model-only checkpoint.

Notebook `18` is an optional single-run driver after training. It invokes only
the genuine Jacobian notebooks `11`, `13`, `14`, `16`, and `17` for one
MuonClip seed, then builds descriptive no-error-bar plots and independent
WeightWatcher/`powerlaw.Fit` diagnostics. It does not replace notebook `15` or
the preregistered three-seed comparisons.

Notebook `21` is the fast one-seed metrics/WeightWatcher dashboard described
above. It is independent of the checkpoint cache and can be run immediately
after the selected training run has a valid completion marker.

Run notebook `19` when only one completed seed is available, or notebook `20`
when the complete three-seed Muon/MuonClip grid is available. These notebooks
are independent of notebook `15`: they test state-level quotient/deconvolution
hypotheses and always retain the raw weight ESD as the assumption-free control.

Every analysis method writes `method_provenance.json` with its exact suite,
method, source-artifact kind, and optimizer/seed-to-protocol-fingerprint grid.
Notebook `15` verifies every dependency manifest and every fit-table row
against the currently completed run grid before combining results, so outputs
from an earlier `--overwrite` run cannot be silently mixed with a new cache.
