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
run notebook `00`, then `01`--`03`, `04`, and finally `10`--`15` in numeric
order. Training notebooks are launch-safe by default: set
`EXECUTE_TRAINING=True` explicitly, while analysis notebooks require completed
artifacts. The checkpoint-based notebooks `10`, `12`, `13`, and `15` require
the verified tail cache and never fall back to training or to the sparse
WeightWatcher checkpoint series. Notebooks `11` and `14` require already saved
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
`12`, `13`, and `15`, or back it up and restore it byte-for-byte before
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
- A calibrated local optimizer response is the derivative of the fully
  specified training step, including its loss batch, optimizer state, and
  calibration perturbation.

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
      method_nulls_stability/
      # every method directory also contains method_provenance.json
```

The independent, model-only trajectory consumed by notebooks `10`, `12`,
`13`, and `15` is stored separately:

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
followed by `04_Fixed_Point_Comparison.ipynb`. The numbered analysis notebooks
reuse the verified final-100 checkpoint cache or the pre-existing dense
captures; they do not retrain a private notebook-local model. Specifically,
`10`, `12`, `13`, and the checkpoint-derived nulls in `15` consume the cache;
`11` and `14` consume captures because their objects require optimizer and
minibatch state not present in a model-only checkpoint.

Every analysis method writes `method_provenance.json` with its exact suite,
method, source-artifact kind, and optimizer/seed-to-protocol-fingerprint grid.
Notebook `15` verifies every dependency manifest and every fit-table row
against the currently completed run grid before combining results, so outputs
from an earlier `--overwrite` run cannot be silently mixed with a new cache.
