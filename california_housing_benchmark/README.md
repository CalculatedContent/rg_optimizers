# California Housing: reproduce a published MLP, then compare optimizers

This experiment anchors the comparison to **Gorishniy et al., Benchmarking Optimizers for MLPs in Tabular Deep Learning (2026)**. It preserves the earlier exploratory California Housing cohort under `california_housing/`.

## What result should we expect?

The published plain-MLP AdamW recipe reports test RMSE **0.4935855 ± 0.0110033**, where ± is **sample SD across 10 seeds**. Targets are in $100,000 units, so this is about **$49,359 RMSE**, not 49% error.

Within that 2026 paper, the best ordinary MLP result is **0.4708079 ± 0.0026432** using **Muon plus exponential weight averaging**. These optimizers were tuned jointly with architecture, so that score is not an otherwise-identical AdamW comparison.

Stronger architectures can improve further. The TabM ICLR 2025 paper's Table 18 gives **0.3964 ± 0.0013** for a TabR ensemble on the same established California split. An official PriorLabs TabPFN example reports MSE 0.1328 (RMSE about 0.3644), but uses a different 90/10 split and a pretrained, fine-tuned model. Neither is the same small MLP trained from scratch. There is no universal score across different splits, target normalizations, and architectures.

Primary sources:

- [2026 optimizer benchmark](https://arxiv.org/html/2604.15297v2)
- [Frozen official implementation](https://github.com/yandex-research/tabular-dl-optimizers/tree/c34e33152db9eea8ed4b2d742b9bef5628d71fb5)
- [TabM paper, Table 18](https://arxiv.org/pdf/2410.24210)
- [Official TabPFN example and reported result](https://github.com/PriorLabs/TabPFN/pull/1101)

The reference configurations and individual seed reports are copied unmodified under `reference/upstream/`, with the upstream Apache 2.0 license. This project recalculates their means and SDs instead of relying on a leaderboard entry. Unverifiable or differently normalized Internet claims are not used as raw-dollar targets.

## Fixed model and data

- Published AdamW architecture: **8 → 784 → 784 → 1**, **623,281 parameters**.
- Two blocks of Linear → ReLU → Dropout; dropout **0.3690073279296028**.
- The actual `tabm.MLPBackbone==0.0.3` defines the backbone; scalar linear output.
- The authors' exact **13,209 / 3,303 / 4,128** train/validation/test arrays are downloaded from an immutable dataset revision and individually SHA256-verified.
- Inputs use the authors' training-only noisy Gaussian quantile transform: 440 quantiles, preprocessing seed 0, Gaussian fit noise 1e-5, then transform the original inputs.
- Targets use training-only float32 mean/std. Scoring reverses this transform.
- Batch 256, one seeded permutation per epoch, final partial minibatch retained: 52 updates/epoch.
- Constant learning rate; global gradient clip 1; CPU float32; one numerical thread per worker.
- The primary checkpoint minimizes validation RMSE before the **first 16-epoch patience stop**. Test metrics never choose checkpoints or hyperparameters.

Raw dataset URL and file hashes are in `reference_manifest.json` and `reference/dataset_archive_manifest.json`. Only a 4 MiB prefix of the authors' 1 GB tar is needed. The full tar hash from the hosting metadata is recorded as metadata, not falsely claimed to have been recomputed.

## Three-optimizer comparison

All three arms share this published architecture, preprocessing, data, minibatches, and validation rules.

**Native AdamW is fixed to the paper's released recipe:**

- LR 0.0002722214565241805
- weight decay 0.4555843429087667 on weight matrices; **zero decay on biases**
- betas (0.9, 0.999), epsilon 1e-8

**Modified AdamW** preserves those Adam settings and moments, while rescaling each of the three weight-matrix gradient updates to the norm of its same-run, zero-dampening momentum reference. Decoupled weight decay remains separate. Biases retain native AdamW updates. This is the established Adam-direction/SGDM-magnitude graft, not an update projection.

**SGD with momentum** uses momentum 0.9, zero dampening, and no Nesterov. Its learning rate and weight decay are selected on development data.

The published SGD row (RMSE 0.5177) used five 272-wide hidden blocks and dampening 0.9. It is contextual information, not the target of our common-architecture SGDM arm.

## Calibration and confirmation

The complete plan is frozen in `plan.json` before confirmation begins.

- Development seeds **2338 and 2339** are excluded from all primary statistics.
- SGDM: LR {0.003, 0.01, 0.03} × weight decay {0, 0.005277362486160341}.
- Modified AdamW: momentum-reference LR {0.001, 0.003, 0.01, 0.03, 0.1}; every native Adam setting stays fixed.
- Select the candidate with the lowest mean best-validation RMSE over the two development seeds. Each development trajectory uses the same patience rule, with a 20,000-update cap; any cap is labeled.
- Native AdamW has **no new tuning**.
- Fresh confirmation seeds **2340–2344**, paired across all three optimizers, follow the frozen selection.
- A separate reference run uses the authors' seed 0 and is excluded from primary seed statistics.

This is a limited predeclared calibration, not a claim of globally optimal SGDM or graft hyperparameters. No confirmation or test result changes the selected settings.

## Primary performance and long-horizon diagnostics

The main reproduction score uses the first validation-stopped checkpoint. Confirmation trajectories also retain **2,000 and 20,000 update** checkpoints and measurements every 500 updates. Continuing after the first patience stop never changes the primary selection.

If the first patience stop occurs after 20,000 updates, training continues until it. The 100,000-update safety limit is explicit; a censored run is not reported as a complete primary reproduction.

Each primary run retains:

- `result.json`: epoch validation history, fixed-update metrics, raw spectral fits, provenance, initial and minibatch hashes;
- `best_checkpoint.pt`: clearly labeled weights-only primary checkpoint;
- `checkpoint_step_2000.pt` and `checkpoint_step_20000.pt`: weights, optimizer and RNG state;
- `resume.pt`: full latest resumable state, including partial-epoch permutation and dropout RNG.

Completed calibration metrics are committed. Interrupted calibration state is retained for recovery. The workflow also creates temporary Actions artifact copies, while primary results and checkpoints are committed to the repository.

## Spectral statistic

Only the **784 × 784 hidden weight matrix** meets the fixed minimum matrix dimension 64. The narrow 784 × 8 input and 1 × 784 output are excluded. Therefore the reported confidence-tail weighted average **equals the eligible matrix's fitted alpha**; this model cannot establish collective marginality across many fitted layers.

The fitter uses ordinary alpha with weight `sqrt(num_pl_spikes)/(D+0.02)`, no extra matrix-size factor, and no substitution of WeightWatcher's `alpha_weighted`. It retains the fit details, tail counts, KS distance, cutoffs and eigenvalues. Measurement preserves all training RNG and weights.

## Statistics and provenance

The analysis requires 15 completed primary runs: five seeds per optimizer. It recalculates the raw-fit averages, equal-weight seed means, sample SD, and **two-sided 95% Student-t confidence intervals across seeds**. Layer dispersion is not used as a seed error bar. These intervals condition on one fixed data split.

It reports the paired modified-minus-native difference and the native-versus-published difference. The latter uses an independent-seed Welch interval and is **not an equivalence test**. Published ±SD and our ±CI are labeled separately.

Python 3.12.9 and the principal numerical packages follow the author's environment; Torch is the 2.7.1 CPU build. Runtime, CPU, packages, threads, source hashes, GitHub run ID and launch commit are recorded. CPU/GPU kernel differences can change exact trajectories despite matching software versions.

## Run locally without a notebook

From the repository root, use Python 3.12.9:

```bash
python -m venv .venv-housing-benchmark
. .venv-housing-benchmark/bin/activate
python -m pip install -r california_housing_benchmark/requirements.txt
python -m california_housing_benchmark.verify
python -m pytest -q california_housing_benchmark/tests
python -m california_housing_benchmark.data --data-dir california_housing_benchmark/.cache/data
python - <<'PY'
import json, subprocess, sys
from pathlib import Path
base = Path('california_housing_benchmark')
plan = base / 'plan.json'
root = base / '.cache/local_reproduction'
data = base / '.cache/data'
common = ['--plan', str(plan), '--data-dir', str(data), '--root', str(root)]
settings = json.loads(plan.read_text())
for candidate in settings['calibration']['candidates']:
    subprocess.run([sys.executable, '-m', 'california_housing_benchmark.calibration',
                    'trial', '--candidate', candidate['id'], *common], check=True)
subprocess.run([sys.executable, '-m', 'california_housing_benchmark.calibration',
                'freeze', *common], check=True)
for seed in settings['confirmation_seeds']:
    subprocess.run([sys.executable, '-m', 'california_housing_benchmark.run_seed',
                    '--seed', str(seed), *common], check=True)
subprocess.run([sys.executable, '-m', 'california_housing_benchmark.analysis',
                '--root', str(root), '--local-reproduction'], check=True)
PY
```

Repeat the same candidate or seed command to resume. Completed matching runs are verified and skipped. Never point local runs at the primary GitHub results directory or mix them with the GitHub cohort.

On GitHub, re-run only failed jobs of the original workflow. Preserve successful jobs and their checkpoints. A new workflow ID may not adopt a completed cohort's frozen identity.

## Initial verification and execution recovery

Before the local workspace disconnected, data preprocessing matched the authors' arrays bit for bit; seven core training tests and five data tests passed. A 100-update native test on excluded seed 2337 matched the actual upstream model/update loop bit for bit for all six parameter tensors. That excluded test is not a primary result.

Source files were recovered through the GitHub API after the disconnection. **The complete test suite runs again on GitHub before any development or confirmation training.** The source manifest and launch commit identify the version actually executed. No successful primary run from the earlier exploratory cohort is replaced.
