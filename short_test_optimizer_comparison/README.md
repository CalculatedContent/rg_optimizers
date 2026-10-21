# Short GPT: five AdamW versus five momentum-SGD models

This is the current experiment, replacing the unfinished ten-AdamW replication.
It compares two separately calibrated recipes using the same five fresh seeds.
The discovery pilot and partial replication remain separate measurements.

The running study's Python source is pinned to commit
`39f2fb75a3c023d8d90abe11ab06308d1bf3919e`. Both this directory and
`../short_test/` are required. Later documentation-only commits preserve those
Python hashes. [frozen_design_20261021.json](frozen_design_20261021.json) is the
exact running study design: source/data hashes, package versions, candidates,
seeds, schedule, selection rules, and metrics. The winning recipes and final
results are produced by the run; they are not known in advance.

## Reproduce from a clean checkout

Use Linux with cgroup v2 and Python 3.12, matching the recorded CPU environment.
The controller reads `/sys/fs/cgroup/memory.max` and `memory.stat` for admission.
No GPU is required. Keep source and dependencies pinned; hardware and numerical
libraries can affect exact floating-point results and runtime.

From the repository root, in a dedicated Python environment:

```bash
set -euo pipefail
python -m pip install -r short_test/requirements.txt
GPT_DATA="$PWD/short_test_optimizer_comparison/input.txt"
GPT_STUDY="$PWD/short_test_optimizer_comparison/outputs/adamw_vs_sgdm"

python short_test/download_data.py --output "$GPT_DATA"
python short_test_optimizer_comparison/test_contract.py --data "$GPT_DATA"
python short_test_optimizer_comparison/comparison.py prepare \
  --output "$GPT_STUDY" --data "$GPT_DATA"

# Must match in the recorded environment; inspect differences if it does not.
cmp "$GPT_STUDY/design.json" \
  short_test_optimizer_comparison/frozen_design_20261021.json
git rev-parse HEAD > "$GPT_STUDY/source_commit.txt"

python -u short_test_optimizer_comparison/comparison.py run \
  --output "$GPT_STUDY" --data "$GPT_DATA"
```

The downloader verifies TinyShakespeare SHA256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
If the design comparison fails, inspect source and package-version differences;
do not edit saved hashes to force a match. Use a separate output directory for
a different experiment.

## Frozen calibration and confirmation

| Property | Setting |
| --- | --- |
| Architecture | 4 blocks, 4 heads, width 128, context 64, 65-character vocabulary |
| Parameters | 804,096 unique parameters; tied token/output weights; no bias or dropout |
| Data | Contiguous 90% training / 5% validation / 5% test TinyShakespeare partitions |
| Batch and compute | 12 sequences, 768 targets per update; CPU float32; one thread |
| Calibration seed | 2027 |
| Confirmation seeds | 1340, 1341, 1342, 1343, 1344 for each optimizer |
| Shared schedule | 100-step warmup; cosine to 10% of peak at step 17,000; 3,000 updates at that floor |
| Gradient clipping | Raw gradient norm 1 before the optimizer update |

| Optimizer | Peak learning rates | Calibration variation | Fixed settings |
| --- | --- | --- | --- |
| AdamW | 0.0003, 0.001, 0.003 | Matrix weight decay 0 or 0.1 | Betas (0.9, 0.99), epsilon 1e-8 |
| SGD with momentum | 0.01, 0.03, 0.1 | Momentum 0.9 or 0.98 | Decay 0, dampening 0, no Nesterov |

1. Run all 12 calibration candidates for 4,000 updates on seed 2027.
2. Rank by minimum full-validation cross entropy. Endpoint validation loss,
   then candidate ID, break ties. Freeze the top two per optimizer in
   `finalists.json`.
3. Continue those four exact optimizer/RNG states to 20,000 total updates.
   The schedule has a 20,000-update horizon even during screening.
4. Freeze one recipe per optimizer in `recipes.json` using the same ranking,
   before any confirmation test evaluation.
5. Run five AdamW and five momentum-SGD confirmations, interleaved. Only the
   seed varies within each optimizer group. Matched seeds share initialization
   and minibatch sequences; all final models receive 20,000 updates.

Calibration is additional to the final ten models. Without numerical failures,
the budget is 312,000 updates: 48,000 screening, 64,000 additional finalist
updates, and 200,000 confirmation updates. This bounded single-seed search can
miss slow starters. It finds the best tested full-budget finalist per optimizer,
not a global optimum. Selected decay can differ, so this compares calibrated
recipes rather than isolating a causal effect of optimizer identity.

## Loss, alpha, and timing

Full training, validation, and test cross entropy (nats per character),
perplexity, and next-character accuracy are recorded separately for the final
endpoint and minimum-validation weights. Every next-character target is scored
once, including incomplete tails, with context reset every 64 characters. The
whole corpus defines the fixed vocabulary; held-out text supplies no gradient
updates. Calibration evaluates neither test metrics nor alpha.

The test partition was exposed in the discovery pilot. That exposure is
reported; test values cannot select or retune recipes. Final 1,500-update
training/validation loss changes are retained as plateau diagnostics. The
training diagnostic uses fixed sampled batches; final training metrics use the
full partition. Reaching 20,000 updates does not establish convergence.

Raw WeightWatcher 0.7.7 defaults fit all 18 unique matrices, including positional
embeddings and the tied token/output matrix counted once. QKV stays packed;
LayerNorm vectors have no matrix spectrum. Per-matrix fits, successful-fit
counts, counts below 2, and arithmetic means are retained. A full-model mean
requires all 18 successful fits. Alpha never selects a model or stops training.

Per-model `result.json` retains these `timing_seconds` fields:

| Field | Includes |
| --- | --- |
| `training_updates` | Minibatches, forward/backward passes, clipping, optimizer updates |
| `training_with_monitoring` | Training loop, validation, checkpoint work; excludes setup and pauses between sessions |
| `final_evaluation` | Full endpoint/selected split evaluations and alpha fits |

The first is a component of the second: do not add them together.
`training_with_monitoring + final_evaluation` measures combined work after setup.
Timings accumulate across resumes; runtime is distinct from convergence.

## Resume, pause, and replace the old batch

Reissue the same `run` command to resume; completed jobs are skipped. Atomic
`last.pt` checkpoints every 250 updates preserve model, optimizer (including SGD
momentum), all RNG states, best-validation tensors, history, and timing. Source
hashes are checked on every worker launch. Nonfinite numerical failures are
explicit records; infrastructure failures remain resumable.

SIGINT/SIGTERM request a checkpointed stop. A `PAUSE` file in the study root
requests the same at the next update boundary. Remove it only when intentionally
resuming. Batch, per-job, and shared GPT locks prevent duplicates. The controller
uses one GPT worker and waits for memory admission.

To replace a historical ten-AdamW batch, stop its controller and preserve its
latest full checkpoint separately. Then include its directory:

```bash
python -u short_test_optimizer_comparison/comparison.py run \
  --output "$GPT_STUDY" --data "$GPT_DATA" \
  --legacy /absolute/path/to/old-ten-seed-study
```

`--legacy` checks/holds the old locks; it does not import a partial old seed as a
confirmation. The shared active environment allows five MLP follow-up workers,
one layer-extension worker, and one GPT worker. Do not add GPT workers.

## Results and portable checkpoints

```bash
python short_test_optimizer_comparison/comparison.py summarize \
  --output "$GPT_STUDY"
python short_test_optimizer_comparison/comparison.py package \
  --output "$GPT_STUDY" --data "$GPT_DATA"
```

| Output | Purpose |
| --- | --- |
| `design.json`, `finalists.json`, `recipes.json` | Frozen design and validation-only selections |
| `calibration/*/summary_4000.json`, `summary_20000.json` | Calibration evidence, separate from confirmation |
| `confirmation/*/selection.json`, `result.json` | Selection, full split metrics, raw layer fits, timing, stop interpretation |
| `results.json`, `results.csv` | Successful confirmations, endpoint and selection separately |
| `paired_statistics.json` | Matched-seed AdamW-minus-SGD differences and Student-t 95% intervals |
| `*/numerical_failure.json` | Explicit failures, never silently counted as successful models |
| `package_index.json`, timestamped ZIP parts | Checkpoint packages, byte counts, and SHA256 hashes |

Summaries refresh after each confirmation. Incomplete paired counts are explicit.
Five-seed intervals describe training variability, not corpus or search
uncertainty, and do not calibrate a general poor-training probability. Final
reporting must audit all ten outcomes, show individual seeds and uncertainty,
and retain failures. `training_finished_analysis_pending` means training ended;
analysis and archive verification still require completion.

Save every package part and its index. Packages contain data and resumable
state; source is recovered from the pinned repository commit. Restore matching
jobs by checkpoint step and never overwrite newer progress. Weights, data, and
output archives are not committed to this source repository.

## Primary references checked October 21, 2026

- [nanoGPT character recipe](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py): peak LR 0.001, beta2 0.99, warmup 100, and a one-tenth floor motivate the search. Architecture and batch differ from ours.
- [Srećković, Geiping, Orvieto (2025), Is your batch size the problem?](https://arxiv.org/abs/2506.12543): small-batch SGD, momentum tuning including 0.98, separate LR sweeps, and clipping motivate optimizer-specific calibration. Large-model optima are not assumed optimal here.
- [PyTorch SGD](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html): standard unnormalized buffer `v = momentum*v + gradient`, then `parameter -= lr*v`; no custom LR rescaling.
