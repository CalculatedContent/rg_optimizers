# Short GPT: five AdamW versus five momentum-SGD seeds

This separate study supersedes the unfinished ten-AdamW-seed replication.
The original pilot and any partial replication checkpoints remain separate.
It imports the frozen architecture, data loader, complete evaluation, and raw
WeightWatcher implementation from `../short_test`. It does not alter that code.

The objective is a comparison of two validation-calibrated optimizer recipes,
with five matched, fresh seeds each. Published recipes motivate the search;
they do not establish globally optimal settings for this tiny model.

## Frozen calibration and confirmation

The model has four blocks, four attention heads, width128, context64,
804096 unique parameters, tied token/output weights, and zero dropout.
The dataset and contiguous90/5/5 split are identical to the earlier pilot.
Batch size is12 sequences (768 targets/update), CPUfloat32, one thread.

Calibration uses seed2027, distinct from confirmation seeds1340–1344:

| Optimizer | Peak learning rates | Additional variation | Fixed settings |
|---|---|---|---|
| AdamW | .0003, .001, .003 | matrix weight decay0 or .1 | betas(.9,.99), epsilon1e-8 |
| SGD with momentum | .01, .03, .1 | momentum .9 or .98 | decay0, dampening0, no Nesterov |

All12 candidates run4000 updates. The top2 per optimizer, ranked by minimum
full-validation cross entropy, continue from their exact states to20000
updates. One recipe per optimizer is then frozen in `recipes.json`, before
any confirmation test evaluation. Only the seed varies within each final group.
This successive-halving search can miss slow starters. It establishes the best
settings among the tested full-budget finalists, not a universal optimum.

Both optimizers use100 warmup updates, cosine decay through17000 updates to
one tenth of peak LR, then3000 updates at that floor. Raw gradients are clipped
at norm1 before optimizer updates. The schedule horizon is20000 even during
calibration screening, so continuation does not reset or change the schedule.
All final runs have the same20000-update budget. Plateau diagnostics are
reported; reaching the cap is not called convergence. The use of different
selected decay settings would make this a recipe comparison rather than an
isolated causal test of optimizer identity.

Full train/validation/test cross entropy, perplexity, and next-character
accuracy are reported separately for the endpoint and best-validation weights.
The test partition has already been exposed in the earlier discovery pilot.
No test scores or alpha values guide calibration, recipe selection, or retuning.
All18 unique matrix alpha values use raw WeightWatcher0.7.7 defaults, including
the tied token embedding/output matrix once. Packed QKV remains packed.
LayerNorm vectors are not weight matrices and cannot receive matrix alpha fits.

Elapsed optimizer-update time, training plus validation time, and final
evaluation time are recorded separately, accumulated across resumed sessions.
Model/optimizer/momentum, Torch, minibatch-generator, NumPy, and Python RNG states
are saved atomically every250 updates. Scientific nonfinite failures are
recorded explicitly. Infrastructure failures remain resumable.

## Run

From the repository root, with the existing TinyShakespeare `input.txt`:

```bash
python short_test_optimizer_comparison/test_contract.py --data /absolute/path/input.txt
python short_test_optimizer_comparison/comparison.py prepare --output /absolute/path/study --data /absolute/path/input.txt
python -u short_test_optimizer_comparison/comparison.py run --output /absolute/path/study --data /absolute/path/input.txt --legacy /absolute/path/old-ten-seed-study
```

Stop the old ten-seed controller before running this replacement. Retain its
last valid full checkpoint. The new controller holds the old batch lock and a
shared GPT lock to prevent concurrency. Only one GPT worker is permitted while
the five MLP follow-up workers and one layer-extension worker remain active.
SIGTERM/SIGINT request a checkpointed stop; `PAUSE` in the study root does the
same. Remove that marker only when intentionally resuming.

`summarize` writes result tables and paired AdamW-minus-SGDM differences with
Student-t95% intervals across seeds. Incomplete pairs are identified, and
these intervals do not account for corpus or hyperparameter-search uncertainty.
`package` creates timestamped ZIP parts below the normal transfer threshold;
save all parts and retain their receipts. Never overwrite newer local states
on restore. Source hashes in `design.json` must match before every worker launch.

## Primary references checked October21,2026

- [nanoGPT Shakespeare character recipe](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py): peakLR.001, beta2.99, warmup100, floorone-tenth. Its architecture and batch size differ from ours.
- [Srećković, Geiping, Orvieto, 2025: Is your batch size the problem?](https://arxiv.org/abs/2506.12543): small-batch SGD, momentum tuning including.98, optimizer-specific LR sweeps and gradient clipping. Its large-model LR optima are not transferred as established optima here.
- [PyTorch SGD](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html): standard momentum buffer `v = momentum*v + gradient`, followed by `parameter -= lr*v`; no unreported learning-rate rescaling.
