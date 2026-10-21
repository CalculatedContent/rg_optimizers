# Short character GPT test

Reproduce the original AdamW character-language-model pilot. This directory
also retains the historical ten-AdamW-seed runner and the common implementation.

**Current experiment:** [five AdamW versus five momentum-SGD models](../short_test_optimizer_comparison/README.md).
That study supersedes the unfinished ten-AdamW replication below. Use its
calibration and confirmation commands for the current comparison. The original
pilot and partial replication remain separate measurements. The
MNIST/FashionMNIST studies are separate as well.

## Measured pilot

Seed 1337 reached 20,000 updates with mean WeightWatcher alpha 2.068395 across
18 unique weight matrices; 14 were below 2. Full train/validation/test
cross-entropy was 1.420652 / 1.565638 / 1.659099 nats per character.
Training updates took 1,817.209 seconds; combined process time including setup
and evaluations was 1,988.448 seconds, excluding the pause between stages.
This was one CPU thread on an AMD EPYC 9V74 host alongside other training jobs.
The run hit its cap without the declared validation plateau. Runtime and
convergence are separate measurements.

`pilot_result/` contains the original measurements. `historical/` preserves
the three exact scripts used to produce them. `vendor/model.py` is unchanged
nanoGPT commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`; its MIT license is
included. Model SHA256:
`7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c`.

## Historical ten-AdamW seed runner (superseded)

These commands reproduce the earlier request, not the current five-versus-five
comparison. Do not start this batch alongside the replacement study.

From this directory, in the recorded Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
python download_data.py --output input.txt
python test_contract.py --data input.txt
python run_seeds.py --data input.txt --output outputs/ten_seeds
```

Default seeds are 1338 through 1347; the discovery pilot 1337 is excluded.
Runs execute sequentially with one CPU thread. Allow roughly 5–6 hours from
the measured pilot pace; throughput depends on contention and stopping steps.
Reissue the same command after interruption. The batch and each run have flock
locks; completed seeds are skipped. `last.pt` is saved atomically every 250
steps with model, AdamW moments, RNG states, selection state, and accumulated
timings. Frozen source/data/config hashes reject accidental changes on resume.
An infrastructure failure is retryable, not a completed scientific result.

The architecture has 4 blocks, 4 heads, width 128, context 64, 65 characters,
804,096 trainable parameters, no bias and no dropout. Batch size is 12. AdamW
uses betas (0.9, 0.99), matrix decay 0.1, and gradient clipping 1.0. There are
100 warmup steps followed by cosine decay to step 2,000, then LR 1e-4.
Full validation is measured every 500 steps from step 2,000. Three checks
without a cumulative 0.005-nat gain halve LR to 5e-5. Another such plateau
stops training. The cap is 20,000 steps and does not certify convergence.

Train/validation/test use contiguous 90/5/5 partitions of Tiny Shakespeare.
Full evaluations score every next-character target once, resetting context
every 64 characters. The entire corpus defines the fixed 65-character
vocabulary. Text from the held-out partitions never enters gradient updates.
The test partition was already inspected in the discovery pilot. The new
batch freezes all ten seeds before training, defers test evaluation until
after each run's selection is fixed, and permits no test-guided retuning.

Both the stopping endpoint and minimum-validation checkpoint are reported.
Raw default WeightWatcher 0.7.7 fits all 18 unique matrices independently,
keeping QKV packed. The tied token embedding/output head is included once;
positional embeddings are included; LayerNorm vectors have no matrix spectrum.
Failed fits are explicit; a full-model mean requires all 18 successful fits.
Small matrix sizes and tail fits limit inference. Alpha never selects models.

`results.json`, `Results.txt`, and `Seed_Results.png` are refreshed after each
seed. Standard deviations describe seed variation, not dataset uncertainty.
Individual seed points have no invented error bars. Ten seeds can show
stability, but do not establish an optimizer-specific effect or a calibrated
probability that a model is poorly trained.

## Reproduce the original two-process pilot

Create a working directory containing `historical/run_pilot.py`,
`historical/continue_to_plateau.py`, and `historical/check_alpha.py` at its root,
copy `vendor/model.py` to `nanoGPT/model.py`, and place the verified input at
`nanoGPT/data/shakespeare_char/input.txt`. Then run the three scripts in that
order. These historical scripts record their own hashes and retain seed 1337.
They require Linux cgroup files, as did the original environment. The portable
ten-seed runner combines both stages, defers the intermediate test evaluation,
and accumulates timing across resumed sessions; the training recipe is the same.

No model weights, credentials, datasets, or unrelated private experiment
sources are checked into this folder.
