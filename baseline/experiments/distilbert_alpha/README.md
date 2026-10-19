# DistilBERT: alpha, generalization, and optimizer experiments

An isolated experiment in `rg_optimizers`, starting with **one full fine-tuning run
on one existing eight-device v5e TPU**. The first run establishes the training,
checkpoint/restart, accuracy, and spectral measurement path. Its default launch
does not start the future sweep.

## First run

| Setting | Value |
|---|---|
| Model | Pretrained `distilbert/distilbert-base-uncased`, full fine-tuning |
| Model revision | `12040accade4e8a0f71eabdb258fecc2e7e948be` |
| Dataset | `stanfordnlp/imdb`, revision `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` |
| Split | 22,500 optimization / 2,500 validation examples from the official training partition; stratified, seed 1701 |
| Test set | Neither downloaded nor evaluated |
| Optimizer | AdamW, peak LR 2e-5, weight decay 0.01, betas 0.9/0.999, epsilon 1e-8 |
| Dropout | Hidden 0.1; attention 0.1; classifier 0.2 |
| Schedule | 10% linear warmup, then linear decay; three full epochs |
| Sequence length | 128, fixed padding and truncation |
| Batch | 16 per device × 8 = 128 global; masked final batch retains every example |
| Precision | bfloat16 forward/backward autocast, FP32 parameters and optimizer states |
| Seed | 401 |
| Gradient clipping | Global norm 1.0, after averaging gradients across all eight devices |
| Measurements | Initialization and each epoch: frozen-model training and validation loss, accuracy, errors, gap, and raw matrix spectra |
| Checkpoints | Every 50 updates, epoch boundaries, and cooperative budget stop; retain last two full states |
| Budget | Default 5.5 hours, maximum 6; automatically shortened to the checked existing lease minus five minutes |

With 22,500 examples and this batch size there are 176 updates per epoch, 528
total. The first **two real updates** form the TPU preflight. The trainer exits,
saves all eight RNG states, and the supervisor resumes the same experiment from
update 2. These updates are included in the 528, not a separate trial.

## Launch from Cloud Shell

Use a clean checkout of the committed source. The launcher needs the existing
Google Cloud login and SSH access; it never creates a TPU, extends a lease, or
stops another training job.

```bash
git clone https://github.com/CalculatedContent/rg_optimizers.git rg-distilbert
cd rg-distilbert
python3 baseline/experiments/distilbert_alpha/cloudshell.py start \
  --zone us-west4-a --hours 5.5
```

`start` verifies the active `v5litepod-8` lease, guest identity, mounted drive,
free disk space, existing RG services, and device occupancy. It fetches the exact
checkout commit to the TPU and starts a bounded systemd service. The existing
TPU environment is preserved; HF/WW dependencies are installed into a private
venv inheriting its torch/torch_xla/libtpu stack. Version equality is checked.
It discovers the sole eligible existing allocation in the zone. If several
exist, choose the intended one explicitly with `--node NODE`; it never guesses
between multiple allocations or assumes an older node is still active.

Status:

```bash
python3 baseline/experiments/distilbert_alpha/cloudshell.py status \
  --zone us-west4-a
```

The initial run is verified on TPU when `preflight.json` reports two saved
updates, `runtime.json` reports `world: 8` and `resumed_step: 2`, and `status.json`
continues advancing beyond update 2. `RESULT.json` distinguishes a completed run
from a partial saved run. Read `ERROR.json` if the service failed.

## Persistent storage and recovery

Everything is written beneath the **existing mounted** drive:

```
/mnt/disks/rg-data/distilbert_alpha/
```

The launcher refuses a plain directory at the mount point. This experiment does
not format, mount, unmount, delete, or resize the drive. Keep the persistent disk
when removing the TPU VM.

Each `initial-TIMESTAMP/` contains source, selected configuration, package versions,
launch/lease provenance, logs, run results, and a prospective sweep plan. Prepared
data, tokenizer and pretrained model have pinned upstream revisions and SHA256
manifests; `data_path.json` records the shared prepared-data directory. No test
parquet file is fetched.

Within `initial/`:

- `status.json`: current step, epoch, LR, and completion/pause state.
- `checkpoints/step_*/`: model, optimizer, cursor, per-rank Python/NumPy/torch/XLA RNG
  states, and SHA256 manifest. `latest_checkpoint.json` is published only after
  every rank has saved. The last two committed generations are retained.
- `measurements/`: training/validation measurements, exact example counts, and
  generalization gap in percentage points.
- `spectral_snapshots/`: immutable raw dense-matrix weights at measured steps.
- `spectra/`: per-layer WeightWatcher results and aggregate alphas joined to the
  matching accuracy measurements.

Resume after an interruption from the **same source commit**, on an idle active
eight-device TPU, using the previous run path:

```bash
python3 baseline/experiments/distilbert_alpha/cloudshell.py start \
  --resume-run /mnt/disks/rg-data/distilbert_alpha/initial-TIMESTAMP/initial \
  --hours 5.5
```

Resume enforces identical data, configuration, source and device count, verifies
the checkpoint hashes, and restores the next minibatch and all RNG states. A
hard process kill may replay work since the last committed checkpoint. The
cooperative training deadline leaves ten minutes for saving and spectral work;
the service has a separate hard time limit. No automatic restart or new billing
allocation is requested.

## Spectral diagnostic

WeightWatcher **0.7.7**, default `analyze()`, runs in a separate CPU process on the
**raw, unnormalized** saved matrices. For full DistilBERT, the primary aggregate
is the unweighted arithmetic mean of 37 dense-matrix alphas: Q/K/V/output and
two FFN matrices in each of six transformer blocks, plus the pre-classifier.
Embeddings are outside this declared aggregate. The binary classifier has only
two singular values and is reported as excluded from power-law fitting.

Every eligible layer's full fit record and validity are retained. The aggregate
is null if any eligible fit is invalid; missing layers cannot silently change
the average. The fraction of eligible layers with alpha < 2 is also reported.

Use the matched checkpoints to study whether alpha < 2 accompanies worsening
validation loss/accuracy, increasing train–validation gap, and continued
improvement in training performance. A low-alpha snapshot and a widening gap
are distinct recorded measurements. This enables a prospective diagnostic
assessment as replicated runs accumulate. The initial seed is a pilot, and
the validation partition remains the selection source.

## Configurable variations

`sweep.py` freezes **234 full-run jobs: 78 unique settings × three seeds**. This
is a matched one-factor-at-a-time design, with duplicates removed and family
memberships retained. It is a prospective plan, not a promise to finish all
234 jobs within the first allocation.

| Variable | Recorded variations |
|---|---|
| Optimizer | AdamW, Adam, SGD |
| LR | 0.25, 0.5, 1, 2, 4 × center; Adam/AdamW center 2e-5, SGD center 0.01 |
| Weight decay | 0, 1e-4, 1e-3, 0.01, 0.1 |
| Hidden dropout | 0, 0.1, 0.2, 0.3, other dropout types fixed |
| Attention dropout | 0, 0.1, 0.2, 0.3, other dropout types fixed |
| Classifier dropout | 0, 0.1, 0.2, 0.3, other dropout types fixed |
| SGD momentum | 0, 0.5, 0.9, 0.95; separate Nesterov setting at 0.9 |
| Adam beta1 | 0.8, 0.9, 0.95 |
| Adam beta2 | 0.95, 0.99, 0.999 |
| Warmup | 0%, 5%, 10% |
| Schedule | Constant after warmup, linear decay, cosine decay |
| Seeds | 401, 402, 403 |

AdamW uses decoupled decay. Adam and SGD use coupled L2, with zero decay at their
centers. Biases and LayerNorm vectors are excluded from decay throughout. TPU
Adam/AdamW use PyTorch's capturable single-tensor path and tensor learning rates
to avoid step-dependent scalar graph constants. TPU SGD implements the reference
SGD/momentum/Nesterov arithmetic with a tensor LR; local parity tests compare it
against PyTorch SGD. These are baseline optimizers; no spectral feedback changes
the learning rate in this pilot.

Create the plan outside the checkout to keep the launch source clean:

```bash
python3 baseline/experiments/distilbert_alpha/sweep.py --output /tmp/distilbert-plan.json
```

After the pilot, `cloudshell.py start --job-id ID` runs exactly one selected job
from that plan under the same time cap. All numerical fields are also available
through `Config` and `train.py --config FILE`; use separate result directories.

## Validation completed before TPU launch

Run local engineering tests with the pinned HF dependencies and a CPU torch:

```bash
cd baseline/experiments/distilbert_alpha
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest test_experiment -v
```

Tests check exact-once padded coverage across eight shards, the stratified split,
decay exclusions, finite sweep/lease bounds, tensor-SGD reference parity, and
interrupted versus uninterrupted tiny DistilBERT training under all three
optimizers. Resume matches model, optimizer, RNG and metrics exactly on CPU;
corrupted checkpoint state is refused. A separate tiny-model spectral end-to-end
check exercises saving, default WW fits and joined metrics.

These CPU checks qualify the implementation path. The first live TPU launch is
the hardware qualification and timing measurement; no TPU throughput or IMDb
accuracy is claimed before that run succeeds.
