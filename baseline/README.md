# RG optimizer baselines

This directory contains the **unmodified reference experiments** used to test
RG-motivated optimizer extensions. No trace-log projection, ECS correction,
WW-PGD retraction, spectral-flow subtraction, or other RG intervention is
applied in these controls.

Three documents define the current baseline version:

- [`BASELINE_RECIPE_AUDIT.md`](BASELINE_RECIPE_AUDIT.md): data, model,
  initialization, optimizer, and schedule rationale;
- [`BASELINE_EXECUTION_REVIEW.md`](BASELINE_EXECUTION_REVIEW.md): notebook,
  restart, RNG, checkpoint-selection, statistical, and executable-test audit;
- [`FINAL_BASELINE_QUALIFICATION.md`](FINAL_BASELINE_QUALIFICATION.md): bounded
  validation-only search and configuration-freezing protocol required before a
  source-backed candidate is called the best baseline for its exact model/data.

## Baseline suite

| Baseline | Dataset / corpus | Reference model | Optimizer controls | Main entry point |
|---|---|---|---|---|
| **MNIST / MLP3** | Fixed 55k/5k split of official MNIST training data; official test monitoring-only | `784 -> 512 -> 512 -> 10` MLP | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`notebooks/MNIST_MLP3_Baseline_Comparison.ipynb`](notebooks/MNIST_MLP3_Baseline_Comparison.ipynb) |
| **CIFAR-10 / small ViT** | Fixed 45k/5k split of official CIFAR-10 training data; official test monitoring-only | 4x4 patches, width 192, 6 blocks, 3 heads | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb`](notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb) |
| **One-head nanoGPT / FineWeb-Edu** | Pinned FineWeb-Edu `sample-10BT`, document-disjoint 10M/1M/1M GPT-2-BPE splits | 1 block, 1 attention head, width 128, context 256 | SGD + Nesterov, AdamW, Muon + auxiliary AdamW | [`nanogpt_one_head/README.md`](nanogpt_one_head/README.md) |
| **nanochat d12** | Native pinned nanochat data/tokenizer pipeline | 12 layers, width 768, context 2048 | Native nanochat Muon + AdamW recipe | [`notebooks/NanoChat_D12_Reference_Baseline.ipynb`](notebooks/NanoChat_D12_Reference_Baseline.ipynb) |
| **nanochat mac_d4** | Separately cached reduced nanochat preparation | 4 layers, width 256, context 512 | Same pinned upstream Muon + AdamW logic | Same notebook; selected automatically on MPS/CPU |

The d12 and mac_d4 nanochat profiles are distinct baseline versions. A mac_d4
result must never be described as a d12 result.

## Scientific contract

- The unit of replication is a complete training run.
- Optimizer arms share architecture, data identities, seeds, evaluation probes,
  and training budget.
- Optimizer-specific learning rates and schedules are allowed because SGD,
  AdamW, and Muon have different update geometries.
- Validation data select hyperparameters and best checkpoints.
- Protected test measurements never change optimization, schedules, stopping,
  or checkpoint selection.
- WeightWatcher metrics come directly from
  `watcher.analyze(ERG=True, randomize=True, ...)`; no fallback alpha, proxy
  trap count, or fabricated ERG gap is allowed.
- Three-seed uncertainty uses two-sided 95% Student-t intervals across complete
  runs. Layers and fit points are repeated measurements, not extra replicates.
- “Best” means the validation winner in the preregistered source-backed
  neighborhood for the exact architecture, data identity, budget, optimizer,
  and precision policy. It does not mean an unbounded global search has been
  mathematically solved.

The candidate grids, validation ranking, and lock-file writer are implemented
in [`rg_baselines/qualification.py`](rg_baselines/qualification.py).

## Persistent output roots

MNIST, ViT, and nanochat use `RG_BASELINE_RUN_ROOT`. MNIST/CIFAR data use
`RG_BASELINE_DATA_DIR`. The isolated one-head nanoGPT suite uses
`RG_NANOGPT_ONE_HEAD_ROOT`. Put long-running outputs under `$HOME`, not `/tmp`.

## 1. MNIST / MLP3

Run the three optimizer notebooks and then the comparison notebook:

1. [`MNIST_MLP3_SGD_Momentum_Baseline.ipynb`](notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb)
2. [`MNIST_MLP3_AdamW_Baseline.ipynb`](notebooks/MNIST_MLP3_AdamW_Baseline.ipynb)
3. [`MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb`](notebooks/MNIST_MLP3_SGD_Momentum_Muon_Baseline.ipynb)
4. [`MNIST_MLP3_Baseline_Comparison.ipynb`](notebooks/MNIST_MLP3_Baseline_Comparison.ipynb)

All profiles use 30 epochs, batch size 128, three seeds, gradient clipping, a
fixed 55k/5k optimization/validation split, and WeightWatcher at epoch zero and
every epoch. Warm-up and cosine decay are applied **before every optimizer
step**, not once per epoch.

| Optimizer | Peak LR | Floor | Warm-up | Other settings |
|---|---:|---:|---:|---|
| SGD + Nesterov | 0.05 | 5e-4 | 2 epochs | momentum 0.90, matrix WD 1e-4 |
| AdamW | 1e-3 | 1e-5 | 1 epoch | betas (0.90, 0.999), matrix WD 1e-2 |
| Muon matrices | 0.02 | 0.002 | 2 epochs | momentum 0.95, Nesterov, 5 NS steps, WD 0.01 |
| Auxiliary AdamW | 3e-4 | 3e-5 | 2 epochs | betas (0.90, 0.95), matrix WD 0.01 |

Every seed writes latest, validation-selected best, final, and per-epoch
checkpoints with model, optimizer, data-generator, Python/NumPy/Torch/CUDA/MPS
RNG state and a protocol fingerprint. Randomized WeightWatcher diagnostics
preserve the training RNG stream. The historical directory key
`sgd_momentum_muon` remains for compatibility; the implementation is Muon plus
auxiliary AdamW.

## 2. CIFAR-10 / small ViT

Run [`CIFAR10_ViT_Optimizer_Baselines.ipynb`](notebooks/CIFAR10_ViT_Optimizer_Baselines.ipynb).
The notebook imports the final public runtime from `rg_baselines.vit_final`.

The reference uses 300 epochs, dropout 0, stochastic depth 0.10, RandAugment,
color jitter, random erasing 0.25, mixup 0.8, CutMix 1.0, label smoothing 0.1,
and gradient clipping. The final model/schedule contract additionally uses:

```text
LayerNorm epsilon:             1e-6
patch projection:              Conv2d fan-in initialization
warm-up:                       explicit low starting LR
main schedule:                 cosine decay
cooldown:                      final 10 epochs at non-zero LR floor
checkpoint selection:          minimum validation loss
```

| Optimizer | Warm-up start | Peak LR | Floor | Warm-up |
|---|---:|---:|---:|---:|
| SGD + Nesterov | 1e-3 | 0.10 | 0.001 | 5 epochs |
| AdamW | 1e-6 | 1.25e-4 | 1e-5 | 5 epochs |
| Muon matrices | 2e-4 | 0.02 | 0.002 | 5 epochs |
| Muon auxiliary AdamW | 3e-6 | 3e-4 | 3e-5 | 5 epochs |

The hardened runtime saves/restores accelerator RNG state, isolates randomized
WeightWatcher diagnostics, skips completed compatible jobs, and writes explicit
final versus validation-selected test summaries. Layer uncertainty is computed
separately for every physical matrix from exactly three runs; transformer
blocks are never pooled as independent replicates.

## 3. One-head nanoGPT / FineWeb-Edu

See [`nanogpt_one_head/README.md`](nanogpt_one_head/README.md).

```text
train:                         10,000,000 tokens
validation:                     1,000,000 tokens
test:                           1,000,000 tokens
tokenizer:                     GPT-2 BPE
model:                         1 block, 1 head, width 128, context 256
reference horizon:             8 passes / approximately 80M tokens
validation probe:              64 common fixed batches
held-out continuation BLEU:    64 common fixed examples
```

The eight-pass horizon is approximately the 12-tokens-per-scaling-parameter
regime for the tied vocabulary head plus the six hidden matrices. Evaluation
probe identities are fixed across optimizer arms and training seeds. The exact
document-disjoint corpus files are verified by dataset identity, pinned
revision, byte count, and SHA-256 before reuse.

The suite uses update-level warm-up/cosine schedules, logs the LR that actually
produced each checkpoint, validation-selected checkpoints, MPS-safe restart
state, next-token loss/accuracy/perplexity, held-out continuation BLEU, and
direct six-matrix WeightWatcher diagnostics. Randomized WeightWatcher analysis
restores CPU, CUDA, and MPS RNG state.

## 4. nanochat

Run [`NanoChat_D12_Reference_Baseline.ipynb`](notebooks/NanoChat_D12_Reference_Baseline.ipynb).

The wrapper pins upstream commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. It installs two exact,
pinned-source runtime patches: the hard-coded global seed becomes configurable,
and `torch.compile` becomes environment-controlled. CUDA d12 retains the native
compiled path; MPS/CPU runs the identical model and optimizer in eager mode.
`RG_NANOCHAT_PROFILE=auto` selects d12 on CUDA and mac_d4 on MPS/CPU. Override
explicitly with `d12` or `mac`.

The wrapper creates the correct platform environment, forces one process on
MPS/CPU, resumes only from checkpoints with model/metadata/all optimizer shards,
appends logs, fingerprints profile/process/device/compile policy, and analyzes
only the six principal hidden matrices in every block. d12 and mac_d4 use
separate caches and result roots. A mismatched `runtime_policy.json` fails rather
than silently reusing incompatible results.

The canonical d12 recipe remains native upstream because nanochat itself derives
the token horizon, total batch size, LR scaling, weight-decay scaling, warm-up,
warmdown, and Muon momentum schedule from the model size. The mac_d4 profile is
separate development evidence and requires its own validation qualification.

## Automated tests

```bash
cd baseline
PYTHONPATH=. python -m unittest discover -s tests -v

cd nanogpt_one_head
PYTHONPATH=src pytest -q
```

The same suites run in `.github/workflows/baseline-tests.yml`, alongside source
and notebook syntax checks. The full long-horizon campaigns remain target-
hardware experiments and are not replaced by CI smoke tests.
