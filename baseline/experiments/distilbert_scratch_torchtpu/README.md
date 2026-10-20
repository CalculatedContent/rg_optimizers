# DistilBERT from scratch with native Google TorchTPU

This independent folder trains **one randomly initialized DistilBERT architecture**
using masked-language modeling (MLM) on the existing eight-device v5e TPU. It uses
[`google-pytorch/torch_tpu`](https://github.com/google-pytorch/torch_tpu),
`device='tpu'`, `tpu_dist`, and `torch.compile`. It has no `torch_xla` imports and
does not reuse the earlier pretrained IMDb fine-tuning training code.

This is a first pretraining pilot. There is **no teacher, pretrained model, or
distillation objective**. Only the existing uncased WordPiece tokenizer is reused.
The goal is to establish a reproducible training/measurement path, then assess
convergence before launching optimizer/seed comparisons. A below-two fit in one
run does not establish optimizer bias or overfitting.

## Baseline

| Setting | Value |
|---|---|
| Architecture | 6 blocks, 12 heads, width 768, FFN 3072, GELU, stock biases/LayerNorm/residuals |
| Parameters | 66,985,530, including tied MLM vocabulary projection |
| Initialization | Random Hugging Face `DistilBertForMaskedLM(DistilBertConfig(...))` |
| Objective | MLM only; 15% candidate selection; selected tokens use 80% mask / 10% random / 10% unchanged |
| Corpus | `Salesforce/wikitext`, `wikitext-103-raw-v1`; official train/validation/test partitions |
| Data provenance | Resolve dataset revision once; save SHA, packed-file SHA256s, row counts, tokenizer revision |
| Seed | Model/dropout 401; independent shared data/masking seed 1701 |
| Optimizer | AdamW; LR 1e-4; decay 0.01; betas .9/.999; epsilon 1e-8 |
| Schedule | 1,000 warmup updates, then linear decay over a fixed 10,000-update horizon |
| Batch | 32 per device × 8 = 256 sequences; length 128 |
| Token budget | At most 327,680,000 sequence positions; exact full-corpus passes depend on prepared row count |
| Precision | BF16 autocast; FP32 parameters and optimizer moments; TPU matmul precision `medium` |
| Regularization | Hidden and attention dropout .1; gradient norm clipping 1.0 after DDP reduction |
| Checkpoints | Every 100 updates and cooperative exit; retain two committed generations |
| Measurements | Initialization, update 2 preflight, every 500 updates, and final cursor if budget permits |
| First launch budget | 5.5 hours including setup; capped by checked existing lease; resumable if incomplete |

LR 1e-4 is a conservative **pretraining pilot setting**, not the earlier 2e-5
fine-tuning LR, nor a tuned optimum. The 10,000-update target is a pilot budget,
not a claim of convergence or equivalence to the original DistilBERT recipe.
Runtime is measured by the actual TPU preflight. No overnight completion estimate
is asserted before that measurement.

## Launch from the existing authenticated Cloud Shell

Use a clean checkout of the experiment branch (or its exact commit):

```bash
git clone --branch codex/distilbert-scratch-torchtpu --single-branch \
  https://github.com/CalculatedContent/rg_optimizers.git rg-distilbert-torchtpu
cd rg-distilbert-torchtpu
python3 baseline/experiments/distilbert_scratch_torchtpu/cloudshell.py start \
  --zone us-west4-a --hours 5.5
```

Project is `tpu-builders-504820`. The launcher discovers the sole eligible
**existing** READY/ACTIVE `v5litepod-8` allocation. If there are multiple eligible
allocations, supply `--node NAME`. It checks the current lease, matching guest IP,
device occupancy, RG services, mounted persistent disk, and 30 GiB free space.
It never provisions a TPU, extends a lease, stops another job, or changes an older
environment. The existing VM must have Python 3.12+ with venv support. If missing,
install Python 3.12 and `python3.12-venv` on that VM before launch.

The bounded systemd service creates its own venv under
`/mnt/disks/rg-data/distilbert_scratch_torchtpu/initial-TIMESTAMP/`, installs pinned
TorchTPU 0.2.0 and HF/WW dependencies, and freezes the entire resolved environment
to `requirements.lock.txt`. Torch/libtpu transitive versions are resolved at the
first install and recorded; they are **not predeclared as hardware-qualified**.
The original venv is reused for resume, which also checks package versions.
TorchTPU supplies the required Torch/libtpu stack; the old XLA venv is not inherited.

The job performs:

1. All-eight-rank native TPU matmul and all-reduce probe.
2. Real WikiText-103 token preparation with immutable hashes.
3. Initialization evaluation, two real full-model training updates, and checkpoint.
4. Process exit and resume from update 2, continuing the same model to its fixed horizon.
5. CPU WeightWatcher analysis on saved weights alongside subsequent training.

Status:

```bash
python3 baseline/experiments/distilbert_scratch_torchtpu/cloudshell.py status \
  --zone us-west4-a
```

Check `probe/probe.json` for eight successful ranks, `preflight.json` for two
saved real updates, `initial/runtime.json` for native TPU and `resumed_step: 2`,
and `initial/status.json` for advancing steps. `RESULT.json` distinguishes a
completed target from a partial saved run. `ERROR.json` and `run.log` explain
failures. A CPU unit test is never recorded as TPU success.

Resume with the same source commit and idle existing TPU:

```bash
python3 baseline/experiments/distilbert_scratch_torchtpu/cloudshell.py start \
  --resume-run /mnt/disks/rg-data/distilbert_scratch_torchtpu/initial-TIMESTAMP/initial \
  --hours 5.5
```

Checkpoints include model, optimizer, update cursor, every rank's CPU/TPU RNG
states, manifest hashes, configuration, data identity, package versions, and source
hash. The LR horizon stays fixed across pauses. Interrupted writes remain
uncommitted; the most recent complete checkpoint is the resume point. A hard kill
can lose work since the last 100-update checkpoint. Exact restart is tested on
CPU; TPU determinism and kernel coverage require the live hardware run.

## Measurements and scientific interpretation

The primary spectral set is exactly **36 encoder matrices**: attention Q/K/V/output
and both FFN weights in each of six blocks. Embeddings, the tied vocabulary output,
and the MLM prediction transform are excluded from that declared aggregate.
Snapshots are raw unnormalized FP32 weights. WeightWatcher 0.7.7's default
`analyze()` fits each matrix independently, matching the prior RG experiment's
estimator. There is no spectral feedback or alpha-dependent learning rate.

Saved columns include alpha, KS distance D, sigma, xmin/xmax, tail count/fraction,
fit status and warnings. Model summaries report mean/min alpha and the fraction
of layers below two. Any invalid or missing expected fit makes the primary mean
null; it never silently changes the layer set. D and tail sizes must accompany
interpretation, especially for untrained or poorly power-law-like spectra.

At every measurement the frozen model evaluates a fixed 2,048-sequence training
subset and the complete held-out validation/test partitions. **These accuracies
are masked-token accuracies, not IMDb/GLUE classification accuracy.** The same
per-example evaluation masks are used at every checkpoint and across seeds.
Training masks change with the update; data order/masking seed is separate from
model initialization. Evaluation restores training RNG state. Test metrics are
monitoring-only and never control configuration, stopping, updates, or selection.

Packing occurs separately inside each official split. Each full block has CLS/SEP
boundary tokens; line boundaries receive SEP. Incomplete tails are dropped and
counted. Special tokens are never prediction targets. Rare zero-target sequences
receive one random eligible target; this nonempty-target safeguard is recorded
in the protocol. Epoch-final batches use loss-masked duplicate padding so every
real sequence contributes exactly once per pass. DDP scales by the global number
of targets, not by a mean of unequal per-worker means.

Outputs in `initial/`:

- `status.json`, `runtime.json`, `identity.json`, `RESULT.json`.
- `checkpoints/`, `measurements/`, `spectral_snapshots/`, `spectra/`.
- `checkpoint_metrics.csv`: alpha summaries joined by step to loss/accuracy.
- `layer_metrics.csv`: every layer's fit joined to the same checkpoint metrics.

Analyze remaining snapshots after training without using TPU devices:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /path/to/run/venv/bin/python \
  baseline/experiments/distilbert_scratch_torchtpu/spectra.py --root /path/to/run/initial
```

## Next phase, after the pilot

The config already exposes AdamW, Adam, SGD/momentum, LR, decay, dropout and seed.
No sweep starts automatically. First assess validation-loss progress, throughput,
fit quality, and whether the training budget reaches a useful regime. Then freeze
the corpus, tokenizer, masking/data schedule, architecture, and token budget;
calibrate optimizer-specific LR ranges; compare seeds 401/402/403 at matched
compute and at comparable validation quality. Preserve per-layer below-two
indicators separately from mean-alpha-below-two. Add Muon only after a separate
implementation/parity check on the new backend.

DistilBERT's stock architecture includes biases, residual additions, softmax and
LayerNorm. It does not enforce the scale invariance of the earlier MLP controls.
This pilot measures spectral trajectories; it cannot by itself test a universal
SGD alpha=2 prediction or attribute alpha<2 causally to AdamW.

## Engineering checks and upstream references

```bash
cd baseline/experiments/distilbert_scratch_torchtpu
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest test_experiment -v
```

Checks cover the full architecture/random initialization path, eight-way exact
sample coverage, fixed masking, invalid-fit handling, tiny-model interrupted
versus uninterrupted AdamW states/metrics, corruption rejection, and in-process eight-shard gradients versus a serial masked-token mean. An
additional opt-in two-process Gloo integration check compares actual DDP against
serial updates, including an uneven final batch; run it with
`RG_RUN_DISTRIBUTED_TESTS=1` on a host that permits Gloo sockets. See `VALIDATION.json` for actual results and limitations.

Implementation references inspected on 2026-10-20:

- [TorchTPU installation](https://github.com/google-pytorch/torch_tpu#installation)
- [Native distributed setup](https://github.com/google-pytorch/torch_tpu/blob/main/docs/distributed_support.md)
- [Supported hardware/features](https://github.com/google-pytorch/torch_tpu/blob/main/docs/feature_set.md)
- [Precision](https://github.com/google-pytorch/torch_tpu/blob/main/docs/numerical_precision.md)
- [Upstream AdamW tests](https://github.com/google-pytorch/torch_tpu/blob/main/tests/adamw_test.py)
- [DistilBERT implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/distilbert/modeling_distilbert.py)

The released 0.2.0 wheel was also inspected directly: its metadata requires
Python >=3.12 (stricter than the current main README), torch >=2.11.0 and <=2.15.0,
and libtpu >=0.0.50. Its Python launch helper sets topology and device addresses;
the launcher derives the eight-device count from those addresses before torchrun.

The inspected TorchTPU main commit was
`905a0dfde04a01df8ee2c100ffd656694f38df84`; installation pins the public 0.2.0 wheel.
