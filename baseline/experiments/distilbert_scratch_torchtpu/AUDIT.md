# Code and literature audit: BERT-6 MLM from scratch

Reviewed 2026-10-20. The experiment is a randomly initialized, six-layer encoder
with ordinary masked-language-model pretraining. The implementation class is
`DistilBertForMaskedLM`; the model is not distilled. The intended first result is
a reproducible training trajectory on native Google TorchTPU, followed by
validation-based calibration and optimizer comparisons.

## Findings and changes

| Area | Audit result |
|---|---|
| Initialization | Uses `DistilBertForMaskedLM(DistilBertConfig(...))`; model weights are never loaded with `from_pretrained`. Only tokenizer files are reused. |
| Objective | Ordinary token cross-entropy; no teacher logits, distillation loss, cosine teacher loss, or next-sentence objective. |
| Optimizer | Stock PyTorch AdamW with bias correction and decoupled decay; biases and LayerNorm vectors receive no decay. Tied embedding/output weights occur once in optimizer groups. |
| Epsilon | Changed from the PyTorch default 1e-8 to 1e-6, matching Google's original BERT implementation and the academic-budget recipe. |
| Distributed loss | Sum losses, normalize by global masked-target count, compensate for DDP gradient averaging; padded examples contribute no loss or metrics. |
| Evaluation | Fixed per-example masks now use an independent 15% evaluation rate. Previously a training-rate change would also change evaluation difficulty. |
| Model retention | Added full best-validation weights, independent of rotating restart checkpoints. A later regression no longer removes the best observed validation model. |
| Configuration | Added Cloud Shell/supervisor config selection. Sequence preparation now uses the selected sequence length; previously it always prepared length 128. |
| Data identity | Pinned WikiText to `b08601e04326c79dfdd32d625aee71d232d685c3`, verified against the dataset API. Subsequent comparisons use the same upstream revision. |
| Restart | Full model, optimizer and each rank's RNG are preserved with hashes and source/config/data/package/world identity. Old runs must resume their original source/config. |
| Spectra | Exactly 36 encoder matrices; missing/failed fits cannot silently change the primary aggregate. No alpha-dependent updates or model selection. |
| Hardware | Native `torch_tpu==0.2.0`, `device='tpu'`, `tpu_dist`, compiler backend `tpu`. Actual v5e execution is still pending. |

## Second complete code review, 2026-10-21

The review covered every Python module, configuration, dependency declaration
and runbook in this experiment folder. It found additional failures that the
original numerical checks did not exercise:

| Finding | Correction |
|---|---|
| Native RNG API | The released 0.2.0 wheel marks `torch.tpu.get_rng_state` internal on PyTorch >=2.13, with a strict runtime error for user code. Use the supported accelerator getter when available. Add a real RNG replay check to every rank's TPU probe. The setter remains the wheel's public experimental API. |
| Diagnostics ahead of resume state | Scheduled evaluation could publish spectra/best weights before committing the matching training checkpoint. Commit state first, including when measurement/checkpoint intervals differ; recover an interrupted scheduled measurement on resume. |
| Unpublished checkpoint reuse | A crash after directory rename but before pointer publication could leave a checkpoint from an earlier replay. Replace that orphan when the cursor is replayed; retain only published state at an already committed cursor. |
| Checkpoint retention | Partial directories and unpublished generations ahead of the current cursor could displace a valid checkpoint. Filter them before numerically selecting the last two committed checkpoints. |
| Final spectral scan | Training could publish its final snapshot between the worker's scan and completion check. Read the completion marker before scanning, ensuring a final scan observes all published measurements. |
| Process cleanup | Analyst timeout handling could itself time out without killing/reaping the process group. Escalate from SIGTERM to SIGKILL, including leftover children after a failed phase. |
| Status near lease expiry | Status discovery applied the 30-minute launch gate, hiding a still-running job near expiry. Apply lease eligibility only to new launches; keep the launch guard unchanged. |
| Invalid configurations | Reject nonfinite optimizer scalars, fractional sizes, invalid seeds, and non-boolean mode flags before training. |

Regression coverage now includes exact restart for AdamW, Adam and SGD, compiled
BF16 restart with dropout, simulated interrupted measurement publication,
unpublished checkpoint replacement, final-scan races and forced process cleanup.
API-selection tests use mocks and do not substitute for the real TPU probe.
No full-corpus repacking was needed: the preparation code and pinned data are
unchanged from the verified corpus check.

## What the papers support

1. **Sanh et al., DistilBERT (2019).** The original student inherited alternating
   teacher layers and used teacher-derived losses alongside MLM. That explains
   the name. Neither its published quality retention nor its speed comparisons
   establish the results of our independently initialized model.
   [Paper](https://arxiv.org/abs/1910.01108)

2. **Turc et al., Well-Read Students Learn Better (2019).** Direct pretraining and
   fine-tuning of compact BERT models is a competitive baseline; distillation is
   an additional stage, not a requirement for using a compact architecture.
   [Paper](https://arxiv.org/abs/1908.08962)

3. **Devlin et al., BERT (2019).** LR 1e-4, batch 256, betas .9/.999, decay .01,
   dropout .1 and short-sequence training provide a reasonable conservative
   starting point. Google's implementation specifies epsilon 1e-6 and clipping
   norm 1. Its legacy Adam update omits bias correction, unlike our stock PyTorch
   AdamW: this experiment is not a bitwise reproduction of BERT optimization.
   [Paper](https://arxiv.org/abs/1810.04805) ·
   [Optimizer source](https://github.com/google-research/bert/blob/master/optimization.py)

4. **Liu et al., RoBERTa (2019).** Dynamic masking, training without NSP, sufficient
   data and sufficient optimization are established improvements. This supports
   the basic objective, but does not validate our short WikiText pilot as a
   fully pretrained general-purpose encoder.
   [Paper](https://arxiv.org/abs/1907.11692)

5. **Izsak et al., How to Train BERT with an Academic Budget (2021).** Tune LR,
   warmup, batch and schedule for the available computation. Larger encoders can
   outperform smaller ones at equal wall time, so six layers is not proved best
   for that metric. Their larger-model, batch-4096+ results do not justify
   copying a 1e-3 or 2e-3 LR into our batch-256 run. Our initial 3-by-2 search is
   an adaptation, not their published recipe or a measured optimum.
   [Paper](https://arxiv.org/abs/2104.07705)

6. **Wettig et al., Should You Mask 15% in Masked Language Modeling? (2023).**
   Optimal masking depended on model size and task: their aggregate medium,
   base and large results preferred 15%, 20% and 40%, respectively. This supports
   retaining 15% for the first compact-model control and testing alternatives.
   Their experiments also motivate a separate all-MASK corruption comparison;
   the first run keeps standard 80/10/10 so only one recipe is established first.
   [Paper](https://aclanthology.org/2023.eacl-main.217/)

7. **Gisserot-Boukhlef et al., Should We Still Pretrain Encoders with MLM?
   (2026 revision).** Their 210M–1B-model experiments found task-dependent masking
   and advantages for CLM-then-MLM under some fixed-compute conditions. This is
   evidence against a universal best MLM recipe, not a reason to change the
   objective during our initial optimizer-controlled experiment.
   [Paper](https://arxiv.org/abs/2507.00994)

8. **Warner et al., ModernBERT (2024).** The released 149M base model combines
   architecture changes with much larger-scale pretraining and GPU-specific
   efficiency work. Benchmark its native TorchTPU implementation before making
   a speed claim or changing the architecture in an optimizer comparison.
   [Paper](https://arxiv.org/abs/2412.13663)

## Calibration and evidence required

- Start one model with `configs/initial.json`. Its 10,000 updates cover at most
  327,680,000 sequence positions, including special tokens, not that many MLM
  targets. It is a pilot, not a convergence guarantee.
- The six committed configurations cover LR 1e-4/3e-4/5e-4 crossed with
  warmup 300/1,000 updates. The other settings and both seeds are identical.
  Each launch runs one configuration; no automatic sweep is enabled.
- Compare runs only with identical prepared-data hashes, resolved software
  versions, token budget and fixed evaluation masks. Select using validation
  loss, expand the LR range if the best setting lies on a boundary, and check
  the selected recipe on fresh seeds. Compare both matched token budget and
  matched measured wall time when evaluating architectures.
- After establishing LR/warmup, test beta2 .98 versus .999 and masking 15/20/30%
  separately. Retune optimizer-specific LRs for SGD/Adam/AdamW instead of reusing
  one optimizer's best LR for all optimizers. Alpha remains an outcome.
- Decide the longer training horizon from actual TPU throughput and validation
  curves. A pause/resume keeps its existing horizon; extending the horizon is a
  different protocol and must not silently change a run's schedule.
  Pilot-selected settings remain provisional until checked at that longer budget.
- WikiText-103 is a convenient controlled corpus. For broader model quality,
  add a larger, fixed corpus and a declared downstream evaluation. MLM validation
  loss alone does not establish optimal downstream performance.
- A 128-token run has not trained longer absolute position embeddings. Any
  512-token stage needs explicit data preparation, schedule and qualification.
- Checkpoint alpha below two should be interpreted with the fit's tail size,
  KS distance and uncertainty, the layer set, held-out quality and repeat seeds.
  A conventional Transformer is not the scale-invariant MLP control.

See `VALIDATION.json` for the checks actually executed. CPU tests are not evidence
of native TPU kernel coverage, training throughput, convergence or observed alpha.
