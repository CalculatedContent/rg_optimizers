# Sources and limits of the proposed SGDM search

This is a source-informed experiment proposal. The sources below do not establish
that any proposed plain-SGDM setting reaches stock GPT-2/FineWeb validation
NLL ≤ 3.28. Distinguish the published/released facts from the transfer decisions
made for this experiment.

## Benchmark and data identity

- [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt):
  upstream speedrun records and the historical GPT-2/FineWeb benchmark. Its modern
  records change architecture and training protocol; they must not be treated as
  identical stock GPT-2 SGD experiments.
- [karpathy/llm.c, pinned source commit](https://github.com/karpathy/llm.c/tree/7ecd8906afe6ed7a2b2cdb731c042f26d525b820):
  reference lineage of the locally vendored stock GPT-2 model. The local source
  bytes are additionally checked by SHA256. The model source identity is not a
  claim that the TPU runtime or token-loader behavior is identical to a C/CUDA run.
- [Pretokenized FineWeb cache, pinned revision](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/tree/889765ea1f903759787add96995d81171b632d0c):
  the existing repository data manifest pins every consumed shard by SHA256.
  The separate tuning window is a new experimental design choice within that
  validation shard; it is not a second published benchmark.

The fixed full training budget is 19,560 × 524,288 = 10,255,073,280 tokens, and the
official evaluation uses 10,485,760 targets from the validation prefix. The old
SGDM run's 3.615982437133789 endpoint is an internal historical result, not a
published target-reaching recipe. The control configuration retains its main
hyperparameters but is explicitly separated from later alpha-controlled variants.

## Released ordinary-SGD grid: orientino/gap

Primary code, pinned commit
`cf069543dbcaeac5d319d9479f09224725038c71`:

- [run_fineweb.sh](https://github.com/orientino/gap/blob/cf069543dbcaeac5d319d9479f09224725038c71/run_fineweb.sh)
- [language/train.py](https://github.com/orientino/gap/blob/cf069543dbcaeac5d319d9479f09224725038c71/language/train.py)
- [language/model.py](https://github.com/orientino/gap/blob/cf069543dbcaeac5d319d9479f09224725038c71/language/model.py)
- [language/prepare_fineweb.py](https://github.com/orientino/gap/blob/cf069543dbcaeac5d319d9479f09224725038c71/language/prepare_fineweb.py)
- [Authors' paper](https://rustem-islamov.github.io/files/publications/AdamSGD/adam_sgd.pdf)

The example script releases SGD LR grid `{0.01,0.03,0.1,0.3,1.0}` and momentum
grid `{0.9,0.95,0.99,0.999}`, with weight decay 0. The training code constructs
ordinary PyTorch SGD with the specified LR and momentum; it does not enable
Nesterov or nonzero dampening. Its default schedule uses 10% warmup and cosine
decay, with gradient clipping at 1.

Transfer limits are substantial. The main released model is approximately 49M
parameters, with six layers, width 384, RMSNorm, QK normalization, RoPE, squared
ReLU and an untied output head. Data preparation takes the first 1B tokens from
FineWeb-Edu with GPT-2 tokenization and a 99/1 split. The example script uses
batch 1, sequence length 256 and accumulation 1. Four parallel GPU processes in
that script are independent trials, not a four-device distributed batch.
The script is an example grid rather than a manifest of every paper result.

The repository's nonzero weight decay is manually decoupled after the optimizer
step and scaled by the schedule factor. That convention is not imported here.
Using its zero-decay setting avoids confusing it with this experiment's
ordinary coupled PyTorch SGD decay. Momentum 0.98, batches 16/64 and stock GPT-2
124M are new transfer choices; the released grid does not prove their result.

## Small-batch evidence and momentum hypothesis

- [Is your batch size the problem? Revisiting the Adam-SGD gap in language
  modeling](https://arxiv.org/html/2506.12543v1), Sreckovic, Geiping and Orvieto:
  the small-batch study considered during planning includes batch 16 sequences,
  context 1,024 and momentum 0.98. It uses a different model/tokenization setup
  (NeoX tokenization), and the language-model dampening convention was not
  verified in this audit. The proposed grid combines its small-batch/momentum
  hypothesis with the unambiguous ordinary-SGD implementation and LR grid from
  `orientino/gap`; this combination is new, not a reproduced published recipe.
- [Small Batch Size Training for Language Models: When Vanilla SGD Works, and
  Why Gradient Accumulation Is Wasteful](https://arxiv.org/html/2507.07101v3),
  with [official code](https://github.com/martin-marek/batch-size): evidence that
  batch size can materially change the SGD/Adam comparison. Its GPT-2-sized
  experiments include vanilla SGD with momentum 0 and a different model/budget
  setup; they are not validation of this momentum 0.98 recipe.

The implementation deliberately states the PyTorch unnormalized momentum
convention. An LR taken from an EMA-normalized momentum implementation cannot be
copied without considering its factor of `1−momentum` and initialization behavior.

## Direct SGDM comparisons that do not establish the target

- [Revisiting the Adam-SGD Gap in LLM Pre-Training: The Role of Large Effective
  Learning Rates](https://arxiv.org/html/2605.17787v1): Table 4 reports GPT-2
  124M/FineWeb at 2.6B training tokens, with validation losses 3.34 for Adam,
  5.00 for standard momentum SGD and 3.60 for SGD-LL. SGD-LL adds specialized
  clipping and uses a different momentum convention. Its LR should not be
  copied into ordinary PyTorch SGD without those differences.
- [Muon with Nesterov Momentum: Heavy-Tailed Noise and (Randomized) Inexact
  Polar Decomposition, v2](https://arxiv.org/html/2605.06884v2): Table 2 reports
  validation perplexity 71.0208 ± 1.8195 for its all-parameter SGD-Nesterov baseline.
  Appendix E.2 describes tuned LR about 4.18e−4 and momentum about 0.86, a modified
  135M nanoGPT, 1,490 iterations, context 2,048 and five seeds. This does not reach
  NLL 3.28 and is not the stock 124M protocol. Earlier v1 has different settings
  and results; do not mix the versions.

These comparisons support conducting a controlled search. They do not establish
that more tokens or any single momentum/LR choice will close the gap. A measured
full-budget official validation result is the deciding evidence.

## New choices made here

The staged budgets, promotion counts, reserved tuning window, seed 42 pilot,
optional 42/43/44 confirmation, immutable configuration hashes, exact diagnostics
completion gate, continuous cross-shard data stream, and optional longer WSD
fresh-start plan are experimental-design decisions made for this implementation.
They are not attributed to a paper and should be reported as such.
