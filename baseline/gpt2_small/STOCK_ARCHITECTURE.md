# GPT-2 Small architecture audit

The current `speedrun.py` paired-seed Muon/AdamW experiment and the single-run
`muon_speedrun/cloudshell.py` launcher now use `muon_speedrun/stock_model.py`.
Both optimizer choices instantiate the same model before building the optimizers.

The previous `muon_speedrun/model.py` was **not stock GPT-2**. It had 12 blocks
but only 6 heads, RoPE, RMSNorm/QK normalization, squared ReLU, value residuals,
learned input/U-Net skips, zero output projections, an untied vocabulary head,
logit soft-capping and a padded vocabulary of 50304. Its historical 25k recipe
and old checkpoints remain identifiable and reproducible. The old one-head
experiments are also historical small-model studies, not GPT-2 Small.

## Current shared architecture

| Property | Value |
|---|---|
| Transformer blocks | 12 independent blocks, indices 0–11 |
| Attention heads per block | 12 |
| Hidden width / head width | 768 / 64 |
| MLP intermediate width | 3072 |
| Context length | 1024 |
| Vocabulary | 50257, no padding |
| Positions | Learned absolute position embeddings |
| Normalization | Pre-LayerNorm with affine weight and bias; epsilon 1e-5 |
| Activation | Original GPT-2 tanh GELU |
| Linear biases | Enabled in all block projections; no output-head bias |
| Output head | Tied to token embedding, one shared parameter |
| Residuals | Standard attention and MLP residuals, no additional skips or mixing |
| Initialization | Normal standard deviation 0.02; residual projections scaled by 1/sqrt(24) |
| Dropout | 0 for both experiments, following nanoGPT pretraining/llm.c |
| Unique trainable parameters | **124,439,808** |

Q, K and V are stored as three matrices instead of one packed QKV matrix so the
existing per-projection Muon updates and 72 WeightWatcher traces remain available.
Concatenating the three weights/biases recovers the standard GPT-2 packed QKV
projection. This storage choice preserves the architecture's forward and backward
equations. Applying Muon independently to Q/K/V is an optimizer choice; it is not
the same update as orthogonalizing their concatenation.

## Every block matrix

Shapes are **stored PyTorch `[output features, input features]`**. Each Q/K/V
matrix covers all 12 heads; one head corresponds to a 64 × 768 row slice.

| Block | W_Q | W_K | W_V | W_O | W_MLP_IN | W_MLP_OUT |
|---|---|---|---|---|---|---|
| L00 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L01 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L02 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L03 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L04 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L05 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L06 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L07 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L08 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L09 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L10 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L11 | 768 × 768 | 768 × 768 | 768 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |

The checkpoint paths are `transformer.h.<block>.attn.c_q.weight`, `c_k.weight`,
`c_v.weight`, `c_proj.weight`, and `transformer.h.<block>.mlp.c_fc.weight`,
`c_proj.weight`. Packed QKV would be **2304 × 768**, an alternate representation
of the same three 768 × 768 matrices, not an additional parameter.

| Other weight | Shape | Unique elements |
|---|---|---:|
| Token embedding `transformer.wte.weight` | 50257 × 768 | 38,597,376 |
| Position embedding `transformer.wpe.weight` | 1024 × 768 | 786,432 |
| Vocabulary output `lm_head.weight` | 50257 × 768 | 0 additional; tied token embedding |

Each block also has two LayerNorm scale/bias pairs of length 768; Q/K/V/O biases
of length 768 each; and MLP biases of length 3072 and 768. Final LayerNorm has
one scale and one bias vector of length 768. There are no other learned tensors.
Each block has 7,087,872 parameters including vectors. The count is
38,597,376 + 786,432 + 12 × 7,087,872 + 1,536 = **124,439,808**.

[stock_weight_matrices.csv](stock_weight_matrices.csv) lists all 75 named matrix
entries: 72 block matrices, 2 embedding matrices and 1 explicitly marked shared
output alias (74 unique matrix parameters). Regenerate the inventory directly:

```bash
python baseline/gpt2_small/muon_speedrun/stock_model.py
```

## Optimizer and experiment integration

The old untied embedding/head rates of 0.6/0.008 cannot be assigned to the same
tied parameter. Both optimizers now use AdamW at 6e-4 for embeddings, biases and
LayerNorm; matrix decay 0.1 and vector decay 0. Muon retains LR 0.04, its momentum
ramp and five Newton–Schulz iterations on the 72 block matrices, without decay.
The AdamW control also updates all block matrices at 6e-4 with decay 0.1.
All trainable tensors have FP32 parameters and optimizer state; TPU activations
and matrix multiplies use BF16 with FP32 LayerNorm statistics and output loss.

The paired 3-seed protocol, 3000 updates, token order/budget, cooldown, validation,
checkpointing and spectral measurement cadence are retained. Plans/manifests
include `gpt2-small-stock-v1` and the full configuration. Reports exclude runs
with a different architecture/configuration, and execution rejects old suite plans.
The flash-attention preflight now tests 12 heads of width 64. The historical
25k worker explicitly requests its original 6-head/width-128 preflight.

The old modified-model NLL target 3.28 remains a labeled historical threshold,
not a predicted stock-model loss. These optimizer settings are not tuned for the
new architecture. Existing running jobs keep their pinned source; this repository
upgrade applies to fresh runs. No TPU training is started by the architecture audit.

## Verification and references

CPU tests compare logits, loss and every gradient against the independent packed
QKV GPT-2 implementation already pinned in
`speedrun30/vendor/llmc_train_gpt2.py`; they also check causality, dimensions,
parameter count, tying, optimizer coverage, both mixed-precision training paths,
checkpoint reload and paired spectral snapshots. Full-model dimensions/counts
are checked on the meta device without allocating weights. Live TPU execution
and convergence require a new run and are not established by these CPU checks.

- [OpenAI GPT-2 implementation](https://github.com/openai/gpt-2/blob/master/src/model.py)
- [Karpathy nanoGPT implementation](https://github.com/karpathy/nanoGPT/blob/master/model.py)
- [Pinned llm.c reference and provenance](speedrun30/README.md)
- [Current paired-seed protocol](muon_speedrun/REPEATED_SEEDS.md)
