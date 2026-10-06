# Unmodified upstream GPT-2 Small architecture

Both MuonClip and AdamW instantiate `GPT` directly from the unchanged
`speedrun30/vendor/llmc_train_gpt2.py`, pinned to llm.c commit
`7ecd8906afe6ed7a2b2cdb731c042f26d525b820`. The adapter checks the complete file's
SHA256 before import. No model class, layer or forward method is rewritten.
The architecture identifier is `gpt2-small-upstream-packed-v2`.

| Property | Value |
|---|---|
| Blocks / heads / hidden width | 12 / 12 / 768 |
| Head width / MLP width | 64 / 3072 |
| Context / vocabulary | 1024 / 50257 |
| Positions | Learned absolute embeddings |
| Normalization | Pre-LayerNorm, affine scales and biases, epsilon 1e-5 |
| Activation | Original GPT-2 tanh GELU |
| Linear biases | Enabled; vocabulary head has no bias |
| Output head | Tied to token embedding |
| Dropout | 0, as in the upstream pretraining model |
| Unique trainable parameters | **124,439,808** |
| Single-run initialization | Upstream initializer, seed 42 |

## Stored layer weight matrices

Shapes use PyTorch `[output features, input features]` storage. **QKV is packed**
in the upstream `attn.c_attn.weight` parameter. The following four matrices occur
in each of the twelve blocks, indexed 0–11:

| Block | QKV `attn.c_attn` | O `attn.c_proj` | MLP IN `mlp.c_fc` | MLP OUT `mlp.c_proj` |
|---|---|---|---|---|
| L00 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L01 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L02 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L03 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L04 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L05 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L06 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L07 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L08 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L09 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L10 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |
| L11 | 2304 × 768 | 768 × 768 | 3072 × 768 | 768 × 3072 |

| Other weight | Shape |
|---|---|
| Token embedding `transformer.wte.weight` | 50257 × 768 |
| Position embedding `transformer.wpe.weight` | 1024 × 768 |
| Vocabulary output `lm_head.weight` | 50257 × 768; alias of token embedding |

There are **48 stored block matrices**, two embedding matrices and one named tied
output alias: **50 unique matrix parameters, 51 named entries**. The full machine
readable list is [stock_weight_matrices.csv](stock_weight_matrices.csv).
Each block also has LayerNorm scales/biases of length 768, packed QKV bias 2304,
attention output bias 768, MLP input bias 3072 and output bias 768. Final LayerNorm
has scale and bias vectors of length 768. Each block contains 7,087,872 parameters.
The upstream causal-mask buffers are not learned parameters.

Q, K and V are each a 768 × 768 **slice** of the packed 2304 × 768 matrix; a single
head occupies 64 × 768 rows. WeightWatcher extracts these slices only from saved
CPU weights, retaining 72 projection traces without changing trainable storage.

MuonClip observes QK logits without modifying the forward output. Its parameter
and bias rescaling occurs in the optimizer update. TPU attention selection and
BF16 autocast live outside the upstream model. The worker requires attention
parity plus a full-sized accumulated optimizer preflight before fresh training.

[Configuration and provenance](muon_speedrun/BENCHMARK.md) ·
[Launch instructions](muon_speedrun/README.md) ·
[Paired-seed protocol](muon_speedrun/REPEATED_SEEDS.md)

Historical six-head modified-model runs and the earlier split-QKV port have
different architecture/protocol identifiers and are excluded from new comparisons.
