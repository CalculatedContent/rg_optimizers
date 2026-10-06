"""GPT-2 equations with separate Q/K/V tensors for per-matrix Muon and spectra.

Reference: ../speedrun30/vendor/llmc_train_gpt2.py (MIT, see its LICENSE).
No RoPE, RMSNorm, QK normalization, value mixing, U-Net or logit soft cap.
Dropout is disabled for both optimizers, as in nanoGPT pretraining/llm.c.
FP32 parameters and optimizer state; optional BF16 activations on TPU.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from stock_config import ARCHITECTURE, GPTConfig

ATTENTION = None


class CastedLinear(nn.Linear):
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        # Accumulate statistics in FP32, then restore the activation dtype.
        return F.layer_norm(x.float(), self.normalized_shape,
                            self.weight.float(), self.bias.float(), self.eps).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = CastedLinear(config.n_embd, config.n_embd)
        self.c_k = CastedLinear(config.n_embd, config.n_embd)
        self.c_v = CastedLinear(config.n_embd, config.n_embd)
        self.c_proj = CastedLinear(config.n_embd, config.n_embd)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = [projection(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
                   for projection in (self.c_q, self.c_k, self.c_v)]
        y = (ATTENTION(q, k, v) if ATTENTION is not None else
             F.scaled_dot_product_attention(q, k, v, is_causal=True))
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = CastedLinear(config.n_embd, 4 * config.n_embd)
        self.c_proj = CastedLinear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        # Original GPT-2's tanh GELU, also used by the pinned llm.c reference.
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    architecture_id = ARCHITECTURE

    def __init__(self, config=GPTConfig(), *, activation_dtype=None):
        super().__init__()
        self.config = config
        self.activation_dtype = activation_dtype
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, eps=1e-5),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.transformer.wte.weight
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _apply(self, fn, recurse=True):
        # XLA/device/dtype conversion can replace the two aliases independently.
        result = super()._apply(fn, recurse=recurse)
        self.lm_head.weight = self.transformer.wte.weight
        return result

    def logits(self, idx):
        if idx.shape[1] > self.config.block_size:
            raise ValueError('Sequence exceeds GPT-2 context length')
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(positions)
        if self.activation_dtype is not None:
            x = x.to(self.activation_dtype)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x)).float()

    def forward(self, idx, target, *, return_token_errors=False):
        logits = self.logits(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        if return_token_errors:
            return loss, (logits.argmax(dim=-1) != target).sum(dtype=torch.int32)
        return loss


def matrix_inventory(model):
    """Stored [out, in] dimensions, with the tied output alias explicitly marked."""
    rows = []
    for name, value in model.named_parameters(remove_duplicate=False):
        if value.ndim == 2:
            rows.append(dict(name=name, shape=list(value.shape), elements=value.numel(),
                             shared_with='transformer.wte.weight' if name == 'lm_head.weight' else None))
    return rows


if __name__ == '__main__':
    import json
    with torch.device('meta'):
        model = GPT()
    print(json.dumps(dict(architecture=ARCHITECTURE,
                         parameters=sum(p.numel() for p in model.parameters()),
                         matrices=matrix_inventory(model)), indent=2))
