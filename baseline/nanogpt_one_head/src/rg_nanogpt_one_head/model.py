from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50_257
    block_size: int = 256
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.n_layer != 1:
            raise ValueError("the reference architecture is fixed to one block")
        if self.n_head < 1:
            raise ValueError("n_head must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size < 2 or self.vocab_size < 2 or self.n_embd < 1:
            raise ValueError("invalid GPT configuration")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class LayerNorm(nn.Module):
    def __init__(self, width: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = float(cfg.dropout)
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "causal_mask",
            torch.ones(
                cfg.block_size,
                cfg.block_size,
                dtype=torch.bool,
            ).tril().view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def _xla_math_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        dropout_p: float,
    ) -> torch.Tensor:
        """TPU-safe mathematical SDPA using core XLA tensor operations."""
        sequence = q.shape[-2]
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = (q @ k.transpose(-2, -1)) * scale
        mask = self.causal_mask[:, :, :sequence, :sequence]
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )
        probabilities = F.softmax(scores, dim=-1)
        if dropout_p:
            probabilities = F.dropout(
                probabilities,
                p=dropout_p,
                training=True,
            )
        return probabilities @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = x.shape
        head_width = channels // self.n_head
        q = self.q_proj(x).view(
            batch, sequence, self.n_head, head_width
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch, sequence, self.n_head, head_width
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch, sequence, self.n_head, head_width
        ).transpose(1, 2)
        dropout_p = self.dropout if self.training else 0.0
        if q.device.type == "xla":
            # Use core matmul/mask/softmax operations on TPU. This avoids
            # depending on accelerator-specific SDPA kernel registration while
            # preserving the same causal attention equation.
            y = self._xla_math_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        y = y.transpose(1, 2).contiguous().view(
            batch, sequence, channels
        )
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.proj(F.gelu(self.fc(x), approximate="tanh"))
        )


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_module)
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0,
                std=residual_std,
            )
            nn.init.normal_(
                block.mlp.proj.weight,
                mean=0.0,
                std=residual_std,
            )

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        _, sequence = idx.shape
        if sequence > self.cfg.block_size:
            raise ValueError("input sequence exceeds model.block_size")
        positions = torch.arange(sequence, device=idx.device)
        x = self.drop(
            self.token_embedding(idx) + self.position_embedding(positions)
        )
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.lm_head(self.hidden_states(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def next_token_logits(self, idx: torch.Tensor) -> torch.Tensor:
        # Apply the expensive vocabulary projection only to the final position.
        hidden = self.hidden_states(idx)[:, -1:, :]
        return self.lm_head(hidden)

    @torch.inference_mode()
    def generate_greedy(
        self,
        prompts: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if prompts.ndim != 2:
            raise ValueError("prompts must be [batch, sequence]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        idx = prompts
        for _ in range(int(max_new_tokens)):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits = self.next_token_logits(idx_cond)
            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def transformer_matrix_items(
    model: GPT,
) -> list[tuple[str, str, int, torch.Tensor]]:
    """Return the six transformer matrices used by WeightWatcher and Muon."""
    items: list[tuple[str, str, int, torch.Tensor]] = []
    for block_index, block in enumerate(model.blocks):
        matrices = (
            ("W_Q", block.attn.q_proj.weight),
            ("W_K", block.attn.k_proj.weight),
            ("W_V", block.attn.v_proj.weight),
            ("W_O", block.attn.out_proj.weight),
            ("W_MLP_IN", block.mlp.fc.weight),
            ("W_MLP_OUT", block.mlp.proj.weight),
        )
        for matrix_type, weight in matrices:
            items.append(
                (
                    f"L{block_index:02d}_{matrix_type}",
                    matrix_type,
                    block_index,
                    weight,
                )
            )
    return items
