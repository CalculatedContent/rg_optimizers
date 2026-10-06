import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .ops import qk_norm_rope_forward, window_attention
class Yarn(nn.Module):
    """RoPE cos/sin tables for one head width, rescaled (YaRN) whenever its layers' window grows.

    Row t of a paired table packs positions 2t and 2t+1 side by side, so every table has max_seq_len
    rows. Each row depends only on its position and the current frequencies, so rebuilding a row range
    gives bitwise the rows a full rebuild would. That makes the partial rebuild legal: once
    GPT.limit_yarn_rebuild sets `rebuild_rows` (right before the clock, to the longest training
    sequence), apply() rebuilds only the rows training reads, and ensure_full() completes the table
    before a validation reads it all (record #360: a window change otherwise rebuilds all 262k rows of
    three tables on the clock).
    """
    ROTARY_DIM = 64

    def __init__(self, head_dim, max_seq_len, paired=False, *, attn_scale: float, device):
        super().__init__()
        assert head_dim >= self.ROTARY_DIM
        self.device = device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.base_attn_scale = attn_scale
        width = head_dim if not paired else 2 * head_dim
        self.factor1 = nn.Buffer(torch.empty(max_seq_len, width, dtype=torch.bfloat16, device=device), persistent=False)
        self.factor2 = nn.Buffer(torch.empty(max_seq_len, width, dtype=torch.bfloat16, device=device), persistent=False)
        self.rebuild_rows = max_seq_len
        self.valid_rows = 0
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.ROTARY_DIM // 2, dtype=torch.float32, device=self.device)
        angular_freq = angular_freq.repeat_interleave(2)
        self.angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim - self.ROTARY_DIM)])
        self._build_rows(0, self.max_seq_len)
        self.attn_scale = self.base_attn_scale

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        self._build_rows(0, min(self.max_seq_len, self.rebuild_rows))
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

    def ensure_full(self):
        """Rebuild the rows the last apply() skipped (before a forward longer than rebuild_rows)."""
        if self.valid_rows < self.max_seq_len:
            self._build_rows(self.valid_rows, self.max_seq_len)

    def _build_rows(self, lo: int, hi: int):
        """Rows [lo, hi) from the current frequencies; rows [0, hi) are then current (lo is 0 or valid_rows)."""
        assert lo == 0 or lo == self.valid_rows
        t = torch.arange(lo, hi, dtype=torch.float32, device=self.device)
        if not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1[lo:hi].copy_(theta.cos())
            self.factor2[lo:hi].copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1[lo:hi].copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2[lo:hi].copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[lo:hi, 1::2] *= -1
        self.valid_rows = hi

@dataclass(slots=True)
class AttnArgs:
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor | None
    aux_v: torch.Tensor | None
    xsa_alpha: torch.Tensor | None
    train_max_seq_len: torch.Tensor
    o_gain: torch.Tensor | None = None

class CausalSelfAttention(nn.Module):
    """One attention layer at its own head widths. No parameters: the weights come from the GPT's banks.

    forward takes this layer's weights already cut to its widths, all in nn.Linear [out, in] layout:
      qk_w [2 * num_heads * qk_dim, dim]   Q rows then K rows
      v_w  [num_heads * v_dim, dim]
      o_w  [dim, num_heads * v_dim]
    """

    def __init__(self, num_heads: int, head_dim: int, qk_dim: int, v_dim: int, val_max_seq_len: int, paired: bool=False):
        super().__init__()
        assert qk_dim <= head_dim and v_dim <= head_dim
        self.val_max_seq_len = val_max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.paired = paired

    def forward(self, x: Tensor, attn_args: AttnArgs, qk_w: Tensor, v_w: Tensor, o_w: Tensor, qkv_fp8=None):
        """qkv_fp8 (training only; None under validation): (weight_f8, weight_f8_t, weight_scale, x_scale,
        grad_scale, x_f8, x_f8_t), this layer's packed weight cache, its static fp8 scales and its
        already-quantized input."""
        B, T = (x.size(0), x.size(1))
        assert B == 1, 'varlen sequences requires B == 1'
        assert T % 16 == 0
        H = self.num_heads
        aux_v, attn_gate_w = (attn_args.aux_v, attn_args.attn_gate_w)
        sa_lambdas, key_offset = (attn_args.sa_lambdas, attn_args.key_offset)
        seqlens, bm_size = (attn_args.seqlens, attn_args.bm_size)
        train_max_seq_len, yarn = (attn_args.train_max_seq_len, attn_args.yarn)
        assert yarn.head_dim == self.qk_dim
        assert not key_offset or self.qk_dim > Yarn.ROTARY_DIM
        qkv_weight = sa_lambdas[0] * torch.cat((qk_w, v_w)).type_as(x)
        qkv = F.linear(x, qkv_weight)
        qk, v = qkv.split((2 * H * self.qk_dim, H * self.v_dim), dim=-1)
        qk = qk.view(B, T, 2 * H, self.qk_dim)
        v = v.view(B, T, H, self.v_dim)
        q, k = qk_norm_rope_forward(qk[0], yarn.factor1[:T], yarn.factor2[:T], H, Yarn.ROTARY_DIM, self.paired, key_offset)
        q, k = (q[None], k[None])
        max_len = train_max_seq_len if self.training else self.val_max_seq_len
        if aux_v is not None:
            aux_v = aux_v.view(B, T, H, self.head_dim)[..., :self.v_dim]
        if not self.paired:
            if aux_v is not None:
                v = v + aux_v
        else:
            v = v.reshape(B, T * 2, H // 2, self.v_dim)
            if aux_v is not None:
                v = v + aux_v.reshape(v.shape)
            seqlens = 2 * seqlens
            max_len = 2 * max_len
        y = window_attention(q[0], k[0], v[0], seqlens, bm_size, yarn.attn_scale)
        y = y.view(B, T, H, self.v_dim)
        if attn_args.xsa_alpha is not None and (not self.paired):
            dot = (y * v).sum(-1, keepdim=True)
            denom = v.square().sum(-1, keepdim=True).clamp_min(1e-08)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(B, T, H, 1)
            y = y - alpha * (dot / denom) * v
        if attn_gate_w is not None:
            y = y * attn_gate_w.type_as(y).view(B, T, H, 1)
        y = y.contiguous().view(B, T, H * self.v_dim)
        o_scale = sa_lambdas[1] if attn_args.o_gain is None else sa_lambdas[1] * attn_args.o_gain
        return F.linear(y, o_scale * o_w.type_as(y))
