from .attention import AttnArgs, CausalSelfAttention, Yarn
from .layers import CastedLinearT, next_multiple_of_n, norm
from .ngram_math import NGRAM_SIGN_POOL_ROWS
from .ops import ngram_embedding, value_embed_lookup, rms_norm_with_head, scale, scale_add, checkpointed_mlp, language_loss
"""The GPT model: embeddings, 11 transformer blocks with MUDD skip connections, and the loss."""
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn
NUM_LAYERS = 11
NO_ATTN_LAYERS = (4, 6, 7, 9)
NO_MLP_LAYERS = (7,)
ATTN_LAYERS = tuple((i for i in range(NUM_LAYERS) if i not in NO_ATTN_LAYERS))
LONG_WINDOW_LAYERS = (3, 10)
PAIRED_HEAD_LAYERS = (0, 2, 5)
WIDE_QK_LAYERS = LONG_WINDOW_LAYERS
HALF_V_LAYERS = (1, 8)
NARROW_QK_LAYERS = tuple((i for i in ATTN_LAYERS if i not in WIDE_QK_LAYERS + HALF_V_LAYERS))
NARROW_HEAD_DIM = 64
NARROW_ATTN_SCALE = 0.13
WIDE_ATTN_SCALE = 0.085
ATTN_WIDTH_GROUPS = {'narrow': NARROW_QK_LAYERS, 'half_v': HALF_V_LAYERS, 'wide': WIDE_QK_LAYERS}
ATTN_BANK_ORDER = sum(ATTN_WIDTH_GROUPS.values(), ())
NUM_QK64_SLOTS = len(NARROW_QK_LAYERS) + len(HALF_V_LAYERS)
assert sorted(ATTN_BANK_ORDER) == list(ATTN_LAYERS), 'every attention layer needs exactly one width group'
assert ATTN_BANK_ORDER[NUM_QK64_SLOTS:] == WIDE_QK_LAYERS
ATTN_GATE_LAYERS = (3, 10)
XSA_LAYERS = (1, 3)
CACHE_LAYERS = (3, 7)
VALUE_EMBED_LAYERS = (1, 2, 8, 10)
VALUE_EMBED_GATE_LAYERS = (1, 2, 8)
VALUE_EMBED_GATE_CHANNELS = 6
X0_INJECT_LAYERS = (0, 1, 2, 4, 5, 7)
BIGRAM_INJECT_LAYERS = (0, 1, 4, 5, 9)
assert not {6, 10} & set(X0_INJECT_LAYERS + BIGRAM_INJECT_LAYERS)
POST_GATE_LAYER = 4
PRE_GATE_X0_LAYERS = tuple((i for i in X0_INJECT_LAYERS if i < POST_GATE_LAYER))
PRE_GATE_BIGRAM_LAYERS = tuple((i for i in BIGRAM_INJECT_LAYERS if i < POST_GATE_LAYER))
POST_GATE_X0_LAYERS = tuple((i for i in X0_INJECT_LAYERS if i >= POST_GATE_LAYER))
POST_GATE_BIGRAM_LAYERS = tuple((i for i in BIGRAM_INJECT_LAYERS if i >= POST_GATE_LAYER))
PRE_GATE_ATTN_GATE_LAYERS = tuple((i for i in ATTN_GATE_LAYERS if i < POST_GATE_LAYER))
POST_GATE_ATTN_GATE_LAYERS = tuple((i for i in ATTN_GATE_LAYERS if i >= POST_GATE_LAYER))
assert all((i < POST_GATE_LAYER for i in XSA_LAYERS)), 'the XSA strengths come from the pre gate'
PARALLEL_MLP_LAYER, PARALLEL_MLP_SLOT = (8, 11)
MUDD_GROUPS = 12
LAST_LAYER_MUDD_COEFS = 14
MUDD_GATE_HEAD_LANES = 6
MUDD_GATE_SCALE = 0.1
NUM_MLP_SLOTS = 12
MLP_HIDDEN_DIM = 2816
EVAL_CE_SLAB_ROWS = 32768

@dataclass(slots=True)
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    prefix_weight: torch.Tensor
    ws_short: int
    ws_long: int
    train_max_seq_len: int
    sampled_loss: object = None

class GPT(nn.Module):
    """Training runs every projection in fp8 (the attention QKV, the MLP, the lm_head loss); validation
    runs the bf16 path under no_grad."""

    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int, *, ngram_dim: int, world_size: int, device: torch.device):
        super().__init__()
        assert num_layers == NUM_LAYERS
        self.world_size = world_size
        self.device = device
        self.ngram_dim = ngram_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.register_buffer('prefix_table', torch.full((self.vocab_size,), -1, dtype=torch.int64), persistent=False)
        self.register_buffer('ngram_cache', torch.zeros(2 * max_seq_len, ngram_dim, dtype=torch.bfloat16, device=device), persistent=False)
        self.register_buffer('canon_mask', torch.zeros(self.vocab_size, self.vocab_size // 8, dtype=torch.uint8), persistent=False)
        self.lm_head = CastedLinearT(model_dim, self.vocab_size, x_s=100 / 448, w_s=2.0 / 448, grad_s=0.75 / 8 / 448)
        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)
        self.embed = nn.Embedding(self.vocab_size, model_dim)
        with torch.no_grad():
            self.embed.weight.copy_(self.lm_head.weight.T)
        self.init_attn(model_dim, head_dim, num_heads, max_seq_len)
        self.init_mlp(model_dim)
        self.init_misc(model_dim, num_layers)
        self.init_mudd(num_layers, model_dim)
        self.init_mudd_gate(model_dim)
        for name, param in self.named_parameters():
            param.label = name.replace('.weight', '')

    def init_attn(self, model_dim, head_dim, num_heads, max_seq_len):
        assert head_dim == 2 * NARROW_HEAD_DIM
        self.attn = nn.ModuleDict({str(layer): CausalSelfAttention(num_heads, head_dim, qk_dim=self.attn_qk_dim(layer), v_dim=self.attn_v_dim(layer), val_max_seq_len=max_seq_len, paired=layer in PAIRED_HEAD_LAYERS) for layer in ATTN_LAYERS})
        self.yarn = Yarn(NARROW_HEAD_DIM, max_seq_len, attn_scale=NARROW_ATTN_SCALE, device=self.device)
        self.yarn_paired_head = Yarn(NARROW_HEAD_DIM, max_seq_len, paired=True, attn_scale=NARROW_ATTN_SCALE, device=self.device)
        self.yarn_wide = Yarn(head_dim, max_seq_len, attn_scale=WIDE_ATTN_SCALE, device=self.device)
        assert not set(PAIRED_HEAD_LAYERS) & set(WIDE_QK_LAYERS), 'no paired rotary table at full width'
        num_ve = len(VALUE_EMBED_LAYERS)
        self.value_embeds = nn.Parameter(0.01 * torch.randn(num_ve * self.vocab_size, model_dim, dtype=torch.bfloat16))
        self.ve_gate_bank = nn.Parameter(torch.zeros(len(VALUE_EMBED_GATE_LAYERS), num_heads, 12))
        num_slots = len(ATTN_BANK_ORDER)
        hdim = num_heads * head_dim
        qk_groups_per_slot = num_heads
        num_qk_groups = num_slots * qk_groups_per_slot
        self._num_qk_groups = num_qk_groups
        num_qk_padded = next_multiple_of_n(num_qk_groups, n=self.world_size)
        self.qk_bank = nn.Parameter(torch.empty(num_qk_padded, 2 * head_dim, model_dim))
        self.qk_bank.reshape = (num_qk_padded, 2 * head_dim, model_dim)
        assert hdim == model_dim
        num_vo_real = 2 * num_slots
        num_vo_padded = next_multiple_of_n(num_vo_real, n=self.world_size)
        self.vo_bank = nn.Parameter(torch.empty(num_vo_padded, hdim, model_dim))
        self.vo_bank.reshape = (num_vo_padded, hdim, model_dim)
        std = 0.5 * model_dim ** (-0.5)
        bound = 3 ** 0.5 * std
        with torch.no_grad():
            self.qk_bank[:num_qk_groups].uniform_(-bound, bound)
            self.qk_bank[num_qk_groups:].zero_()
            self.vo_bank[:num_vo_real].uniform_(-bound, bound)
            self.vo_bank[num_vo_real:].zero_()

    def attn_qk_dim(self, layer: int) -> int:
        return self.head_dim if layer in WIDE_QK_LAYERS else NARROW_HEAD_DIM

    def attn_v_dim(self, layer: int) -> int:
        return NARROW_HEAD_DIM if layer in HALF_V_LAYERS else self.head_dim

    def init_mlp(self, model_dim):
        self.mlp_hdim = MLP_HIDDEN_DIM
        self.mlp_bank = nn.Parameter(torch.empty(NUM_MLP_SLOTS, 2, self.mlp_hdim, model_dim))
        self.mlp_bank.reshape = (2 * NUM_MLP_SLOTS, self.mlp_hdim, model_dim)
        self.mlp_bank.frozen_matrices = frozenset((2 * layer + j for layer in NO_MLP_LAYERS for j in (0, 1)))
        std = 0.5 * model_dim ** (-0.5)
        bound = 3 ** 0.5 * std
        with torch.no_grad():
            self.mlp_bank[:, 0, :, :].uniform_(-bound, bound)
            self.mlp_bank[:, 1, :, :].zero_()

    def init_misc(self, model_dim, num_layers):
        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)
        ngram_sign_pool = torch.randn(NGRAM_SIGN_POOL_ROWS, self.ngram_dim).sign().to(torch.bfloat16)
        self.register_buffer('ngram_sign_pool', ngram_sign_pool)
        self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))
        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1 ** 0.5))
        pad = (-num_layers * 2 - 2) % self.world_size
        self.scalars = nn.Parameter(torch.cat([*[torch.tensor([0.5, 1.0]) for _ in range(num_layers)], torch.zeros(1), -1.5 * torch.ones(1), torch.ones(pad)]))

    def init_mudd(self, num_layers: int, model_dim: int):
        """
        Multiway Dynamic Dense Connections @lishengping. https://arxiv.org/abs/2502.12170
        Expressive and efficient mechanism for data dependent skip connections.
        Given current activation x, return n skip coefficients computed via ~mlp(x).
        Trimmed for speedrun: invoked at start of last layer and post-loop only.

        Start of last layer produces LAST_LAYER_MUDD_COEFS (14) coefficients:
          mu[0..2]  = v_mudd source coefs  (cache[0], cache[7], x)   -> added into V
          mu[3..5]  = residual source coefs (cache[0], cache[7], x)  -> residual recombination
          mu[6..7]  = per-pair ve_gate (2 channels, tiled to num_heads)
          mu[8..9]  = resid_attn / post_attn lambdas (dynamic)
          mu[10..11]= x0 / bigram injection lambdas (dynamic)
          mu[12..13]= resid_mlp / post_mlp lambdas (dynamic)

        Post-loop produces 5 residual coefs over
          {cache[0], cache[7], cache[9], ve_bank0, cache[3]}.
        """
        num_mudd_layers = 2
        self._mudd_scale = 0.1
        mudd_dim = 64
        max_num_coef = LAST_LAYER_MUDD_COEFS
        self.mudd_w1 = nn.Parameter(torch.empty(num_mudd_layers, mudd_dim, model_dim))
        for j in range(num_mudd_layers):
            nn.init.kaiming_uniform_(self.mudd_w1.data[j], a=math.sqrt(5))
        self.mudd_w2 = nn.Parameter(torch.zeros(num_mudd_layers, max_num_coef, mudd_dim))
        bs_init = torch.zeros(num_mudd_layers, max_num_coef)
        bs_init[0, 6] = 2.0 / self._mudd_scale
        bs_init[0, 7] = 2.0 / self._mudd_scale
        bs_init[0, 8] = 1.1 ** 0.5 / self._mudd_scale
        bs_init[0, 9] = 1.0 / self._mudd_scale
        bs_init[0, 10] = 0.0
        bs_init[0, 11] = 0.05 / self._mudd_scale
        bs_init[0, 12] = 1.1 ** 0.5 / self._mudd_scale
        bs_init[0, 13] = 1.0 / self._mudd_scale
        bs_init[1, 1] = -0.5 / self._mudd_scale
        self.mudd_b2 = nn.Parameter(bs_init)
        self.mudd_w2g = nn.Parameter(torch.zeros(max_num_coef, MUDD_GROUPS, mudd_dim))

    def forward_mudd(self, x, id, num_coef):
        """Returns `num_coef` per-token MUDD coefficients from block `id` (0 or 1)."""
        x = F.gelu(F.linear(x, self.mudd_w1[id]))
        x = (F.linear(x, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        return x.split(1, dim=-1)

    def forward_mudd_grouped(self, x, id, num_coef):
        """forward_mudd's coefficients plus (B, T, num_coef, MUDD_GROUPS) per-channel-group deltas."""
        h = F.gelu(F.linear(x, self.mudd_w1[id]))
        y = (F.linear(h, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        deltas = torch.einsum('btd,kgd->btkg', h, self.mudd_w2g[:num_coef]) * self._mudd_scale
        return (y.split(1, dim=-1), deltas)

    def init_mudd_gate(self, model_dim: int):
        self._mudd_gate_scale = nn.Parameter(torch.tensor(MUDD_GATE_SCALE))
        mudd_gate_dim = 64
        H = MUDD_GATE_HEAD_LANES
        assert self.num_heads == H
        pre_attn_start = H * len(XSA_LAYERS)
        pre_x0_start = pre_attn_start + H * len(PRE_GATE_ATTN_GATE_LAYERS)
        pre_bigram_start = pre_x0_start + len(PRE_GATE_X0_LAYERS)
        self._mudd_gate_pre_num_coef = pre_bigram_start + len(PRE_GATE_BIGRAM_LAYERS)
        post_x0_start = H * len(POST_GATE_ATTN_GATE_LAYERS)
        post_bigram_start = post_x0_start + len(POST_GATE_X0_LAYERS)
        post_skip_lane = post_bigram_start + len(POST_GATE_BIGRAM_LAYERS)
        self._mudd_gate_post_num_coef = post_skip_lane + 1
        max_num_coef = max(self._mudd_gate_pre_num_coef, self._mudd_gate_post_num_coef)
        self.mudd_gate_w1 = nn.Parameter(torch.empty(2, mudd_gate_dim, model_dim))
        self.mudd_gate_w2 = nn.Parameter(torch.zeros(2, max_num_coef, mudd_gate_dim))
        for j in range(2):
            nn.init.kaiming_uniform_(self.mudd_gate_w1.data[j], a=math.sqrt(5))
        bs_init = torch.zeros(2, max_num_coef)
        attn_gate_bias = 0.25 / MUDD_GATE_SCALE
        bigram_gate_bias = 0.05 / MUDD_GATE_SCALE
        skip_gate_bias = 0.5 / MUDD_GATE_SCALE
        bs_init[0, pre_attn_start:pre_x0_start].fill_(attn_gate_bias)
        bs_init[0, pre_bigram_start:self._mudd_gate_pre_num_coef].fill_(bigram_gate_bias)
        bs_init[1, 0:post_x0_start].fill_(attn_gate_bias)
        bs_init[1, post_bigram_start:post_skip_lane].fill_(bigram_gate_bias)
        bs_init[1, post_skip_lane].fill_(skip_gate_bias)
        self.mudd_gate_b2 = nn.Parameter(bs_init)

    def forward_mudd_gate(self, x, id, num_coef):
        x = F.gelu(F.linear(x, self.mudd_gate_w1[id]))
        return (F.linear(x, self.mudd_gate_w2[id, :num_coef]) + self.mudd_gate_b2[id, :num_coef]) * self._mudd_gate_scale.type_as(x)

    @staticmethod
    def _unpack_lanes(gate, start, layers, gates, width=1):
        """gates[layer] = the next `width`-lane slice of `gate` for each of `layers`; returns the next free lane."""
        for k, layer in enumerate(layers):
            lo = start + k * width
            gates[layer] = gate[..., lo:lo + width]
        return start + len(layers) * width

    def unpack_pre_mudd_gate(self, gate, xsa_alphas, attn_gates, x0_gates, bigram_gates):
        lane = self._unpack_lanes(gate, 0, XSA_LAYERS, xsa_alphas, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_ATTN_GATE_LAYERS, attn_gates, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_X0_LAYERS, x0_gates)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_BIGRAM_LAYERS, bigram_gates)
        assert lane == self._mudd_gate_pre_num_coef

    def unpack_post_mudd_gate(self, gate, attn_gates, x0_gates, bigram_gates):
        """Unpacks the post gate; returns the layer-6 skip gate."""
        lane = self._unpack_lanes(gate, 0, POST_GATE_ATTN_GATE_LAYERS, attn_gates, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, POST_GATE_X0_LAYERS, x0_gates)
        lane = self._unpack_lanes(gate, lane, POST_GATE_BIGRAM_LAYERS, bigram_gates)
        assert lane + 1 == self._mudd_gate_post_num_coef
        return gate[..., lane:lane + 1]

    def _inject(self, i, x, x0, x0_bigram, x0_gates, bigram_gates):
        """Layer i's gated x0 and n-gram injections into the residual stream (layer 0's n-gram is pre-loop)."""
        if i in X0_INJECT_LAYERS:
            x = x + x0 * x0_gates[i]
        if i in BIGRAM_INJECT_LAYERS and i != 0:
            x[..., :self.ngram_dim] = x[..., :self.ngram_dim] + x0_bigram * bigram_gates[i]
        return x

    def _mlp(self, x_normed, c_fc, c_proj, fp8_args):
        return checkpointed_mlp(x_normed, c_fc, c_proj)

    def _attn_weights(self):
        """Per attention layer: (qk_w, v_w, o_w) from its bank slot, cut to the layer's head widths.

        qk_w is [2 * num_heads * qk_dim, dim] (Q heads, then K heads), v_w is [num_heads * v_dim, dim]
        and o_w is [dim, num_heads * v_dim] (nn.Linear layout). Each bank is unbound once, and the
        d_qk = 64 rows of all their slots are cut in one copy: per-layer indexing of a batched view
        would add a select_backward kernel per access (the same thing mlp_bank's unbind avoids).
        """
        H, head_dim, dim = (self.num_heads, self.head_dim, self.qk_bank.shape[-1])
        num_slots = len(ATTN_BANK_ORDER)
        qk_heads = self.qk_bank[:self._num_qk_groups].view(num_slots, 2 * H, head_dim, dim)
        qk_full = qk_heads.flatten(1, 2).unbind(0)
        qk_narrow = qk_heads[:NUM_QK64_SLOTS, :, :NARROW_HEAD_DIM].reshape(NUM_QK64_SLOTS, 2 * H * NARROW_HEAD_DIM, dim).unbind(0)
        vo = self.vo_bank[:2 * num_slots].unbind(0)
        weights = {}
        for slot, layer in enumerate(ATTN_BANK_ORDER):
            qk_w = qk_full[slot] if layer in WIDE_QK_LAYERS else qk_narrow[slot]
            v_w, o_w = (vo[2 * slot], vo[2 * slot + 1])
            v_dim = self.attn_v_dim(layer)
            if v_dim < head_dim:
                v_w = v_w.view(H, head_dim, dim)[:, :v_dim].reshape(H * v_dim, dim)
                o_w = o_w.view(dim, H, head_dim)[:, :, :v_dim].reshape(dim, H * v_dim)
            weights[layer] = (qk_w, v_w, o_w)
        return weights

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, bigram_input_seq: Tensor, schedule_cfg: ForwardScheduleConfig, ngram_sink: Tensor | None=None, value_embed_grad: Tensor | None=None):
        """Per-token loss for one packed varlen batch (B=1, documents separated by `seqlens`).

        bigram_input_seq: [2T] int32 slots in `ngram_cache` of each token's bigram (first T) and
        trigram (last T) row. It carries cache slots, not hashes; the name is the keyword
        evals/hellaswag.py calls with.
        ngram_sink receives per-occurrence gradients for host sparse updates.
        value_embed_grad is an unused compatibility argument; value_embeds uses ordinary autograd.

        Layer topology (11 layers, 0-indexed):
          - attention on ATTN_LAYERS (0, 1, 2, 3, 5, 8, 10); short sliding window except layers 3 and 10
            (long window, partial key offset). Layer 6 adds a gated skip from layer 3 instead of attention;
            layers 4 and 9 run only their MLP; layer 7 only rescales and re-injects x0 / bigram
          - head widths: query/key 128 on the long-window layers 3, 10 and 64 elsewhere; value/output
            64 on layers 1, 8 and 128 elsewhere
          - paired-head attention on layers 0, 2, 5; token value embeddings added to V on 1, 2, 8, 10
          - MUDD gates are computed from x0 (for layers 0-3) and at the start of layer 4 (for layers 4-10);
            the last layer and the post-loop mix use MUDD dense connections over cached layer outputs
        """
        assert input_seq.ndim == 1
        mtp_weights, train_max_seq_len = (schedule_cfg.mtp_weights, schedule_cfg.train_max_seq_len)
        prefix_weight = schedule_cfg.prefix_weight
        ws_short, ws_long = (schedule_cfg.ws_short, schedule_cfg.ws_long)
        bm_sizes = [ws_long if i in LONG_WINDOW_LAYERS else ws_short for i in range(self.num_layers)]
        key_offset = [i in LONG_WINDOW_LAYERS for i in range(self.num_layers)]
        use_fp8 = False
        attn_f8_weights = None
        mlp_up_f8 = None
        sa_lambdas = self.scalars[:2 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[2 * self.num_layers]
        skip_lambda = self.scalars[2 * self.num_layers + 1]
        resid_lambdas_attn = self.resid_lambdas[:, 0].bfloat16().unbind(0)
        resid_lambdas_mlp = self.resid_lambdas[:, 1].bfloat16().unbind(0)
        post_lambdas_attn = self.post_lambdas[:, 0].bfloat16().unbind(0)
        post_lambdas_mlp = self.post_lambdas[:, 1].bfloat16().unbind(0)
        ve_gates = [None] * self.num_layers
        for layer, gate in zip(VALUE_EMBED_GATE_LAYERS, self.ve_gate_bank.unbind(0)):
            ve_gates[layer] = gate
        attn_gates = [None] * self.num_layers
        xsa_alphas = [None] * self.num_layers
        x0_gates = [None] * self.num_layers
        bigram_gates = [None] * self.num_layers
        attn_weights = self._attn_weights()
        mlp_all = self.mlp_bank.flatten(0, 1).unbind(0)
        mlp_fcs = mlp_all[0::2]
        mlp_projs = mlp_all[1::2]
        x = self.embed(input_seq)
        x0_bigram = ngram_embedding(self.ngram_cache, bigram_input_seq, self.ngram_sign_pool, input_seq, ngram_sink)[None]
        ve_planes = value_embed_lookup(self.value_embeds, input_seq, len(VALUE_EMBED_LAYERS), value_embed_grad)
        ve = [None] * self.num_layers
        for layer, plane in zip(VALUE_EMBED_LAYERS, ve_planes):
            ve[layer] = plane
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])
        pre_gate = self.forward_mudd_gate(x0, id=0, num_coef=self._mudd_gate_pre_num_coef)
        self.unpack_pre_mudd_gate(pre_gate, xsa_alphas, attn_gates, x0_gates, bigram_gates)
        x = x0.clone()
        x[..., :self.ngram_dim] = x[..., :self.ngram_dim] + x0_bigram * bigram_gates[0]
        skip_gate_out = None
        post_skip_gate = None
        cache = {0: x}
        late_attn_in = None
        for i in range(self.num_layers):
            c_fc = mlp_fcs[i]
            c_proj = mlp_projs[i]
            mu = None
            if i == POST_GATE_LAYER:
                post_gate = self.forward_mudd_gate(x, id=1, num_coef=self._mudd_gate_post_num_coef)
                post_skip_gate = self.unpack_post_mudd_gate(post_gate, attn_gates, x0_gates, bigram_gates)
            if i == 6:
                assert post_skip_gate is not None
                skip_gate_out = torch.sigmoid(skip_lambda) * post_skip_gate
                x = x + skip_gate_out * cache[3]
            elif i in NO_ATTN_LAYERS:
                x = self._inject(i, scale(resid_lambdas_attn[i], x), x0, x0_bigram, x0_gates, bigram_gates)
            else:
                ve_gate_head = None
                if late_attn_in is not None:
                    attn_in_normed, attn_x_f8 = late_attn_in
                else:
                    attn_in = cache.get(7, x)
                    if i in VALUE_EMBED_GATE_LAYERS:
                        attn_in_normed, ve_gate_head = rms_norm_with_head(attn_in, VALUE_EMBED_GATE_CHANNELS)
                    else:
                        attn_in_normed = norm(attn_in)
                    attn_x_f8 = None
                    if 7 in cache:
                        late_attn_in = (attn_in_normed, attn_x_f8)
                qkv_fp8 = None
                B, T = (attn_in_normed.size(0), attn_in_normed.size(1))
                if i == self.num_layers - 1:
                    cache[9] = x
                    mu = self.forward_mudd(x, id=0, num_coef=LAST_LAYER_MUDD_COEFS)
                    v_mudd = mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * x
                    v_mudd = v_mudd.view(B, T, self.num_heads, self.head_dim)
                    x = (1 + mu[5]) * x + mu[3] * cache[0] + mu[4] * cache[7]
                    ve_gate = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(self.num_heads // 2, dim=-1).unsqueeze(-1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate * ve_view + v_mudd).view(B, T, -1)
                elif ve[i] is not None:
                    gate_in = torch.cat([ve_gate_head, ve[i][None, ..., :VALUE_EMBED_GATE_CHANNELS]], dim=-1)
                    ve_gate_out = 2 * torch.sigmoid(F.linear(gate_in, ve_gates[i])).view(B, T, self.num_heads, 1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate_out * ve_view).view(B, T, -1)
                else:
                    aux_v = None
                if i in WIDE_QK_LAYERS:
                    yarn = self.yarn_wide
                elif i in PAIRED_HEAD_LAYERS:
                    yarn = self.yarn_paired_head
                else:
                    yarn = self.yarn
                attn_args = AttnArgs(sa_lambdas=sa_lambdas[i], seqlens=seqlens, bm_size=bm_sizes[i], yarn=yarn, key_offset=key_offset[i], attn_gate_w=attn_gates[i] if i in ATTN_GATE_LAYERS else None, aux_v=aux_v, xsa_alpha=xsa_alphas[i], train_max_seq_len=train_max_seq_len, o_gain=post_lambdas_attn[i] if mu is None else None)
                qk_w, v_w, o_w = attn_weights[i]
                attn_out = self.attn[str(i)](attn_in_normed, attn_args, qk_w, v_w, o_w, qkv_fp8)
                if mu is not None:
                    x = mu[8] * x + mu[9] * attn_out + mu[10] * cache[0]
                    x[..., :self.ngram_dim] = x[..., :self.ngram_dim] + mu[11] * x0_bigram
                else:
                    x = scale(resid_lambdas_attn[i], x) + attn_out
                    x = self._inject(i, x, x0, x0_bigram, x0_gates, bigram_gates)
            if i in NO_MLP_LAYERS:
                x = scale(resid_lambdas_mlp[i], x)
                if i in CACHE_LAYERS:
                    cache[i] = x
                continue
            mlp_in = norm(x)
            fold_p = post_lambdas_mlp[i] if False and mu is None else None
            fp8_args = None
            mlp_out = self._mlp(mlp_in, c_fc, c_proj, fp8_args)
            if mu is not None:
                x = mu[12] * x + mu[13] * mlp_out
            elif fold_p is not None:
                x = scale(resid_lambdas_mlp[i], x) + mlp_out
            else:
                x = scale_add(resid_lambdas_mlp[i], x, post_lambdas_mlp[i], mlp_out)
            if i == PARALLEL_MLP_LAYER:
                k = PARALLEL_MLP_SLOT
                parallel_args = None
                parallel_out = self._mlp(mlp_in, mlp_fcs[k], mlp_projs[k], parallel_args)
                x = x + (parallel_out if fold_p is not None else scale(post_lambdas_mlp[i], parallel_out))
            if i in CACHE_LAYERS:
                cache[i] = x
        assert late_attn_in is not None, 'norm(cache[7]) is not bound at loop exit'
        sources = [cache[0], cache[7], cache[9], ve[1][None].to(dtype=x.dtype), cache[3], ve[2][None].to(dtype=x.dtype), ve[10][None].to(dtype=x.dtype), ve[8][None].to(dtype=x.dtype), mlp_in, late_attn_in[0]]
        mu, deltas = self.forward_mudd_grouped(x, id=1, num_coef=len(sources))
        grouped = lambda t: t.unflatten(-1, (MUDD_GROUPS, -1))
        mixed = grouped(x)
        for k, src in enumerate(sources):
            mixed = mixed + (mu[k] + deltas[..., k, :]).unsqueeze(-1) * grouped(src)
        x = mixed.flatten(-2)
        return self._loss(norm(x), input_seq, target_seq, mtp_weights, prefix_weight, schedule_cfg.sampled_loss)

    def _loss(self, x, input_seq, target_seq, mtp_weights, prefix_weight, sampled_loss):
        return language_loss(self, x, input_seq, target_seq, mtp_weights, prefix_weight, sampled_loss)

    def cast_matrix_weights_bf16(self):
        """Matrix weights train in bf16 (lm_head and value_embeds are created bf16); the scalar and
        lambda parameters stay fp32. Call once, before the optimizer is built."""
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                m.weight.data = m.weight.data.bfloat16()
        for param in (self.ve_gate_bank, self.qk_bank, self.vo_bank, self.mlp_bank, self.mudd_w1, self.mudd_w2, self.mudd_w2g, self.mudd_b2, self.mudd_gate_w1, self.mudd_gate_w2, self.mudd_gate_b2):
            param.data = param.data.bfloat16()

    @property
    def yarns(self) -> tuple[Yarn, ...]:
        return (self.yarn, self.yarn_paired_head, self.yarn_wide)

    def limit_yarn_rebuild(self, rows: int):
        """From now on a window change rebuilds only the first `rows` rotary rows (the longest training
        sequence); complete_yarn_tables() fills in the rest before a validation (model/attention.py Yarn)."""
        for yarn in self.yarns:
            yarn.rebuild_rows = rows

    def complete_yarn_tables(self):
        for yarn in self.yarns:
            yarn.ensure_full()
