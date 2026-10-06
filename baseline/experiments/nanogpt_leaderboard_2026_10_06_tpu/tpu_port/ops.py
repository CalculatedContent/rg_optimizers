"""Portable BF16 math; bounded attention/logit slabs instead of CUDA kernels."""
import torch
import torch.nn.functional as F
from .residual import scale, scale_add, rms_norm_with_head


def checkpoint(function, *args):
    if not torch.is_grad_enabled() or not any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
        return function(*args)
    if any(isinstance(a, torch.Tensor) and a.device.type == 'xla' for a in args):
        from torch_xla.utils.checkpoint import checkpoint as run
    else:
        from torch.utils.checkpoint import checkpoint as run
    return run(function, *args, use_reentrant=True, preserve_rng_state=False)


def checkpointed_mlp(x, up, down):
    def mlp(a, b, c):
        return F.linear(F.relu(F.linear(a, b)).square(), c.T)
    return checkpoint(mlp, x, up, down)


def ngram_embedding(cache, slots, sign_pool, tokens, sink=None):
    n = tokens.numel()
    rows = cache[slots.long()]
    if sink is not None:
        rows = rows + sink
    tokens = tokens.to(torch.int32)
    previous = F.pad(tokens[:-1], (1, 0))
    older = F.pad(tokens[:-2], (2, 0))
    positions = torch.arange(n, device=tokens.device)
    mask = sign_pool.size(0) - 1
    b = torch.where(positions >= 1, ((30011 * previous) ^ (48271 * tokens)) & mask, 0)
    t = torch.where(positions >= 2, ((26801 * older) ^ (39779 * previous) ^ (58699 * tokens)) & mask, 0)
    return rows[:n] * sign_pool[b.long()] + rows[n:] * sign_pool[t.long()]


def value_embed_lookup(weight, tokens, planes, grad_accum=None):
    # Ordinary BF16 parameter gradients replace the reference's FP16 atomic cache.
    # Accumulation precision/order differs; convergence requires a measured TPU run.
    vocab = weight.size(0) // planes
    return tuple(F.embedding(tokens.long() + i * vocab, weight) for i in range(planes))


def qk_norm_rope_forward(qk, factor1, factor2, heads, rotary_dim, paired, key_offset):
    n, _, d = qk.shape
    f = qk.float()
    r = torch.rsqrt(f.square().mean(-1, keepdim=True) + 1.1920928955078125e-7)
    flipped = f.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    if paired:
        h = torch.arange(2 * heads, device=qk.device) % heads
        f1 = factor1.reshape(n, 2, d)[:, (h % 2).long()]
        f2 = factor2.reshape(n, 2, d)[:, (h % 2).long()]
    else:
        f1, f2 = factor1[:, None], factor2[:, None]
    y = f1.float() * (f * r) + f2.float() * (flipped * r)
    q, k = y[:, :heads], y[:, heads:]
    if key_offset:
        normalized = f[:, heads:] * r[:, heads:]
        stationary = torch.cat((k[:1, ..., rotary_dim:], normalized[:-1, ..., rotary_dim:]), 0)
        k = torch.cat((k[..., :rotary_dim], stationary), -1)
    if paired:
        q, k = q.reshape(2 * n, heads // 2, d), k.reshape(2 * n, heads // 2, d)
    return q.to(qk.dtype), k.to(qk.dtype)


def window_attention(q, k, v, seqlens, window, softmax_scale, query_rows=128):
    """Exact causal/document/window mask, at most query_rows*(window+query_rows) scores/head.

    The key interval includes the left endpoint, matching FA3 window_size=(window,0).
    Checkpoint each slab so backward does not retain all attention score matrices.
    """
    n = q.size(0)
    pos = torch.arange(n, device=q.device)
    # Repeated trailing ends and a possible duplicate leading 0 are legal upstream.
    doc = (pos[:, None] >= seqlens[None, :]).sum(-1)
    outputs = []
    for start in range(0, n, query_rows):
        end = min(start + query_rows, n)
        lo, hi = max(0, start - window), end
        qp, kp = pos[start:end], pos[lo:hi]
        mask = ((kp[None] <= qp[:, None]) & (kp[None] >= qp[:, None] - window)
                & (doc[start:end, None] == doc[None, lo:hi]))
        def attend(a, b, c, m):
            # Explicit FP32 softmax; BF16 projections and output, no dense packed-stream mask.
            scores = torch.matmul(a.transpose(0, 1).float(), b.permute(1, 2, 0).float()) * softmax_scale
            probability = scores.masked_fill(~m[None], float('-inf')).softmax(-1).to(c.dtype)
            return torch.matmul(probability, c.transpose(0, 1)).transpose(0, 1)
        outputs.append(checkpoint(attend, q[start:end], k[lo:hi], v[lo:hi], mask))
    return torch.cat(outputs)


def language_loss(model, x, inputs, targets, mtp_weights, prefix_weight, sampled, slab=256):
    x = x.reshape(-1, x.size(-1))
    n = x.size(0)
    train = model.training
    if train and sampled is not None:
        candidates, target_pos, prefix_pos = sampled
        weight = model.lm_head.weight[:, candidates]
        target, prefix = target_pos, prefix_pos
    else:
        weight = model.lm_head.weight
        target, prefix = targets, model.prefix_table[targets]
    pieces = []
    for start in range(0, n, slab):
        stop = min(n, start + slab)
        # Bind offsets now: checkpoint invokes this closure again during backward.
        def loss_rows(hidden, w, next_targets, prefix_targets, mask_rows, mtp, pw, start=start, stop=stop):
            raw = hidden @ w
            if train:
                logits = 23 * torch.sigmoid((raw.float() + 5) / 7.5)
            else:
                logits = (23 * torch.sigmoid((raw + 5) / 7.5)).float()
                shifts = torch.arange(8, device=hidden.device, dtype=torch.int32)
                mask = ((mask_rows.to(torch.int32)[..., None] >> shifts) & 1).flatten(-2).bool()
                logits = logits.masked_fill(mask, -60)
            logp = F.log_softmax(logits, -1)
            if not train:
                return -logp.gather(1, next_targets[start:stop, None]).squeeze(1)
            result = logits.new_zeros(stop - start)
            for offset in range(mtp.numel()):
                positions = torch.arange(start, stop, device=hidden.device) + offset
                valid = positions < n
                labels = next_targets[positions.clamp(max=n - 1)]
                ce = -logp.gather(1, labels[:, None]).squeeze(1)
                result = result + mtp[offset] * torch.where(valid, ce, 0)
                # Upstream custom CE backward counts all MTP normalizer weights at the tail.
                lse = logits.logsumexp(-1)
                result = result + mtp[offset] * (~valid) * (lse - lse.detach())
            valid = prefix_targets[start:stop] >= 0
            ce = -logp.gather(1, prefix_targets[start:stop].clamp(min=0)[:, None]).squeeze(1)
            return result + pw.reshape(()) * torch.where(valid, ce, 0)
        mask_rows = (model.canon_mask[inputs[start:stop].long()] if not train
                     else torch.empty(0, dtype=torch.uint8, device=x.device))
        pieces.append(checkpoint(loss_rows, x[start:stop], weight, target, prefix, mask_rows, mtp_weights, prefix_weight))
    return torch.cat(pieces)
