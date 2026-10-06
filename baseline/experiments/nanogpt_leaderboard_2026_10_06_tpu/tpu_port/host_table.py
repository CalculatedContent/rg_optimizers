"""Full-size CPU-owned BF16 n-gram shards and sparse row-Adam; no table autograd."""
import torch
import torch.distributed as dist
from .ngram_math import NGRAM_VOCAB_SIZE, NGRAM_DIM, is_update_step, adam_beta2_and_wd_mul


def exchange(send, receive_rows, width, dtype, rank, world):
    """Gloo CPU point-to-point exchange. Caller bounds each payload by active rows."""
    received = [torch.empty((n, width), dtype=dtype) for n in receive_rows]
    handles = []
    for peer in range(world):
        if peer == rank:
            received[peer].copy_(send[peer])
            continue
        if received[peer].numel():
            handles.append(dist.irecv(received[peer], src=peer))
    for peer in range(world):
        if peer != rank and send[peer].numel():
            handles.append(dist.isend(send[peer].contiguous(), dst=peer))
    for handle in handles:
        handle.wait()
    return received


class HostTable:
    def __init__(self, rank, world, *, rows=NGRAM_VOCAB_SIZE, width=NGRAM_DIM):
        if rows % world:
            raise ValueError('Row count must divide evenly among owners')
        self.rank, self.world = rank, world
        self.rows, self.width = rows, width
        self.local_rows = rows // world
        self.first_row = rank * self.local_rows
        self.shard = torch.zeros(self.local_rows, width, dtype=torch.bfloat16, device='cpu')
        self.exp_avg_sq = torch.zeros(self.local_rows, dtype=torch.float32)
        self.last_event = torch.zeros(self.local_rows, dtype=torch.int32)
        self.history = [1.0]
        self.pending = []
        self.last_update_step = -1

    def _requests(self, ids):
        ids = ids.to(device='cpu', dtype=torch.int64).contiguous()
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.rows):
            raise ValueError('N-gram row outside full table')
        ids = ids.unique(sorted=True)
        requests = [None] * self.world
        if self.world == 1:
            requests[0] = ids
        else:
            dist.all_gather_object(requests, ids)
        bounds = [i * self.local_rows for i in range(self.world + 1)]
        local_wants = [ids[(ids >= bounds[p]) & (ids < bounds[p + 1])] for p in range(self.world)]
        peers_want = [r[(r >= self.first_row) & (r < self.first_row + self.local_rows)] for r in requests]
        return ids, local_wants, peers_want

    def lookup(self, row_ids):
        ids, wants, peer_wants = self._requests(row_ids)
        send = [self.shard[r - self.first_row] for r in peer_wants]
        rows = exchange(send, [r.numel() for r in wants], self.width, torch.bfloat16, self.rank, self.world)
        cache = torch.cat(rows)
        slots = torch.searchsorted(ids, row_ids.long())
        return ids, cache, slots

    def accumulate(self, ids, slots, gradient):
        # Per-cycle FP16 accumulation, separate from the full-size BF16 table.
        out = torch.zeros(ids.numel(), self.width, dtype=torch.float16)
        out.index_add_(0, slots.long(), gradient.detach().cpu().to(torch.float16))
        self.pending.append((ids.clone(), out))

    @torch.no_grad()
    def update(self, step, lr, weight_decay=0.005, eps=1e-10):
        if not is_update_step(step):
            raise ValueError('Sparse Adam invoked outside its reference cadence')
        if step <= self.last_update_step or not self.pending:
            raise ValueError('Repeated/out-of-order event or missing cycle gradients')
        all_ids = torch.cat([r for r, _ in self.pending]).unique(sorted=True)
        accumulated = torch.zeros(all_ids.numel(), self.width, dtype=torch.float16)
        for ids, grad in self.pending:
            accumulated.index_add_(0, torch.searchsorted(all_ids, ids), grad)
        self.pending.clear()
        _, wants, peer_wants = self._requests(all_ids)
        send = [accumulated[torch.searchsorted(all_ids, rows)].to(torch.bfloat16) for rows in wants]
        received = exchange(send, [r.numel() for r in peer_wants], self.width, torch.bfloat16, self.rank, self.world)
        unique = torch.cat(peer_wants).unique(sorted=True)
        merged = torch.zeros(unique.numel(), self.width, dtype=torch.bfloat16)
        for rows, grad in zip(peer_wants, received):
            merged.index_add_(0, torch.searchsorted(unique, rows), grad)
        self.apply_rows(step, unique - self.first_row, merged, lr, weight_decay, eps)
        self.last_update_step = step

    @torch.no_grad()
    def apply_rows(self, step, local_ids, merged, lr, weight_decay=0.005, eps=1e-10, chunk_rows=4096):
        """One update per unique row, replaying missed beta2 events in order."""
        if local_ids.unique().numel() != local_ids.numel():
            raise ValueError('Sparse Adam must receive unique rows')
        if not torch.isfinite(merged).all():
            raise FloatingPointError('Nonfinite sparse gradient')
        beta, wd_mul = adam_beta2_and_wd_mul(step)
        self.history.append(beta)
        event = len(self.history) - 1
        lr = lr * 70.0
        step_size = lr * (1 - beta ** event) ** 0.5
        decay = lr * lr * weight_decay * wd_mul
        for lo in range(0, local_ids.numel(), chunk_rows):
            ids = local_ids[lo:lo + chunk_rows].long()
            g = merged[lo:lo + chunk_rows].float()
            v = self.exp_avg_sq[ids]
            last = self.last_event[ids]
            for missed in range(int(last.min()) + 1, event):
                v = torch.where(last < missed, v * self.history[missed], v)
            v = v * beta + g.square().mean(-1) * ((1 - beta) / self.world ** 2)
            u = (g / self.world) / (v.sqrt()[:, None] + eps) * step_size
            p = self.shard[ids].float()
            u = u + torch.where(u * p > 0, p * decay, 0)
            self.shard[ids] = (p - u).bfloat16()
            self.exp_avg_sq[ids] = v
            self.last_event[ids] = event
