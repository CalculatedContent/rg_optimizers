"""MuonClip optimizer for unchanged upstream packed-QKV GPT-2.

The observation hook returns None and cannot change the projection output.
QK maxima cover every causal token pair, head and accumulation microbatch.
Only optimizer updates (including Q/K row and bias rescaling) change weights.
"""
import math
import torch
from optim import Muon, orthogonalize


class MuonClip(Muon):
    peak_lr = 0.02
    beta = 0.95
    weight_decay = 0.1
    rms_scale = 0.2

    def __init__(self, named_parameters, rt, model, threshold=100.0, balance=0.5):
        super().__init__(named_parameters, rt)
        if threshold <= 0 or not 0 <= balance <= 1:
            raise ValueError('Invalid QK clipping parameters')
        self.model, self.threshold, self.balance = model, threshold, balance
        self.max_logits = [None for _ in model.transformer.h]
        self.last_diagnostics = None
        self.handles = []
        for index, block in enumerate(model.transformer.h):
            self.handles.append(block.attn.c_attn.register_forward_hook(self._observer(index, block.attn)))

    def reset_qk_tracking(self):
        self.max_logits = [None for _ in self.model.transformer.h]

    def _observer(self, index, attention):
        @torch.no_grad()
        def observe(module, inputs, output):
            if not self.model.training:
                return None
            batch, length, _ = output.shape
            width, heads = attention.n_embd, attention.n_head
            q, k, _ = output.detach().split(width, dim=-1)
            with torch.autocast(output.device.type, enabled=False):
                q = q.reshape(batch, length, heads, width//heads).transpose(1, 2).float()
                k = k.reshape(batch, length, heads, width//heads).transpose(1, 2).float()
                maxima = torch.zeros(heads, device=output.device, dtype=torch.float32)
                positions = torch.arange(length, device=output.device)
                # Query tiling bounds temporary memory without approximating maxima.
                for start in range(0, length, 128):
                    scores = (q[:,:,start:start+128] @ k.transpose(-2,-1)) * (width//heads)**-0.5
                    causal = positions[None,:] <= positions[start:start+128,None]
                    scores = scores.masked_fill(~causal, float('-inf'))
                    maxima = torch.maximum(maxima, scores.amax(dim=(0,2,3)))
            self.rt.replicate(maxima)
            previous = self.max_logits[index]
            self.max_logits[index] = maxima if previous is None else torch.maximum(previous, maxima)
            return None
        return observe

    @torch.no_grad()
    def step(self, lr, beta):
        if any(value is None for value in self.max_logits):
            raise RuntimeError('MuonClip requires training QK observations for every block')
        for group in self.groups:
            params = [p for _,p in group['entries']]
            if any(p.grad is None for p in params):
                raise RuntimeError('Missing MuonClip gradient')
            grads = torch.stack([p.grad for p in params])
            if group['padded'] > len(params):
                grads = torch.cat([grads, torch.zeros((group['padded']-len(params), *group['shape']),
                                                     device=grads.device, dtype=grads.dtype)])
            self.rt.shard_matrices(grads)
            buf = group['buffer']
            buf.mul_(beta).add_(grads)
            updates = orthogonalize(grads + beta*buf)
            updates *= self.rms_scale * math.sqrt(max(group['shape']))
            self.rt.replicate(updates)
            for index,p in enumerate(params):
                p.mul_(1-lr*self.weight_decay)
                p.add_(updates[index].to(p.dtype)*(-lr))

    @torch.no_grad()
    def clip_qk(self):
        gammas = []
        for block,maximum in zip(self.model.transformer.h, self.max_logits):
            if maximum is None:
                raise RuntimeError('Missing QK observation')
            gamma = torch.clamp(self.threshold / maximum.clamp_min(self.threshold), max=1.0)
            attention = block.attn
            heads, width = attention.n_head, attention.n_embd
            weights = attention.c_attn.weight.view(3,heads,width//heads,width)
            bias = attention.c_attn.bias.view(3,heads,width//heads)
            for index,exponent in ((0,self.balance),(1,1-self.balance)):
                scale = gamma.pow(exponent)
                weights[index].mul_(scale[:,None,None])
                bias[index].mul_(scale[:,None])
            gammas.append(gamma)
        self.last_diagnostics = dict(max_logit=torch.stack(self.max_logits).amax(),
                                     min_gamma=torch.stack(gammas).amin(),
                                     clipped_heads=sum((g<1).sum() for g in gammas))
        self.reset_qk_tracking()

    def state_dict(self):
        return dict(groups=super().state_dict(), threshold=self.threshold, balance=self.balance)

    def load_state_dict(self, state):
        if state['threshold'] != self.threshold or state['balance'] != self.balance:
            raise ValueError('MuonClip configuration mismatch')
        super().load_state_dict(state['groups'])
        self.reset_qk_tracking()
