"""One-process SPMD runtime; TPU flash attention has explicit scale and sharding."""
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class Runtime:
    def __init__(self, device='tpu', cache=None):
        self.tpu = device == 'tpu'
        if self.tpu:
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr
            import torch_xla.distributed.spmd as xs
            xr.use_spmd()
            if cache is not None:
                xr.initialize_cache(str(cache), readonly=False)
            if xr.global_runtime_device_count() != 8 or xr.addressable_runtime_device_count() != 8:
                raise RuntimeError('Requires one host with eight TPU chips')
            self.xm, self.xs = xm, xs
            self.mesh = xs.Mesh(np.arange(8), (8,), ('data',))
            self.device = xm.xla_device()
        else:
            self.device = torch.device('cpu')

    def scalar(self, value):
        return torch.tensor(value, dtype=torch.float32).to(self.device) if self.tpu else value

    def put(self, value):
        value = value.to(self.device)
        if self.tpu:
            self.xs.mark_sharding(value, self.mesh, ('data',)+(None,)*(value.ndim-1))
        return value

    def replicate(self, value):
        if self.tpu:
            self.xs.mark_sharding(value, self.mesh, (None,)*value.ndim)

    def shard_matrices(self, value):
        if self.tpu:
            self.xs.mark_sharding(value, self.mesh, ('data', None, None))

    def step(self, wait=False):
        if self.tpu:
            self.xm.mark_step()
            if wait:
                self.xm.wait_device_ops()

    def attention(self, backend):
        if backend == 'flash':
            if not self.tpu:
                raise ValueError('TPU flash attention requires TPU')
            from torch_xla.experimental.custom_kernel import flash_attention
            def flash(q, k, v):
                return flash_attention(q, k, v, causal=True, sm_scale=q.shape[-1]**-0.5,
                                       partition_spec=('data', None, None, None), mesh=self.mesh)
            return flash
        return lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)


def attention_check(root, microbatch=64):
    """Check forward/backward at the actual head/sequence shape in an isolated child."""
    rt = Runtime(cache=Path(root)/'xla-cache')
    from torch_xla.experimental.custom_kernel import jax_import_guard
    jax_import_guard()
    import jax
    gen = torch.Generator().manual_seed(43)
    cpu = [torch.randn((microbatch, 6, 1024, 128), generator=gen).bfloat16() for _ in range(3)]
    upstream = torch.randn(cpu[0].shape, generator=gen).bfloat16()
    results = []
    for kind in ('math', 'flash'):
        values = [rt.put(t).detach().requires_grad_() for t in cpu]
        output = rt.attention(kind)(*values)
        (output.float()*rt.put(upstream).float()).sum().backward()
        rt.step(wait=True)
        results.append([output.detach().cpu()] + [t.grad.detach().cpu() for t in values])
    errors = []
    for a, b in zip(*results):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError('Nonfinite attention check')
        relative = float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12))
        errors.append(relative)
        if relative > .03:
            raise RuntimeError('TPU flash/math relative L2 disagreement: '+str(relative))
    result = {'status':'passed', 'relative_l2_output_dq_dk_dv':errors, 'jax':jax.__version__,
              'global_batch':microbatch, 'batch_per_chip':microbatch//8}
    (Path(root)/'ATTENTION_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
