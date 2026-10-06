"""XLA CPU lowering tests, not hardware qualification. Run in a fresh CPU PJRT process."""
import os
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_xla_anvil_optimizer_and_tail():
    if os.environ.get('PJRT_DEVICE')!='CPU':
        pytest.skip('Set PJRT_DEVICE=CPU to run XLA lowering qualification')
    torch_xla=pytest.importorskip('torch_xla')
    from tpu_port.optimizer import Optimizer,ADAM,BANKS,cascade
    from tpu_port.runtime import schedule
    from tpu_port.tail import TailAverages
    from torch_xla.backends import set_mat_mul_precision
    set_mat_mul_precision('highest')
    device=torch_xla.device()
    # Rank-one gradients exposed an unstable BF16 polynomial lowering in XLA.
    for shape in [(2,4,2),(2,2,4)]:
        gradient=torch.ones(shape,dtype=torch.bfloat16)
        cpu=cascade(gradient,torch.zeros(2,*shape),torch.tensor(.85),torch.tensor(.85),torch.tensor(1.))
        xla=cascade(gradient.to(device),torch.zeros(2,*shape,device=device),
                    torch.tensor(.85,device=device),torch.tensor(.85,device=device),torch.tensor(1.,device=device)).cpu()
        assert torch.isfinite(xla).all()
        torch.testing.assert_close(xla,cpu,atol=.004,rtol=.02)
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in (*ADAM,*BANKS):
                shape=(24,4,2) if name=='mlp_bank' else (2,4,2) if name in BANKS else (2,4) if name=='lm_head' else (4,2)
                p=torch.nn.Parameter(torch.full(shape,.1,dtype=torch.bfloat16,device=device));p.label=name
                if name in BANKS:p.reshape=shape
                if name=='mlp_bank':p.frozen_matrices=[14,15]
                self.register_parameter(name,p)
    model=Fixture();sched=schedule();opt=Optimizer(model,sched);tail=TailAverages(opt.params)
    for step in [0,1,514,515,965,967,1175,1192,1193]:
        for p in model.parameters():p.grad=torch.ones_like(p)
        opt.step(step);tail.tick(step);torch_xla.sync(wait=True)
    tail.ship();torch_xla.sync(wait=True)
    assert all(torch.isfinite(p.cpu()).all() for p in model.parameters())
