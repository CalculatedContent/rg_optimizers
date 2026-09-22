import json
from pathlib import Path
import torch

from am_runtime import baseline, make_model, reset_qk_clip_stats, apply_qk_clip

HERE=Path(__file__).resolve().parents[1]


def test_qk_clip_scales_only_excess_heads():
    cfg=json.loads((HERE/'protocol_four_head_muon_qkclip.json').read_text())
    source=baseline(cfg)
    model=make_model(source,cfg,2027,'cpu')
    attn=model.blocks[0].attn
    reset_qk_clip_stats(model)
    attn._qk_clip_max_logits=torch.tensor([25.0,100.0,400.0,1600.0])
    q0=attn.q_proj.weight.detach().clone()
    k0=attn.k_proj.weight.detach().clone()
    events=apply_qk_clip(model,100.0,0.5)
    d=q0.shape[0]//4
    assert len(events)==2
    assert torch.equal(attn.q_proj.weight[:2*d],q0[:2*d])
    assert torch.equal(attn.k_proj.weight[:2*d],k0[:2*d])
    assert torch.allclose(attn.q_proj.weight[2*d:3*d],q0[2*d:3*d]*0.5)
    assert torch.allclose(attn.k_proj.weight[2*d:3*d],k0[2*d:3*d]*0.5)
    assert torch.allclose(attn.q_proj.weight[3*d:],q0[3*d:]*0.25)
    assert torch.allclose(attn.k_proj.weight[3*d:],k0[3*d:]*0.25)
