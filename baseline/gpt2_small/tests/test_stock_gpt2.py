"""Upstream model identity, independent numerical parity, and optimizer integration."""
import ast
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
import types
import pytest
import torch
from torch import nn
import torch.nn.functional as F

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'muon_speedrun'))
import stock_model as stock
from optim import make_optimizers, apply_update, clip_gradients, orthogonalize
from runtime import Runtime


def reference():
    allowed={'NewGELU','CausalSelfAttention','MLP','Block','GPTConfig','GPT'}
    nodes=[n for n in ast.parse(stock.SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name in allowed]
    namespace=dict(torch=torch,nn=nn,F=F,math=math,dataclass=dataclass,FLASH=1)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(stock.SOURCE),'exec'),namespace)
    return types.SimpleNamespace(**namespace)


def tiny(seed=42):
    return stock.make_model(stock.GPTConfig(vocab_size=128,block_size=32,n_layer=2,n_head=3,n_embd=24),seed)


def test_exact_upstream_class_hash_shapes_and_parameter_count():
    assert stock.GPT is stock.reference.GPT
    assert hashlib.sha256(stock.SOURCE.read_bytes()).hexdigest()==stock.SOURCE_SHA256
    with torch.device('meta'): model=stock.make_model()
    assert sum(p.numel() for p in model.parameters())==124439808
    assert model.lm_head.weight is model.transformer.wte.weight
    assert len(stock.matrix_inventory(model))==51
    for block in model.transformer.h:
        assert block.attn.n_head==12
        assert block.attn.c_attn.weight.shape==(2304,768)
        assert block.attn.c_proj.weight.shape==(768,768)
        assert block.mlp.c_fc.weight.shape==(3072,768)
        assert block.mlp.c_proj.weight.shape==(768,3072)
        assert type(block.ln_1) is nn.LayerNorm
    assert stock.GPT.forward.__code__.co_filename==str(stock.SOURCE)


def test_initialization_logits_gradients_and_clipped_adamw_update_match_upstream():
    torch.set_num_threads(1)
    ref=reference(); actual=tiny(); expected=ref.GPT(ref.GPTConfig(**vars(actual.config)))
    for name,p in actual.state_dict().items():
        torch.testing.assert_close(p,expected.state_dict()[name],rtol=0,atol=0)
    x=torch.randint(128,(2,13)); y=torch.randint(128,(2,13))
    logits,loss=actual(x,y); ref_logits,ref_loss=expected(x,y)
    torch.testing.assert_close(logits,ref_logits,rtol=0,atol=0)
    loss.backward(); ref_loss.backward()
    for p,q in zip(actual.parameters(),expected.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    rt=Runtime('cpu'); muon,adam=make_optimizers(actual,rt,'adamw')
    expected_adam=torch.optim.AdamW([
        dict(params=[p for p in expected.parameters() if p.ndim>=2],weight_decay=.1),
        dict(params=[p for p in expected.parameters() if p.ndim<2],weight_decay=0)],
        lr=.0006/700,betas=(.9,.95),eps=1e-8,foreach=False,fused=False)
    torch.testing.assert_close(clip_gradients(actual),torch.nn.utils.clip_grad_norm_(expected.parameters(),1.),rtol=1e-5,atol=1e-6)
    apply_update(muon,adam,rt,0); expected_adam.step()
    for p,q in zip(actual.parameters(),expected.parameters()):
        torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-7)
    other=tiny(43)
    assert not torch.equal(tiny(42).transformer.wte.weight,other.transformer.wte.weight)


@pytest.mark.parametrize('kind',['muon_clip','muon','adamw'])
def test_optimizer_coverage_update_checkpoint_and_spectra(kind,tmp_path):
    torch.set_num_threads(1)
    model=tiny(); rt=Runtime('cpu'); muon,adam=make_optimizers(model,rt,kind)
    owned=[p for g in adam.param_groups for p in g['params']]
    if muon: owned += [p for g in muon.groups for _,p in g['entries']]
    assert len(owned)==len({id(p) for p in owned})==len(list(model.parameters()))
    x=torch.randint(128,(2,16)); losses=[]
    for step in range(3):
        model.zero_grad(set_to_none=False)
        _,loss=model(x,x); loss.backward(); clip_gradients(model)
        apply_update(muon,adam,rt,700+step); losses.append(float(loss.detach()))
        assert all(torch.isfinite(p).all() for p in model.parameters())
    assert losses[-1]<losses[0]
    import importlib.util
    spec=importlib.util.spec_from_file_location('upstream_runner',BASE/'muon_speedrun/run.py')
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    validation=dict(step=250,val_nll=losses[-1],val_token_error=.5,full_benchmark_evaluation=True,evaluation_tokens=10485760)
    run.save_checkpoint(tmp_path,model,muon,adam,types.SimpleNamespace(shard=1,position=0),250,
                        {'architecture':stock.ARCHITECTURE},rt,validation,float('inf'))
    saved=torch.load(tmp_path/'checkpoint_latest.pt',weights_only=False)
    restored=tiny();restored.load_state_dict(saved['model'])
    rm,ra=make_optimizers(restored,rt,kind);ra.load_state_dict(saved['adam'])
    if rm: rm.load_state_dict(saved['muon'])
    assert restored.lm_head.weight is restored.transformer.wte.weight
    torch.testing.assert_close(restored(x,x)[0],model(x,x)[0],rtol=0,atol=0)
    snapshot=torch.load(tmp_path/'tracking/snapshots/0000250.pt',weights_only=False)
    assert len(snapshot['matrices'])==12
    torch.testing.assert_close(snapshot['matrices']['L00_W_Q'],model.transformer.h[0].attn.c_attn.weight[:24])


def test_muonclip_observer_preserves_forward_backward_and_exact_causal_maxima():
    torch.set_num_threads(1)
    model=tiny(); baseline=tiny(); rt=Runtime('cpu')
    muon,_=make_optimizers(model,rt,'muon_clip')
    x=torch.randint(128,(2,13)); y=torch.randint(128,(2,13))
    projections=[]
    handle=model.transformer.h[0].attn.c_attn.register_forward_hook(lambda m,i,o:projections.append(o.detach()))
    logits,loss=model(x,y); expected,ref_loss=baseline(x,y)
    torch.testing.assert_close(logits,expected,rtol=0,atol=0)
    loss.backward();ref_loss.backward()
    for p,q in zip(model.parameters(),baseline.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    maxima=[]
    for data in (projections[0],):
        q,k,_=data.chunk(3,dim=-1)
        q=q.reshape(2,13,3,8).transpose(1,2);k=k.reshape(2,13,3,8).transpose(1,2)
        scores=q@k.transpose(-2,-1)/math.sqrt(8)
        scores=scores.masked_fill(~torch.ones(13,13,dtype=torch.bool).tril(),float('-inf'))
        maxima.append(scores.amax(dim=(0,2,3)).clamp_min(0))
    torch.testing.assert_close(muon.max_logits[0],maxima[0])
    before=muon.max_logits[0].clone();model(x,y)
    torch.testing.assert_close(muon.max_logits[0],before)
    model.eval();model(x,y);torch.testing.assert_close(muon.max_logits[0],before)
    handle.remove()


def test_muonclip_update_math_and_forced_head_clipping_include_biases():
    model=tiny();rt=Runtime('cpu');muon,adam=make_optimizers(model,rt,'muon_clip')
    # Independent one-step optimizer equation, followed by forced known QK scales.
    torch.manual_seed(11)
    before={}
    for group in muon.groups:
        for name,p in group['entries']:
            p.grad=torch.randn_like(p);before[name]=p.detach().clone()
    muon.max_logits=[torch.tensor([400.,100.,25.]) for _ in model.transformer.h]
    with torch.no_grad():
        for block in model.transformer.h: block.attn.c_attn.bias.fill_(2.)
    muon.step(.02,.95)
    for group in muon.groups:
        for name,p in group['entries']:
            update=orthogonalize(p.grad+.95*p.grad)
            update.mul_(.2*math.sqrt(max(p.shape)))
            expected=before[name]*(1-.02*.1)-.02*update.float()
            torch.testing.assert_close(p,expected,rtol=2e-4,atol=2e-6)
    packed=[block.attn.c_attn.weight.detach().clone().view(3,3,8,24) for block in model.transformer.h]
    muon.clip_qk()
    for block,old in zip(model.transformer.h,packed):
        new=block.attn.c_attn.weight.view(3,3,8,24)
        torch.testing.assert_close(new[0,0],old[0,0]*.5)
        torch.testing.assert_close(new[1,0],old[1,0]*.5)
        torch.testing.assert_close(new[2],old[2],rtol=0,atol=0)
        torch.testing.assert_close(new[:2,1:],old[:2,1:],rtol=0,atol=0)
        bias=block.attn.c_attn.bias.view(3,3,8)
        assert torch.all(bias[:2,0]==1) and torch.all(bias[2]==2)
    assert float(muon.last_diagnostics['min_gamma'])==.25
    assert int(muon.last_diagnostics['clipped_heads'])==2
    assert all(m is None for m in muon.max_logits)
    with pytest.raises(RuntimeError,match='observations'): muon.step(.02,.95)


def test_tpu_training_requires_matching_upstream_optimizer_preflight(tmp_path):
    import importlib.util,json
    spec=importlib.util.spec_from_file_location('preflight_runner',BASE/'muon_speedrun/run.py')
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    a=types.SimpleNamespace(root=tmp_path,optimizer='muon_clip',microbatch=64,attention='flash',seed=42,device='tpu')
    proof=dict(status='passed',identity=run.preflight_identity(a))
    (tmp_path/'MODEL_PREFLIGHT.json').write_text(json.dumps(proof))
    assert run.require_preflight(a)==proof
    for field,value in [('optimizer','adamw'),('device','cpu'),('seed',1337),('microbatch',32)]:
        bad=types.SimpleNamespace(**{**vars(a),field:value})
        with pytest.raises(RuntimeError,match='mismatched'): run.require_preflight(bad)


def test_attention_hardware_adapter_does_not_patch_torch_or_model_definition():
    original=F.scaled_dot_product_attention
    seen=[]
    def attention(q,k,v):
        seen.append(q.shape)
        return original(q,k,v,is_causal=True)
    model=tiny(); x=torch.randint(128,(2,8))
    expected=model(x,x)[0]
    try:
        stock.configure_attention(attention)
        actual=model(x,x)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert len(seen)==2
        assert F.scaled_dot_product_attention is original
        assert model.forward.__func__ is stock.reference.GPT.forward
    finally: stock.configure_attention()
