"""CPU parity tests against the pinned record, plus target/checkpoint guard tests."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]/'muon_speedrun'
sys.path.insert(0, str(BASE))
import model as port
from optim import Muon, make_optimizers, apply_update, schedule
from runtime import Runtime
from data import FineWeb, required_shards


def original():
    tree = ast.parse((BASE/'vendor/record_source.py').read_text())
    # Load only definitions. The original top-level CUDA/DDP launcher must not run.
    accepted = {'Rotary','apply_rotary_emb','CastedLinear','CausalSelfAttention','MLP','Block',
                'GPTConfig','GPT','zeropower_via_newtonschulz5','Hyperparameters'}
    nodes = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef,ast.ClassDef)) and n.name in accepted:
            if isinstance(n,ast.FunctionDef):
                n.decorator_list = []
            nodes.append(n)
    import dataclasses
    namespace = dict(torch=torch, nn=torch.nn, F=F, dataclass=dataclasses.dataclass)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'record-definitions','exec'),namespace)
    return types.SimpleNamespace(**namespace)


def small(module):
    result = module.GPT(module.GPTConfig(vocab_size=128,n_layer=2,n_head=2,n_embd=16)).bfloat16()
    for layer in result.modules():
        if isinstance(layer,module.CastedLinear):
            layer.float()
    return result


def test_model_forward_backward_matches_pinned_record():
    torch.set_num_threads(1)
    source = original()
    torch.manual_seed(1337)
    expected = small(source)
    actual = small(port)
    actual.load_state_dict(expected.state_dict())
    # Un-zero output projections to exercise every block, not just the first head update.
    with torch.no_grad():
        for model in (expected,actual):
            gen = torch.Generator().manual_seed(8)
            for name,p in model.named_parameters():
                if 'c_proj' in name or 'lm_head' in name:
                    p.copy_(torch.randn(p.shape,generator=gen)*.02)
    x = torch.randint(128,(4,8)); y = torch.randint(128,(4,8))
    a,b = expected(x,y),actual(x,y)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.backward(); b.backward()
    for p,q in zip(expected.parameters(),actual.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_batched_muon_matches_record_updates_and_restores_state():
    torch.set_num_threads(1)
    source = original()
    rt = Runtime('cpu')
    torch.manual_seed(5)
    params = [torch.nn.Parameter(torch.randn(shape)) for shape in ((8,8),(8,8),(16,8),(8,16))]
    expected = [p.detach().clone() for p in params]
    buffers = [torch.zeros_like(p) for p in params]
    opt = Muon([(str(i),p) for i,p in enumerate(params)],rt)
    for step in range(4):
        beta = .85+.1*step/500
        grads = [torch.randn_like(p) for p in params]
        for p,g in zip(params,grads): p.grad = g.clone()
        opt.step(.04,beta)
        for i,(p,g) in enumerate(zip(expected,grads)):
            buffers[i].mul_(beta).add_(g)
            update = source.zeropower_via_newtonschulz5(g+beta*buffers[i],steps=5)
            update *= max(1,update.size(0)/update.size(1))**.5
            p.add_(update.to(p.dtype),alpha=-.04)
        for a,b in zip(params,expected):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=3e-7)
        state = opt.state_dict()
        opt = Muon([(str(i),p) for i,p in enumerate(params)],rt)
        opt.load_state_dict(state)


def test_complete_mixed_dtype_recipe_learns_on_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = small(port)
    rt = Runtime('cpu')
    muon,adam = make_optimizers(model,rt)
    x = torch.randint(128,(4,8)); y = x.clone()
    losses=[]
    for step in range(6):
        model.zero_grad(set_to_none=False)
        loss = model(x,y)
        assert torch.isfinite(loss)
        loss.backward()
        apply_update(muon,adam,rt,step)
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(p).all() for p in model.parameters())


def runner():
    spec = importlib.util.spec_from_file_location('muon_recipe_runner',BASE/'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_or_nonfinite_validation_cannot_pass():
    run = runner()
    good = dict(full_benchmark_evaluation=True,evaluation_tokens=10485760,val_nll=3.28)
    assert run.target_met(good)
    for change in (dict(full_benchmark_evaluation=False),dict(evaluation_tokens=1048576),
                   dict(val_nll=float('nan')),dict(val_nll=3.29)):
        assert not run.target_met({**good,**change})


def test_best_checkpoint_survives_latest_replacement(tmp_path):
    run = runner(); rt=Runtime('cpu')
    model=small(port); muon,adam=make_optimizers(model,rt)
    stream=types.SimpleNamespace(shard=1,position=131072)
    validation=dict(full_benchmark_evaluation=True,evaluation_tokens=10485760,val_nll=3.5)
    run.save_checkpoint(tmp_path,model,muon,adam,stream,125,{},rt,validation,4.)
    run.save_checkpoint(tmp_path,model,muon,adam,stream,250,{},rt,{**validation,'val_nll':3.6},3.5)
    best=torch.load(tmp_path/'checkpoint_best.pt',weights_only=False)
    latest=torch.load(tmp_path/'checkpoint_latest.pt',weights_only=False)
    assert best['step']==125 and latest['step']==250
    assert latest['data_cursor']==dict(shard=1,position=131072)
    assert latest['muon'] and 'adam' in latest and 'rng' in latest


def test_schedule_and_full_corpus_budget(tmp_path):
    source=original().Hyperparameters()
    assert source.num_iterations==3000 and source.warmdown_iters==900
    assert schedule(0)==schedule(2099)==1.
    assert schedule(2550)==.5 and schedule(3000)==0.
    files=FineWeb(tmp_path,0)
    for micro in (64,128):
        names=required_shards(files,micro)
        assert len(names)==17 # validation + 16 distinct training shards
        usable=sum(((files.manifest['files'][n]['size']-1024)//2-1)//(micro*1024)*(micro*1024)
                   for n in names if '_train_' in n)
        assert usable >= 3000*524288


def test_launcher_does_not_duplicate_active_run(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('muon_recipe_launcher',BASE/'cloudshell.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pointer=tmp_path/'MUON_SPEEDRUN_LATEST.json'; pointer.write_text(json.dumps({'unit':'active.service'}))
    monkeypatch.setattr(module,'BASE',tmp_path); monkeypatch.setattr(module,'LATEST',pointer)
    monkeypatch.setattr(module.os,'geteuid',lambda:0)
    monkeypatch.setattr(module.os.path,'ismount',lambda p:True)
    monkeypatch.setattr(module,'active',lambda u:True)
    seen=[]; monkeypatch.setattr(module,'status_remote',lambda:seen.append(True))
    monkeypatch.setattr(module,'run',lambda *a,**k:pytest.fail('must not launch twice'))
    module.start_remote('a'*40)
    assert seen==[True]
