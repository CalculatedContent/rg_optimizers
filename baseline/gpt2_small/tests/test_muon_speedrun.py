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
    for micro in (32,64,128):
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


def test_pallas_install_is_pinned_and_isolated(tmp_path):
    spec=importlib.util.spec_from_file_location('pallas_dependency_setup',BASE/'pallas_dependencies.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    target=tmp_path/'pallas-deps'
    command=module.install_command(target)
    assert '--no-deps' in command and '--only-binary=:all:' in command
    assert command[command.index('--target')+1]==str(target)
    assert 'jax==0.4.38' in command and 'jaxlib==0.4.38' in command
    assert not any(p.startswith(('torch==','torch_xla==','libtpu==','numpy==','scipy==')) for p in command)


@pytest.mark.parametrize('attention',['flash','auto'])
def test_flash_failure_never_starts_training(monkeypatch,tmp_path,attention):
    spec=importlib.util.spec_from_file_location('muon_worker_strict_attention',BASE/'worker.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import time
    monkeypatch.setattr(sys,'argv',['worker.py',str(tmp_path),str(time.time()+10800),'--attention',attention])
    phases=[]
    def bounded(command,seconds,root,label,watch=False):
        phases.append((label,command))
        rc=1 if 'forward/backward' in label else 0
        return {'exit_code':rc,'timed_out':False,'phase':label}
    monkeypatch.setattr(module,'bounded',bounded)
    monkeypatch.setenv('PYTHONPATH','test-original')
    assert module.main()==1
    assert not any('3,000-update' in label for label,_ in phases)
    assert json.loads((tmp_path/'RUN_STATUS.json').read_text())['status']=='failed'
    assert 'fallback' in json.loads((tmp_path/'status.json').read_text())['error']


def test_worker_defaults_use_smaller_microbatch_and_verified_flash(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('muon_worker_defaults',BASE/'worker.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import time
    monkeypatch.setattr(sys,'argv',['worker.py',str(tmp_path),str(time.time()+10800)])
    phases=[]
    def bounded(command,seconds,root,label,watch=False):
        phases.append((label,command))
        if '3,000-update' in label:
            (root/'status.json').write_text(json.dumps(dict(status='target_reached',target_met=True,step=2875)))
        return {'exit_code':0,'timed_out':False,'phase':label}
    monkeypatch.setattr(module,'bounded',bounded)
    monkeypatch.setattr(module,'start_tracking',lambda *args:object())
    monkeypatch.setattr(module,'finish_tracking',lambda *args:{'status':'complete'})
    monkeypatch.setenv('PYTHONPATH','test-original')
    assert module.main()==0
    training=next(command for label,command in phases if '3,000-update' in label)
    assert training[training.index('--microbatch')+1]=='64'
    assert training[training.index('--attention')+1]=='flash'
    assert phases[0][0]=='pinned Pallas dependencies'


def test_token_error_uses_the_same_logits_without_changing_loss_or_gradients():
    model = small(port)
    with torch.no_grad():
        model.lm_head.weight.normal_(std=.02)
    x = torch.randint(128, (2, 8)); y = torch.randint(128, (2, 8))
    captured = []
    hook = model.lm_head.register_forward_hook(lambda module, args, out:captured.append(out.detach()))
    loss = model(x, y)
    loss.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    measured, errors = model(x, y, return_token_errors=True)
    measured.backward()
    hook.remove()
    torch.testing.assert_close(loss, measured, rtol=0, atol=0)
    for old, p in zip(gradients, model.parameters()):
        torch.testing.assert_close(old, p.grad, rtol=0, atol=0)
    logits = (30 * torch.tanh(captured[-1] / 30)).float()
    assert int(errors) == int((logits.argmax(-1) != y).sum())


def test_evaluation_pairs_exact_token_count_with_unchanged_nll(monkeypatch,tmp_path):
    import numpy as np
    import time
    run=runner(); model=small(port); rt=Runtime('cpu')
    monkeypatch.setattr(run,'VAL_TOKENS',2048)
    tokens=np.arange(2049,dtype=np.int64)%128
    # Zero-initialized head predicts token zero everywhere. 1/128 targets are zero.
    row=run.evaluate(model,tokens,rt,tmp_path,125,time.time()+60,1,time.time())
    assert row['evaluation_tokens']==2048
    assert row['val_error_count']==2032
    assert row['val_token_error']==2032/2048
    assert row['val_accuracy']==16/2048
    assert row['val_nll']==pytest.approx(float(torch.tensor(128.).log()),abs=1e-6)
    assert model.training


def test_spectral_snapshot_is_immutable_and_paired_with_validation(tmp_path):
    import tracking
    rt=Runtime('cpu'); model=small(port); muon,adam=make_optimizers(model,rt)
    stream=types.SimpleNamespace(shard=1,position=0)
    v=dict(step=125,full_benchmark_evaluation=True,evaluation_tokens=10485760,
           val_nll=3.5,val_token_error=.6)
    state=torch.get_rng_state().clone()
    runner().save_checkpoint(tmp_path,model,muon,adam,stream,125,{},rt,v,4.)
    assert torch.equal(state,torch.get_rng_state())
    path=tmp_path/'tracking/snapshots/0000125.pt'
    before=path.read_bytes()
    with torch.no_grad():
        model.transformer.h[0].attn.c_q.weight.add_(1)
    runner().save_checkpoint(tmp_path,model,muon,adam,stream,250,{},rt,{**v,'step':250},4.)
    assert path.read_bytes()==before
    payload=torch.load(path,weights_only=False)
    assert payload['validation']==v and payload['step']==125
    assert len(payload['matrices'])==12
    assert set(payload['matrices'])=={f'L{i:02d}_W_{role}' for i in range(2) for role in tracking.ROLES.values()}


def test_weightwatcher_raw_alpha_never_falls_back_to_clipped():
    import tracking
    frame=types.SimpleNamespace(to_dict=lambda orient:[
        dict(longname='L00_W_Q',status='success',alpha=1.9,raw_alpha=float('nan')),
        dict(longname='L00_W_K',status='failed',alpha=1.8,raw_alpha=1.7)])
    rows=tracking.normalize_rows(frame,['L00_W_Q','L00_W_K','L00_W_V'],{'step':125,'val_token_error':.6})
    assert len(rows)==3 and all(r['alpha_raw'] is None for r in rows)
    assert rows[0]['alpha_clip_xmax']==1.9
    assert rows[1]['alpha_clip_xmax'] is None
    assert rows[2]['status']=='not_returned'
    s=tracking.summary(rows,{'step':125})
    assert s['alpha_raw_valid_count']==0 and s['alpha_raw_mean'] is None
    assert s['alpha_clip_xmax_valid_count']==1


def test_replace_stops_only_the_recorded_muon_service(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('muon_restart_test',BASE/'cloudshell.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pointer=tmp_path/'pointer.json'
    unit='rg-muon-speedrun-20261005-020737.service'
    pointer.write_text(json.dumps({'unit':unit}))
    monkeypatch.setattr(module,'LATEST',pointer)
    monkeypatch.setattr(module,'active',lambda name:False)
    calls=[]; monkeypatch.setattr(module,'run',lambda command,**kwargs:calls.append(command))
    module.stop_current()
    assert calls==[['systemctl','stop',unit]]
    pointer.write_text(json.dumps({'unit':'unrelated.service'}))
    with pytest.raises(RuntimeError,match='Unexpected service'):
        module.stop_current()
    assert len(calls)==1


def test_real_weightwatcher_pairs_snapshot_and_writes_tables(tmp_path):
    pytest.importorskip('weightwatcher')
    import tracking, time
    validation=dict(step=125,val_nll=4.5,val_token_error=.8,evaluation_tokens=10485760,
                    full_benchmark_evaluation=True)
    matrices={f'transformer.h.0.{suffix}.weight':torch.randn(64,64) for suffix in tracking.ROLES}
    tracking.queue_snapshot(tmp_path,dict(step=125,tokens_seen=65536000,config={'n_layer':1},
                            model=matrices,validation=validation,manifest={}))
    (tmp_path/'tracking/TRAINING_DONE').touch()
    assert tracking.watch(tmp_path,time.time()+60)==0
    result=json.loads((tmp_path/'tracking/measurements/0000125.json').read_text())
    assert len(result['layers'])==6
    assert all(row['step']==125 and row['val_token_error']==.8 for row in result['layers'])
    assert result['summary']['alpha_raw_valid_count']==6
    assert (tmp_path/'tracking/layers.csv').is_file()
    assert (tmp_path/'tracking/summary.csv').is_file()
    assert json.loads((tmp_path/'TRACKING_STATUS.json').read_text())['status']=='complete'


def test_ssh_retry_cannot_replace_the_run_it_just_launched(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('muon_idempotent_restart',BASE/'cloudshell.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pointer=tmp_path/'pointer.json'; pointer.write_text(json.dumps({'launch_id':'same-request'}))
    monkeypatch.setattr(module,'BASE',tmp_path); monkeypatch.setattr(module,'LATEST',pointer)
    monkeypatch.setattr(module.os,'geteuid',lambda:0)
    monkeypatch.setattr(module.os.path,'ismount',lambda p:True)
    seen=[]; monkeypatch.setattr(module,'status_remote',lambda:seen.append(True))
    monkeypatch.setattr(module,'run',lambda *a,**k:pytest.fail('retry must not stop or launch'))
    module.start_remote('a'*40,replace_current=True,launch_id='same-request')
    assert seen==[True]
