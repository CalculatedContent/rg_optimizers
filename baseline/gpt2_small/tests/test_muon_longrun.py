"""State replay, sample order, long scheduler, real WW fields, and lease safety."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import random
import sys
import types
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]/'muon_longrun'
sys.path.insert(0,str(HERE))
from common import Schedule, BATCH_TOKENS, ww_due, val_due, adapt_tracking, PERMANENT
from long_data import Stream, FineWeb, corpus_metadata
import checkpoint
import train_long
import track_long
from model import GPT, GPTConfig, CastedLinear
from runtime import Runtime
from optim import make_optimizers, apply_update


def test_schedule_exposure_and_measurement_boundaries():
    s=Schedule()
    assert 25000*BATCH_TOKENS==13107200000
    assert s.factor(0)==s.factor(3000)==s.factor(17500)==1
    assert s.factor(21250)==.5 and s.factor(24999)==1/7500 and s.factor(25000)==0
    assert s.phase(17499)=='main' and s.phase(17500)=='cooldown'
    assert all(val_due(x) for x in (0,1000,2000,3000,5000,7500,10000,12500,15000,17500,20000,22500,25000))
    assert all(ww_due(x) and ww_due(x,True) for x in (0,100,250,500,750,1000,1500,2000,2500,3000,17500,25000))
    assert ww_due(5000) and not ww_due(5000,True) and ww_due(6000,True)
    assert not adapt_tracking(1000,.2,1.3,1.28,{})
    assert adapt_tracking(3000,.11,1.3,1.28,{})
    assert adapt_tracking(4000,.01,1.5,1.28,{'status':'measuring'})
    assert not adapt_tracking(4000,.01,1.5,1.28,{'status':'waiting'})


def source():
    arrays={'fineweb_train_000000.bin':np.arange(33)%128,
            'fineweb_train_000001.bin':(np.arange(49)+17)%128}
    return types.SimpleNamespace(manifest={'files':dict.fromkeys(arrays)},array=arrays.__getitem__)


def model():
    m=GPT(GPTConfig(vocab_size=128,n_layer=2,n_head=2,n_embd=16)).bfloat16()
    for layer in m.modules():
        if isinstance(layer,CastedLinear): layer.float()
    return m


def train_step(m,muon,adam,stream,rt,step,s):
    m.zero_grad(set_to_none=False); x,y=stream.next_batch(); loss=m(x,y); loss.backward()
    norm=train_long.gradient_norm(m)
    train_long.apply_update(muon,adam,rt,step,s)
    return loss.detach().clone(),norm.detach().clone()


def equal(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    else: assert a==b


def test_full_state_resume_matches_uninterrupted_across_shard_boundary(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(1337); np.random.seed(1337); random.seed(1337)
    rt=Runtime('cpu'); s=Schedule(); m=model(); mu,ad=make_optimizers(m,rt); stream=Stream(source(),2,8)
    for i in range(2): train_step(m,mu,ad,stream,rt,i,s)
    payload=checkpoint.state(m,mu,ad,stream,rt,2,s,{'test':'fixed'})
    checkpoint.save(tmp_path,payload)
    expected_loss=[]
    for i in range(2,9): expected_loss.append(train_step(m,mu,ad,stream,rt,i,s))
    expected=checkpoint.state(m,mu,ad,stream,rt,9,s,{'test':'fixed'})
    expected_rng=(torch.rand(3),np.random.rand(3),random.random())
    # Reinitialize unrelated RNG/model/optimizers, then restore from disk.
    torch.manual_seed(9); m2=model(); mu2,ad2=make_optimizers(m2,rt); stream2=Stream(source(),2,8)
    saved=torch.load(tmp_path/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    assert checkpoint.restore(saved,m2,mu2,ad2,stream2,rt,s,{'test':'fixed'})==2
    for i in range(2,9): equal(train_step(m2,mu2,ad2,stream2,rt,i,s),expected_loss[i-2])
    actual=checkpoint.state(m2,mu2,ad2,stream2,rt,9,s,{'test':'fixed'})
    equal(actual,expected); equal(expected_rng,(torch.rand(3),np.random.rand(3),random.random()))
    with pytest.raises(RuntimeError): checkpoint.restore(saved,m2,mu2,ad2,stream2,rt,Schedule(25000,900),{'test':'fixed'})
    with pytest.raises(RuntimeError): checkpoint.restore(saved,m2,mu2,ad2,stream2,rt,s,{'test':'changed'})


def test_long_update_matches_original_at_peak_and_cooldown_is_stretched():
    torch.manual_seed(33); rt=Runtime('cpu'); a=model(); b=model(); b.load_state_dict(a.state_dict())
    am,aa=make_optimizers(a,rt); bm,ba=make_optimizers(b,rt)
    for step in range(4):
        x=torch.randint(128,(2,8)); y=torch.randint(128,(2,8))
        for m in (a,b): m.zero_grad(set_to_none=False); m(x,y).backward()
        before=[p.grad.clone() for p in b.parameters()]
        norm=train_long.gradient_norm(b)
        assert torch.isfinite(norm)
        for p,g in zip(b.parameters(),before): equal(p.grad,g)
        apply_update(am,aa,rt,step)
        train_long.apply_update(bm,ba,rt,step,Schedule())
        equal(a.state_dict(),b.state_dict()); equal(am.state_dict(),bm.state_dict()); equal(aa.state_dict(),ba.state_dict())


def test_same_process_startup_replay_gate(tmp_path):
    torch.manual_seed(1337); rt=Runtime('cpu'); s=Schedule(); m=model(); mu,ad=make_optimizers(m,rt)
    stream=Stream(source(),2,8); identity={'test':'startup'}
    checkpoint.save(tmp_path,checkpoint.state(m,mu,ad,stream,rt,0,s,identity))
    for step in range(2): train_long.update(m,mu,ad,stream,rt,step,s,True)
    train_long.verify_startup_replay(tmp_path,m,mu,ad,stream,rt,s,identity)
    assert json.loads((tmp_path/'RESUME_PARITY.json').read_text())['exact_equal']


def test_cursor_preserves_original_order_and_cycles():
    import long_data
    data=source(); original=long_data.reference.TrainStream(data,2,8); actual=Stream(data,2,8)
    for _ in range(14): equal(original.next_batch(),actual.next_batch())
    assert actual.cycles==2
    restored=Stream(data,2,8); restored.load_state_dict(actual.state_dict())
    equal(actual.next_batch(),restored.next_batch())
    with pytest.raises(ValueError): restored.load_state_dict({**actual.state_dict(),'position':3})


def test_entire_pinned_corpus_and_exposure(tmp_path):
    metadata=corpus_metadata(FineWeb(tmp_path,0))
    assert metadata['train_shards']==103
    assert metadata['corpus_tokens']==10255324043
    assert 1.27 < 25000*BATCH_TOKENS/metadata['usable_tokens_per_epoch'] < 1.29


def test_atomic_checkpoint_retention(tmp_path):
    for step in [0,2500,3000,5000,7500,10000,12500,15000,17500,20000,22500,25000]:
        checkpoint.save(tmp_path,{'step':step,'tokens_seen':step*BATCH_TOKENS,
                                  'next_update':Schedule().values(step),'scheduler':asdict(Schedule())})
    steps={int(p.stem.split('_')[1]) for p in (tmp_path/'checkpoints').glob('*.pt')}
    assert steps==PERMANENT|{20000,22500}
    assert torch.load(tmp_path/'checkpoint_latest.pt',weights_only=False)['step']==25000
    assert not list(tmp_path.rglob('*.tmp'))


def launcher():
    spec=importlib.util.spec_from_file_location('long_launcher',HERE/'launch.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_live_lease_requires_explicit_expiry_and_margin():
    mod=launcher(); now=1000000000
    expiry=mod.dt.datetime.fromtimestamp(now+14*3600,mod.dt.timezone.utc).isoformat()
    node={'state':'READY','acceleratorType':'v5litepod-8','schedulingConfig':{'terminationTimestamp':expiry}}
    queue={'name':'projects/p/locations/z/queuedResources/q','state':{'state':'ACTIVE'},
           'createTime':'2001-01-01T00:00:00Z','runDuration':{'maxRunDuration':'172800s'}}
    assert mod.lease_from(node,queue,now)['remaining_hours']==14
    with pytest.raises(RuntimeError): mod.lease_from(node,queue,now+2*3600)
    with pytest.raises(RuntimeError): mod.lease_from({**node,'schedulingConfig':{}},queue,now)
    with pytest.raises(RuntimeError): mod.lease_from({**node,'state':'STOPPED'},queue,now)


def test_real_ww_zero_and_randomized_fields(tmp_path):
    torch.manual_seed(3)
    payload={'matrices':{'L00_W_Q':torch.randn(64,64),'L00_W_O':torch.zeros(64,64)},
             'step':0,'tokens_seen':0,'validation':{'step':0,'val_nll':11.},'run_id':'test'}
    path=tmp_path/'0000000.pt'; torch.save(payload,path)
    out=track_long.measure(path); rows={r['matrix_name']:r for r in out['layers']}
    assert out['summary']['matrix_count']==2
    assert rows['L00_W_O']['matrix_rank']==0 and rows['L00_W_O']['alpha_raw'] is None
    assert rows['L00_W_O']['status']=='zero_matrix'
    assert rows['L00_W_Q']['randomized_status']=='available'
    for key in ('alpha_raw','alpha_clip_xmax','alpha_weighted','log_alpha_norm','matrix_rank','max_rand_eval'):
        assert key in rows['L00_W_Q']
    assert out['summary']['weightwatcher_seconds']>0


def test_captured_cloud_error_is_visible(monkeypatch,capsys):
    mod=launcher()
    def denied(*a,**k):
        raise mod.subprocess.CalledProcessError(1,a[0],stderr='Permission denied: tpu.nodes.get')
    monkeypatch.setattr(mod.subprocess,'run',denied)
    with pytest.raises(RuntimeError,match='No trainer started'): mod.live_lease()
    assert 'Permission denied: tpu.nodes.get' in capsys.readouterr().err


@pytest.mark.parametrize('uid',[0,1000])
def test_direct_mode_does_not_ssh_or_change_credentials(monkeypatch,uid):
    mod=launcher(); calls=[]
    monkeypatch.setattr(mod.os,'geteuid',lambda:uid)
    monkeypatch.setattr(mod.subprocess,'run',lambda c:calls.append(c) or types.SimpleNamespace(returncode=0))
    command=['sudo','python3','-c','code','start','--on-tpu']
    assert mod.dispatch(command,True)==0
    assert calls==[command if uid else command[1:]]


def test_direct_start_requires_matching_live_node(monkeypatch):
    import io
    mod=launcher()
    monkeypatch.setattr(mod.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'10.0.0.7'))
    mod.verify_local_host({'node_internal_ips':['10.0.0.7']})
    with pytest.raises(RuntimeError,match='does not match'):
        mod.verify_local_host({'node_internal_ips':['10.0.0.8']})
