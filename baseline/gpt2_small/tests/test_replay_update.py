import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from rg_gpt2_small import experiment as g, replay_update as replay
from rg_nanogpt_one_head.checkpoints import optimizer_state_sha256
from rg_nanogpt_one_head.muonclip import MuonClip
from test_gpt2_experiment import config, data


def script(name):
    folder=Path(__file__).parents[1]/'scripts'
    sys.path.insert(0,str(folder))
    try:
        spec=importlib.util.spec_from_file_location(name,folder/(name+'.py'))
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    finally: sys.path.pop(0)


@pytest.fixture
def saved_step(tmp_path):
    torch.set_num_threads(1)
    c=config(); c['optimizer']=config('fineweb_muonclip_baseline')['optimizer']
    c['metrics_steps']=[1]; c['finite_update_guard']=True
    d=data(tmp_path,c); out=tmp_path/'original'
    g.train(c,d,out,stop_after=1)
    return c,d,out/'checkpoints/step_000000001.pt'


def test_scalar_checks_do_not_stack_or_copy_arrays(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('stack/cat forbidden in diagnostic')
    monkeypatch.setattr(torch,'stack',forbidden); monkeypatch.setattr(torch,'cat',forbidden)
    original=torch.Tensor.cpu
    def scalar_cpu(tensor,*args,**kwargs):
        assert tensor.numel()==1
        return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',scalar_cpu)
    tensors=[('weight/test',torch.ones(4,8)),('aux/test/exp_avg_sq',torch.tensor([-1.,0.]))]
    with pytest.raises(RuntimeError,match='First invalid stage: moments'):
        replay.check_finite(tensors,tmp_path,'moments',2,'cpu')
    bad=json.loads((tmp_path/'FIRST_INVALID.json').read_text())
    assert bad['invalid_tensors'][0]['negative_second_moment'] is True
    assert bad['invalid_tensors'][0]['tensor']=='aux/test/exp_avg_sq'


def test_exact_cpu_update_replay_and_original_preserved(saved_step,tmp_path):
    c,d,checkpoint=saved_step; before=replay.sha256(checkpoint)
    reference=tmp_path/'reference'; g.train(c,d,reference,stop_after=2)
    output=tmp_path/'replay'; replay.replay(checkpoint,d,output,'cpu')
    expected=torch.load(reference/'checkpoints/step_000000002.pt',weights_only=False)
    actual=torch.load(output/'update_state.pt',weights_only=False)
    for name in expected['model']:
        torch.testing.assert_close(actual['model'][name],expected['model'][name],rtol=0,atol=0)
    assert optimizer_state_sha256(actual['optimizers'])==optimizer_state_sha256(expected['optimizers'])
    assert torch.equal(actual['next_data_rng'],expected['data_rng'])
    assert actual['diagnostic_only'] and not actual['resumable']
    assert actual['completed_roles']==['primary','auxiliary']
    assert replay.sha256(checkpoint)==before
    metrics=json.loads((output/'after_update_evaluation.json').read_text())
    original=json.loads((reference/'metrics/000000002.json').read_text())
    for key in metrics: assert metrics[key]==original[key]
    assert json.loads((output/'REPLAY_STATUS.json').read_text())['status']=='one_update_passed'


def test_invalid_primary_stops_before_auxiliary(saved_step,tmp_path,monkeypatch):
    _,d,checkpoint=saved_step; before=replay.sha256(checkpoint)
    original=MuonClip.step
    def corrupt(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        with torch.no_grad(): self.param_groups[0]['params'][0].view(-1)[0]=float('nan')
        return result
    monkeypatch.setattr(MuonClip,'step',corrupt)
    def forbidden(*args,**kwargs): raise AssertionError('Auxiliary applied after invalid primary')
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    output=tmp_path/'bad'
    with pytest.raises(RuntimeError,match='First invalid stage: after_primary'):
        replay.replay(checkpoint,d,output,'cpu')
    assert json.loads((output/'FIRST_INVALID.json').read_text())['stage']=='after_primary'
    state=torch.load(output/'update_state.pt',weights_only=False)
    assert state['completed_roles']==['primary']
    assert (output/'TPU_PORT_FAILURE.json').exists()
    assert replay.sha256(checkpoint)==before


def test_postupdate_state_survives_evaluation_failure(saved_step,tmp_path,monkeypatch):
    _,d,checkpoint=saved_step; evaluate=replay.evaluate_splits
    def broken(*args,**kwargs):
        if args[-1]=='after_update_evaluation': raise RuntimeError('Injected evaluation failure')
        return evaluate(*args,**kwargs)
    monkeypatch.setattr(replay,'evaluate_splits',broken)
    output=tmp_path/'eval-failure'
    with pytest.raises(RuntimeError,match='Injected evaluation failure'):
        replay.replay(checkpoint,d,output,'cpu')
    state=torch.load(output/'update_state.pt',weights_only=False)
    assert state['completed_roles']==['primary','auxiliary']
    assert all(torch.isfinite(value).all() for value in state['model'].values())


def test_launch_is_bounded_and_preserves_original(tmp_path,monkeypatch):
    module=script('replay_muonclip'); old=tmp_path/'allocation'; old.mkdir()
    (old/'allocation.json').write_text(json.dumps({'validation_deadline_unix':100000.}))
    source=tmp_path/'original'; source.mkdir(); checkpoint=source/'step1.pt'; checkpoint.write_bytes(b'evidence')
    monkeypatch.setattr(module,'BASE',tmp_path); monkeypatch.setattr(module,'OLD',old)
    monkeypatch.setattr(module,'SOURCE',source); monkeypatch.setattr(module,'CHECKPOINT',checkpoint)
    monkeypatch.setattr(module,'LATEST',tmp_path/'latest.json')
    monkeypatch.setattr(module.os,'geteuid',lambda:0)
    monkeypatch.setattr(module.os.path,'ismount',lambda _:True)
    monkeypatch.setattr(module,'assert_idle',lambda:None)
    original=Path.is_file
    monkeypatch.setattr(Path,'is_file',lambda p:True if str(p).endswith('/data/train.bin') else original(p))
    monkeypatch.setattr(module.time,'time',lambda:1000.)
    calls=[]; monkeypatch.setattr(module,'run',lambda args,**kwargs:calls.append(args))
    module.launch_remote('a'*40)
    record=json.loads(module.LATEST.read_text())
    assert record['training_deadline_unix']==2800. and record['service_deadline_unix']==3400.
    command=next(c for c in calls if c[0]=='systemd-run')
    assert '--property=RuntimeMaxSec=2400' in command and '--property=Restart=no' in command
    assert command[-1]==str(checkpoint) and checkpoint.read_bytes()==b'evidence'
    assert not any(c[0]=='gcloud' for c in calls)


@pytest.mark.parametrize('name',['run_muonclip','retry_adamw','replay_muonclip'])
def test_all_launchers_refuse_active_replay(tmp_path,monkeypatch,name):
    module=script(name)
    monkeypatch.setattr(module,'BASE',tmp_path)
    monkeypatch.setattr(module,'LATEST',tmp_path/'absent.json')
    (tmp_path/'MUONCLIP_REPLAY_LATEST.json').write_text(json.dumps({'unit':'replay.service'}))
    if name=='replay_muonclip': monkeypatch.setattr(module,'LATEST',tmp_path/'MUONCLIP_REPLAY_LATEST.json')
    monkeypatch.setattr(module,'active',lambda unit:unit=='replay.service')
    with pytest.raises(RuntimeError,match='[Rr]eplay'): module.assert_idle()


def test_supervisor_preserves_native_abort_and_child_report(tmp_path):
    module=script('supervise_replay'); stage=tmp_path/'diagnostic/diagnostics'; stage.mkdir(parents=True)
    (stage/'current_stage.json').write_text(json.dumps({'stage':'after_primary_started','update':2}))
    module.preserve_child_failure(tmp_path,RuntimeError('aborted'),{'child_exit_code':-6})
    path=tmp_path/'diagnostic/TPU_PORT_FAILURE.json'; raw=path.read_bytes()
    assert json.loads(raw)['last_stage']['stage']=='after_primary_started'
    assert json.loads(raw)['child_exit_code']==-6
    module.preserve_child_failure(tmp_path,RuntimeError('later error'),{})
    assert path.read_bytes()==raw
