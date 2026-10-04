import copy
import json

import pytest
import torch

from rg_gpt2_small import experiment as g, execution_checks as checks
from rg_gpt2_small.run_backup import RunBackup
from rg_nanogpt_one_head.checkpoints import optimizer_state_sha256
from rg_nanogpt_one_head.muonclip import MuonClip
from test_gpt2_experiment import config, data


def synced_config():
    c=config(); c['optimizer']=config('fineweb_muonclip_baseline')['optimizer']
    c.update(synchronized_finite_checks=True,checkpoint_before_evaluation=True,
             finite_update_guard=True,progress_reporting=True,benchmark_sync_every_step=True,
             metrics_steps=[1,2,4])
    return c


def load_last(root):
    pointer=json.loads((root/'checkpoints/latest.json').read_text())
    return torch.load(root/'checkpoints'/pointer['file'],weights_only=False)


def same_training_state(a,b):
    for name in a['model']: torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
    assert optimizer_state_sha256(a['optimizers'])==optimizer_state_sha256(b['optimizers'])
    assert torch.equal(a['data_rng'],b['data_rng'])


def test_25_updates_continue_in_one_process_and_match_cpu_reference(tmp_path,monkeypatch):
    torch.set_num_threads(1); c=synced_config()
    c['training'].update(max_steps=25,max_tokens=400,schedule_steps=50)
    c['metrics_interval']=5; d=data(tmp_path,c)
    reference=copy.deepcopy(c); reference['synchronized_finite_checks']=False
    g.train(reference,d,tmp_path/'reference')
    calls=[]; original=checks.check_finite
    def checked(*args,**kw):
        calls.append((args[2],args[3])); return original(*args,**kw)
    monkeypatch.setattr(checks,'check_finite',checked)
    out=tmp_path/'synced'; g.train(c,d,out)
    for step in range(1,26):
        stages=[label for label,index in calls if index==step]
        wanted=['before_clipping','after_clipping','after_primary','after_auxiliary']
        assert [label for label in stages if label in wanted]==wanted
    assert calls.count(('initial_state',0))==1
    assert not (out/'resume_verified.json').exists()
    state=load_last(out); assert state['step']==25 and state['tokens_seen']==400
    assert not state['measurement_pending']
    same_training_state(state,load_last(tmp_path/'reference'))
    assert len(list((out/'checkpoints').glob('step_*.pt')))==3
    assert (out/'diagnostics/latest-after_auxiliary.json').is_file()
    assert not (out/'diagnostics/000024-after_auxiliary.json').exists()


@pytest.mark.parametrize('failed_stage',['evaluation','spectra'])
def test_checkpoint_precedes_measurement_and_resume_preserves_weights(tmp_path,monkeypatch,failed_stage):
    torch.set_num_threads(1); c=synced_config(); d=data(tmp_path,c)
    if failed_stage=='spectra':
        c['ww'].update(enabled=True,interval=2)
        monkeypatch.setattr(g,'measure_ww',lambda *args:{'records':[],'seconds':0.01})
    original_eval=checks.evaluate_splits; original_ww=g.measure_ww
    output=tmp_path/'failed'
    def evaluate(*args,**kw):
        if args[5]==2 and failed_stage=='evaluation':
            saved=load_last(output)
            assert saved['step']==2 and saved['measurement_pending']
            raise RuntimeError('Injected evaluation failure')
        return original_eval(*args,**kw)
    def spectra(model,cfg,identity,metrics):
        if identity['step']==2:
            saved=load_last(output)
            assert saved['step']==2 and saved['measurement_pending']
            raise RuntimeError('Injected spectra failure')
        return original_ww(model,cfg,identity,metrics)
    monkeypatch.setattr(checks,'evaluate_splits',evaluate)
    if failed_stage=='spectra': monkeypatch.setattr(g,'measure_ww',spectra)
    with pytest.raises(RuntimeError,match='Injected'): g.train(c,d,output)
    before=(output/'metrics/000000001.json').read_bytes()
    assert not (output/'metrics/000000002.json').exists()
    assert (output/'TPU_PORT_FAILURE.json').exists()
    monkeypatch.setattr(checks,'evaluate_splits',original_eval)
    monkeypatch.setattr(g,'measure_ww',original_ww)
    g.train(c,d,output,resume=True)
    g.train(c,d,tmp_path/'reference')
    same_training_state(load_last(output),load_last(tmp_path/'reference'))
    assert (output/'metrics/000000001.json').read_bytes()==before
    assert (output/'metrics/000000002.json').exists()


def test_invalid_primary_does_not_apply_auxiliary(tmp_path,monkeypatch):
    torch.set_num_threads(1); c=synced_config(); d=data(tmp_path,c)
    original=MuonClip.step
    def broken(self,*args,**kw):
        original(self,*args,**kw)
        with torch.no_grad(): self.param_groups[0]['params'][0].fill_(float('nan'))
    monkeypatch.setattr(MuonClip,'step',broken)
    def forbidden(*args,**kwargs): raise AssertionError('Auxiliary applied after invalid primary')
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    output=tmp_path/'invalid'
    with pytest.raises(RuntimeError,match='First invalid stage: after_primary'): g.train(c,d,output)
    report=json.loads((output/'FIRST_INVALID.json').read_text())
    assert report['update']==1 and report['invalid_tensors']
    assert load_last(output)['step']==0


def test_cloud_refreshes_current_diagnostics(tmp_path):
    class Sink:
        def __init__(self): self.files=[]
        def file(self,path,name):
            self.files.append((name,path.read_bytes()))
            return {'generation':'1','bytes':1,'crc32c':'test'}
        def json(self,*args): pass
    (tmp_path/'manifest.json').write_text('{}')
    folder=tmp_path/'diagnostics'; folder.mkdir()
    latest=folder/'latest-after_primary.json'; latest.write_text('first')
    fixed=folder/'000001-after_primary.json'; fixed.write_text('fixed')
    checkpoint=tmp_path/'checkpoint.pt'; checkpoint.write_bytes(b'state')
    sink=Sink(); backup=RunBackup(tmp_path,'unused',sink)
    backup.publish(checkpoint,1); latest.write_text('second'); backup.publish(checkpoint,2)
    assert [data for name,data in sink.files if name.endswith(latest.name)]==[b'first',b'second']
    assert [data for name,data in sink.files if name.endswith(fixed.name)]==[b'fixed']
