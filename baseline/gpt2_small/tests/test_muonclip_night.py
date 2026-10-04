import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import torch

from rg_gpt2_small import experiment as g
from rg_gpt2_small.run_backup import RunBackup
from test_gpt2_experiment import config, data


def script(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def night_config():
    c=config(); c['optimizer']=config('fineweb_muonclip_baseline')['optimizer']
    c.update(validation_tensor_checks=False, validation_gradient_checks=False,
             finite_update_guard=True, progress_reporting=True, benchmark_sync_every_step=True,
             metrics_interval=2, metrics_steps=[1])
    return c


def test_muonclip_trains_with_diagnostic_removed(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    def forbidden(*args,**kwargs): raise AssertionError('Removed diagnostic was called')
    monkeypatch.setattr(g.port_debug,'check_tensors',forbidden)
    c=night_config(); out=tmp_path/'run'
    g.train(c,data(tmp_path,c),out)
    assert json.loads((out/'status.json').read_text())['step']==4
    assert json.loads((out/'progress.json').read_text())['stage']=='completed'
    initial=json.loads((out/'metrics/000000000.json').read_text())
    final=json.loads((out/'metrics/000000004.json').read_text())
    assert final['train_nll']<initial['train_nll']
    assert final['test_accuracy']+final['test_token_error']==1
    assert not list((out/'diagnostics').glob('*-before_clipping.json'))
    assert len(list((out/'checkpoints').glob('step_*.pt')))==3


def test_basic_guard_still_stops_before_invalid_update(tmp_path,monkeypatch):
    torch.set_num_threads(1); c=night_config(); original=g.GPT
    def broken(*args,**kwargs):
        model=original(*args,**kwargs)
        model.token_embedding.weight.register_hook(lambda gradient:gradient*float('nan'))
        return model
    monkeypatch.setattr(g,'GPT',broken)
    def forbidden(*args,**kwargs): raise AssertionError('Invalid optimizer update applied')
    monkeypatch.setattr(g.optimizers,'optimizer_step',forbidden)
    out=tmp_path/'bad'
    with pytest.raises(RuntimeError,match='BEFORE update 1'): g.train(c,data(tmp_path,c),out)
    assert json.loads((out/'checkpoints/latest.json').read_text())['step']==0
    assert (out/'TPU_PORT_FAILURE.json').is_file()


def test_config_retains_model_data_and_optimizer():
    module=script('supervise_muonclip')
    template=Path(__file__).parents[1]/'configs/gpt2_small_fineweb_muonclip_long_ww.yaml'
    c=module.prepare_config(template,'fresh')
    old=config('fineweb_muonclip_long_ww')
    for key in ('model','dataset','optimizer','training'): assert c[key]==old[key]
    assert c['validation_tensor_checks'] is c['validation_gradient_checks'] is False
    assert c['finite_update_guard'] and c['cloud_checkpoints']
    assert c['metrics_interval']==25 and c['ww']['interval']==100


def test_cloud_keeps_previous_pointer_if_upload_fails(tmp_path):
    class Sink:
        def __init__(self): self.fail=False; self.files=[]; self.pointers=[]
        def file(self,path,relative):
            if self.fail: raise IOError('upload failed')
            self.files.append(relative)
            return {'generation':str(len(self.files)),'crc32c':'test','bytes':1}
        def json(self,value,relative): self.pointers.append(dict(value))
    (tmp_path/'manifest.json').write_text('{}')
    checkpoint=tmp_path/'state.pt'; checkpoint.write_bytes(b'x')
    sink=Sink(); backup=RunBackup(tmp_path,'unused',sink)
    for i in range(4): backup.publish(checkpoint,i*25)
    assert [x['file'] for x in sink.pointers]==[
        'muonclip/checkpoints/slot_0.pt','muonclip/checkpoints/slot_1.pt',
        'muonclip/checkpoints/slot_2.pt','muonclip/checkpoints/slot_0.pt']
    assert 'muonclip/checkpoints/initial.pt' in sink.files
    restored=RunBackup(tmp_path,'unused',sink)
    assert restored.sequence==4  # Never reuse the currently referenced slot on reopen.
    before=(tmp_path/'cloud_checkpoint.json').read_bytes(); sink.fail=True
    with pytest.raises(IOError): backup.publish(checkpoint,100)
    assert len(sink.pointers)==4 and (tmp_path/'cloud_checkpoint.json').read_bytes()==before


def test_watch_stops_stall_instead_of_waiting_for_allocation(tmp_path,monkeypatch):
    module=script('supervise_muonclip'); ticks=iter([0,0,1801])
    monkeypatch.setattr(module.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(module.time,'time',lambda:1000.)
    class Child:
        def poll(self): return None
        def wait(self,timeout): raise subprocess.TimeoutExpired('trainer',timeout)
    with pytest.raises(RuntimeError,match='30 minutes'):
        module.watch(Child(),tmp_path,10000.,10300.)


def test_launch_reuses_allocation_and_preserves_evidence(tmp_path,monkeypatch):
    module=script('run_muonclip'); old=tmp_path/'old'; old.mkdir()
    (old/'allocation.json').write_text(json.dumps({'validation_deadline_unix':100000.}))
    evidence=tmp_path/'port-check-old'; evidence.mkdir(); (evidence/'run.log').write_text('crash')
    monkeypatch.setattr(module,'BASE',tmp_path); monkeypatch.setattr(module,'OLD',old)
    monkeypatch.setattr(module,'LATEST',tmp_path/'latest.json')
    monkeypatch.setattr(module.os,'geteuid',lambda:0)
    monkeypatch.setattr(module.os.path,'ismount',lambda path:True)
    monkeypatch.setattr(module,'assert_idle',lambda:None)
    original=Path.is_file
    monkeypatch.setattr(Path,'is_file',lambda path:True if str(path).endswith('/data/train.bin') else original(path))
    monkeypatch.setattr(module.time,'time',lambda:1000.)
    calls=[]; monkeypatch.setattr(module,'run',lambda args,**kwargs:calls.append(args))
    module.launch_remote('a'*40)
    record=json.loads(module.LATEST.read_text())
    assert record['training_deadline_unix']==99400.
    systemd=next(c for c in calls if c[0]=='systemd-run')
    assert '--property=RuntimeMaxSec=99000' in systemd and '--property=Restart=no' in systemd
    assert not any(c[0]=='gcloud' for c in calls)
    assert (evidence/'run.log').read_text()=='crash'


def test_new_launch_refuses_existing_muonclip_service(tmp_path,monkeypatch):
    module=script('run_muonclip')
    pointer=tmp_path/'latest.json'; pointer.write_text(json.dumps({'unit':'night.service'}))
    monkeypatch.setattr(module,'LATEST',pointer)
    monkeypatch.setattr(module,'active',lambda unit:unit=='night.service')
    with pytest.raises(RuntimeError,match='already active'): module.assert_idle()
