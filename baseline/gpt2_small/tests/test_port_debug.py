import importlib.util
import json
from pathlib import Path
import types

import pytest
import torch

from rg_gpt2_small import port_debug as debug
from rg_gpt2_small import experiment as g
from test_gpt2_experiment import config,data


def test_nonfinite_layer_report_uses_one_small_transfer(tmp_path,monkeypatch):
    shapes=[]; original=torch.Tensor.cpu
    def cpu(tensor,*args,**kwargs):
        shapes.append(tuple(tensor.shape)); return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',cpu)
    with pytest.raises(RuntimeError,match='before_clipping'):
        debug.check_tensors([('gradient/good',torch.ones(128,128)),
                             ('gradient/bad',torch.tensor([float('nan'),1.]))],
                            tmp_path,'before_clipping',3,'cpu')
    row=json.loads((tmp_path/'diagnostics/first_invalid_tensors.json').read_text())
    assert [x['tensor'] for x in row['invalid_tensors']]==['gradient/bad']
    assert shapes==[(2,3)]
    assert row['invalid_tensors'][0]['min']=='nan'


def test_negative_second_moment_detected(tmp_path):
    with pytest.raises(RuntimeError,match='after_optimizer'):
        debug.check_tensors([('primary/layer/exp_avg_sq',torch.tensor([-.1,1.]))],
                            tmp_path,'after_optimizer',2,'cpu')
    row=json.loads((tmp_path/'diagnostics/first_invalid_tensors.json').read_text())
    assert row['invalid_tensors'][0]['negative_second_moment'] is True


def test_cpu_diagnostic_records_stages_inputs_and_states(tmp_path):
    torch.set_num_threads(1)
    c=config(); c.update(validation_tensor_checks=True,validation_gradient_checks=True,metrics_interval=1)
    out=tmp_path/'run'; g.train(c,data(tmp_path,c),out,stop_after=2)
    assert json.loads((out/'status.json').read_text())['step']==2
    for step in (1,2):
        for stage in ('before_clipping','after_clipping','after_optimizer'):
            row=json.loads((out/f'diagnostics/{step:06d}-{stage}.json').read_text())
            assert not row['invalid_tensors']
        windows=json.loads((out/f'diagnostics/{step:06d}-input-windows.json').read_text())
        assert len(windows['microbatch_offsets'])==c['training']['grad_accum_steps']
    assert (out/'diagnostics/environment.json').is_file()


def test_training_failure_keeps_initial_checkpoint_and_layer_evidence(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    c=config(); c.update(validation_tensor_checks=True,validation_gradient_checks=True)
    original=g.GPT
    def broken_model(*args,**kwargs):
        model=original(*args,**kwargs)
        model.token_embedding.weight.register_hook(lambda gradient:gradient*float('nan'))
        return model
    monkeypatch.setattr(g,'GPT',broken_model)
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='before_clipping'):
        g.train(c,data(tmp_path,c),out)
    report=json.loads((out/'TPU_PORT_FAILURE.json').read_text())
    assert report['attribution'].startswith('unconfirmed')
    assert json.loads((out/'checkpoints/latest.json').read_text())['step']==0
    bad=json.loads((out/'diagnostics/first_invalid_tensors.json').read_text())['invalid_tensors']
    assert 'gradient/token_embedding.weight' in [r['tensor'] for r in bad]


def load_retry():
    spec=importlib.util.spec_from_file_location('retry',Path(__file__).parents[1]/'scripts/retry_adamw.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_retry_refuses_active_training(monkeypatch):
    module=load_retry()
    monkeypatch.setattr(module,'active',lambda unit:unit=='rg-gpt2-validation.service')
    with pytest.raises(RuntimeError,match='existing training service'):
        module.assert_idle()


def test_retry_is_bounded_and_never_allocates(tmp_path,monkeypatch):
    module=load_retry(); old=tmp_path/'old'; old.mkdir()
    (old/'allocation.json').write_text(json.dumps({'validation_deadline_unix':100000.}))
    monkeypatch.setattr(module,'BASE',tmp_path); monkeypatch.setattr(module,'OLD',old)
    monkeypatch.setattr(module,'LATEST',tmp_path/'latest.json')
    monkeypatch.setattr(module.os,'geteuid',lambda:0)
    monkeypatch.setattr(module.os.path,'ismount',lambda path:True)
    monkeypatch.setattr(module,'assert_idle',lambda:None)
    original=Path.is_file
    monkeypatch.setattr(Path,'is_file',lambda path:True if str(path).endswith('/data/train.bin') else original(path))
    monkeypatch.setattr(module.time,'time',lambda:1000.)
    calls=[]
    monkeypatch.setattr(module,'run',lambda args,**kwargs:calls.append(args))
    module.launch_remote('a'*40)
    record=json.loads(module.LATEST.read_text())
    assert record['training_deadline_unix']==2200.
    systemd=next(c for c in calls if c[0]=='systemd-run')
    assert '--property=RuntimeMaxSec=1800' in systemd
    assert '--property=Restart=no' in systemd
    assert not any(c[0]=='gcloud' for c in calls)
