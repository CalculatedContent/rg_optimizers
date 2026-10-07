"""CPU-only hardware-probe routing, synthetic batches and failure reporting. No TPU launch."""
import builtins
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import experiment
from tpu_port.probe import synthetic_batch
from tpu_port.probe_report import FLAGS, STEP_STAGES, ProbeReport, summarize


@pytest.mark.parametrize('tokens',[16,49152])
def test_synthetic_batch_shapes_and_bounds(tokens):
    batch=synthetic_batch(tokens)
    assert batch.inputs.shape==batch.targets.shape==(tokens,)
    assert batch.inputs.dtype==torch.int32 and batch.targets.dtype==torch.int64
    assert batch.cache.shape==batch.sink.shape==(2*tokens,768)
    assert batch.cache.dtype==batch.sink.dtype==torch.bfloat16
    assert batch.sink.is_leaf and batch.sink.requires_grad
    assert batch.slots.shape==(2*tokens,)
    assert int(batch.slots.min())>=0 and int(batch.slots.max())<batch.cache.shape[0]
    assert batch.seqlens[0]==0 and batch.seqlens[-1]==tokens
    assert ((batch.seqlens>=0)&(batch.seqlens<=tokens)).all()
    assert ((batch.seqlens.diff()>0)&(batch.seqlens.diff()<=2560)).all()
    assert batch.inputs.min()>=0 and batch.inputs.max()<50257
    assert batch.targets.min()>=0 and batch.targets.max()<50257
    expected=batch.inputs.clone(); del batch
    # Different process/model RNG does not change the synthetic batch.
    torch.manual_seed(42)
    evaluation=synthetic_batch(tokens,training=False)
    assert evaluation.sink is None
    assert torch.equal(evaluation.inputs,expected)


def test_environment_versions_without_xla_import(monkeypatch):
    for key in ('XLA_USE_SPMD','XLA_AUTO_SPMD','XLA_USE_BF16','XLA_DOWNCAST_BF16',
                'NUM_SCHEDULED_ITERATIONS','TRAIN_SEED'):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('PJRT_DEVICE','TPU')
    monkeypatch.setattr(experiment.importlib.metadata,'version',lambda name:'2.9.0')
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='torch_xla' or name.startswith('torch_xla.'):
            pytest.fail('Launch grandparent imported torch_xla')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert experiment.runtime_environment()['torch_xla']=='2.9.0'


def complete_reports(root):
    for rank in range(8):
        value={'rank':rank,'status':'complete','finite_loss':True,
               'stages':[{'stage':stage,'status':'complete','finite_loss':True} for stage in STEP_STAGES],**FLAGS}
        (root/f'probe-rank-{rank:02d}.json').write_text(json.dumps(value))


@pytest.mark.parametrize('action',['check','probe'])
def test_probe_and_check_skip_data_tracking_and_capacity_gates(monkeypatch,tmp_path,action):
    def forbidden(*args,**kwargs): pytest.fail('Data, tracking or extra subprocess requested')
    monkeypatch.setattr(experiment,'verify_reference',lambda:None)
    monkeypatch.setattr(experiment,'runtime_environment',lambda:{'torch':'2.9.0','torch_xla':'2.9.0','host_available_bytes':0})
    monkeypatch.setattr(experiment,'source_fingerprint',lambda:{})
    monkeypatch.setattr(experiment.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    monkeypatch.setattr(experiment,'data_manifest',forbidden)
    monkeypatch.setattr(experiment,'tracking_command',forbidden)
    monkeypatch.setattr(experiment.subprocess,'run',forbidden)
    seen=[]
    def fake_probe(cmd,root):
        seen.append(json.loads(cmd[-1])['action']); complete_reports(root); return 0
    monkeypatch.setattr(experiment,'probe_subprocess',fake_probe)
    def fake_popen(cmd,**kwargs):
        assert action=='check'
        seen.append(json.loads(cmd[-1])['action'])
        return SimpleNamespace(stdout=[],wait=lambda:0)
    monkeypatch.setattr(experiment.subprocess,'Popen',fake_popen)
    root=experiment.launch(action,tmp_path/'nonexistent-data',tmp_path/'results')
    manifest=json.loads((root/'run_manifest.json').read_text())
    assert manifest['data'] is None and manifest['status']=='complete'
    assert seen==[action]
    assert not (root/'PREFLIGHT_COMPLETE.json').exists()
    assert 'weightwatcher_complete' not in manifest


def test_probe_cli_routes_action(monkeypatch):
    seen=[]
    monkeypatch.setattr(sys,'argv',['experiment.py','probe'])
    monkeypatch.setattr(experiment,'launch',lambda *args:seen.append(args[0]))
    experiment.main()
    assert seen==['probe']


def test_worker_oom_records_failure_and_does_not_continue(monkeypatch,tmp_path):
    from tpu_port import runtime
    def fail(index,options,report):
        report.begin('train_49152_compile')
        raise torch.OutOfMemoryError('simulated CPU-only OOM')
    monkeypatch.setattr(runtime,'_worker',fail)
    with pytest.raises(torch.OutOfMemoryError):
        runtime.worker(0,{'action':'probe','output':str(tmp_path)})
    value=json.loads((tmp_path/'probe-rank-00.json').read_text())
    assert value['status']=='failed' and value['failed_stage']=='train_49152_compile'
    assert 'simulated CPU-only OOM' in value['exception']
    assert all(value[key] is expected for key,expected in FLAGS.items())
    assert summarize(tmp_path,final=True)['status']=='failed'


def test_incomplete_workers_cannot_be_green(tmp_path):
    report=ProbeReport(tmp_path,0)
    report.begin('eval_262144_compile')
    assert summarize(tmp_path,final=True)['status']=='failed'


def test_full_eval_is_required_for_green_summary(tmp_path):
    complete_reports(tmp_path)
    assert summarize(tmp_path,final=True)['status']=='complete'
    path=tmp_path/'probe-rank-07.json'
    report=json.loads(path.read_text())
    report['stages']=[s for s in report['stages'] if s['stage']!='eval_262144_compile']
    path.write_text(json.dumps(report))
    assert summarize(tmp_path,final=True)['status']=='failed'


def test_failed_launch_finalizes_json_without_success(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(experiment,'verify_reference',lambda:None)
    monkeypatch.setattr(experiment,'runtime_environment',lambda:{'host_available_bytes':0})
    monkeypatch.setattr(experiment,'source_fingerprint',lambda:{})
    monkeypatch.setattr(experiment,'probe_subprocess',lambda *args:1)
    with pytest.raises(RuntimeError,match='TPU subprocess failed'):
        experiment.launch('probe',tmp_path/'no-data',tmp_path/'results')
    root=next((tmp_path/'results').iterdir())
    assert json.loads((root/'PROBE_SUMMARY.json').read_text())['status']=='failed'
    assert json.loads((root/'run_manifest.json').read_text())['status']=='failed'
    assert 'Results:' not in capsys.readouterr().out
