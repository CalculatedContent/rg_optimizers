"""Exercise provisioning without network access or real cloud mutations."""
import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('machines,hours', [(1,6),(2,4)])
def test_cloud_budget_without_cloudshell_preparation(tmp_path, monkeypatch, machines, hours):
    file = Path(__file__).resolve().parents[1]/'continuous8/cloudshell.py'
    spec = importlib.util.spec_from_file_location('pilot_launcher', file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(m.tempfile,'gettempdir',lambda:str(tmp_path))
    monkeypatch.setattr(m.subprocess,'check_output',
                        lambda cmd,**kw: '\n' if 'status' in cmd else 'abcdef123\n')
    def inventory(*args):
        if args[:3] == ('storage','buckets','list'): return [{'name':m.BUCKET}]
        if args[:3] == ('iam','service-accounts','list'): return [{'email':m.SA}]
        return []
    monkeypatch.setattr(m,'inventory',inventory)
    calls = []
    def run(cmd,**kw):
        raise AssertionError('Launcher must not install dependencies or prepare data in Cloud Shell: '+str(cmd))
    monkeypatch.setattr(m.subprocess,'run',run)
    def gc(*args,**kw):
        calls.append(args)
        if args[:5] == ('alpha','compute','tpus','queued-resources','create'):
            assert f'--max-run-duration={hours}h' in args
            source = next(x.split('=',2)[-1] for x in args if x.startswith('--metadata-from-file='))
            startup = Path(source).read_text()
            assert '__' not in startup
            assert f'+{hours-0.5}*3600' in startup
            assert 'Restart=no' in startup
            assert 'RG_CONTINUOUS_DATA_URI=gs://' in startup
    monkeypatch.setattr(m,'gc',gc)
    m.provision(machines,hours)
    create = [c for c in calls if c[:5] == ('alpha','compute','tpus','queued-resources','create')]
    assert len(create) == machines
    assert len({c[5] for c in create}) == machines
    assert all('--accelerator-type=v5litepod-8' in c for c in create)
    import json
    record=json.loads((tmp_path/'continuous8-launch.json').read_text())
    assert record['status']=='submitted'


def test_dataset_writer_import_requires_no_torch():
    import sys
    source = Path(__file__).resolve().parents[1]/'src/rg_nanogpt_one_head/data.py'
    code = (
        'import importlib.util,sys; '
        f's=importlib.util.spec_from_file_location("writer",{str(source)!r}); '
        'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
        'assert "torch" not in sys.modules'
    )
    subprocess.run([sys.executable,'-c',code],check=True)


def test_empty_status_is_explicit(tmp_path,monkeypatch,capsys):
    file=Path(__file__).resolve().parents[1]/'continuous8/cloudshell.py'
    spec=importlib.util.spec_from_file_location('status_launcher',file)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    monkeypatch.setattr(m,'inventory',lambda *args:[])
    m.status()
    assert 'NO TPU REQUEST' in capsys.readouterr().out


def test_launch_failure_persists_exact_phase(tmp_path,monkeypatch):
    import json
    import sys
    file=Path(__file__).resolve().parents[1]/'continuous8/cloudshell.py'
    spec=importlib.util.spec_from_file_location('failed_launcher',file)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    monkeypatch.setattr(sys,'argv',[str(file),'launch'])
    def fail(*args):
        m.launch_record(phase='granting experiment bucket access')
        raise RuntimeError('permission denied in test')
    monkeypatch.setattr(m,'provision',fail)
    with pytest.raises(SystemExit) as caught:
        m.main()
    assert caught.value.code==1
    record=json.loads((tmp_path/'continuous8-launch.json').read_text())
    assert record['status']=='failed'
    assert record['phase']=='granting experiment bucket access'
    assert 'permission denied' in record['error']


def test_parallel_data_matches_serial(tmp_path):
    from rg_nanogpt_one_head.data import write_token_splits
    class Encoder:
        n_vocab=64
        eot_token=63
        def encode_ordinary(self,text):return [int(x) for x in text.split()]
    texts=['1 2 3','4 5','6 7 8 9','10','11 12 13 14','15 16']*10
    common=dict(train_tokens=25,val_tokens=13,test_tokens=19,progress_every_documents=0)
    serial=write_token_splits(texts,Encoder(),tmp_path/'serial',**common)
    parallel=write_token_splits(iter(texts),Encoder(),tmp_path/'parallel',encoding_workers=4,
                                encoding_batch_size=3,**common)
    assert serial==parallel
    for name in ('train.bin','val.bin','test.bin','meta.json'):
        assert (tmp_path/'serial'/name).read_bytes()==(tmp_path/'parallel'/name).read_bytes()
