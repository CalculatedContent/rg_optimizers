"""Exercise provisioning without network access or real cloud mutations."""
import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('machines,hours', [(1,6),(2,4)])
def test_cloud_budget_and_data_before_allocation(tmp_path, monkeypatch, machines, hours):
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
    prepared = False
    calls = []
    def run(cmd,**kw):
        nonlocal prepared
        if any(str(x).endswith('prepare_cloud_data.py') for x in cmd): prepared = True
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(m.subprocess,'run',run)
    def gc(*args,**kw):
        calls.append(args)
        if args[:5] == ('alpha','compute','tpus','queued-resources','create'):
            assert prepared, 'No TPU allocation before CPU preparation succeeds'
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
