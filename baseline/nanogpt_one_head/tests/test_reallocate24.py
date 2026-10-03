"""Simulate the destructive provisioning sequence without contacting GCP."""
import importlib.util
from pathlib import Path
import subprocess
import pytest


def launcher(monkeypatch):
    folder = Path(__file__).resolve().parents[1]/'continuous8'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('reallocation',folder/'reallocate24.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_startup_reuses_disk_without_format_and_has_full_deadline(monkeypatch):
    m = launcher(monkeypatch)
    source = m.make_startup('a'*40)
    subprocess.run(['bash','-n'],input=source,text=True,check=True)
    assert 'mkfs' not in source and '__' not in source
    assert '+23.5*3600' in source
    assert 'Restart=no' in source
    assert 'BASE='+m.BASE in source
    assert 'RG_CONTINUOUS_SHARED_BASE='+m.SHARED in source
    assert 'muonclip_continuous8_24h.yaml' in source


@pytest.mark.parametrize('existing,fail_delete',[(False,False),(True,False),(False,True)])
def test_delete_before_create_preserve_disk_and_avoid_duplicate(tmp_path,monkeypatch,existing,fail_delete):
    m = launcher(monkeypatch)
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw:'' if 'status' in args else 'a'*40+'\n')
    # Keep real bash syntax validation; only cloud calls are simulated.
    queues = {z:[{'name':f'old-{z}'}] for z in m.ZONES}
    if existing: queues[m.ZONE].append({'name':m.QUEUE})
    nodes = {z:[] for z in m.ZONES}
    nodes[m.ZONE] = [{'name':m.OLD_NODE}]
    nodes['us-east5-a'] = [{'name':'standalone-old'}]
    calls = []
    def inventory(*args):
        zone = next((x.split('=',1)[1] for x in args if x.startswith('--zone=')),m.ZONE)
        if 'queued-resources' in args: return queues[zone]
        if 'tpu-vm' in args: return nodes[zone]
        if args[:3] == ('compute','disks','describe'): return {'name':m.DISK,'zone':m.ZONE,'users':[]}
        return {}
    def gc(*args,**kw):
        calls.append(args)
        zone = next((x.split('=',1)[1] for x in args if x.startswith('--zone=')),m.ZONE)
        if 'delete' in args:
            if fail_delete: raise RuntimeError('deletion failed')
            if 'queued-resources' in args:
                queues[zone] = []
                if zone == m.ZONE: nodes[zone] = []
            else: nodes[zone] = []
        if 'create' in args:
            assert all(not values for values in queues.values())
            assert all(not values for values in nodes.values())
            assert '--max-run-duration=24h' in args
            assert '--accelerator-type=v5litepod-8' in args
            assert any(m.DISK in x for x in args if x.startswith('--data-disk='))
    monkeypatch.setattr(m,'inventory',inventory)
    monkeypatch.setattr(m,'gc',gc)
    if fail_delete:
        with pytest.raises(RuntimeError,match='deletion failed'):m.main()
    else:m.main()
    creates = [c for c in calls if 'create' in c]
    assert len(creates) == (0 if existing or fail_delete else 1)
    assert not any('disks' in c and 'delete' in c for c in calls)
    if existing: assert calls == []
