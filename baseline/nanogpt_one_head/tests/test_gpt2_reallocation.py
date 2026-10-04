"""Exercise allocation ordering and preservation gates without contacting GCP."""
import importlib.util
from pathlib import Path
import subprocess

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / 'gpt2small/reallocate_validation.py'
    spec = importlib.util.spec_from_file_location('gpt2_reallocation', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_startup_mounts_only_existing_disk_and_never_restarts():
    m = load(); script = m.make_startup('a' * 40)
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)
    assert '__' not in script and 'mkfs' not in script
    assert 'readlink -f "$DEVICE"' in script
    assert '+4*3600-600' in script and 'Restart=no' in script
    assert 'STARTED_ONCE' in script and 'replacement_worker.sh' in script
    worker = Path(m.__file__).with_name('replacement_worker.sh').read_text()
    subprocess.run(['bash', '-n'], input=worker, text=True, check=True)
    assert worker.index('receipt=sink.file') < worker.index('gpt2small/validate.py')
    assert 'prepare_tpu_data' not in worker and '--allow-long-run' not in worker


@pytest.mark.parametrize('case', ['normal', 'duplicate', 'wrong_disk', 'delete_failure', 'other_user'])
def test_replacement_sequence(tmp_path, monkeypatch, case):
    m = load(); calls = []
    old_present = case != 'other_user'
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(m.subprocess, 'check_output',
                        lambda args, **kw: '' if 'status' in args else 'a'*40 + '\n')

    def inventory(*args):
        if 'queued-resources' in args:
            if 'describe' in args:
                return {'tpu': {'nodeSpec': [{'nodeId': m.OLD_NODE}]}}
            return ([{'name': m.QUEUE}] if case == 'duplicate' else
                    [{'name': m.OLD_QUEUE}] if old_present else [])
        if 'tpu-vm' in args:
            if 'describe' in args:
                return {'state': 'READY', 'dataDisks': [{'sourceDisk':
                        'wrong' if case == 'wrong_disk' else m.DISK_PATH}]}
            return [{'name': m.OLD_NODE, 'state': 'READY'}] if old_present else []
        if args[:3] == ('compute', 'disks', 'describe'):
            return {'zone': m.ZONE, 'users': ['an-attached-vm'] if old_present or case == 'other_user' else []}
        raise AssertionError(args)

    def gc(*args, **kwargs):
        nonlocal old_present
        calls.append(args)
        if 'delete' in args:
            if case == 'delete_failure':
                raise RuntimeError('delete failed')
            assert 'queued-resources' in args and m.OLD_QUEUE in args
            old_present = False
        if 'create' in args:
            assert not old_present
            assert '--max-run-duration=4h' in args
            assert '--accelerator-type=v5litepod-8' in args
            assert '--data-disk=source=' + m.DISK_PATH + ',mode=read-write' in args

    monkeypatch.setattr(m, 'inventory', inventory)
    monkeypatch.setattr(m, 'gc', gc)
    if case in ('wrong_disk', 'delete_failure', 'other_user'):
        with pytest.raises(RuntimeError):
            m.launch()
    else:
        m.launch()
    assert sum('create' in call for call in calls) == (case == 'normal')
    assert not any('disks' in call and 'delete' in call for call in calls)
    if case in ('duplicate', 'wrong_disk', 'other_user'):
        assert calls == []
