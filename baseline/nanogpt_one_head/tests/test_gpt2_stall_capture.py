import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'stall_capture', Path(__file__).parents[1] / 'gpt2small/capture_stall.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def service(root, active='active', pid='6800'):
    return {'ExecStart': f'/bin/bash {root}/repo/baseline/nanogpt_one_head/gpt2small/replacement_worker.sh {root}',
            'ActiveState': active, 'MainPID': pid}


def test_process_selection_excludes_other_training_runs(tmp_path):
    root = Path('/mnt/disks/rg-data/gpt2small/current')
    for pid, args in {
        11: ['python', '-m', 'rg_nanogpt_one_head.gpt2_experiment', '--output', str(root / 'adamw')],
        12: ['python', '-m', 'rg_nanogpt_one_head.gpt2_experiment', '--output', str(root / 'muonclip')],
        13: ['python', '-m', 'rg_nanogpt_one_head.gpt2_experiment', '--output', str(root.parent / 'other/adamw')],
        14: ['python', 'validate.py', '--root', str(root)],
    }.items():
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / 'cmdline').write_bytes(('\0'.join(args) + '\0').encode())
    assert capture.training_pids(root, tmp_path) == [11, 12]


def test_stop_refuses_other_service_root(monkeypatch):
    monkeypatch.setattr(capture, 'service_info', lambda: service('/some/other/run'))
    commands = []
    monkeypatch.setattr(capture.subprocess, 'run', lambda args, **kwargs: commands.append(args))
    with pytest.raises(RuntimeError, match='another run'):
        capture.stop_service(capture.ROOT, {})
    assert commands == []


def test_clean_service_stop_does_not_force_kill(monkeypatch):
    states = iter([service(capture.ROOT), service(capture.ROOT, 'inactive', '0')])
    monkeypatch.setattr(capture, 'service_info', lambda: next(states))
    commands = []
    monkeypatch.setattr(capture.subprocess, 'run', lambda args, **kwargs: commands.append(args))
    report = {}
    capture.stop_service(capture.ROOT, report)
    assert report == {'service_stopped': True}
    assert commands == [['systemctl', 'stop', '--no-block', capture.SERVICE]]


def test_stuck_service_kill_is_scoped_and_verified(monkeypatch):
    states = iter([service(capture.ROOT)] * 32 + [service(capture.ROOT, 'failed', '0')])
    monkeypatch.setattr(capture, 'service_info', lambda: next(states))
    monkeypatch.setattr(capture.time, 'sleep', lambda seconds: None)
    commands = []
    monkeypatch.setattr(capture.subprocess, 'run', lambda args, **kwargs: commands.append(args))
    report = {}
    capture.stop_service(capture.ROOT, report)
    assert report == {'service_stopped': True, 'forced_service_kill': True}
    assert commands[-1] == ['systemctl', 'kill', '--kill-who=all', '--signal=KILL', capture.SERVICE]


def test_changed_service_is_not_force_killed(monkeypatch):
    states = iter([service(capture.ROOT)] * 31 + [service('/different/run')])
    monkeypatch.setattr(capture, 'service_info', lambda: next(states))
    monkeypatch.setattr(capture.time, 'sleep', lambda seconds: None)
    commands = []
    monkeypatch.setattr(capture.subprocess, 'run', lambda args, **kwargs: commands.append(args))
    with pytest.raises(RuntimeError, match='another run'):
        capture.stop_service(capture.ROOT, {})
    assert len(commands) == 1
