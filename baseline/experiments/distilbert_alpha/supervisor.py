"""One initial experiment: setup, full-model two-update preflight, resume, spectra."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def write(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def run(command, deadline, env=None):
    remaining = deadline - time.time()
    if remaining < 30:
        raise TimeoutError('Insufficient remaining budget; mounted-drive data preserved')
    print('RUN', command, flush=True)
    proc = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        returncode = proc.wait(timeout=remaining)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        raise


def main(root, deadline, resume=None, job_id=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tracker = None
    try:
        # A private venv inherits the proven framework; only HF/WW dependencies added.
        venv = root / 'venv'
        run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], deadline - 600)
        python = str(venv / 'bin/python')
        framework = {k: importlib.metadata.version(k) for k in ('torch', 'torch-xla', 'libtpu')}
        write(root / 'framework_before.json', framework)
        run([python, '-m', 'pip', 'install', '-r', str(HERE / 'requirements.txt')], deadline - 600)
        check = "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ('torch','torch-xla','libtpu')}))"
        after = json.loads(subprocess.check_output([python, '-c', check], text=True))
        if framework != after:
            raise RuntimeError('Private environment changed framework versions; abort')
        with (root / 'pip-freeze.txt').open('w') as f:
            subprocess.run([python, '-m', 'pip', 'freeze'], check=True, stdout=f)
        runroot = Path(resume) if resume else root / 'initial'
        run([python, str(HERE / 'sweep.py'), '--output', str(root / 'prospective_sweep.json')], deadline - 600)
        config = root / 'selected_config.json'
        if resume:
            if job_id:
                raise ValueError('Resume restores its original config; omit --job-id')
            selected = json.loads((runroot / 'config.json').read_text())
        elif job_id:
            jobs = json.loads((root / 'prospective_sweep.json').read_text())['jobs']
            matches = [j for j in jobs if j['id'] == job_id]
            if len(matches) != 1:
                raise ValueError('Unknown sweep job ID')
            selected = matches[0]['config']
        else:
            selected = json.loads((HERE / 'configs/initial.json').read_text())
        write(config, selected)
        normalize = "import sys,json; sys.path.insert(0,sys.argv[1]); from common import read_config,config_dict; print(json.dumps(config_dict(read_config(sys.argv[2]))))"
        selected = json.loads(subprocess.check_output([python, '-c', normalize, str(HERE), str(config)], text=True))
        write(config, selected)
        if resume:
            data = Path(json.loads((runroot.parent / 'data_path.json').read_text())['path'])
        else:
            # Share immutable prepared inputs across later optimizer/dropout/seed trials.
            # Include all overrides conservatively; prepare.py checks the full data identity.
            keys = ('model', 'dataset', 'model_revision', 'dataset_revision', 'split_seed',
                    'validation_per_class', 'max_length', 'synthetic')
            identity = {k: selected.get(k) for k in keys}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            data = root.parent / 'prepared' / key
        write(root / 'data_path.json', {'path': str(data)})
        run([python, str(HERE / 'prepare.py'), '--data', str(data), '--config', str(config)], deadline - 600)
        env = os.environ.copy()
        env.update(PJRT_DEVICE='CPU', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        # A separate process does all SVDs on CPU; it never initializes TPU devices.
        tracklog = (root / 'spectra.log').open('a')
        tracker = subprocess.Popen([python, str(HERE / 'spectra.py'), '--root', str(runroot), '--watch',
                                    '--deadline', str(deadline - 60)], env=env, stdout=tracklog, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        command = [python, '-u', str(HERE / 'train.py'), '--root', str(runroot), '--data', str(data),
                   '--config', str(config), '--backend', 'xla', '--deadline', str(deadline - 600)]
        if not (runroot / 'latest_checkpoint.json').exists():
            # Real model, real data, all eight devices; these two updates belong to the same run.
            run(command + ['--stop-after', '2'], deadline - 300)
            status = json.loads((runroot / 'status.json').read_text())
            if status['step'] != 2 or status['status'] != 'paused_resumable':
                raise RuntimeError('Full-model preflight did not reach two saved updates')
            write(root / 'preflight.json', {'passed': True, 'step': 2, 'same_experiment_resumed_next': True})
        run(command, deadline - 300)
        status = json.loads((runroot / 'status.json').read_text())
        write(runroot / 'TRAINING_DONE.json', status)
        try:
            tracker.wait(timeout=max(1, deadline - 90 - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(tracker.pid, signal.SIGTERM)
            tracker.wait(timeout=15)
        expected = {p.name for p in (runroot / 'measurements').glob('*.json')}
        actual = {p.name for p in (runroot / 'spectra').glob('*.json')}
        write(root / 'RESULT.json', {'training': status, 'spectra_complete': expected <= actual,
              'spectra_exit_code': tracker.returncode, 'run_root': str(runroot),
              'status': 'complete' if status['status'] == 'training_complete' and expected <= actual and tracker.returncode == 0 else 'partial_saved',
              'test_accessed': False, 'time': time.time()})
    except BaseException as exc:
        write(root / 'ERROR.json', {'error': repr(exc), 'traceback': traceback.format_exc(), 'time': time.time()})
        raise
    finally:
        if tracker is not None and tracker.poll() is None:
            os.killpg(tracker.pid, signal.SIGTERM)
            try:
                tracker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(tracker.pid, signal.SIGKILL)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--deadline', required=True, type=float)
    p.add_argument('--resume-run')
    p.add_argument('--job-id', help='One ID from prospective_sweep.json; still launches only one experiment')
    a = p.parse_args()
    main(a.root, a.deadline, a.resume_run, a.job_id)
