"""Bounded install, probe, two real updates, resume, and CPU spectral analysis."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def terminate(proc, grace=20):
    """Reap the owned process group, including children that ignore SIGTERM."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        else:
            proc.wait()


def run(command, deadline, env=None):
    if deadline <= time.time():
        raise RuntimeError('Budget exhausted before next phase')
    proc = subprocess.Popen([str(x) for x in command], start_new_session=True, env=env)
    try:
        code = proc.wait(timeout=max(1, deadline - time.time()))
    except subprocess.TimeoutExpired:
        terminate(proc)
        raise RuntimeError('Phase exceeded its bounded deadline')
    if code:
        terminate(proc)
        raise RuntimeError(f'Phase failed ({code}): {command}')


def main(a):
    root = Path(a.root)
    here = Path(__file__).resolve().parent
    training_deadline = a.deadline - 600
    if a.resume_run:
        output = Path(a.resume_run)
        # Resume uses the original environment, not a freshly resolved package stack.
        python = output.parent / 'venv/bin/python'
        config = output.parent / 'config.json'
        data = Path(json.loads((output.parent / 'data_path.json').read_text())['path'])
        if not python.is_file():
            raise RuntimeError('Original run environment is required for exact resume')
    else:
        if sys.version_info < (3, 12):
            raise RuntimeError('Python >=3.12 required')
        venv = root / 'venv'
        run([sys.executable, '-m', 'venv', venv], training_deadline)
        python = venv / 'bin/python'
        run([python, '-m', 'pip', 'install', '--upgrade', 'pip'], training_deadline)
        run([python, '-m', 'pip', 'install', '-r', here / 'requirements.txt',
             '--extra-index-url', 'https://download.pytorch.org/whl/cpu'], training_deadline)
        run([python, '-m', 'pip', 'check'], training_deadline)
        frozen = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
        (root / 'requirements.lock.txt').write_text(frozen)
        config = root / 'config.json'
        requested = (here / (a.config or 'configs/initial.json')).resolve()
        if not requested.is_relative_to(here) or requested.suffix != '.json':
            raise ValueError('Config must be a JSON file inside this experiment folder')
        config.write_text(requested.read_text())
        data = root / 'data'
        output = root / 'initial'
    run([python, here / 'launch.py', '--probe-only', '--output', root / 'probe'], training_deadline)
    if not a.resume_run:
        length = json.loads(config.read_text()).get('sequence_length', 128)
        run([python, here / 'prepare.py', '--output', data, '--length', str(length)], training_deadline)
        write(root / 'data_path.json', dict(path=str(data)))
    if training_deadline - time.time() < 900:
        raise RuntimeError('Less than 15 minutes remain after setup; data saved, training not started')
    cmd = [python, here / 'launch.py', '--config', config, '--data', data,
           '--output', output, '--deadline', str(training_deadline)]
    if not a.resume_run:
        run([*cmd, '--stop-after=2'], a.deadline - 300)
        state = json.loads((output / 'status.json').read_text())
        runtime = json.loads((output / 'runtime.json').read_text())
        if state['step'] != 2 or not state['checkpoint_saved'] or runtime['world'] != 8:
            raise RuntimeError('Two-update hardware preflight did not pass')
        write(root / 'preflight.json', dict(passed=True, saved_updates=2, world=8,
                                         initialization='random', pretrained_weights=False))
    # The analyst never touches TPU tensors or changes optimization.
    (output / 'TRAINING_DONE.json').unlink(missing_ok=True)
    analyst_env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (root / 'spectra.log').open('a') as log:
        analyst = subprocess.Popen([str(python), str(here / 'spectra.py'), '--root', str(output),
                    '--watch', '--deadline', str(a.deadline - 60)], env=analyst_env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            run([*cmd, '--resume'], a.deadline - 300)
            write(output / 'TRAINING_DONE.json', dict(done=True))
            try:
                analyst.wait(timeout=max(1, a.deadline - time.time() - 60))
            except subprocess.TimeoutExpired:
                terminate(analyst)
            result = json.loads((output / 'RESULT.json').read_text())
            result['spectra_exit_code'] = analyst.returncode
            result['spectra_complete'] = (analyst.returncode == 0 and
                {p.name for p in (output / 'measurements').glob('step_*.json')} ==
                {p.name for p in (output / 'spectra').glob('step_*.json')})
            result['output'] = str(output)
            write(root / 'RESULT.json', result)
        finally:
            if analyst.poll() is None:
                terminate(analyst)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--deadline', type=float, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--resume-run')
    mode.add_argument('--config', help='JSON path relative to this experiment folder; default configs/initial.json')
    args = p.parse_args()
    try:
        main(args)
    except Exception:
        write(Path(args.root) / 'ERROR.json', dict(error=traceback.format_exc(), time=time.time()))
        raise
