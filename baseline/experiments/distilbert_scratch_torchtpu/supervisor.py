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


def run(command, deadline, env=None):
    if deadline <= time.time():
        raise RuntimeError('Budget exhausted before next phase')
    proc = subprocess.Popen([str(x) for x in command], start_new_session=True, env=env)
    try:
        code = proc.wait(timeout=max(1, deadline - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise RuntimeError('Phase exceeded its bounded deadline')
    if code:
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
        config.write_text((here / 'configs/initial.json').read_text())
        data = root / 'data'
        output = root / 'initial'
    run([python, here / 'launch.py', '--probe-only', '--output', root / 'probe'], training_deadline)
    if not a.resume_run:
        run([python, here / 'prepare.py', '--output', data], training_deadline)
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
                os.killpg(analyst.pid, signal.SIGTERM)
                analyst.wait(timeout=20)
            result = json.loads((output / 'RESULT.json').read_text())
            result['spectra_exit_code'] = analyst.returncode
            result['spectra_complete'] = (analyst.returncode == 0 and
                len(list((output / 'measurements').glob('step_*.json'))) ==
                len(list((output / 'spectra').glob('step_*.json'))))
            result['output'] = str(output)
            write(root / 'RESULT.json', result)
        finally:
            if analyst.poll() is None:
                os.killpg(analyst.pid, signal.SIGTERM)
                analyst.wait(timeout=20)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--resume-run')
    args = p.parse_args()
    try:
        main(args)
    except Exception:
        write(Path(args.root) / 'ERROR.json', dict(error=traceback.format_exc(), time=time.time()))
        raise
