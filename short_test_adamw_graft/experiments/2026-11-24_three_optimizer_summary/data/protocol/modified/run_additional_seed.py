"""Continue the frozen reference-0.4 graft design on one new confirmation seed."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'frozen_source/short_test_adamw_graft'))
import reference_lr_study as study
import run


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify():
    recorded = json.loads((ROOT/'existing_seed1340/design.json').read_text())
    actual = {name: importlib.metadata.version(name) for name in recorded['packages']}
    if actual != recorded['packages']:
        raise RuntimeError(f'Package versions differ from the existing seed: {actual}')
    if platform.python_version() != recorded['device']['python']:
        raise RuntimeError('Python version differs from the existing seed')
    for relative, expected in recorded['sources'].items():
        path = ROOT/'frozen_source'/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen training source changed: {relative}')
    driver = ROOT/'frozen_source/short_test_adamw_graft/reference_lr_study.py'
    if run.base.sha(driver) != recorded['driver_sha256']:
        raise RuntimeError('Frozen reference-rate study driver changed')
    if run.base.sha(ROOT/'data/input.txt') != recorded['input_sha256']:
        raise RuntimeError('Corpus changed')
    return actual


def job_path(seed, output):
    return output/'confirmation/rate_0.4/graft'/f'seed_{seed}'


def history_export(job, snapshot):
    run.base.write_json(job/'history.json', snapshot['progress']['history'])


def train(seed):
    packages = verify()
    root = ROOT/'runs'/f'seed_{seed}'
    root.mkdir(parents=True, exist_ok=True)
    state = dict(seed=seed, status='running', pid=os.getpid(), started_at=utc(), packages=packages,
                 reference_peak_lr=0.4, native_peak_lr=0.001, target_steps=20000)
    run.base.write_json(root/'worker_state.json', state)
    study.CONFIRMATION_SEED = seed
    args = SimpleNamespace(device='cpu', phase='confirmation', rate=0.4,
                           steps=2000, data=ROOT/'data/input.txt', output=root)
    job = job_path(seed, root)
    try:
        result = job/'result.json'
        if result.exists():
            result_data = json.loads(result.read_text())
            if not result_data.get('completed') or result_data['endpoint']['step'] != 20000:
                raise RuntimeError('Existing result is not a completed endpoint')
        else:
            saved = job/'checkpoint_2000.pt'
            progress_path = job/'progress.json'
            progress = json.loads(progress_path.read_text())['step'] if progress_path.exists() else 0
            if progress <= 2000:
                study.worker(args)
                snapshot = run.torch.load(job/'last.pt', map_location='cpu', weights_only=False)
                if snapshot['progress']['step'] != 2000:
                    raise RuntimeError('The preserved intermediate checkpoint is not step2000')
                if not saved.exists():
                    shutil.copyfile(job/'last.pt', saved)
                history_export(job, snapshot)
            elif not saved.exists():
                raise RuntimeError('Resumed beyond2k without its preserved checkpoint')
            args.steps = 20000
            study.worker(args)
        result_data = json.loads(result.read_text())
        assert result_data['seed'] == seed
        assert [x['step'] for x in result_data['history']] == list(range(0,20001,500))
        assert result_data['endpoint']['step'] == 20000 and result_data['completed']
        shutil.copyfile(job/'last.pt', job/'checkpoint_20000.pt')
        run.base.write_json(job/'history.json', result_data['history'])
        state.update(status='completed', finished_at=utc(), result=str(result.relative_to(ROOT)))
        run.base.write_json(root/'worker_state.json', state)
        print(json.dumps(state), flush=True)
    except Exception as exc:
        state.update(status='failed', failed_at=utc(), error=str(exc))
        run.base.write_json(root/'worker_state.json', state)
        raise


def preflight():
    packages = verify()
    run.runtime.DEVICE = 'cpu'
    run.runtime.initialize()
    run.RATIO = 400.0
    data = run.base.load_data(ROOT/'data/input.txt')
    run.preflight(data)
    report = dict(status='passed', completed_at=utc(), packages=packages,
                  python=platform.python_version(), reference_lr_ratio=run.RATIO,
                  checks='shared initialization; exact CPU checkpoint replay;18successfulmatrixfits')
    run.base.write_json(ROOT/'preflight.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(1340,1341,1342,1343,1344))
    parser.add_argument('--preflight', action='store_true')
    args=parser.parse_args()
    if args.preflight: preflight()
    elif args.seed is not None: train(args.seed)
    else: parser.error('Choose --preflight or --seed')

