"""Pinned upstream leaderboard reference. plan/verify need only the standard library."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor'


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory=VENDOR):
    expected = read_json(HERE / 'upstream_git_files.json')
    for name, info in expected.items():
        path = directory / name
        if not path.is_file() or sha256(path) != info['sha256']:
            raise RuntimeError('Upstream source missing or changed: ' + name)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*.py')
              if '__pycache__' not in p.parts}
    if actual != {p for p in expected if p.endswith('.py')}:
        raise RuntimeError('Unexpected Python source in upstream snapshot')
    return len(expected)


def check_environment():
    import torch
    if not torch.__version__.startswith('2.10.') or torch.version.cuda != '12.8':
        raise RuntimeError('Requires upstream PyTorch 2.10 from the cu128 wheel index')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError('Requires exactly eight visible NVIDIA H100 GPUs; TPU/CPU unsupported')
    devices = [torch.cuda.get_device_properties(i) for i in range(8)]
    if any('H100' not in p.name or p.total_memory < 75 * 1024**3 for p in devices):
        raise RuntimeError('Requires eight H100 GPUs with 80 GB memory each')
    # The pinned FA3 wheel links libcudart.so.13, independently of torch's cu128 build.
    import ctypes
    try:
        ctypes.CDLL('libcudart.so.13')
    except OSError as error:
        raise RuntimeError('CUDA 13 runtime missing; use the vendored upstream Dockerfile') from error
    return {'torch':torch.__version__, 'torch_cuda':torch.version.cuda,
            'devices':[{'name':p.name, 'bytes':p.total_memory} for p in devices]}


def required_data(root):
    return [root/'data/fineweb10B'/name for name in
            ['fineweb_val_000000.bin'] + [f'fineweb_train_{i:06d}.bin' for i in range(1, 10)]]


def prepare(root):
    """Run the unchanged upstream nine-shard downloader outside the source snapshot."""
    verify()
    folder = root / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / 'cached_fineweb10B.py'
    if script.exists() and script.read_bytes() != (VENDOR/'data/cached_fineweb10B.py').read_bytes():
        raise RuntimeError('Refusing to overwrite a different data preparation script')
    shutil.copyfile(VENDOR/'data/cached_fineweb10B.py', script)
    subprocess.run([sys.executable, str(script), '9'], cwd=root, check=True)


def stage_run(root, data_root, environment, save_weights):
    verify()
    files = required_data(data_root)
    if any(not path.is_file() for path in files):
        raise RuntimeError('Missing FineWeb shards; run prepare first')
    if any(key in os.environ for key in ('TRAIN_SEED', 'NUM_SCHEDULED_ITERATIONS')):
        raise RuntimeError('Unset TRAIN_SEED and NUM_SCHEDULED_ITERATIONS to use the pinned defaults')
    root.mkdir(parents=True)  # never overwrite or restart an existing run
    source = root/'source'
    shutil.copytree(VENDOR, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    verify(source)
    # Preserve the observation adapter with the run too, without putting it in upstream source.
    for name in ('rank_entry.py', 'weight_export.py'):
        shutil.copyfile(HERE/name, root/name)
    manifest = {**read_json(HERE/'upstream.json'), 'root':str(root),
                'environment':environment, 'save_weights':save_weights,
                'mode':'upstream-plus-postrun-weight-export' if save_weights else 'upstream-default',
                'data_sha256':{path.name:sha256(path) for path in files},
                'source_files':read_json(HERE/'upstream_git_files.json'),
                'wrapper_sha256':{name:sha256(HERE/name) for name in
                                  ('experiment.py', 'rank_entry.py', 'weight_export.py')},
                'hardware_execution_verified_before_launch':False}
    write_json(root/'manifest.json', manifest)
    env = dict(os.environ, DATA_PATH=str(data_root))
    if save_weights:
        entry = root/'rank_entry.py'
        env['RG_LEADERBOARD_EXPORT_ROOT'] = str(root/'weights')
    else:
        entry = source/'train_gpt.py'
        env.pop('RG_LEADERBOARD_EXPORT_ROOT', None)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=8', str(entry)]
    write_json(root/'launch.json', {'command':command, 'cwd':str(root),
                                  'data_root':str(data_root), 'save_weights':save_weights})
    return command, env


def run(root, data_root, save_weights):
    verify()
    environment = check_environment()
    if save_weights:
        # The sparse BF16 table alone is 130 GB; retain headroom for dense weights and metadata.
        parent = root.parent
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < 160 * 1024**3:
            raise RuntimeError('Final weight export requires at least 160 GiB free disk space')
    command, env = stage_run(root, data_root, environment, save_weights)
    write_json(root/'RUN_STATUS.json', {'status':'running'})
    try:
        with (root/'console.log').open('w') as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Training exited {result.returncode}; inspect {root}/console.log')
        if save_weights:
            from weight_export import verify_export
            verify_export(root/'weights', world_size=8, total_rows=84_602_880)
        write_json(root/'RUN_STATUS.json', {'status':'finished', 'exit_code':0,
                   'weights_exported':save_weights,
                   'target_check':'Read the final validation loss in logs; process completion does not assert target attainment.'})
    except BaseException as error:
        write_json(root/'RUN_STATUS.json', {'status':'failed', 'error':str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'verify', 'check', 'prepare', 'run'))
    parser.add_argument('--data-root', type=Path, default=HERE/'cache')
    parser.add_argument('--results-root', type=Path, default=HERE/'results')
    parser.add_argument('--save-weights', action='store_true', help='Export dense weights and all sparse shards after final validation (about 130 GB)')
    args = parser.parse_args()
    if args.action == 'plan':
        print(json.dumps(read_json(HERE/'upstream.json'), indent=2))
    elif args.action == 'verify':
        print(f'{verify()} upstream files verified')
    elif args.action == 'check':
        verify(); print(json.dumps(check_environment(), indent=2))
    elif args.action == 'prepare':
        prepare(args.data_root.resolve())
    else:
        name = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        root = args.results_root.resolve()/name
        print('Results: '+str(root), flush=True)
        run(root, args.data_root.resolve(), args.save_weights)


if __name__ == '__main__':
    main()
