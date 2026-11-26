"""Run one paired seed, resuming or skipping already archived matching results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from .calibrate import digest, write_json
from .train import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--publish-branch')
    args = p.parse_args()
    root = Path(args.root)
    plan = json.loads(Path(args.plan).read_text())
    frozen_path = root / 'frozen_selection.json'
    frozen = json.loads(frozen_path.read_text())
    checks = {'plan_file_sha256': digest(args.plan),
              'data_manifest_sha256': digest(Path(args.data_dir) / 'manifest.json'),
              'source_manifest_sha256': digest(Path(__file__).with_name('SOURCE_SHA256.json'))}
    if not frozen.get('calibration_complete') or any(frozen.get(k) != v for k, v in checks.items()):
        raise RuntimeError('Frozen selection/source/data/plan contract failed')
    if args.seed not in plan['confirmation_seeds'] or args.seed == plan['development_seed']:
        raise ValueError('Seed is not a confirmation seed')
    seed_dir = root / 'runs' / f'seed_{args.seed}'
    failures = []
    for name in plan['optimizers']:
        try:
            print(f'Confirmation seed={args.seed} optimizer={name}', flush=True)
            result = train(plan, frozen['selected'][name], args.data_dir,
                           seed_dir / name, args.seed, phase='confirmation', resume=True)
            result['frozen_selection_sha256'] = digest(frozen_path)
            result['source_manifest_sha256'] = checks['source_manifest_sha256']
            write_json(seed_dir / name / 'result.json', result)
        except Exception as error:
            failures.append(f'{name}: {type(error).__name__}: {error}')
            (seed_dir / name).mkdir(parents=True, exist_ok=True)
            (seed_dir / name / 'failure.log').write_text(traceback.format_exc())
            print(failures[-1], flush=True)
        if args.publish_branch:
            subprocess.run([sys.executable, '-m', 'california_housing.publish', 'publish',
                            '--branch', args.publish_branch, '--paths', str(seed_dir)], check=True)
    write_json(seed_dir / 'github_execution.json', {
        'seed': args.seed, 'github_run_id': os.getenv('GITHUB_RUN_ID'),
        'github_run_attempt': os.getenv('GITHUB_RUN_ATTEMPT'), 'github_job': os.getenv('GITHUB_JOB'),
        'github_sha': os.getenv('GITHUB_SHA'), 'frozen_selection_sha256': digest(frozen_path),
        'failures': failures, 'complete_three_optimizers': not failures})
    if failures:
        raise RuntimeError('; '.join(failures))


if __name__ == '__main__':
    main()
