"""Run or resume a paired confirmation seed with frozen settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .calibration import check_identity, identity, write_json
from .train import train

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--optimizer', choices=['sgdm', 'adamw', 'modified_adamw'])
    parser.add_argument('--publish-branch')
    args = parser.parse_args()
    root = Path(args.root)
    plan = json.loads(Path(args.plan).read_text())
    frozen = json.loads((root / 'frozen_selection.json').read_text())
    check_identity(frozen, identity(args.plan, args.data_dir))
    if not frozen['calibration_complete'] or args.seed not in plan['confirmation_seeds']:
        raise RuntimeError('Unfrozen settings or unexpected confirmation seed')
    group = root / 'runs' / f'seed_{args.seed}'
    execution = {'seed': args.seed, 'github_run_id': os.getenv('GITHUB_RUN_ID'),
        'launch_commit': os.getenv('GITHUB_SHA')}
    metadata = group / 'github_execution.json'
    if metadata.exists():
        old = json.loads(metadata.read_text())
        for key in ('seed', 'github_run_id', 'launch_commit'):
            if old.get(key) != execution.get(key):
                raise RuntimeError('Cannot mix GitHub cohorts or local runs')
    write_json(metadata, execution)
    frozen_hash = hashlib.sha256((root / 'frozen_selection.json').read_bytes()).hexdigest()
    for arm in ([args.optimizer] if args.optimizer else plan['optimizers']):
        output = group / arm
        result = train(plan, frozen['selected'][arm], args.data_dir, output,
                       args.seed, phase='confirmation', resume=True)
        result['frozen_selection_sha256'] = frozen_hash
        result['source_manifest_sha256'] = frozen['source_manifest_sha256']
        write_json(output / 'result.json', result)
        if args.publish_branch:
            subprocess.run([sys.executable, '-m', 'california_housing_benchmark.publish',
                'publish', '--branch', args.publish_branch, '--paths', str(output)], check=True)
    print(f'Completed requested optimizer runs for seed {args.seed}', flush=True)

if __name__ == '__main__':
    main()
