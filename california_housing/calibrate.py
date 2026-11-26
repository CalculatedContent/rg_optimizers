"""Run the preregistered development grid, then freeze validation-only choices."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from .train import train


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    plan_hash = digest(args.plan)
    data_hash = digest(Path(args.data_dir) / 'manifest.json')
    code_hash = digest(Path(__file__).with_name('SOURCE_SHA256.json'))
    frozen_path = root / 'frozen_selection.json'
    identity = {'plan_file_sha256': plan_hash, 'data_manifest_sha256': data_hash,
                'source_manifest_sha256': code_hash}
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if any(frozen.get(k) != v for k, v in identity.items()):
            raise RuntimeError('Existing frozen selection belongs to a different design/data/source')
        if not frozen.get('calibration_complete'):
            raise RuntimeError('Existing frozen selection is not complete')
        if (os.getenv('GITHUB_RUN_ID') and frozen.get('github_run_id')
                and str(frozen['github_run_id']) != os.getenv('GITHUB_RUN_ID')):
            raise RuntimeError('This GitHub cohort already exists under another run ID. Re-run only failed jobs of the original workflow; do not launch a duplicate cohort.')
        for name, config in frozen['selected'].items():
            path = root / 'configs' / f'{name}.json'
            if path.exists() and json.loads(path.read_text()) != config:
                raise RuntimeError(f'Frozen config changed: {name}')
            write_json(path, config)
        print('Reusing archived frozen selection; no calibration or confirmation retraining.', flush=True)
        return

    trials, selected = [], {}
    for name in plan['optimizers']:
        choices = []
        for index, config in enumerate(plan['calibration']['candidates'][name]):
            trial_dir = root / 'calibration' / name / f'candidate_{index}'
            row = {'optimizer': name, 'candidate_index': index, 'config': config,
                   'seed': plan['development_seed']}
            try:
                print(f'Calibration {name} candidate {index}: {config}', flush=True)
                result = train(plan, config, args.data_dir, trial_dir,
                               plan['development_seed'], phase='calibration',
                               stop_step=plan['calibration']['updates'], resume=True)
                last = next(h for h in result['history']
                            if h['step'] == plan['calibration']['updates'])
                rmse = float(last['validation']['rmse'])
                if not math.isfinite(rmse):
                    raise ValueError('Non-finite validation RMSE')
                row.update(status='complete', validation_rmse=rmse)
                rate = config['peak_lr'] * (config.get('reference_lr_ratio') or 1.0)
                choices.append((rmse, rate, index, config))
            except Exception as error:
                numerical = isinstance(error, FloatingPointError) or (
                    isinstance(error, RuntimeError) and 'non-finite' in str(error).lower())
                row.update(status='numerical_failure' if numerical else 'infrastructure_failure',
                           error=f'{type(error).__name__}: {error}')
                trial_dir.mkdir(parents=True, exist_ok=True)
                (trial_dir / 'error.log').write_text(traceback.format_exc())
                print(row['error'], flush=True)
                if not numerical:
                    trials.append(row)
                    write_json(root / 'calibration' / 'trials.json', {'trials': trials, **identity})
                    raise
            trials.append(row)
            write_json(root / 'calibration' / 'trials.json', {'trials': trials, **identity})
        if not choices:
            raise RuntimeError(f'All preregistered candidates failed for {name}; confirmation blocked')
        selected[name] = min(choices, key=lambda t: t[:3])[3]

    frozen = {'schema_version': 1, 'experiment_id': plan['experiment_id'], **identity,
              'calibration_complete': True, 'development_seed': plan['development_seed'],
              'confirmation_seeds': plan['confirmation_seeds'], 'selection': plan['calibration']['selection_metric'],
              'selected': selected, 'trials': trials, 'github_run_id': os.getenv('GITHUB_RUN_ID'),
              'launch_commit': os.getenv('GITHUB_SHA'),
              'comparison_scope': 'separately calibrated optimizer recipes; modified Adam base LR fixed at 0.001'}
    for name, config in selected.items():
        write_json(root / 'configs' / f'{name}.json', config)
    write_json(frozen_path, frozen)
    write_json(root / 'plan.json', plan)
    (root / 'calibration' / 'data_manifest.json').write_bytes((Path(args.data_dir) / 'manifest.json').read_bytes())
    with np.load(Path(args.data_dir) / 'prepared.npz', allow_pickle=False) as data:
        write_json(root / 'calibration' / 'split_indices.json', {
            name: data[f'indices_{name}'].tolist() for name in ('train', 'validation', 'test')})
    print(json.dumps({'frozen_selection': selected}, indent=2), flush=True)


if __name__ == '__main__':
    main()
