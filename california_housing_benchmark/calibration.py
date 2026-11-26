"""Evaluate the frozen development grid and freeze validation-only settings."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def identity(plan_path, data_dir):
    return {'plan_file_sha256': digest(plan_path),
            'data_manifest_sha256': digest(Path(data_dir) / 'manifest.json'),
            'source_manifest_sha256': digest(Path(__file__).with_name('SOURCE_SHA256.json')),
            'github_run_id': os.getenv('GITHUB_RUN_ID'),
            'launch_commit': os.getenv('GITHUB_SHA')}

def check_identity(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(f'Design, runtime cohort, or source mismatch: {key}')

def trial(plan, plan_path, data_dir, root, candidate_id):
    from .train import train
    candidates = {c['id']: c for c in plan['calibration']['candidates']}
    candidate = candidates[candidate_id]
    output = root / 'calibration' / candidate_id
    record_path = output / 'trial.json'
    ident = identity(plan_path, data_dir)
    if record_path.exists():
        old = json.loads(record_path.read_text())
        check_identity(old, ident)
        if old['status'] == 'complete':
            print(f'Reusing completed development candidate {candidate_id}', flush=True)
            return
    record = {**ident, 'candidate_id': candidate_id, 'config': candidate['config'],
              'status': 'running', 'seed_results': []}
    write_json(record_path, record)
    for seed in plan['development_seeds']:
        destination = output / f'seed_{seed}'
        try:
            result = train(plan, candidate['config'], data_dir, destination, seed,
                           phase='calibration', stop_step=plan['calibration']['max_updates'], resume=True)
            value = float(result['primary_best_validation']['validation']['rmse'])
            if not math.isfinite(value):
                raise FloatingPointError('Non-finite best validation RMSE')
            record['seed_results'].append({'seed': seed, 'status': 'complete',
                'best_validation_rmse': value, 'completed_updates': result['completed_updates'],
                'paper_stopping': result.get('paper_stopping'),
                'result_sha256': digest(destination / 'result.json')})
        except Exception as error:
            numerical = isinstance(error, FloatingPointError)
            record['seed_results'].append({'seed': seed,
                'status': 'numerical_failure' if numerical else 'execution_failure',
                'error': f'{type(error).__name__}: {error}'})
            output.mkdir(parents=True, exist_ok=True)
            (output / f'error_seed_{seed}.log').write_text(traceback.format_exc())
            write_json(record_path, record)
            if not numerical:
                raise
        write_json(record_path, record)
    valid = all(r['status'] == 'complete' for r in record['seed_results'])
    record['status'] = 'complete'
    record['eligible'] = valid
    record['mean_best_validation_rmse'] = (
        sum(r['best_validation_rmse'] for r in record['seed_results']) / len(record['seed_results'])
        if valid else None)
    write_json(record_path, record)
    print(json.dumps(record, indent=2), flush=True)

def freeze(plan, plan_path, data_dir, root):
    ident = identity(plan_path, data_dir)
    path = root / 'frozen_selection.json'
    if path.exists():
        existing = json.loads(path.read_text())
        check_identity(existing, ident)
        if not existing.get('calibration_complete'):
            raise RuntimeError('Existing frozen selection is incomplete')
        print('Reusing frozen selection; no new calibration.', flush=True)
        return
    trials = []
    for candidate in plan['calibration']['candidates']:
        row = json.loads((root / 'calibration' / candidate['id'] / 'trial.json').read_text())
        check_identity(row, ident)
        if row['status'] != 'complete' or row['config'] != candidate['config']:
            raise RuntimeError(f'Candidate is incomplete or changed: {candidate["id"]}')
        if [r['seed'] for r in row['seed_results']] != plan['development_seeds']:
            raise RuntimeError('Wrong development seeds')
        trials.append(row)
    selected = {'adamw': plan['published_adamw_config']}
    for arm in ('sgdm', 'modified_adamw'):
        eligible = [r for r in trials if r['eligible'] and r['config']['optimizer'] == arm]
        if not eligible:
            raise RuntimeError(f'No finite development candidate for {arm}')
        winner = min(eligible, key=lambda r: (r['mean_best_validation_rmse'], r['candidate_id']))
        selected[arm] = winner['config']
    value = {**ident, 'schema_version': 1, 'experiment_id': plan['experiment_id'],
        'calibration_complete': True, 'development_seeds': plan['development_seeds'],
        'confirmation_seeds': plan['confirmation_seeds'], 'selected': selected,
        'trials': trials, 'selection': plan['calibration']['selection_metric'],
        'comparison_scope': 'Common published AdamW architecture; native AdamW recipe fixed from literature; modified AdamW differs only in matrix-update magnitudes; SGDM LR/weight decay calibrated on development seeds.'}
    for arm, config in selected.items():
        write_json(root / 'configs' / f'{arm}.json', config)
    (root / 'calibration').mkdir(parents=True, exist_ok=True)
    (root / 'calibration' / 'data_manifest.json').write_bytes((Path(data_dir) / 'manifest.json').read_bytes())
    (root / 'SOURCE_SHA256.json').write_bytes(Path(__file__).with_name('SOURCE_SHA256.json').read_bytes())
    write_json(root / 'plan.json', plan)
    write_json(path, value)
    print(json.dumps({'selected': selected}, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['trial', 'freeze'])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--candidate')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.command == 'trial':
        if args.candidate is None:
            parser.error('--candidate is required for trial')
        trial(plan, args.plan, args.data_dir, root, args.candidate)
    else:
        freeze(plan, args.plan, args.data_dir, root)

if __name__ == '__main__':
    main()
