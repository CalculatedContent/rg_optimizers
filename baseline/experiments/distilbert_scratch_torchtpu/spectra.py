"""CPU WeightWatcher fits; encoder matrices only; no feedback into training."""
import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from common import atomic_json, file_hash


def summarize(rows, expected):
    valid = [r for r in rows if r['valid']]
    complete = len(rows) == expected and len(valid) == expected
    alphas = np.array([r['alpha'] for r in valid])
    return dict(expected_layers=expected, valid_layers=len(valid), all_fits_valid=complete,
                mean_alpha=float(alphas.mean()) if complete else None,
                min_alpha=float(alphas.min()) if complete else None,
                fraction_alpha_below_2=float((alphas < 2).mean()) if complete else None,
                any_alpha_below_2=bool((alphas < 2).any()) if complete else None,
                mean_D=float(np.mean([r['D'] for r in valid])) if complete else None)


def analyze(root):
    import weightwatcher as ww
    version = importlib.metadata.version('weightwatcher')
    if version != '0.7.7':
        raise RuntimeError('This protocol pins WeightWatcher 0.7.7')
    root = Path(root)
    torch.set_num_threads(1)
    for path in sorted((root / 'measurements').glob('step_*.json')):
        target = root / 'spectra' / path.name
        measured = json.loads(path.read_text())
        if target.exists():
            continue
        source = root / measured['snapshot']
        if file_hash(source) != measured['sha256']:
            raise RuntimeError('Snapshot hash mismatch')
        matrices = torch.load(source, map_location='cpu', weights_only=True)
        rows = []
        for name, w in matrices.items():
            row = dict(name=name, shape=list(w.shape), valid=False)
            try:
                layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
                layer.weight.data.copy_(w)
                # Identical default WW estimator to the existing RG experiment.
                df = ww.WeightWatcher(model=torch.nn.Sequential(layer)).analyze()
                fits = json.loads(df.to_json(orient='records'))
                fit = fits[0] if fits else {}
                row['fit'] = fit
                for key in ('alpha', 'D', 'sigma', 'xmin', 'xmax', 'num_pl_spikes', 'num_evals', 'warning', 'status'):
                    row[key] = fit.get(key)
                row['valid'] = (isinstance(row['alpha'], (float, int)) and math.isfinite(row['alpha'])
                                and row['alpha'] > 0 and isinstance(row['D'], (float, int))
                                and 0 <= row['D'] <= 1 and fit.get('status') == 'success')
                n, total = row['num_pl_spikes'], row['num_evals']
                row['tail_fraction'] = n / total if n is not None and total else None
            except Exception as e:
                row['error'] = repr(e)
            rows.append(row)
        atomic_json(target, {**measured, **summarize(rows, measured['expected_layers']),
                    'layers': rows, 'weightwatcher': version,
                    'aggregation': 'Unweighted mean of all 36 raw encoder matrix fits; null if any invalid',
                    'excluded': 'Embeddings, tied vocabulary projection and MLM prediction transform',
                    'interpretation': 'One trajectory cannot establish optimizer bias or overfitting'})
    export(root)


def export(root):
    models, layers = [], []
    for path in sorted((root / 'spectra').glob('step_*.json')):
        item = json.loads(path.read_text())
        base = {k: item[k] for k in ('step', 'mean_alpha', 'min_alpha', 'mean_D',
                  'fraction_alpha_below_2', 'any_alpha_below_2', 'valid_layers', 'expected_layers')}
        for split in ('train', 'validation', 'test'):
            for metric in ('mlm_loss', 'masked_token_accuracy'):
                base[split + '_' + metric] = item[split][metric]
        models.append(base)
        for row in item['layers']:
            layers.append({**base, **{k: row.get(k) for k in ('name', 'valid', 'alpha', 'D',
                           'sigma', 'xmin', 'xmax', 'num_pl_spikes', 'num_evals', 'tail_fraction', 'status', 'warning')}})
    for name, rows in [('checkpoint_metrics.csv', models), ('layer_metrics.csv', layers)]:
        if rows:
            path = root / name
            temp = path.with_suffix('.tmp')
            with temp.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temp.replace(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--deadline', type=float, default=float('inf'))
    args = p.parse_args()
    while time.time() < args.deadline:
        analyze(args.root)
        if not args.watch or (Path(args.root) / 'TRAINING_DONE.json').exists():
            break
        time.sleep(15)
