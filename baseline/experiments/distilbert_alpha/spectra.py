"""CPU-only default WeightWatcher 0.7.7 fits on saved raw dense matrices."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from common import atomic_json, file_hash


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def summarize(rows, measured):
    """Promote saved fit fields without refitting or changing fit parameters."""
    for row in rows:
        fit = row.get('fit', {})
        row['included_in_mean'] = (min(row['shape']) >= 10 and finite(row.get('alpha')) and row['alpha'] > 0
                                  and fit.get('status', 'success') == 'success')
        for key in ('D', 'sigma', 'xmin', 'xmax', 'num_pl_spikes', 'num_evals'):
            row[key] = fit.get(key)
        row['fit_status'] = fit.get('status')
        row['fit_warning'] = fit.get('warning')
        row['D_valid'] = (min(row['shape']) >= 10 and finite(row['D']) and 0 <= row['D'] <= 1
                          and fit.get('status', 'success') == 'success')
        row['tail_fraction'] = (row['num_pl_spikes'] / row['num_evals']
                                if finite(row['num_pl_spikes']) and finite(row['num_evals']) and row['num_evals'] > 0 else None)
    alphas = [r['alpha'] for r in rows if r['included_in_mean']]
    distances = [r['D'] for r in rows if r['D_valid']]
    expected = sum(min(r['shape']) >= 10 for r in rows)
    complete = len(alphas) == expected and expected > 0
    d_complete = len(distances) == expected and expected > 0
    return {'schema': 2, 'step': measured['step'], 'epoch': measured.get('epoch'),
            'weightwatcher': '0.7.7', 'raw_matrices': True, 'layers': rows,
            'valid_layers': len(alphas), 'expected_layers': expected, 'valid_D_layers': len(distances),
            'all_eligible_fits_valid': complete, 'all_eligible_D_valid': d_complete,
            'mean_alpha': float(np.mean(alphas)) if complete else None,
            'fraction_alpha_below_2': float(np.mean(np.array(alphas) < 2)) if complete else None,
            'mean_D': float(np.mean(distances)) if d_complete else None,
            'median_D': float(np.median(distances)) if d_complete else None,
            'max_D': float(np.max(distances)) if d_complete else None,
            'aggregation': 'Unweighted dense-matrix averages excluding embeddings and rank < 10; require every eligible fit',
            'D_definition': 'Kolmogorov-Smirnov tail distance; lower means closer fit; not a p-value',
            'train': measured['train'], 'validation': measured['validation'], 'test': measured.get('test'),
            'generalization_gap_pp': measured['generalization_gap_pp'],
            'train_test_gap_pp': measured.get('train_test_gap_pp'),
            'test_used_for_selection': False,
            'synthetic_engineering_only': measured['synthetic_engineering_only']}


def analyze(root):
    import weightwatcher as ww
    if importlib.metadata.version('weightwatcher') != '0.7.7':
        raise RuntimeError('Protocol requires WeightWatcher 0.7.7')
    torch.set_num_threads(1)
    root = Path(root)
    for measurement in sorted((root / 'measurements').glob('step_*.json')):
        target = root / 'spectra' / measurement.name
        measured = json.loads(measurement.read_text())
        if target.exists():
            old = json.loads(target.read_text())
            if old.get('schema') != 2 or old.get('test') != measured.get('test'):
                atomic_json(target, summarize(old['layers'], measured))
            continue
        snapshot = root / measured['spectral_snapshot']
        if file_hash(snapshot) != measured['snapshot_sha256']:
            raise RuntimeError('Spectral snapshot hash mismatch')
        matrices = torch.load(snapshot, map_location='cpu', weights_only=True)
        rows = []
        for name, weight in matrices.items():
            row = {'name': name, 'shape': list(weight.shape), 'included_in_mean': False}
            if min(weight.shape) < 10:
                row.update(status='excluded_insufficient_rank', reason='binary classification head has only two singular values')
            else:
                try:
                    module = torch.nn.Sequential(torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False))
                    module[0].weight.data.copy_(weight)
                    # No custom fit, normalization, xmin, or power-law parameters.
                    df = ww.WeightWatcher(model=module).analyze()
                    fits = json.loads(df.to_json(orient='records'))
                    fit = fits[0] if fits else {}
                    alpha = fit.get('alpha')
                    valid = isinstance(alpha, (int, float)) and math.isfinite(alpha) and alpha > 0
                    row.update(status='fit' if valid else 'invalid_fit', fit=fit, alpha=alpha,
                               included_in_mean=valid)
                except Exception as exc:
                    row.update(status='fit_error', error=repr(exc))
            rows.append(row)
        atomic_json(target, summarize(rows, measured))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--deadline', type=float, default=float('inf'))
    a = p.parse_args()
    while time.time() < a.deadline:
        analyze(a.root)
        if not a.watch:
            break
        status = Path(a.root) / 'TRAINING_DONE.json'
        if status.exists():
            analyze(a.root)
            break
        time.sleep(15)
