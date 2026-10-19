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


def analyze(root):
    import weightwatcher as ww
    if importlib.metadata.version('weightwatcher') != '0.7.7':
        raise RuntimeError('Protocol requires WeightWatcher 0.7.7')
    torch.set_num_threads(1)
    root = Path(root)
    for measurement in sorted((root / 'measurements').glob('step_*.json')):
        target = root / 'spectra' / measurement.name
        if target.exists():
            continue
        measured = json.loads(measurement.read_text())
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
        valid = [r['alpha'] for r in rows if r['included_in_mean']]
        expected = sum(min(r['shape']) >= 10 for r in rows)
        complete = len(valid) == expected
        atomic_json(target, {'step': measured['step'], 'weightwatcher': '0.7.7', 'raw_matrices': True,
                    'layers': rows, 'valid_layers': len(valid), 'expected_layers': expected,
                    'all_eligible_fits_valid': complete,
                    'mean_alpha': float(np.mean(valid)) if valid and complete else None,
                    'fraction_alpha_below_2': float(np.mean(np.array(valid) < 2)) if valid and complete else None,
                    'aggregation': 'Unweighted arithmetic mean over every non-embedding dense matrix with rank >= 10; requires all fits valid',
                    'train': measured['train'], 'validation': measured['validation'],
                    'generalization_gap_pp': measured['generalization_gap_pp'],
                    'synthetic_engineering_only': measured['synthetic_engineering_only']})


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
