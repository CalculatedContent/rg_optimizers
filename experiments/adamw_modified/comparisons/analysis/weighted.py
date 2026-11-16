"""Pure, keyed, baseline-weighted spectral comparison. No training or network."""
import numpy as np
import pandas as pd
from scipy.stats import t

KEY = ['seed', 'batch_size', 'multiplier']

def paired_contributions(frame, baseline, treatment, layers, weighting, target=2.):
    a = frame[frame.arm == baseline].set_index(KEY)
    b = frame[frame.arm == treatment].set_index(KEY)
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError('Duplicate paired model key')
    if set(a.index) != set(b.index) or a.empty:
        raise ValueError('Missing or empty paired model keys; no silent intersection')
    b = b.reindex(a.index)
    rows = []
    for key, model in a.iterrows():
        for layer in layers:
            aa, bb = float(model['alpha_' + layer]), float(b.loc[key, 'alpha_' + layer])
            da, db = abs(aa-target), abs(bb-target)
            if weighting == 'absolute_baseline_distance':
                w = da
            elif weighting == 'sub2_baseline_deficit':
                w = max(0., target-aa) if np.isfinite(aa) else np.nan
            else:
                raise ValueError('Unknown weighting: ' + weighting)
            rows.append(dict(zip(KEY, key)) | dict(
                layer=layer, baseline_alpha=aa, treatment_alpha=bb,
                baseline_distance=da, treatment_distance=db, weight=w,
                weighted_baseline=w*da, weighted_treatment=w*db,
                weighted_improvement=w*(da-db),
                baseline_status=model.get('status', 'UNKNOWN'),
                treatment_status=b.loc[key].get('status', 'UNKNOWN')))
    return pd.DataFrame(rows)

def seed_scores(contributions):
    rows = []
    for seed, g in contributions.groupby('seed'):
        cols = ['weight','weighted_baseline','weighted_treatment','weighted_improvement']
        finite = np.isfinite(g[cols].to_numpy()).all()
        total = float(g.weight.sum()) if finite else np.nan
        valid = finite and total > 0
        rows.append(dict(seed=int(seed), total_weight=total, paired_layers=len(g),
            score_status='VALID' if valid else ('ZERO_WEIGHT' if finite else 'NONFINITE'),
            weighted_baseline_distance=g.weighted_baseline.sum()/total if valid else np.nan,
            weighted_treatment_distance=g.weighted_treatment.sum()/total if valid else np.nan,
            improvement=g.weighted_improvement.sum()/total if valid else np.nan))
    return pd.DataFrame(rows)

def interval(values):
    v = np.asarray(values, float)
    n = len(v)
    # Missing seed scores are never silently removed to improve the result.
    if n < 2 or not np.isfinite(v).all():
        return dict(independent_seeds=n, mean=np.nan, ci_low=np.nan, ci_high=np.nan,
                    status='UNDEFINED', valid_seed_scores=int(np.isfinite(v).sum()))
    mean = v.mean(); half = t.ppf(.975, n-1)*v.std(ddof=1)/np.sqrt(n)
    return dict(independent_seeds=n, mean=mean, ci_low=mean-half, ci_high=mean+half,
                status='VALID', valid_seed_scores=n)

def performance_scores(frame, baseline, treatment):
    a = frame[frame.arm==baseline].set_index(KEY)
    b = frame[frame.arm==treatment].set_index(KEY).reindex(a.index)
    values = b[['test_accuracy_pct','test_loss']] - a[['test_accuracy_pct','test_loss']]
    values.columns = ['accuracy_change_pp','ce_change']
    return values.groupby('seed').agg(lambda x:x.mean(skipna=False)).reset_index()
