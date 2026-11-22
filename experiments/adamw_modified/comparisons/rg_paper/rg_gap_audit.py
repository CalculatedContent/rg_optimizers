"""Measure the signed WeightWatcher ERG rank gap in the archived MLP runs.

Usage: python rg_gap_audit.py PATH_TO_ARCHIVES layer_gaps.csv.gz
Requires NumPy and pandas. The input directory holds the original experiment
ZIPs (including the GitHub artifact wrappers). No retraining or refitting.
"""
import argparse
import io
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = ('fc1', 'fc2', 'fc3')
KEY = ['cohort', 'seed', 'batch', 'rate', 'arm', 'epoch', 'layer']


def measure(values, fit):
    eigen = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    if not np.all(np.isfinite(eigen)) or eigen[-1] < 0 or eigen[0] <= 0:
        raise ValueError('A Gram spectrum has invalid eigenvalues')
    if eigen[0] == eigen[-1]:
        raise ValueError('A flat spectrum has no determinant crossing')
    m = eigen.size
    # This is WeightWatcher.detX_constraint in log space. Include the
    # crossing mode. If none crosses earlier, WeightWatcher returns M.
    with np.errstate(divide='ignore'):
        logs = np.log(eigen / eigen.mean())
    cumulative = np.cumsum(logs)
    crossed = np.flatnonzero(cumulative < 0)
    erg_rank = int(crossed[0]) + 1 if crossed.size else m
    if fit.get('small_n'):
        f = fit['small_n']
        k = int(f['n_tail'])
    else:
        f = fit
        k = f.get('num_pl_spikes')
    alpha, distance = f.get('alpha'), f.get('D')
    if k is None or not (0 < k <= m) or alpha is None or distance is None:
        return dict(M=m, k_erg=erg_rank, fit_valid=False)
    k = int(k)
    gap = erg_rank-k
    return dict(M=m, k_erg=erg_rank, k_pl=k, gap=gap, g=gap/m,
                trace_log=cumulative[k-1]/k, alpha=alpha, D=distance,
                weight=np.sqrt(k)/(distance+.02), fit_valid=True)


def cohort_name(name):
    if name.startswith('Experiment01_Seed'):
        return 'Digits sweep'
    if name.startswith(('Experiment02_', 'Fashion-Recovery-')):
        return 'Fashion original'
    if name.startswith(('Fashion_Clean_', 'Fashion_Retry01_', 'Fashion-Retry01-', 'Fashion-NoAug120-')):
        return 'Fashion continuation'
    if name.startswith('Fashion-Scale-Checkpoints'):
        return 'Fashion augmented matched'
    if name.startswith('MNIST-AdamW-Augmented120-Seed'):
        return 'Digits augmented native'
    if name.startswith('Fashion-AdamW-Paired120-Seed'):
        return 'Fashion paired native'
    return None


def records(archive, cohort):
    names = set(archive.namelist())
    for wrapper in sorted(n for n in names if n.endswith('.zip')):
        with tempfile.TemporaryFile() as local:
            with archive.open(wrapper) as source:
                shutil.copyfileobj(source, local, length=1024*1024)
            with zipfile.ZipFile(local) as nested:
                yield from records(nested, cohort)
    for member in sorted(names):
        if not (member.startswith('runs/') and member.endswith('/history.json')):
            continue
        prefix = member.rsplit('/', 1)[0]
        parts = prefix.split('/')
        seed = int(parts[-1].removeprefix('seed_'))
        group = cohort
        if len(parts) == 4:
            batch, rate = re.fullmatch(r'b(\d+)_lr([\d.]+)', parts[1]).groups()
            batch, rate, arm = int(batch), float(rate), parts[2]
        elif len(parts) == 3 and parts[1] in ('augmented','unaugmented'):
            batch, rate, arm = 256, 1., 'adamw'
            if group == 'Fashion paired native':
                group += ' '+parts[1]
        else:
            raise ValueError(prefix)
        for row in json.loads(archive.read(member)):
            epoch = int(row['epoch'])
            spec = f'{prefix}/spectra/epoch_{epoch:03d}.npz'
            if spec not in names or not row.get('spectra'):
                continue
            with np.load(io.BytesIO(archive.read(spec))) as spectra:
                for layer in LAYERS:
                    yield dict(cohort=group,seed=seed,batch=batch,rate=rate,
                               arm=arm,epoch=epoch,layer=layer,
                               test_accuracy_pct=100*(row.get('test') or {}).get('accuracy', np.nan),
                               validation_accuracy_pct=100*(row.get('validation') or {}).get('accuracy', np.nan),
                               **measure(spectra[layer],row['spectra'][layer]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive_directory', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify-paper', action='store_true',
                        help='Compare the rebuilt results with the paper checkpoint table.')
    args = parser.parse_args()
    frames = []
    for path in sorted(args.archive_directory.glob('*.zip')):
        cohort = cohort_name(path.name)
        if cohort:
            with zipfile.ZipFile(path) as archive:
                table = pd.DataFrame(records(archive,cohort))
            print(path.name, len(table), flush=True)
            if len(table):
                frames.append(table)
    data = pd.concat(frames,ignore_index=True)
    repeated = data.duplicated(KEY,keep=False)
    measures = ['k_erg','k_pl','gap','M','alpha','D']
    variation = data[repeated].groupby(KEY)[measures].nunique()
    if variation.gt(1).any().any():
        raise ValueError('Conflicting duplicates in the restored seed records')
    data = data.drop_duplicates(KEY).sort_values(KEY)
    if args.verify_paper:
        paper = pd.read_csv(Path(__file__).with_name('optimizer_checkpoints.csv'))
        matched = paper.merge(data, on=KEY, how='left', validate='one_to_one',
                              suffixes=('_paper', '_raw'), indicator=True)
        if not matched['_merge'].eq('both').all():
            raise ValueError('Missing archived checkpoints required by the paper')
        for col in ['M', 'k_pl', 'k_erg', 'gap', 'D', 'weight']:
            np.testing.assert_allclose(matched[col+'_paper'], matched[col+'_raw'])
        np.testing.assert_allclose(matched.alpha_layer, matched.alpha)
        print(f'Verified layer fits, weights and integer counts for {len(matched)} paper rows.')
    data.to_csv(args.output,index=False)
    print(data.groupby('cohort').size().to_string(), flush=True)


if __name__ == '__main__':
    main()
