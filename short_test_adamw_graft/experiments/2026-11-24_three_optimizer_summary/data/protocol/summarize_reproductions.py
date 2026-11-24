"""Summarize all 15 independent reruns without training or spectral refitting."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

T4 = 2.7764451051977934
ARMS = ("sgdm", "adamw", "graft")
SEEDS = range(1340, 1345)


def model_alpha(spectrum):
    rows = spectrum["matrices"]
    assert len(rows) == 18 and len({r['matrix'] for r in rows}) == 18
    weighted, weights = [], []
    for row in rows:
        assert row['matrix'] != 'lm_head.weight'
        assert len(row['fits']) == 1
        f = row['fits'][0]
        assert f['status'] == 'success' and math.isfinite(f['alpha']) and f['alpha'] > 0
        assert math.isfinite(f['D']) and 0 <= f['D'] <= 1
        assert type(f['num_pl_spikes']) is int and 0 < f['num_pl_spikes'] <= f['num_evals']
        w = math.sqrt(f['num_pl_spikes']) / (f['D'] + 0.02)
        weights.append(w)
        weighted.append(w * f['alpha'])
    return math.fsum(weighted) / math.fsum(weights)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_root', type=Path)
    a = p.parse_args()
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            path = a.output_root / arm / f'seed_{seed}' / 'state' / arm / f'seed_{seed}' / 'result.json'
            r = json.loads(path.read_text())
            assert r['completed'] and r['arm'] == arm and r['seed'] == seed
            assert r['endpoint']['step'] == 20000
            assert [x['step'] for x in r['history']] == list(range(0, 20001, 500))
            for q in r['history']:
                ce = q['validation_loss']
                rows.append(dict(arm=arm, seed=seed, step=q['step'], alpha=model_alpha(q['alpha']),
                                 validation_ce=ce, validation_ppl=math.exp(ce),
                                 validation_accuracy=q['validation']['next_character_accuracy']))
    summary = []
    for arm in ARMS:
        for step in range(0, 20001, 500):
            g = [r for r in rows if r['arm'] == arm and r['step'] == step]
            assert len(g) == 5
            item = dict(arm=arm, step=step, n=5)
            for metric in ('alpha', 'validation_ce', 'validation_ppl', 'validation_accuracy'):
                v = [r[metric] for r in g]
                mean, sd = st.fmean(v), st.stdev(v)
                half = T4 * sd / math.sqrt(5)
                item.update({f'{metric}_mean': mean, f'{metric}_sd': sd,
                             f'{metric}_ci95_low': mean-half, f'{metric}_ci95_high': mean+half})
            summary.append(item)
    out = a.output_root / 'analysis'
    out.mkdir(exist_ok=True)
    write_csv(out/'per_seed.csv', rows)
    write_csv(out/'trajectory.csv', summary)
    write_csv(out/'endpoints.csv', [r for r in summary if r['step'] in (2000, 20000)])
    (out/'status.json').write_text(json.dumps(dict(completed=True, n_seeds_per_arm=5,
        n_runs=15, n_measurements=len(rows), cohort='independent_local_reproduction',
        uncertainty='Equal seed weights; sampleSD;pointwise95%Student-t(df4)CI'), indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
