"""Write evidence tables using the fixed five-seed analysis, with no selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import pandas as pd
from analyze import interval, LABELS

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parents[1] / 'repository_update/experiments/adamw_modified/2026-11-13_mnist_mlp_experiment01'


def fmt(r, digits=3, signed=False):
    spec = ('+' if signed else '') + f'.{digits}f'
    return f'{r["mean"]:{spec}} [{r["ci_low"]:{spec}}, {r["ci_high"]:{spec}}]'


def main():
    audit = json.loads((ROOT / 'Final_Verification.json').read_text())
    assert audit['passed'] and audit['completed_runs'] == 135
    e = pd.read_csv(ROOT / 'Endpoints.csv')
    overall = pd.read_csv(ROOT / 'Overall_Summaries.csv')
    paired = pd.read_csv(ROOT / 'Paired_Summaries.csv')
    arms = ('adamw', 'magnitude_graft', 'sgdm')
    lines = ['# Experiment 01: complete results', '',
             'All 135 runs reached epoch 120. Every table retains all nine preset batch/rate settings.', '',
             '## Average over the nine settings', '',
             'Each entry is a mean [95% Student t interval]. Average the nine outcomes within each seed first, then use the five seed averages. The test set is fixed; these intervals describe seed variability, are not multiplicity-adjusted, and do not include uncertainty from resampling the dataset.', '',
             '| Arm | FC1 α | FC2 α | Mean hidden α | Mean hidden \\|α−2\\| | Test accuracy (%) |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        row = [LABELS[arm]]
        for metric in ('alpha_fc1', 'alpha_fc2', 'mean_alpha', 'mean_abs_alpha_minus_2', 'test_accuracy_pct'):
            r = overall[(overall.arm == arm) & (overall.metric == metric)].iloc[0]
            assert r['n'] == 5
            row.append(fmt(r))
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.extend(['', '## Paired average effects', '',
                  'Positive Δα means upward movement. Negative Δmean|α−2| means closer to 2. Positive Δaccuracy means higher accuracy. These are distinct outcomes.', '',
                  '| Treatment − reference | ΔFC1 α | ΔFC2 α | Δmean \\|α−2\\| | Δaccuracy (pp) |',
                  '|---|---:|---:|---:|---:|'])
    for treatment, reference in (('magnitude_graft', 'adamw'), ('sgdm', 'adamw'), ('magnitude_graft', 'sgdm')):
        q = paired[(paired.scope == 'overall') & (paired.treatment == treatment) & (paired.reference == reference)]
        row = [LABELS[treatment] + ' − ' + LABELS[reference]]
        for metric in ('alpha_fc1', 'alpha_fc2', 'mean_abs_alpha_minus_2', 'test_accuracy_pct'):
            row.append(fmt(q[q.metric == metric].iloc[0], signed=True))
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.extend(['', '## All settings: magnitude graft minus native AdamW', '',
                  'Each row uses the same five paired seeds. Entries are paired mean differences [95% t interval].', '',
                  '| Batch | LR multiplier | ΔFC1 α | ΔFC2 α | Δmean \\|α−2\\| | Δaccuracy (pp) |',
                  '|---:|---:|---:|---:|---:|---:|'])
    q = paired[(paired.scope == 'setting') & (paired.treatment == 'magnitude_graft') & (paired.reference == 'adamw')]
    for b in (128, 256, 512):
        for m in (.5, 1., 2.):
            g = q[(q.batch_size == b) & (q.multiplier == m)]
            row = [str(b), f'{m:g}']
            for metric in ('alpha_fc1', 'alpha_fc2', 'mean_abs_alpha_minus_2', 'test_accuracy_pct'):
                row.append(fmt(g[g.metric == metric].iloc[0], signed=True))
            lines.append('| ' + ' | '.join(row) + ' |')
    lines.extend(['', '## Per-setting arm means', '',
                  'Entries below are mean ± seed SD, not confidence intervals. FC3 has only ten eigenvalues and is exploratory; it is excluded from hidden-layer means and distance metrics.', '',
                  '| Batch | LR | Arm | FC1 α | FC2 α | FC3 α (exploratory) | Mean \\|α−2\\| | Accuracy (%) |',
                  '|---:|---:|---|---:|---:|---:|---:|---:|'])
    for b in (128, 256, 512):
        for m in (.5, 1., 2.):
            for arm in arms:
                g = e[(e.batch_size == b) & (e.multiplier == m) & (e.arm == arm)]
                assert len(g) == 5
                row = [str(b), f'{m:g}', LABELS[arm]]
                for metric in ('alpha_fc1', 'alpha_fc2', 'alpha_head', 'mean_abs_alpha_minus_2', 'test_accuracy_pct'):
                    row.append(f'{g[metric].mean():.3f} ± {g[metric].std(ddof=1):.3f}')
                lines.append('| ' + ' | '.join(row) + ' |')
    lines.extend(['', '## Low-accuracy outcomes', '',
                  'The following is a descriptive count with an explicit threshold; no run is excluded from any estimate.', '',
                  '| Arm | Runs with test accuracy <50% | Minimum test accuracy | Maximum test accuracy |',
                  '|---|---:|---:|---:|'])
    for arm in arms:
        g = e[e.arm == arm]
        lines.append(f'| {LABELS[arm]} | {int((g.test_accuracy_pct < 50).sum())}/45 | {g.test_accuracy_pct.min():.2f}% | {g.test_accuracy_pct.max():.2f}% |')
    lines.extend(['', '## Fit support and interpretation', '',
                  f'{audit["analysis_audit"]["final_hidden_supported"]} of 270 final hidden-layer fits satisfy the prespecified support rule: at least 50 eigenvalues, at least 20 tail eigenvalues, KS D ≤0.2. [Common-support comparisons](results/common_support_pairs.csv) retain the exact paired eligible layers.', '',
                  'WeightWatcher 0.7.7 uses its default tail selection, with no α-targeted cutoff. [Layer measurements](results/all_layer_measurements.csv) include fit support, tail size, KS D, norms, and fixed-rank estimates. [Accuracy correlations](results/accuracy_correlations.csv) are descriptive within-arm correlations; the 45 setting/seed points are not 45 independent seeds.', '',
                  'This intervention tests the consequence of changing update magnitude while retaining Adam direction and local moment history. Spectral shifts alone do not establish an RG Jacobian condition or a universal optimizer advantage. Coupling/Jacobian calculations can be performed from the saved states in subsequent analysis.', ''])
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'RESULTS.md').write_text('\n'.join(lines))
    print(DEST / 'RESULTS.md')


if __name__ == '__main__':
    main()
