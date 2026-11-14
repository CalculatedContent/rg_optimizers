"""Post-hoc presentation cleanup; no training and no deletion of original outcomes.

Usage: python build_stable_review.py original_endpoints.csv replication_endpoints.csv output_dir
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

original, replication, dest = map(Path, sys.argv[1:])
dest.mkdir(parents=True, exist_ok=True)
for name in ('results', 'plots'):
    (dest / name).mkdir(exist_ok=True)
e = pd.read_csv(original); rep = pd.read_csv(replication)
assert len(e) == 135 and len(rep) == 3
bad_setting = (e.batch_size == 128) & (e.multiplier == 2)
stable = e[~bad_setting].copy()
assert len(stable) == 120 and stable.groupby(['seed', 'arm']).size().eq(8).all()
assert stable.supported_hidden_layers.eq(2).all()
stable.to_csv(dest / 'results/stable_endpoints.csv', index=False)
all_runs = pd.concat([e.assign(cohort='original'), rep.assign(cohort='replication')], ignore_index=True)
bad = (all_runs.batch_size == 128) & (all_runs.multiplier == 2)
all_runs['outcome'] = np.where(bad & (all_runs.arm == 'magnitude_graft'), 'failed_hyperparameter_setting', 'trained_without_observed_collapse')
all_runs['presentation_role'] = np.where(all_runs.outcome == 'failed_hyperparameter_setting', 'failure_register', np.where((all_runs.cohort == 'original') & ~bad, 'stable_comparison', 'retained_control_outside_shared_stable_subset'))
all_runs.to_csv(dest / 'results/run_disposition.csv', index=False)
failures = all_runs[all_runs.outcome == 'failed_hyperparameter_setting']
assert len(failures) == 6 and np.allclose(failures.test_accuracy_pct, 11.35)
failures.to_csv(dest / 'results/hyperparameter_failures.csv', index=False)
metrics = ['alpha_fc1', 'alpha_fc2', 'mean_abs_alpha_minus_2', 'test_accuracy_pct']
arms = ['adamw', 'magnitude_graft', 'sgdm']
labels = dict(zip(arms, ['AdamW', 'Magnitude graft', 'SGD + momentum']))
seed = stable.groupby(['seed', 'arm'])[metrics].mean()
seed.to_csv(dest / 'results/stable_seed_means.csv')
def interval(values):
    a = np.asarray(values, float); assert len(a) == 5
    m = a.mean(); sd = a.std(ddof=1); half = t.ppf(.975, 4) * sd / np.sqrt(5)
    return dict(mean=m, seed_sd=sd, ci_low=m-half, ci_high=m+half)
summary = []
for arm in arms:
    for metric in metrics:
        summary.append(dict(arm=arm, metric=metric, **interval(seed.xs(arm, level='arm')[metric])))
summary = pd.DataFrame(summary); summary.to_csv(dest / 'results/stable_summary.csv', index=False)
pairs = []
for treatment, reference in [('magnitude_graft','adamw'), ('sgdm','adamw'), ('magnitude_graft','sgdm')]:
    delta = seed.xs(treatment, level='arm') - seed.xs(reference, level='arm')
    for metric in metrics:
        pairs.append(dict(treatment=treatment, reference=reference, metric=metric, **interval(delta[metric])))
pairs = pd.DataFrame(pairs); pairs.to_csv(dest / 'results/stable_paired_effects.csv', index=False)
def formatted(row, signed=False):
    f = '+.3f' if signed else '.3f'
    return f"{row['mean']:{f}} [{row['ci_low']:{f}}, {row['ci_high']:{f}}]"
lines = ['# Experiment 01: stable-setting comparison', '',
    '**120 original runs: eight shared settings × five paired seeds × three optimizers.**', '',
    'This is a post-hoc stable-setting analysis requested after the batch128/LR×2 graft failures were observed. It is conditional on the eight settings where all arms trained successfully; it does not replace the prespecified full-grid analysis. Exclude the entire failing setting from this comparison, including its successful controls, to retain identical settings and seeds across arms. Do not filter on α or on which optimizer wins.', '',
    'Included settings: batch128 at LR×0.5/×1; batch256 and batch512 at LR×0.5/×1/×2. All 240 hidden-layer fits in this subset satisfy the original fit-support rule. FC3 has only ten eigenvalues and remains exploratory.', '',
    '## Means over stable settings', '',
    'Entries are mean [95% Student t interval]. Average eight settings within each seed first; then use the five independent seed averages. These descriptive intervals do not account for post-hoc setting selection or multiplicity.', '',
    '| Optimizer | FC1 α | FC2 α | Mean hidden \\|α−2\\| | Test accuracy (%) |', '|---|---:|---:|---:|---:|']
for arm in arms:
    rows = summary[summary.arm == arm].set_index('metric')
    lines.append('| '+labels[arm]+' | '+' | '.join(formatted(rows.loc[m]) for m in metrics)+' |')
lines += ['', '## Paired effects', '',
    'Positive Δα is upward movement; negative Δ|α−2| is movement closer to 2; positive Δaccuracy is higher accuracy. These are distinct outcomes.', '',
    '| Treatment − reference | ΔFC1 α | ΔFC2 α | Δmean \\|α−2\\| | Δaccuracy (pp) |', '|---|---:|---:|---:|---:|']
for (treatment, reference), group in pairs.groupby(['treatment', 'reference'], sort=False):
    rows=group.set_index('metric')
    lines.append('| '+labels[treatment]+' − '+labels[reference]+' | '+' | '.join(formatted(rows.loc[m], True) for m in metrics)+' |')
lines += ['', '![Stable-setting comparison](plots/Stable_Setting_Comparison.png)', '',
    '## Every included setting', '', 'Five-seed mean test accuracy; settings are retained regardless of treatment benefit.', '',
    '| Batch | LR multiplier | AdamW | Graft | SGDM |', '|---:|---:|---:|---:|---:|']
for (b,m),g in stable.groupby(['batch_size','multiplier']):
    means=g.groupby('arm').test_accuracy_pct.mean()
    lines.append(f'| {b} | {m:g} | '+ ' | '.join(f'{means[a]:.3f}%' for a in arms)+' |')
lines += ['', '## Interpretation and preserved evidence', '',
    'The prespecified central setting remains batch256/LR×1, with its original five-seed estimates. Stable training does not establish optimal tuning: AdamW shows increasing late validation cross-entropy despite strong accuracy. Future rate selection should use validation behavior, with α retained as an outcome.', '',
    '[Hyperparameter failures](HYPERPARAMETER_FAILURES.md) · [Original full-grid results](FULL_GRID_RESULTS.md) · [Original full-grid narrative](FULL_GRID_ANALYSIS.md) · [All run dispositions](results/run_disposition.csv) · [Stable endpoints](results/stable_endpoints.csv) · [Paired effects](results/stable_paired_effects.csv) · [Checkpoints and raw data](ARTIFACTS.md)', '']
(dest / 'RESULTS.md').write_text('\n'.join(lines))
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,4,figsize=(14,4.6))
colors=['#2d6fb7','#df7934','#398958']
for ax,metric,title in zip(axs,metrics,['FC1 α','FC2 α','Mean hidden |α − 2|','Test accuracy (%)']):
    for i,arm in enumerate(arms):
        row=summary[(summary.arm==arm)&(summary.metric==metric)].iloc[0]
        ax.errorbar(i,row['mean'],yerr=[[row['mean']-row.ci_low],[row.ci_high-row['mean']]],fmt='o',color=colors[i],capsize=5)
        vals=seed.xs(arm,level='arm')[metric].values
        ax.scatter(i+np.linspace(-.09,.09,5),vals,s=17,color=colors[i],alpha=.5)
    if metric.startswith('alpha'):ax.axhline(2,color='gray',ls='--',lw=1)
    ax.set_xticks(range(3),['AdamW','Graft','SGDM'],rotation=20);ax.set_title(title);ax.grid(axis='y',alpha=.2)
fig.suptitle('Eight shared stable settings · five paired seeds',fontsize=16)
fig.text(.5,.025,'Post-hoc subset: batch128/LR×2 excluded for every arm. Dots = seed averages; bars = descriptive 95% t intervals.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(dest/'plots/Stable_Setting_Comparison.png',dpi=160);plt.close(fig)
print(pairs.to_string(index=False))
print('Verified disposition: 120 balanced comparison runs, 6 failures, 12 retained controls outside the subset; no training.')
