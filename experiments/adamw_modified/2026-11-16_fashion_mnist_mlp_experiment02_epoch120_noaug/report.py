"""All-outcome descriptive analysis of the fixed continuation; no selection."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from step_accounting import accounting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'Protocol.json').read_text())
OUT = ROOT / 'results'
PLOTS = ROOT / 'plots'
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)
ARMS = ['adamw', 'magnitude_graft', 'sgdm']
COLORS = dict(zip(ARMS, ['#2867A6', '#CD6B27', '#35815B']))
KEY = ['seed', 'batch_size', 'multiplier', 'arm']
METRICS = ['alpha_fc1', 'alpha_fc2', 'alpha_fc3', 'distance_fc1', 'distance_fc2', 'distance_fc3',
           'mean_alpha', 'mean_abs_alpha_minus_2', 'alpha_layer_variance',
           'test_accuracy_pct', 'test_loss', 'validation_loss', 'parameter_weighted_alpha', 'harmonic_alpha',
           'capped_alpha', 'sub2_deficit', 'rms_sub2_deficit', 'fraction_sub2', 'test_ece15', 'test_brier']


def read(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def write(p, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if hasattr(x, 'item'): x = x.item()
        return None if isinstance(x, float) and not np.isfinite(x) else x
    p.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')


def run_id(sp):
    return f"b{sp['batch_size']}_lr{sp['multiplier']:g}/{sp['arm']}/seed_{sp['seed']}"


def measurements(r):
    row = {**r['spec'], 'epoch': r['endpoint_epoch'], 'status': r['status'],
           'test_accuracy_pct': 100 * r['test']['accuracy'], 'test_loss': r['test']['loss'],
           'validation_loss': r['validation']['loss']}
    for n in ['fc1', 'fc2', 'fc3']:
        alpha = r['spectra'][n]['alpha']
        row['alpha_' + n] = np.nan if alpha is None else alpha
        row['distance_' + n] = abs(row['alpha_' + n] - 2)
        row['supported_' + n] = r['spectra'][n]['supported']
    a = np.array([row['alpha_fc1'], row['alpha_fc2']])
    row.update(mean_alpha=a.mean(), mean_abs_alpha_minus_2=abs(a - 2).mean(), alpha_layer_variance=a.var())
    deficit = np.maximum(0, 2-a)
    row.update(parameter_weighted_alpha=float(np.average(a,weights=[784*256,256*256])),
               harmonic_alpha=float(2/np.sum(1/a)) if np.all(a>0) else np.nan,
               capped_alpha=float(np.minimum(a,2).mean()), sub2_deficit=float(deficit.mean()),
               rms_sub2_deficit=float(np.sqrt((deficit**2).mean())), fraction_sub2=float((a<2).mean()))
    d=(ROOT/'parent_states' if r['endpoint_epoch']==60 else ROOT)/'runs'/run_id(r['spec'])
    logits_path=d/'final_test_logits.npz'
    row['test_ece15']=row['test_brier']=np.nan
    if logits_path.exists():
        z=np.load(logits_path); logits=z['logits'].astype(float); labels=z['labels'].astype(int)
        probs=np.exp(logits-logits.max(axis=1,keepdims=True)); probs/=probs.sum(axis=1,keepdims=True)
        confidence=probs.max(axis=1); correct=probs.argmax(axis=1)==labels
        bins=np.minimum((confidence*15).astype(int),14)
        row['test_ece15']=float(sum(np.mean(bins==b)*abs(correct[bins==b].mean()-confidence[bins==b].mean()) for b in range(15) if np.any(bins==b)))
        row['test_brier']=float(np.mean(np.sum((probs-np.eye(10)[labels])**2,axis=1)))
        assert abs(100*correct.mean()-row['test_accuracy_pct'])<1e-10
    return row


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2 or not np.isfinite(x).all():
        return {'independent_seeds': n, 'mean': float(x.mean()) if n else np.nan,
                'ci_low': np.nan, 'ci_high': np.nan, 'seed_sd': np.nan}
    sd = x.std(ddof=1)
    mean = x.mean()
    half = t.ppf(.975, n - 1) * sd / math.sqrt(n)
    return {'independent_seeds': n, 'mean': mean, 'ci_low': mean - half, 'ci_high': mean + half, 'seed_sd': sd}


rows, statuses, layers, curves = [], [], [], []
audit = {'planned': 120, 'completed': 0, 'paired_initialization': True, 'paired_minibatches': True,
         'fixed_epoch120': True, 'restored_replay': True, 'source_checkpoint_match': True,
         'step_and_lr_accounting': True, 'graft_nonzero_records': 0, 'maximum_graft_norm_relative_error': 0.}
fingerprints = {}
for sp in PLAN['specifications']:
    d = ROOT / 'runs' / run_id(sp)
    parent = read(ROOT / 'parent_states/runs' / run_id(sp) / 'done.json') or read(d / 'parent_done.json')
    r = read(d / 'done.json')
    if parent:
        rows.append(measurements(parent))
    if not r:
        failure = read(d / 'failure.json')
        statuses.append({**sp, 'status': 'ERROR' if failure else 'INCOMPLETE', 'error': failure.get('error') if failure else None,
                         'last_completed_epoch': read(d / 'history.json', [{}])[-1].get('epoch')})
    else:
        row = measurements(r)
        rows.append(row)
        statuses.append({**row, 'verification_passed': r['verification']['passed']})
        audit['completed'] += 1
        expected=accounting(sp,120)
        audit['step_and_lr_accounting'] &= r['accounting']['observed_cumulative_updates']==expected['cumulative_updates'] and r['accounting']['optimizer_layer_lrs']==expected['optimizer_layer_lrs']
        audit['fixed_epoch120'] &= r['endpoint_epoch'] == 120
        audit['restored_replay'] &= r['verification']['checkpoint_replay_exact'] and r['verification']['restored_validation_and_spectra_exact'] and r['verification']['optimizer_state_restored_exact']
        audit['source_checkpoint_match'] &= bool(parent and r['parent_checkpoint_sha256'] == parent['checkpoint_hashes']['final.pt'])
        hist = read(d / 'history.json', [])
        fp = (r['initial_hash'], [h['batch_order_sha256'] for h in hist])
        group = (sp['seed'], sp['batch_size'], sp['multiplier'])
        if group in fingerprints:
            audit['paired_initialization'] &= fp[0] == fingerprints[group][0]
            audit['paired_minibatches'] &= fp[1] == fingerprints[group][1]
        fingerprints[group] = fp
        for n, fit in r['spectra'].items():
            layers.append({**sp, 'epoch': 120, 'layer': n, **fit})
    for h in read(d / 'history.json', []):
        row = {**sp, 'epoch': h['epoch'], 'validation_loss': h['validation']['loss'],
               'validation_accuracy_pct': 100 * h['validation']['accuracy'],
               'train_loss': h.get('train', {}).get('loss', np.nan)}
        if 'spectra' in h:
            for n in ['fc1', 'fc2', 'fc3']:
                row['alpha_' + n] = h['spectra'][n]['alpha']
                row['supported_' + n] = h['spectra'][n]['supported']
        curves.append(row)
    if sp['arm'] == 'magnitude_graft':
        for path in (d / 'statistics').glob('epoch_*.npz'):
            values = np.load(path)['values']
            target, actual = values[:, :, 3], values[:, :, 4]
            good = (target > 0) & np.isfinite(target) & np.isfinite(actual)
            audit['graft_nonzero_records'] += int(good.sum())
            if good.any():
                audit['maximum_graft_norm_relative_error'] = max(audit['maximum_graft_norm_relative_error'], float(np.max(abs(actual[good] / target[good] - 1))))

e=pd.DataFrame(rows,columns=KEY+['epoch','status']+METRICS+['supported_fc1','supported_fc2','supported_fc3'])
tr=pd.DataFrame(curves)
account_rows=[]
for sp in PLAN['specifications']:
    for epoch in [20,60,120]:
        a=accounting(sp,epoch)
        account_rows.append({**sp,**{k:v for k,v in a.items() if not isinstance(v,list)},**{f'{k}_fc{i+1}':v for k,ls in a.items() if isinstance(ls,list) for i,v in enumerate(ls)}})
pd.DataFrame(account_rows).to_csv(OUT/'step_lr_accounting.csv',index=False)
pd.DataFrame(statuses).to_csv(OUT / 'model_status.csv', index=False)
e.to_csv(OUT / 'endpoints60_and120.csv', index=False)
pd.DataFrame(layers).to_csv(OUT / 'layer_measurements.csv', index=False)
tr.to_csv(OUT / 'trajectories.csv', index=False)
audit['parent_validation'] = read(ROOT / 'Parent_Validation.json', {'passed': False, 'count': 0})
audit['continuation_gate'] = read(ROOT / 'Continuation_Gate.json', {'passed': False})
audit['optimizer_contract'] = read(ROOT / 'Optimizer_Contract.json', {'passed': False})
audit['resumed_optimizer_contract'] = read(ROOT / 'Resumed_Optimizer_Contract.json', {'passed': False})
audit['passed'] = audit['completed'] == 120 and all(audit[k] for k in ['paired_initialization', 'paired_minibatches', 'fixed_epoch120', 'restored_replay', 'source_checkpoint_match', 'step_and_lr_accounting']) and all(audit[k]['passed'] for k in ['parent_validation', 'continuation_gate', 'optimizer_contract', 'resumed_optimizer_contract']) and audit['parent_validation']['count'] == 120 and audit['graft_nonzero_records'] > 0 and audit['maximum_graft_norm_relative_error'] < 2e-4
audit['replay_negative_controls'] = read(ROOT / 'Replay_Negative_Controls.json', {'passed': False})
audit['passed'] = audit['passed'] and audit['replay_negative_controls']['passed']
records = audit['parent_validation'].get('records', [])
audit['parent_cross_runtime_exact_count'] = sum(bool(x.get('parent_replay_epoch21_to22_exact')) for x in records)
audit['parent_current_runtime_exact_count'] = sum(bool(x.get('current_runtime_replay_exact')) for x in records)
audit['maximum_parent_replay_loss_error'] = max((x.get('maximum_replay_loss_error', 0) for x in records), default=0)
audit['passed'] = audit['passed'] and audit['parent_current_runtime_exact_count'] == 120
write(OUT / 'Verification.json', audit)
complete = audit['completed'] == 120
means, effects, changes, interactions = [], [], [], []
if complete:
    assert len(e) == 240 and not e.duplicated(KEY + ['epoch']).any()
    assert e.groupby(['seed', 'arm', 'epoch']).size().eq(8).all()
    scopes = [('all_settings', e)] + [(f'b{b}_lr{m:g}', g) for (b, m), g in e.groupby(['batch_size', 'multiplier'])]
    for scope, frame in scopes:
        avg = frame.groupby(['seed', 'arm', 'epoch'])[METRICS].agg(lambda x: x.mean(skipna=False))
        for epoch in [60, 120]:
            at = avg.xs(epoch, level='epoch')
            for arm in ARMS:
                for metric in METRICS:
                    means.append({'scope': scope, 'epoch': epoch, 'arm': arm, 'metric': metric, **ci(at.xs(arm, level='arm')[metric])})
            for arm in ARMS[1:]:
                delta = at.xs(arm, level='arm') - at.xs('adamw', level='arm')
                for metric in METRICS:
                    effects.append({'scope': scope, 'epoch': epoch, 'treatment': arm, 'reference': 'adamw', 'metric': metric, **ci(delta[metric])})
        change = avg.xs(120, level='epoch') - avg.xs(60, level='epoch')
        for arm in ARMS:
            for metric in METRICS:
                changes.append({'scope': scope, 'arm': arm, 'metric': metric, 'contrast': 'epoch120_minus_epoch60', **ci(change.xs(arm, level='arm')[metric])})
        for arm in ARMS[1:]:
            delta = change.xs(arm, level='arm') - change.xs('adamw', level='arm')
            for metric in METRICS:
                interactions.append({'scope': scope, 'treatment': arm, 'reference': 'adamw', 'metric': metric, 'contrast': 'change_in_optimizer_effect_120_minus60', **ci(delta[metric])})
for name, data in [('seed_summaries', means), ('paired_effects', effects), ('within_arm_changes', changes), ('effect_changes', interactions)]:
    pd.DataFrame(data).to_csv(OUT / (name + '.csv'), index=False)
write(OUT / 'NoAug120_Result_Summary.json', {'dataset': 'Fashion-MNIST', 'planned': 120,
      'completed': audit['completed'], 'complete': complete, 'verification_passed': audit['passed'],
      'parent_epoch': 60, 'fixed_final_epoch': 120, 'independent_seeds': 5, 'exploratory_extension': True,
      'paired_effects': effects, 'within_arm_changes': changes, 'effect_changes': interactions,
      'failures_retained': True, 'gpt_on_hold': True})

if len(tr):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    for ax, ((batch, mult), group) in zip(axes.flat, tr.groupby(['batch_size', 'multiplier'])):
        for arm, g in group.groupby('arm'):
            for seed, h in g.groupby('seed'):
                ax.plot(h.epoch, h.validation_loss, color=COLORS[arm], alpha=.35, lw=1)
            mean = g.groupby('epoch').validation_loss.mean()
            ax.plot(mean.index, mean.values, color=COLORS[arm], label=arm, lw=2)
        ax.axvline(60, color='black', ls=':', lw=1)
        ax.set(xlabel='Epoch', title=f'Batch {batch}; LR ×{mult:g}', yscale='log')
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel('Validation cross-entropy')
    axes[1, 0].set_ylabel('Validation cross-entropy')
    fig.suptitle('Raw validation curves; continuation begins after epoch 60\nLogarithmic loss axis; all finite values shown')
    fig.tight_layout()
    fig.savefig(PLOTS / 'Validation_Trajectories.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, layer in zip(axes, ['fc1', 'fc2']):
        metric = 'alpha_' + layer
        for arm, g in tr.groupby('arm'):
            h = g.dropna(subset=[metric]).groupby(['seed', 'epoch'])[metric].mean().groupby('epoch').mean()
            ax.plot(h.index, h.values, color=COLORS[arm], label=arm)
        ax.axhline(2, color='black', ls='--', lw=1)
        ax.axvline(60, color='gray', ls=':')
        ax.set(title=layer.upper(), xlabel='Epoch', ylabel='Mean layer α (all settings)')
        ax.grid(alpha=.15)
    axes[1].legend()
    fig.suptitle('Alpha movement; approaching 2 is a separate outcome')
    fig.tight_layout()
    fig.savefig(PLOTS / 'Alpha_Trajectories.png', dpi=160)
    plt.close(fig)
if complete:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, label in zip(axes, ['test_accuracy_pct', 'test_loss', 'mean_abs_alpha_minus_2'],
                                ['Test accuracy (%)', 'Test cross-entropy', 'Mean hidden |α − 2|']):
        for i, arm in enumerate(ARMS):
            g = e[e.arm == arm].groupby(['seed', 'epoch'])[metric].mean().unstack('epoch')
            for _, row in g.iterrows():
                ax.plot([60, 120], [row[60], row[120]], color=COLORS[arm], alpha=.25, lw=1)
            center = [ci(g[x]) for x in [60, 120]]
            ax.errorbar([60 + (i - 1) * .6, 120 + (i - 1) * .6], [x['mean'] for x in center],
                        yerr=[x['ci_high'] - x['mean'] for x in center], color=COLORS[arm], label=arm,
                        marker='o', capsize=4, lw=2)
        ax.set(xlabel='Fixed endpoint epoch', ylabel=label, xticks=[60, 120])
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=9)
    fig.suptitle('Five seed means; 95% t intervals across seeds\nAll eight settings within each seed; no outcome filtering')
    fig.tight_layout()
    fig.savefig(PLOTS / 'Endpoint_Comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    end = e[e.epoch == 120]
    for arm, g in end.groupby('arm'):
        axes[0].scatter(g.mean_alpha, g.test_accuracy_pct, color=COLORS[arm], label=arm, alpha=.65)
        axes[1].scatter(g.mean_abs_alpha_minus_2, g.test_accuracy_pct, color=COLORS[arm], label=arm, alpha=.65)
    axes[0].axvline(2, color='gray', ls='--')
    axes[0].set_xlabel('Mean hidden α')
    axes[1].set_xlabel('Mean hidden |α − 2|')
    for ax in axes:
        ax.set_ylabel('Epoch120 test accuracy (%)')
        ax.grid(alpha=.15)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(PLOTS / 'Alpha_Accuracy.png', dpi=160)
    plt.close(fig)


if complete:
    weighted=[]
    for arm in ARMS:
        for seed in PLAN['independent_seeds']:
            g=e[(e.arm==arm)&(e.seed==seed)]
            pre=g[g.epoch==60].set_index(['batch_size','multiplier']).sort_index()
            post=g[g.epoch==120].set_index(['batch_size','multiplier']).sort_index()
            for mode in ['symmetric','sub2']:
                a=pre[['alpha_fc1','alpha_fc2']].to_numpy();b=post[['alpha_fc1','alpha_fc2']].to_numpy()
                d0=abs(a-2) if mode=='symmetric' else np.maximum(0,2-a)
                d1=abs(b-2) if mode=='symmetric' else np.maximum(0,2-b)
                denom=d0.sum()
                score=float(np.sum(d0*(d0-d1))/denom) if denom>0 and np.isfinite(d0).all() and np.isfinite(d1).all() else np.nan
                weighted.append({'arm':arm,'seed':seed,'metric':mode+'_severity_weighted_rescue','score':score,'weight_sum':float(denom)})
    w=pd.DataFrame(weighted);w.to_csv(OUT/'severity_weighted_seed_scores.csv',index=False)
    wr=[]
    for (arm,metric),g in w.groupby(['arm','metric']):wr.append({'arm':arm,'metric':metric,**ci(g.score)})
    pd.DataFrame(wr).to_csv(OUT/'severity_weighted_intervals.csv',index=False)

lines = ['# Fashion-MNIST fixed continuation: epoch 60 → 120', '',
         f"Completed: {audit['completed']}/120. Full verification passed: **{audit['passed']}**.", '',
         'Same120 unaugmented trajectories, restored from their exact epoch60 model/optimizer/RNG checkpoints. The continuation holds the actual terminal LR at1% of the original peak for epochs61-120. No warmup restart, augmentation, new seeds or retuning. Every endpoint is fixed before continuation. All outcomes and unsupported fits remain; FC3 is exploratory with10 eigenvalues.', '',
         'This is an exploratory follow-up after seeing prior results, using the previously seen test set. Five seeds, not eight settings, are independent. Settings are averaged within each seed before paired95%t intervals. These are descriptive unadjusted intervals. The separate augmented120epoch cohort uses different seeds and a different schedule; it is not a controlled augmentation comparison.', '',
         'At B256/LR1, epoch60 has12900 optimizer updates and epoch120 has25800. AdamW LR remains0.00001; SGDM matrix target rates remain[0.0004,0.002,0.008]. Other batches/multipliers retain their original settings. Full per-layer cumulative LR, squared LR, sample counts and explicit decay are in results/step_lr_accounting.csv. Cumulative LR does not fully describe adaptive effective steps.', '',
         '[Parent epoch60 results](../2026-11-16_fashion_mnist_mlp_experiment02_extension_retry01/RESULTS.md) · [All model statuses](results/model_status.csv) · [Epoch60/120 endpoints](results/endpoints60_and120.csv) · [Paired effects](results/paired_effects.csv) · [Within-arm changes](results/within_arm_changes.csv) · [Effect changes](results/effect_changes.csv) · [Verification](results/Verification.json)', '']
if not complete:
    lines += ['**Partial result: no complete five-seed inference is claimed.**', '']
else:
    lines += ['| Epoch | Arm | FC1 α | FC2 α | Mean hidden distance to2 | Test accuracy | Test CE |',
              '|---|---|---:|---:|---:|---:|---:|']
    for (epoch, arm), group in e.groupby(['epoch', 'arm']):
        mean = group[METRICS].agg(lambda x:x.mean(skipna=False))
        lines.append(f"| {epoch} | {arm} | {mean.alpha_fc1:.4f} | {mean.alpha_fc2:.4f} | {mean.mean_abs_alpha_minus_2:.4f} | {mean.test_accuracy_pct:.3f}% | {mean.test_loss:.4f} |")
    lines += ['', 'Graft − native AdamW at epoch120 (settings averaged within each seed):', '',
              '| Metric | Mean difference | Paired 95% interval |', '|---|---:|---:|']
    for row in effects:
        if row['scope'] == 'all_settings' and row['epoch'] == 120 and row['treatment'] == 'magnitude_graft':
            lines.append(f"| {row['metric']} | {row['mean']:+.5f} | [{row['ci_low']:+.5f}, {row['ci_high']:+.5f}] |")
    lines += ['', 'Increasing α, decreasing |α−2|, increasing accuracy and decreasing CE are different outcomes. Positive accuracy differences favor graft; negative distance/CE differences favor graft. Intervals including0 do not establish improvement. CIs are descriptive and unadjusted for multiple comparisons. Functional quality is labeled separately from spectral alignment.', '']
for path in sorted(PLOTS.glob('*.png')):
    lines += [f'![{path.stem}](plots/{path.name})', '']
(ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
print({'completed': audit['completed'], 'verified': audit['passed']}, flush=True)
if complete and not audit['passed']:
    raise RuntimeError('Numerical verification failed; all available outcomes retained')
