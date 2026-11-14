"""Verify and describe the single fresh-seed replication; never retrain."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import study as s
from verification_helpers import direction_check
from checkpoint_diagnostics import diagnose

OUT = s.OUT
COLORS = {'adamw': '#2d6fb7', 'magnitude_graft': '#df7934', 'sgdm': '#398958'}
LABELS = {'adamw': 'AdamW', 'magnitude_graft': 'Magnitude graft', 'sgdm': 'SGD + momentum'}


def endpoint(spec, row):
    a = np.array([row['spectra'][n]['alpha'] for n in ('fc1', 'fc2')], dtype=float)
    d = {**spec, 'epoch': row['epoch'], 'alpha_fc1': a[0], 'alpha_fc2': a[1],
         'alpha_head': row['spectra']['fc3']['alpha'], 'mean_alpha': a.mean(),
         'alpha_variance': a.var(), 'mean_abs_alpha_minus_2': np.abs(a - 2).mean(),
         'train_accuracy_pct': row['train']['accuracy'] * 100,
         'validation_accuracy_pct': row['validation']['accuracy'] * 100,
         'train_loss': row['train']['loss'], 'validation_loss': row['validation']['loss'],
         'supported_hidden_layers': sum(row['spectra'][n]['supported'] for n in ('fc1', 'fc2'))}
    if 'test' in row:
        d.update(test_accuracy_pct=row['test']['accuracy'] * 100, test_loss=row['test']['loss'])
    return d


def plots(e, t, ref):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, metric, title in zip(axes, ('alpha_fc1', 'alpha_fc2', 'mean_abs_alpha_minus_2', 'test_accuracy_pct'),
                                ('FC1 α', 'FC2 α', 'Mean hidden |α − 2|', 'Test accuracy (%)')):
        for j, arm in enumerate(s.ARMS):
            old = ref[ref.arm == arm][metric].to_numpy()
            ax.scatter(j + np.linspace(-.13, .13, len(old)), old, s=30, facecolors='none',
                       edgecolors='#89919c', alpha=.8, label='Original five seeds' if j == 0 else None)
            value = float(e[e.arm == arm][metric].iloc[0])
            ax.scatter(j, value, marker='*', color=COLORS[arm], s=170, zorder=5,
                       label='Fresh seed 31006' if j == 0 else None)
        if metric.startswith('alpha'):
            ax.axhline(2, color='#7e8794', ls='--', lw=1)
        if metric == 'test_accuracy_pct':
            ax.set_ylim(0, 102)
        ax.set_xticks(range(3), ['AdamW', 'Graft', 'SGDM'], rotation=15)
        ax.set_title(title); ax.grid(axis='y', alpha=.15)
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle('Batch 128, LR ×2: one fresh paired seed versus the original five', fontsize=15)
    fig.text(.5, .025, 'All outcomes at epoch 120. Hollow circles: original seeds 31001–31005. Stars: new seed 31006. No single-seed confidence intervals.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .93]); fig.savefig(OUT / 'Replication_vs_Original.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5))
    for ax, metric, title in zip(axes, ('alpha_fc1', 'alpha_fc2', 'validation_accuracy_pct'),
                                ('FC1 α', 'FC2 α', 'Validation accuracy (%)')):
        for arm in s.ARMS:
            g = t[(t.arm == arm) & (t.epoch >= 10)]
            ax.plot(g.epoch, g[metric], color=COLORS[arm], label=LABELS[arm])
        if metric.startswith('alpha'):
            ax.axhline(2, color='#7e8794', ls='--', lw=1)
        else:
            ax.set_ylim(0, 102)
        ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.grid(alpha=.15)
    axes[-1].legend(fontsize=8, loc='best')
    fig.suptitle('Fresh seed 31006: layer exponents and validation accuracy', fontsize=15)
    fig.text(.5, .025, 'Batch 128 · LR ×2 · unchanged training rule. Displayed epochs 10–120; complete trajectories include initialization.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .93]); fig.savefig(OUT / 'Replication_Trajectories.png', dpi=180); plt.close(fig)


def main():
    protocol = json.loads((OUT / 'Replication_Protocol.json').read_text())
    completion = json.loads((OUT / 'Replication_Completion.json').read_text())
    launch = json.loads((OUT / 'Replication_Launch.json').read_text())
    assert completion['completed'] == 3 and not completion['failures']
    assert s.sha(OUT / 'study.py') == protocol['source_sha256']
    for name, digest in protocol['data_hashes'].items():
        assert s.sha(OUT / 'data' / name) == digest
    x, y, xv, yv, xt, yt = s.load_data()
    rows = []; traces = []; layers = []; hashes = []; checks = []; activities = []; histories = []
    norm_errors = []; norm_count = 0
    for spec in protocol['specifications']:
        d = OUT / 'runs' / s.run_id(spec)
        done = json.loads((d / 'done.json').read_text())
        assert done['status'] == 'complete' and done['last_epoch'] == 120
        history = json.loads((d / 'history.json').read_text()); histories.append(history)
        assert [r['epoch'] for r in history] == list(range(121))
        for r in history:
            if 'spectra' in r:
                traces.append(endpoint(spec, r))
                for n, z in r['spectra'].items():
                    layers.append({**spec, 'epoch': r['epoch'], 'layer': n, **{k: z[k] for k in ('alpha', 'D', 'num_pl_spikes', 'num_evals', 'supported', 'norm', 'spectral_norm')}})
        rows.append(endpoint(spec, done['final']))
        net = s.MLP()
        for ep in (0, 30, 60, 90, 120):
            cp = d / f'checkpoints/epoch_{ep:03}.pt'
            hashes.append({'path': str(cp.relative_to(OUT)), 'sha256': s.sha(cp), 'bytes': cp.stat().st_size})
        initial = s.torch.load(d / 'checkpoints/epoch_000.pt', weights_only=False)
        net.load_state_dict(initial['model'])
        assert s.weight_hash(net) == launch['initial_hash'] == done['initial_hash']
        final = s.torch.load(d / 'checkpoints/epoch_120.pt', weights_only=False)
        net.load_state_dict(final['model']); assert s.weight_hash(net) == done['final_weight_hash']
        dest = OUT / 'verification' / s.run_id(spec); (dest / 'spectra').mkdir(parents=True, exist_ok=True)
        restored = s.spectra(net, dest, 120)
        alpha_error = max(abs(restored[n]['alpha'] - done['final']['spectra'][n]['alpha']) for n in s.LAYERS)
        assert alpha_error < 1e-10
        old = np.load(d / 'spectra/epoch_120.npz'); new = np.load(dest / 'spectra/epoch_120.npz')
        assert all(np.array_equal(old[n], new[n]) for n in s.LAYERS)
        z = np.load(d / 'test_logits_epoch_120.npz')
        acc = float((z['logits'].argmax(1) == z['labels']).mean())
        ce = float((logsumexp(z['logits'].astype(float), axis=1) - z['logits'][np.arange(len(yt)), z['labels']]).mean())
        assert abs(acc - done['final']['test']['accuracy']) < 1e-14
        assert abs(ce - done['final']['test']['loss']) < 2e-6
        ck = s.torch.load(d / 'checkpoints/epoch_030.pt', weights_only=False)
        net.load_state_dict(ck['model']); opt = s.make_opt(net, spec); opt.load_state_dict(ck['optimizer']); s.torch.set_rng_state(ck['torch_rng'])
        order = s.order(spec['seed'], 31); stats = np.load(d / 'statistics/epoch_031.npz')
        for j in range(11):
            ix = order[j * 128:(j + 1) * 128]; record = j in (0, 1, 10)
            loss, values = s.train_step(net, opt, spec, x[ix], y[ix], s.schedule(31), record)
            if record:
                k = int(np.flatnonzero(stats['batch_index'] == j)[0])
                assert loss == float(stats['loss'][k])
                assert np.array_equal(np.array(values, np.float32), stats['values'][k], equal_nan=True)
        checks.append({'spec': spec, 'initial_and_final_hashes_match': True, 'alpha_max_abs_error': alpha_error,
                       'spectra_identical': True, 'eleven_step_replay_exact': True, 'saved_accuracy_matches': True,
                       'saved_cross_entropy_abs_error': abs(ce - done['final']['test']['loss'])})
        activities.append(diagnose(spec, xt, yt))
        if spec['arm'] == 'magnitude_graft':
            graft_check = direction_check(spec, x, y)
            early_graft_check = direction_check(spec, x, y, from_epoch=0)
            assert all(z['direction_defined_for_nonzero_update'] and not z['frozen_denominator_floor_active']
                       for z in early_graft_check['layers'])
            for ep in range(1, 121):
                a = np.load(d / f'statistics/epoch_{ep:03}.npz')['values']
                valid = a[:, :, 3] > 1e-20
                errors = np.abs(a[:, :, 4][valid] / a[:, :, 3][valid] - 1)
                norm_count += len(errors)
                if len(errors): norm_errors.append(float(errors.max()))
    for ep in range(1, 121):
        expected = hashlib.sha256(s.order(protocol['seed'], ep).tobytes()).hexdigest()
        assert all(h[ep]['batch_order_sha256'] == expected for h in histories)
    assert max(norm_errors) < 3e-6
    e = pd.DataFrame(rows); t = pd.DataFrame(traces); ref = pd.read_csv(OUT / 'Reference_Endpoints.csv')
    e.to_csv(OUT / 'Endpoints.csv', index=False); t.to_csv(OUT / 'Trajectories.csv', index=False)
    pd.DataFrame(layers).to_csv(OUT / 'Layer_Measurements.csv', index=False)
    deltas = []
    for treatment in ('magnitude_graft', 'sgdm'):
        a = e[e.arm == treatment].iloc[0]; b = e[e.arm == 'adamw'].iloc[0]
        deltas.append({'treatment': treatment, 'reference': 'adamw', 'seed': protocol['seed'],
                       **{k: float(a[k] - b[k]) for k in ('alpha_fc1', 'alpha_fc2', 'mean_abs_alpha_minus_2', 'test_accuracy_pct')}})
    pd.DataFrame(deltas).to_csv(OUT / 'Paired_Differences.csv', index=False)
    audit = {'passed': True, 'completed_runs': 3, 'source_sha256': protocol['source_sha256'],
             'data_hashes_match': True, 'common_initialization': True, 'common_minibatches': True,
             'checkpoints_hashed': len(hashes), 'checkpoint_checks': checks,
             'graft_check': graft_check, 'nonzero_initial_step_graft_check': early_graft_check,
             'recorded_nondegenerate_magnitudes_checked': norm_count,
             'max_recorded_magnitude_relative_error': max(norm_errors), 'activity_checks': activities}
    s.writej(OUT / 'Checkpoint_Manifest.json', hashes); s.writej(OUT / 'Replication_Verification.json', audit)
    summary = {'status': 'complete_and_verified', 'seed': protocol['seed'], 'batch_size': 128, 'lr_multiplier': 2,
               'completed_runs': 3, 'epochs': 120, 'source_sha256': protocol['source_sha256'],
               'endpoints': rows, 'paired_differences': deltas, 'test_activity': activities,
               'single_new_seed_no_confidence_interval': True, 'original_five_seed_analysis_unchanged': True,
               'training_wall_seconds': completion['wall_seconds'], 'next_experiment_started': False}
    s.writej(OUT / 'Replication_Result_Summary.json', summary)
    plots(e, t, ref)
    print(e[['arm', 'alpha_fc1', 'alpha_fc2', 'alpha_head', 'mean_abs_alpha_minus_2', 'test_accuracy_pct']].to_string(index=False))
    print('VERIFIED 3 runs, 15 checkpoints; exact replay and graft checks passed.')


if __name__ == '__main__':
    main()
