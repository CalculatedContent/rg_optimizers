"""Additional fixed-pass figures; no checkpoint or hyperparameter selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from analyze import COLORS, LABELS, style

OUT = Path(__file__).resolve().parent


def main():
    style()
    e = pd.read_csv(OUT / 'Endpoints.csv')
    p = pd.read_csv(OUT / 'Paired_Summaries.csv')
    assert len(e) == 135 and e.seed.nunique() == 5
    # Replace the frozen script's crowded annotations without changing its data.
    cells = p[(p.scope == 'setting') & (p.treatment == 'magnitude_graft') & (p.reference == 'adamw')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for ax, metric, title in zip(axes,
                                ('alpha_fc1', 'mean_abs_alpha_minus_2', 'test_accuracy_pct'),
                                ('FC1 α change', 'Mean |α − 2| change', 'Test accuracy change (pp)')):
        g = cells[cells.metric == metric]
        grid = g.pivot(index='batch_size', columns='multiplier', values='mean').loc[[128, 256, 512], [.5, 1., 2.]]
        lim = max(float(np.abs(grid.values).max()), 1e-6)
        im = ax.imshow(grid, cmap='coolwarm', vmin=-lim, vmax=lim, aspect='auto')
        for i, b in enumerate(grid.index):
            for j, m in enumerate(grid.columns):
                r = g[(g.batch_size == b) & (g.multiplier == m)].iloc[0]
                ax.text(j, i, f'{r["mean"]:+.3f}\n[{r.ci_low:+.3f},\n{r.ci_high:+.3f}]',
                        ha='center', va='center', fontsize=10,
                        color='white' if abs(r['mean']) > .7 * lim else '#111111')
        ax.set_xticks(range(3), ['0.5', '1', '2'])
        ax.set_yticks(range(3), ['128', '256', '512'])
        ax.set_xlabel('Learning-rate multiplier'); ax.set_ylabel('Batch size'); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.75, pad=.03)
    fig.suptitle('Magnitude graft minus native AdamW: paired effects', fontsize=16)
    fig.text(.5, .025, 'Cell: paired mean change [95% t interval], five seeds. Negative distance change means closer to 2.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .06, 1, .94])
    fig.savefig(OUT / 'Paired_Effects_All_Settings.png', dpi=180)
    plt.close(fig)
    settings = [(b, m) for b in (128, 256, 512) for m in (.5, 1., 2.)]
    q = p[(p.scope == 'setting') & (p.reference == 'adamw')]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.8), sharey=True)
    metrics = [('alpha_fc1', 'FC1 α change', 'Upward / downward movement'),
               ('mean_abs_alpha_minus_2', 'Mean |α − 2| change', 'Negative = closer to 2'),
               ('test_accuracy_pct', 'Test accuracy change (pp)', 'Positive = higher accuracy')]
    for ax, (metric, title, note) in zip(axes, metrics):
        ax.axvline(0, color='#7d8897', ls='--', lw=1)
        for arm, offset in [('magnitude_graft', -.13), ('sgdm', .13)]:
            g = q[(q.treatment == arm) & (q.metric == metric)]
            for j, (b, m) in enumerate(settings):
                r = g[(g.batch_size == b) & (g.multiplier == m)].iloc[0]
                ax.errorbar(r['mean'], j + offset,
                            xerr=[[r['mean'] - r.ci_low], [r.ci_high - r['mean']]],
                            fmt='o', ms=5, color=COLORS[arm], capsize=3,
                            label=LABELS[arm] if j == 0 else None)
        ax.set_title(title + '\n' + note, fontsize=11)
        ax.grid(axis='x', alpha=.15)
        ax.set_xlabel('Treatment minus native AdamW')
    axes[0].set_yticks(range(len(settings)), [f'Batch {b} · LR ×{m:g}' for b, m in settings])
    axes[0].invert_yaxis()
    axes[-1].legend(loc='best', fontsize=9)
    fig.suptitle('MNIST magnitude graft and SGDM: all prespecified comparisons', fontsize=16)
    fig.text(.5, .02, 'Points: paired mean differences. Bars: 95% t intervals over five seeds per setting. All runs end at epoch 120.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .05, 1, .95])
    fig.savefig(OUT / 'Paired_Effects_With_SGDM.png', dpi=180)
    plt.close(fig)

    # Fixed zoom is explicitly separate from the complete outcome range above it.
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8), sharex='col')
    for j, m in enumerate((.5, 1., 2.)):
        for i in (0, 1):
            ax = axes[i, j]
            for arm in ('adamw', 'magnitude_graft', 'sgdm'):
                for b, marker in ((128, 'o'), (256, 's'), (512, '^')):
                    g = e[(e.arm == arm) & (e.batch_size == b) & (e.multiplier == m)]
                    ax.scatter(g.mean_alpha, g.test_accuracy_pct, color=COLORS[arm], marker=marker,
                               s=34, alpha=.7, label=LABELS[arm] if b == 128 else None)
            ax.axvline(2, color='#7d8897', ls='--', lw=1)
            ax.grid(alpha=.15)
            if i == 0:
                ax.set_ylim(0, 100)
                ax.set_title(f'LR ×{m:g} · complete accuracy range')
            else:
                ax.set_ylim(97, 99)
                ax.set_title('Detail: 97–99% accuracy')
                ax.set_xlabel('Mean hidden-layer α')
                excluded = int(((e.multiplier == m) & ((e.test_accuracy_pct < 97) | (e.test_accuracy_pct > 99))).sum())
                ax.text(.03, .04, f'{excluded} of 45 points outside this detail view', transform=ax.transAxes,
                        fontsize=8, color='#566070')
            if j == 0:
                ax.set_ylabel('Test accuracy (%)')
    axes[0, 2].legend(fontsize=9, loc='center right')
    fig.suptitle('Accuracy versus mean α: all 135 runs and a labeled detail view', fontsize=16)
    fig.text(.5, .02, 'Batch: circle 128 · square 256 · triangle 512. Top row includes every run; bottom row uses a fixed display zoom.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .05, 1, .95])
    fig.savefig(OUT / 'Accuracy_vs_Alpha_Detail.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for i, b in enumerate((128, 256, 512)):
        for j, m in enumerate((.5, 1., 2.)):
            ax = axes[i, j]
            for k, arm in enumerate(('adamw', 'magnitude_graft', 'sgdm')):
                g = e[(e.arm == arm) & (e.batch_size == b) & (e.multiplier == m)]
                ax.scatter(np.full(len(g), k), g.alpha_head, color=COLORS[arm], s=23, alpha=.4)
                ax.errorbar(k, g.alpha_head.mean(), g.alpha_head.std(ddof=1),
                            fmt='o', color=COLORS[arm], capsize=3)
            ax.axhline(2, color='#7d8897', ls='--', lw=1)
            ax.set_xticks((0, 1, 2), ('AdamW', 'Graft', 'SGDM'))
            ax.set_title(f'Batch {b} · LR ×{m:g}')
            ax.grid(axis='y', alpha=.15)
            if j == 0:
                ax.set_ylabel('FC3 α (exploratory)')
    fig.suptitle('Output-layer α: exploratory fits with only 10 eigenvalues', fontsize=16)
    fig.text(.5, .02, 'Five seeds per setting; bars are seed SD. FC3 is excluded from primary hidden-layer means and distances.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .05, 1, .95])
    fig.savefig(OUT / 'Head_Alpha_Exploratory.png', dpi=180)
    plt.close(fig)

    t = pd.read_csv(OUT / 'Trajectories.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, metric, title in zip(axes, ('alpha_fc1', 'alpha_fc2', 'test_accuracy_pct'),
                                ('FC1 α', 'FC2 α', 'Test accuracy (%)')):
        for arm in ('adamw', 'magnitude_graft', 'sgdm'):
            g = t[(t.batch_size == 256) & (t.multiplier == 1) & (t.arm == arm) & (t.epoch > 0)]
            g = g.groupby('epoch')[metric].agg(['mean', 'std']).dropna()
            ax.plot(g.index, g['mean'], color=COLORS[arm], label=LABELS[arm])
            ax.fill_between(g.index, g['mean'] - g['std'], g['mean'] + g['std'], color=COLORS[arm], alpha=.16)
        if metric.startswith('alpha'):
            ax.axhline(2, color='#7d8897', ls='--', lw=1)
        ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.grid(alpha=.15)
    axes[-1].legend(fontsize=8)
    fig.suptitle('Central setting: detail after initialization (batch 256, LR ×1)', fontsize=15)
    fig.text(.5, .025, 'Five-seed means ± seed SD. Epoch 0 omitted only from this detail view; the full trajectory plot retains it.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .065, 1, .94])
    fig.savefig(OUT / 'Central_Setting_Trajectories_Detail.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
