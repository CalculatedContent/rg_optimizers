"""Export fit/accuracy tables and stratified correlations from saved measurements."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from common import atomic_json, digest

SPLITS = ('train', 'validation', 'test')
AGGREGATE_METRICS = ('mean_alpha', 'mean_D', 'median_D', 'max_D', 'fraction_alpha_below_2')
LAYER_METRICS = ('alpha', 'D', 'sigma', 'tail_fraction', 'xmin', 'num_pl_spikes')


def finite(v):
    return isinstance(v, (float, int)) and math.isfinite(v)


def correlate(rows, x, y):
    pairs = [(r.get(x), r.get(y)) for r in rows if finite(r.get(x)) and finite(r.get(y))]
    result = {'metric': x, 'accuracy': y, 'n': len(pairs), 'pearson_r': None, 'spearman_rho': None}
    if len(pairs) < 3:
        return dict(result, status='insufficient_points')
    values = np.array(pairs, dtype=float)
    if np.ptp(values[:, 0]) == 0 or np.ptp(values[:, 1]) == 0:
        return dict(result, status='constant_variable')
    return dict(result, status='descriptive', pearson_r=float(pearsonr(values[:, 0], values[:, 1]).statistic),
                spearman_rho=float(spearmanr(values[:, 0], values[:, 1]).statistic))


def export_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_rows(roots):
    checkpoints, layers, exclusions = [], [], []
    for root in sorted({Path(p).resolve() for p in roots}):
        if not (root / 'config.json').exists():
            exclusions.append({'run': str(root), 'reason': 'config_missing'})
            continue
        config = json.loads((root / 'config.json').read_text())
        runtime = json.loads((root / 'runtime.json').read_text()) if (root / 'runtime.json').exists() else {}
        status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        setting = digest({k: v for k, v in config.items() if k != 'seed'})[:16]
        # Accuracy/fit comparisons stay within the same model, data, compute recipe and optimizer.
        fields = ('model', 'model_revision', 'dataset', 'dataset_revision', 'max_length', 'epochs',
                  'precision', 'split_seed', 'validation_per_class', 'synthetic')
        protocol = {k: config.get(k) for k in fields}
        protocol.update(data_sha256=runtime.get('data_manifest_sha256'), global_batch=runtime.get('global_batch'),
                        test_sha256=runtime.get('test_manifest_sha256'))
        base = {'run': str(root), 'setting': setting, 'seed': config.get('seed'),
                'protocol': digest(protocol)[:16], 'optimizer': config.get('optimizer'),
                'synthetic': bool(config.get('synthetic')), 'config_json': json.dumps(config, sort_keys=True)}
        for path in sorted((root / 'spectra').glob('step_*.json')):
            obj = json.loads(path.read_text())
            row = dict(base, step=obj['step'], epoch=obj.get('epoch'),
                       completed_endpoint=status.get('status') == 'training_complete' and obj['step'] == status.get('step'))
            for metric in AGGREGATE_METRICS:
                row[metric] = obj.get(metric)
            row.update(valid_alpha_layers=obj.get('valid_layers'), valid_D_layers=obj.get('valid_D_layers'),
                       expected_layers=obj.get('expected_layers'))
            for split in SPLITS:
                measured = obj.get(split) or {}
                row[split + '_accuracy_pct'] = 100 * measured['accuracy'] if finite(measured.get('accuracy')) else None
                for metric in ('loss', 'errors', 'examples'):
                    row[split + '_' + metric] = measured.get(metric)
            row['train_validation_gap_pp'] = obj.get('generalization_gap_pp')
            row['train_test_gap_pp'] = obj.get('train_test_gap_pp')
            checkpoints.append(row)
            for layer in obj['layers']:
                fit = layer.get('fit', {})
                item = {k: row[k] for k in ('run', 'setting', 'seed', 'protocol', 'optimizer', 'synthetic',
                                            'step', 'epoch', 'completed_endpoint')}
                item.update(layer=layer['name'], rows=layer['shape'][0], columns=layer['shape'][1],
                            status=layer.get('status'), fit_status=fit.get('status'), fit_warning=fit.get('warning'))
                for key in ('alpha', 'D', 'sigma', 'xmin', 'xmax', 'num_pl_spikes', 'num_evals', 'tail_fraction'):
                    item[key] = layer.get(key, fit.get(key))
                # Excluded/failed fits stay visible in CSV, with null correlation inputs.
                if not layer.get('included_in_mean'):
                    for key in ('alpha', 'sigma', 'xmin', 'xmax', 'num_pl_spikes', 'tail_fraction'):
                        item[key] = None
                if not layer.get('D_valid'):
                    item['D'] = None
                for split in SPLITS:
                    item[split + '_accuracy_pct'] = row[split + '_accuracy_pct']
                layers.append(item)
    return checkpoints, layers, exclusions


def seed_means(rows, metrics, expected_seeds=(401, 402, 403), layerwise=False):
    groups = {}
    for row in rows:
        if row['synthetic'] or not row['completed_endpoint']:
            continue
        key = (row['protocol'], row['optimizer'], row['setting'], row.get('layer') if layerwise else 'model')
        groups.setdefault(key, []).append(row)
    output, excluded = [], []
    for key, group in groups.items():
        seeds = [r['seed'] for r in group]
        if sorted(seeds) != sorted(expected_seeds):
            excluded.append({'key': list(key), 'seeds': seeds, 'reason': 'requires_exactly_one_complete_run_per_expected_seed'})
            continue
        item = {'protocol': key[0], 'optimizer': key[1], 'setting': key[2], 'layer': key[3],
                'n_seeds': len(seeds), 'seeds': ','.join(map(str, sorted(seeds)))}
        for metric in tuple(metrics) + tuple(s + '_accuracy_pct' for s in SPLITS):
            values = [r.get(metric) for r in group]
            valid = all(finite(v) for v in values)
            item[metric] = float(np.mean(values)) if valid else None
            item[metric + '_sd'] = float(np.std(values, ddof=1)) if valid else None
        output.append(item)
    return output, excluded


def correlations(checkpoints, layers, means, layer_means):
    out = []
    # Time-series observations are explicitly separated from independent settings.
    for source, metrics, population, keys in (
        (checkpoints, AGGREGATE_METRICS, 'within_run_trajectory', ('run',)),
        (layers, LAYER_METRICS, 'within_run_layer_trajectory', ('run', 'layer')),
        (means, AGGREGATE_METRICS, 'between_complete_three_seed_settings', ('protocol', 'optimizer')),
        (layer_means, LAYER_METRICS, 'between_complete_three_seed_settings_by_layer', ('protocol', 'optimizer', 'layer'))):
        groups = {}
        for row in source:
            groups.setdefault(tuple(row[k] for k in keys), []).append(row)
        for key, group in groups.items():
            for metric in metrics:
                for split in SPLITS:
                    item = correlate(group, metric, split + '_accuracy_pct')
                    item.update(population=population, group=dict(zip(keys, key)),
                                unit='checkpoint' if population.startswith('within') else 'configuration_mean_over_three_seeds',
                                synthetic=any(r.get('synthetic', False) for r in group))
                    out.append(item)
    return out


def plot_report(output, checkpoints, layers, means, layer_means):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def figure(group, xkeys, title, errorbars=False):
        fig, axes = plt.subplots(len(xkeys), 3, figsize=(13, 3.5 * len(xkeys)), squeeze=False, constrained_layout=True)
        for i, xkey in enumerate(xkeys):
            for j, split in enumerate(SPLITS):
                ax = axes[i, j]
                ykey = split + '_accuracy_pct'
                valid = [r for r in group if finite(r.get(xkey)) and finite(r.get(ykey))]
                if valid:
                    x, y = [r[xkey] for r in valid], [r[ykey] for r in valid]
                    if errorbars:
                        ax.errorbar(x, y, xerr=[r.get(xkey + '_sd', 0) for r in valid],
                                    yerr=[r.get(ykey + '_sd', 0) for r in valid], fmt='o', markersize=4,
                                    color='#176b87', alpha=.75, capsize=2)
                    else:
                        dots = ax.scatter(x, y, c=[r['step'] for r in valid], cmap='viridis', s=42)
                        fig.colorbar(dots, ax=ax, label='Training step')
                    result = correlate(valid, xkey, ykey)
                    rho = result['spearman_rho']
                    ax.set_title(f'{split.title()} · n={len(valid)}' + (f' · Spearman ρ={rho:.2f}' if rho is not None else ''))
                else:
                    ax.text(.5, .5, 'No measured pairs yet', transform=ax.transAxes, ha='center')
                    ax.set_title(split.title())
                ax.set_xlabel(xkey.replace('_', ' ') + (' (lower = closer fit)' if xkey.endswith('D') else ''))
                ax.set_ylabel('Accuracy (%)')
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.ticklabel_format(axis='x', style='plain', useOffset=False)
                ax.grid(alpha=.15)
        fig.suptitle(title, fontsize=13)
        return fig

    # Every run has its own trajectory; one seed is clearly labeled.
    for run in sorted({r['run'] for r in checkpoints}):
        group = [r for r in checkpoints if r['run'] == run]
        label = 'Synthetic engineering check' if group[0]['synthetic'] else 'Single-run trajectory'
        name = digest(run)[:10]
        fig = figure(group, ('mean_D', 'mean_alpha'), f'{label} · {group[0]["optimizer"]} · seed {group[0]["seed"]}')
        fig.savefig(output / f'trajectory_{name}.png', dpi=160)
        plt.close(fig)
        with PdfPages(output / f'layer_trajectories_{name}.pdf') as pdf:
            for layer in sorted({r['layer'] for r in layers if r['run'] == run}):
                group_layer = [r for r in layers if r['run'] == run and r['layer'] == layer]
                if not any(finite(r.get('D')) or finite(r.get('alpha')) for r in group_layer):
                    continue
                fig = figure(group_layer, ('D', 'alpha'), label + '\n' + layer)
                pdf.savefig(fig)
                plt.close(fig)
    for protocol, optimizer in sorted({(r['protocol'], r['optimizer']) for r in means}):
        group = [r for r in means if (r['protocol'], r['optimizer']) == (protocol, optimizer)]
        title = f'{optimizer} · {len(group)} complete settings · three-seed means ± SD · protocol {protocol}'
        fig = figure(group, ('mean_D', 'mean_alpha'), title, True)
        fig.savefig(output / f'settings_{protocol}_{optimizer}.png', dpi=160)
        plt.close(fig)
        with PdfPages(output / f'layers_{protocol}_{optimizer}.pdf') as pdf:
            for layer in sorted({r['layer'] for r in layer_means if (r['protocol'], r['optimizer']) == (protocol, optimizer)}):
                selected = [r for r in layer_means if (r['protocol'], r['optimizer'], r['layer']) == (protocol, optimizer, layer)]
                if not any(finite(r.get('D')) or finite(r.get('alpha')) for r in selected):
                    continue
                fig = figure(selected, ('D', 'alpha'), title + '\n' + layer, True)
                pdf.savefig(fig)
                plt.close(fig)


def report(roots, output, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints, layers, missing = load_rows(roots)
    means, excluded = seed_means(checkpoints, AGGREGATE_METRICS)
    layer_means, layer_excluded = seed_means(layers, LAYER_METRICS, layerwise=True)
    for name, rows in [('checkpoint_metrics', checkpoints), ('layer_metrics', layers),
                       ('configuration_means', means), ('layer_configuration_means', layer_means)]:
        export_csv(output / (name + '.csv'), rows)
    corr = correlations(checkpoints, layers, means, layer_means)
    atomic_json(output / 'correlations.json', corr)
    atomic_json(output / 'report.json', {'runs': len({r['run'] for r in checkpoints}), 'checkpoints': len(checkpoints),
                'complete_three_seed_settings': len(means), 'missing': missing, 'excluded_settings': excluded,
                'excluded_layer_settings': layer_excluded,
                'D': 'KS tail distance; lower is closer fit; not a p-value',
                'uncertainty': 'Plot error bars are sample SD across three seeds, not confidence intervals',
                'correlation_units': 'Within-run checkpoint trajectories and between-configuration seed means are separate. Optimizers/protocols/layers are not pooled.',
                'test_policy': 'Repeated monitoring exposure, excluded from selection; missing test entries remain null'})
    if plots and checkpoints:
        plot_report(output, checkpoints, layers, means, layer_means)
    return {'checkpoints': len(checkpoints), 'layer_rows': len(layers), 'complete_three_seed_settings': len(means)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', required=True, nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--no-plots', action='store_true')
    a = p.parse_args()
    print(json.dumps(report(a.runs, a.output, plots=not a.no_plots)))
