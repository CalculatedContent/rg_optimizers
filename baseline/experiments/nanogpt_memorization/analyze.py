"""Read-only analysis of saved verbatim probes; never loads or trains a model."""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from statistics import mean

ID = ['run', 'stage', 'recipe', 'condition', 'optimizer', 'seed']
METRICS = ['exact_match', 'continuation_token_accuracy', 'teacher_forced_accuracy', 'nll']
GROUP = re.compile(r'^dose_(\d+)/prefix_(\d+)$')


def read_file(path: Path, sources: dict) -> str:
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8')


def history(text: str, warnings: list[str]) -> list[dict]:
    """Permit only an unfinished last JSONL line, not an invalid completed row."""
    lines, rows = text.splitlines(keepends=True), []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not line.endswith('\n'):
                warnings.append('Ignored an unfinished last metrics line (live snapshot).')
                break
            raise ValueError(f'Malformed completed metrics line {i + 1}.')
        if not isinstance(row, dict) or 'step' not in row or 'audit' not in row:
            raise ValueError(f'Invalid metrics schema at line {i + 1}.')
        step = row['step']
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('Checkpoint step must be a nonnegative integer.')
        if rows and step <= rows[-1]['step']:
            raise ValueError('Duplicate or nonmonotonic checkpoint steps; inspect the source log.')
        rows.append(row)
    return rows


def number(value, *, probability=False) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0 or (probability and x > 1):
        raise ValueError(f'Invalid behavioral measurement: {value!r}')
    return x


def read_run(path: Path, sources: dict, warnings: list[str]) -> dict:
    manifest = json.loads(read_file(path / 'manifest.json', sources))
    meta = {k: manifest[k] for k in ['stage', 'recipe', 'condition', 'seed']}
    # The historical manifest stores optimizer identity inside profile.family.
    meta['optimizer'] = manifest['profile']['family']
    meta['run'] = '/'.join(str(meta[k]) for k in ['stage', 'recipe', 'condition', 'optimizer', 'seed'])
    rows = history(read_file(path / 'metrics.jsonl', sources), warnings) if (path / 'metrics.jsonl').is_file() else []
    inventory = json.loads(read_file(path / 'probe_inventory.json', sources)) if (path / 'probe_inventory.json').is_file() else []
    by_id = {r['eid']: r for r in inventory}
    if len(by_id) != len(inventory):
        raise ValueError(f'Duplicate canary IDs: {path}')
    slots = defaultdict(list)
    schedule_present = (path / 'injection_schedule.json').is_file()
    if schedule_present:
        schedule = json.loads(read_file(path / 'injection_schedule.json', sources))
        for slot, eid in schedule.items():
            if int(slot) < 0 or eid not in by_id:
                raise ValueError(f'Invalid injection schedule: {path}')
            slots[eid].append(int(slot))
        for entries in slots.values():
            entries.sort()
    if schedule_present:
        for eid, item in by_id.items():
            expected = int(item['group'].split('_')[-1])
            if len(slots[eid]) != expected:
                raise ValueError(f'Schedule total disagrees with intended dose for {eid}.')
    block = int(manifest['source_model']['block_size'])
    positions = int(manifest['input_tokens_per_step'])
    if block <= 0 or positions % block:
        raise ValueError('Invalid input-token geometry for exposure reconstruction.')
    batch = positions // block
    target_steps = int(manifest['suite']['stages'][meta['stage']]['steps'])
    boundary = max(1, target_steps // 2)
    for entries in slots.values():
        if any(slot >= boundary * batch for slot in entries):
            raise ValueError('Injection scheduled after the configured withdrawal boundary.')
    last_step = rows[-1]['step'] if rows else None
    status = 'incomplete_or_running'
    if (path / 'launcher_status.json').is_file():
        launch = json.loads(read_file(path / 'launcher_status.json', sources))
        if launch.get('state') in ('failed', 'interrupted'):
            status = launch['state']
    if (path / 'complete.json').is_file():
        done = json.loads(read_file(path / 'complete.json', sources))
        if (done.get('status') == 'complete' and done.get('steps') == target_steps
                and done.get('fingerprint') == manifest.get('fingerprint') and last_step == target_steps):
            status = 'complete'
        else:
            warnings.append(f'{meta["run"]}: completion marker does not match the observed final row; not treated as complete.')
    behavior, examples, spectra = [], [], []
    for row in rows:
        step = row['step']
        if step > target_steps:
            raise ValueError('Recorded checkpoint exceeds the declared run horizon.')
        for group, values in row['audit'].items():
            match = GROUP.fullmatch(group)
            if not match:
                raise ValueError(f'Unexpected verbatim group {group!r}.')
            dose, prefix = map(int, match.groups())
            cohort = values['examples']
            if len(cohort) != values['n'] or not cohort:
                raise ValueError(f'Probe denominator mismatch for {group}, step {step}.')
            if len({r['eid'] for r in cohort}) != len(cohort):
                raise ValueError(f'Duplicate evaluated probes in {group}, step {step}.')
            common = {**meta, 'step': step, 'group': group, 'dose': dose, 'prefix_tokens': prefix,
                      'phase': 'acquisition' if step <= boundary else 'withdrawal',
                      'model_sha256': row.get('model_sha256', '')}
            parsed = []
            for item in cohort:
                r = {k: number(item[k], probability=(k != 'nll')) for k in METRICS}
                if r['exact_match'] not in (0, 1):
                    raise ValueError('Per-probe whole-sequence exact match must be 0 or 1.')
                eid = item['eid']
                if eid not in by_id or by_id[eid]['group'] != f'dose_{dose}':
                    raise ValueError(f'Probe missing or dose disagrees with inventory: {eid}')
                item_slots = slots[eid]
                count = (0 if meta['condition'] == 'verbatim_absent' else
                         bisect_left(item_slots, step * batch) if schedule_present else None)
                last_visit = item_slots[count - 1] // batch + 1 if count else None
                example = {**common, 'eid': eid, 'target_tokens': len(by_id[eid]['target']),
                           'presentations_from_schedule': count,
                           'last_presentation_update_from_schedule': last_visit, **r}
                examples.append(example)
                parsed.append(r)
            aggregate = {k: mean(r[k] for r in parsed) for k in METRICS}
            if any(not math.isclose(aggregate[k], float(values['mean'][k]), rel_tol=1e-6, abs_tol=1e-7)
                   for k in METRICS):
                raise ValueError(f'Saved aggregate disagrees with examples: {group}, step {step}.')
            behavior.append({**common, 'n': len(cohort), **aggregate})
        spectral_path = path / 'spectral' / f'step_{step:08d}.csv'
        if spectral_path.is_file():
            table = list(csv.DictReader(io.StringIO(read_file(spectral_path, sources))))
            if 'n_layer' in manifest['source_model'] and len(table) != 6 * int(manifest['source_model']['n_layer']):
                warnings.append(f'{meta["run"]}: incomplete hidden-matrix spectral inventory at step {step}.')
            names = [r.get('matrix_name') for r in table]
            if not names or None in names or len(set(names)) != len(names):
                raise ValueError(f'Invalid spectral matrix identities: {spectral_path}')
            for r in table:
                if int(r['step']) != step:
                    raise ValueError(f'Spectral step mismatch: {spectral_path}')
                spectra.append({**r, **meta, 'step': step})
        else:
            warnings.append(f'{meta["run"]}: no spectral file at step {step}.')
    return {'meta': meta, 'manifest': manifest, 'inventory': inventory,
            'status': status, 'last_step': last_step, 'target_steps': target_steps,
            'boundary': boundary, 'behavior': behavior, 'examples': examples, 'spectra': spectra}


def summarize(runs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    doses, canaries, contrasts = [], [], []
    for run in runs:
        grouped = defaultdict(list)
        per_canary = defaultdict(list)
        for row in run['behavior']:
            grouped[row['group']].append(row)
        for row in run['examples']:
            per_canary[(row['group'], row['eid'])].append(row)
        for group, rows in grouped.items():
            latest = rows[-1]
            observed = [r['step'] for r in rows if r['exact_match'] > 0]
            doses.append({**run['meta'], 'status': run['status'], 'group': group,
                          'dose': latest['dose'], 'prefix_tokens': latest['prefix_tokens'],
                          'n': latest['n'], 'checkpoints': len(rows), 'latest_step': latest['step'],
                          'peak_observed_exact_match': max(r['exact_match'] for r in rows),
                          'latest_exact_match': latest['exact_match'], 'latest_nll': latest['nll'],
                          'first_observed_exact_step': min(observed) if observed else None})
        for (group, eid), rows in per_canary.items():
            positive = [r for r in rows if r['exact_match'] == 1]
            first = positive[0] if positive else None
            canaries.append({**run['meta'], 'group': group, 'eid': eid,
                             'checkpoints': len(rows), 'first_observed_exact_step': first['step'] if first else None,
                             'presentations_at_first_exact_from_schedule': first['presentations_from_schedule'] if first else None,
                             'latest_exact_match': rows[-1]['exact_match'], 'latest_step': rows[-1]['step']})
        controls = {(r['step'], r['prefix_tokens']): r for r in run['behavior'] if r['dose'] == 0}
        for row in run['behavior']:
            control = controls.get((row['step'], row['prefix_tokens']))
            if row['dose'] == 0 or control is None:
                continue
            contrasts.append({**run['meta'], 'step': row['step'], 'group': row['group'],
                              'n_exposed': row['n'], 'n_control': control['n'],
                              'exact_match_minus_zero_dose': row['exact_match'] - control['exact_match'],
                              'token_accuracy_minus_zero_dose': row['continuation_token_accuracy'] - control['continuation_token_accuracy'],
                              'zero_dose_nll_minus_exposed_nll': control['nll'] - row['nll']})
    return doses, canaries, contrasts


def compare(runs: list[dict], warnings: list[str]) -> list[dict]:
    """Only identical recorded steps, probes, seed, recipe and hardware; no pooling."""
    blocks = defaultdict(dict)
    for r in runs:
        meta = r['meta']
        blocks[(meta['stage'], meta['recipe'], meta['condition'], meta['seed'])][meta['optimizer']] = r
    output = []
    required = ['suite', 'initial_model_sha256', 'data_sha256', 'source_model', 'runner_sha256', 'device']
    for key, arms in blocks.items():
        if not {'adamw', 'muon'} <= arms.keys():
            continue
        a, b = arms['adamw'], arms['muon']
        mismatches = [k for k in required if k not in a['manifest'] or k not in b['manifest']
                      or a['manifest'][k] != b['manifest'][k]]
        if a['inventory'] != b['inventory']:
            mismatches.append('probe_inventory')
        if mismatches:
            warnings.append(f'No optimizer comparison for {key}: mismatched {mismatches}.')
            continue
        left = {(r['step'], r['group']): r for r in a['behavior']}
        # Also verify that the exact evaluated target cohort is paired.
        cohort_a = defaultdict(set)
        cohort_b = defaultdict(set)
        for r in a['examples']:
            cohort_a[(r['step'], r['group'])].add(r['eid'])
        for r in b['examples']:
            cohort_b[(r['step'], r['group'])].add(r['eid'])
        for row in b['behavior']:
            match = (row['step'], row['group'])
            other = left.get(match)
            if other is None:
                continue
            if cohort_a[match] != cohort_b[match]:
                warnings.append(f'Unpaired evaluated probes at {key}, {match}; comparison omitted.')
                continue
            output.append({'stage': key[0], 'recipe': key[1], 'condition': key[2], 'seed': key[3],
                           'step': row['step'], 'group': row['group'], 'n': row['n'],
                           **{f'{m}_muon_minus_adamw': row[m] - other[m] for m in METRICS}})
    return output


def write_csv(path: Path, rows: list[dict], defaults: list[str]) -> None:
    from study import atomic_text
    keys = list(dict.fromkeys(defaults + [k for r in rows for k in r]))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def make_plots(runs: list[dict], out: Path) -> None:
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/nanogpt_memorization_plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for run in runs:
        dest = out / 'figures' / run['meta']['run']
        dest.mkdir(parents=True, exist_ok=True)
        prefix = int(run['manifest']['suite']['canary_prefix_tokens'])
        rows = [r for r in run['behavior'] if r['prefix_tokens'] == prefix]
        for metric in METRICS:
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for dose in sorted({r['dose'] for r in rows}):
                cohort = [r for r in rows if r['dose'] == dose]
                ax.plot([r['step'] for r in cohort], [r[metric] for r in cohort], label=f'lifetime dose {dose}')
            ax.axvline(run['boundary'], linestyle='--', label='presentations stop')
            ax.set(xlabel='Completed updates', ylabel=metric, title=run['meta']['run'])
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(dest / f'{metric}.png', dpi=120)
            plt.close(fig)
        for matrix in sorted({r['matrix_name'] for r in run['spectra']}):
            cohort = [r for r in run['spectra'] if r['matrix_name'] == matrix]
            safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', matrix)
            for filename, metrics in [('alpha', ['alpha_clip_xmax', 'alpha_raw']),
                                      ('ERG_gap', ['ERG_gap']), ('num_traps', ['num_traps']),
                                      ('rand_distance', ['rand_distance']), ('D', ['D'])]:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                plotted = False
                for metric in metrics:
                    def finite(value):
                        try:
                            x = float(value)
                            return x if math.isfinite(x) else math.nan
                        except (ValueError, TypeError):
                            return math.nan
                    values = [finite(r.get(metric)) for r in cohort]
                    if any(math.isfinite(x) for x in values):
                        ax.plot([r['step'] for r in cohort], values, label=metric)
                        plotted = True
                if plotted:
                    ax.set(xlabel='Completed updates', ylabel=filename,
                           title=f'{run["meta"]["run"]}: {matrix}\nDescriptive fits; inspect support and randomized controls')
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(dest / f'{safe_name}_{filename}.png', dpi=120)
                plt.close(fig)


def analyze(root: Path, *, plots=True) -> Path:
    from study import atomic_text, exclusive
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Results directory does not exist: {root}')
    paths = ([root / 'manifest.json'] if (root / 'manifest.json').is_file() else
             sorted(root.glob('*/*/verbatim*/*/seed_*/manifest.json')))
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    with exclusive(out / '.analysis.lock'):
        sources, warnings, runs = {}, [], []
        for path in paths:
            # Inspect scope before reading full probe data; exclude smoke evidence.
            meta = json.loads(read_file(path, sources))
            if meta.get('stage') not in ('pilot', 'full') or meta.get('condition') not in ('verbatim', 'verbatim_absent'):
                continue
            runs.append(read_run(path.parent, sources, warnings))
        ids = [r['meta']['run'] for r in runs]
        if len(ids) != len(set(ids)):
            raise ValueError('Multiple directories identify the same run; analyze them separately.')
        behavior = [row for r in runs for row in r['behavior']]
        examples = [row for r in runs for row in r['examples']]
        spectra = [row for r in runs for row in r['spectra']]
        doses, canaries, contrasts = summarize(runs)
        comparison = compare(runs, warnings)
        write_csv(out / 'behavior.csv', behavior, ID + ['step', 'group', 'n'] + METRICS)
        write_csv(out / 'examples.csv', examples, ID + ['step', 'group', 'eid'] + METRICS)
        write_csv(out / 'dose_summary.csv', doses, ID + ['group', 'status'])
        write_csv(out / 'canary_summary.csv', canaries, ID + ['group', 'eid'])
        write_csv(out / 'zero_dose_contrasts.csv', contrasts, ID + ['step', 'group'])
        write_csv(out / 'optimizer_comparison.csv', comparison, ['stage', 'recipe', 'condition', 'seed', 'step', 'group'])
        write_csv(out / 'spectral.csv', spectra, ID + ['step', 'matrix_name'])
        plot_status = 'disabled'
        if plots:
            try:
                make_plots(runs, out)
                plot_status = 'generated'
            except ImportError as exc:
                plot_status = 'unavailable'
                warnings.append(f'Plots unavailable ({exc}); CSV tables and report still written.')
        lines = ['# Memorization analysis', '', f'Results: `{root}`', '',
                 'Observed saved checkpoints only. No training, checkpoint modification, or new WeightWatcher fitting was performed.', '',
                 '| Run | Status | Last observed update | Planned updates |', '|---|---|---:|---:|']
        for r in runs:
            lines.append(f'| {r["meta"]["run"]} | {r["status"]} | {r["last_step"]} | {r["target_steps"]} |')
        if not behavior:
            lines += ['', 'No non-smoke behavioral measurements are available yet.']
        lines += ['', '## Exact recall by exposure group', '',
                  '| Run / group | Probes | Audits | First observed exact recall | Peak exact rate | Latest exact rate |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in doses:
            onset = r['first_observed_exact_step']
            lines.append(f'| {r["run"]} / {r["group"]} | {r["n"]} | {r["checkpoints"]} | '
                         f'{onset if onset is not None else "not observed"} | {r["peak_observed_exact_match"]:.4f} | {r["latest_exact_match"]:.4f} |')
        lines += ['', '## Interpretation and limits', '',
                  'Exact match is reproduction of the complete scored suffix. Compare positive-dose cohorts with zero-dose controls at the same update and prefix length; NLL reduction or partial accuracy alone is not exact extraction.', '',
                  'A dose is its intended lifetime total, not its exposure count at an early checkpoint. Per-example presentation counts are reconstructed from the saved slot schedule and completed updates, not independently measured counters at every checkpoint. Missing schedules remain missing.', '',
                  'Onset means the first saved checkpoint showing an exact match, not the exact update at which it was learned. Prefix sweeps measured only at the final checkpoint have one observation. No observed exact matches does not prove absence of weaker or unprobed memory.', '',
                  'Incomplete, failed and running runs are not final results. Resume may replace uncommitted trailing diagnostic rows; the report hashes the exact files read and does not validate optimizer checkpoint contents.', '',
                  'Optimizer differences use only common recorded updates and matching seed, data, probe inventory, source, recipe and hardware. No extrapolation, pooled checkpoint confidence intervals, or optimizer superiority claims are made. One paired seed is descriptive.', '',
                  'Spectral CSV rows are aligned by recorded step only. Historical spectral files do not contain a model-state hash. All raw fit columns are retained; finite alpha alone is not evidence of a valid power-law tail or memorization. Inspect fit quality, support, and randomized separation matrix by matrix. Spectral timing does not localize individual canaries or establish causation.', '',
                  f'Paired optimizer comparison rows: {len(comparison)}. Plot status: {plot_status}.']
        if warnings:
            lines += ['', '## Data warnings', ''] + [f'- {w}' for w in warnings]
        atomic_text(out / 'summary.md', '\n'.join(lines) + '\n')
        atomic_text(out / 'source_hashes.json', json.dumps(sources, indent=2, sort_keys=True) + '\n')
        for r in doses:
            if r['prefix_tokens'] == 64:
                print(f'{r["run"]} {r["group"]}: peak exact={r["peak_observed_exact_match"]:.3f}, '
                      f'latest exact={r["latest_exact_match"]:.3f}, latest NLL={r["latest_nll"]:.3f} [{r["status"]}]')
        for warning in warnings:
            print('WARNING:', warning)
    return out
