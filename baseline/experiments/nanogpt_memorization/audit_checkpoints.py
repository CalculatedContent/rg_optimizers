#!/usr/bin/env python3
"""Evaluate a completed verbatim run's existing checkpoints. Never trains a model.

Default: all saved model snapshots; teacher-forced replay/fresh-background tests
at each snapshot, and richer greedy/counterfactual tests at the final snapshot.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL_PATH = 'baseline/nanogpt_one_head/src/rg_nanogpt_one_head/model.py'
RUN_PATH = 'baseline/experiments/nanogpt_memorization/run.py'
METRICS = ('nll', 'teacher_forced_accuracy', 'support_mass', 'conditional_nll',
           'confidence', 'brier', 'first_token_nll', 'exact_match', 'token_accuracy',
           'em_first_1', 'em_first_4', 'em_first_8', 'em_first_16', 'em_first_32')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def records_key(prompt):
    return hashlib.sha256(np.asarray(prompt, dtype='<i8').tobytes()).digest()


def background_target(prompt, length):
    return tuple(16 + (int(v) - 16 + 1) % 256 for v in prompt[-length:])


def inventory(run_dir):
    manifest = load_json(run_dir / 'manifest.json')
    if manifest.get('condition') != 'verbatim' or manifest.get('profile', {}).get('family') != 'adamw':
        raise ValueError('This audit is scoped to the completed AdamW verbatim run.')
    horizon = int(manifest['suite']['stages'][manifest['stage']]['steps'])
    done = load_json(run_dir / 'complete.json')
    if (done.get('status') != 'complete' or done.get('steps') != horizon
            or done.get('fingerprint') != manifest.get('fingerprint')):
        raise ValueError('A matching completion marker is required. Do not audit an active run here.')
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
    steps = [r['step'] for r in rows]
    if not rows or steps != sorted(set(steps)) or steps[-1] != horizon:
        raise ValueError('Invalid or incomplete recorded metric trajectory.')
    found = {}
    for path in sorted(run_dir.glob('model_step_*.pt')):
        match = re.fullmatch(r'model_step_(\d+)\.pt', path.name)
        if match and int(match[1]) in steps:
            found[int(match[1])] = path
    if horizon not in found and (run_dir / 'checkpoint_latest.pt').is_file():
        found[horizon] = run_dir / 'checkpoint_latest.pt'
    if 0 not in found or horizon not in found or len(found) < 2:
        raise ValueError('Initialization, final, and at least two saved model snapshots are required.')
    return manifest, rows, sorted(found.items())


def verify_source(manifest):
    if sha(REPO / RUN_PATH) != manifest['runner_sha256']:
        raise ValueError('run.py differs from the training manifest. Restore its original version first.')
    raw = (REPO / MODEL_PATH).read_bytes()
    git_sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if git_sha != manifest['suite']['source_blobs'][MODEL_PATH]:
        raise ValueError('Model source differs from the source used for training.')


def make_probe(record, variant, *, category='canary', dose=None, pad_shift=0):
    return {'record': record, 'variant': variant, 'category': category,
            'dose': dose, 'pad_shift': pad_shift}


def replay_cohort(parent, study, start, stop, n, seed, group):
    if start < 0 or stop <= start or n > (stop - start) * study.batch:
        raise ValueError('Replay interval has insufficient examples.')
    injected = sum(start * study.batch <= slot < stop * study.batch for slot in study.injections)
    if n > (stop - start) * study.batch - injected:
        raise ValueError('Replay interval has too few non-canary records.')
    rng = np.random.default_rng(seed)
    selected, keys, used = [], set(), set()
    while len(selected) < n:
        slot = int(rng.integers(start * study.batch, stop * study.batch))
        if slot in used:
            continue
        used.add(slot)
        step, index = divmod(slot, study.batch)
        record = study.sample(step)[index]
        key = records_key(record.prompt)
        if record.eid != 'background' or key in keys:
            continue
        keys.add(key)
        selected.append(replace(record, eid=f'background_slot_{slot}', group=group))
    return selected


def final_probes(canaries, fresh, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for r in canaries:
        groups[r.group].append(r)
    out = []
    for group, members in groups.items():
        dose = int(group.split('_')[-1])
        for i, r in enumerate(members):
            wrong = members[(i + 1) % len(members)].prompt
            if len(members) < 2:
                raise ValueError('Need at least two canaries per dose for wrong-prefix controls.')
            out.append(make_probe(r, 'true_prefix', dose=dose))
            out.append(make_probe(replace(r, prompt=wrong), 'wrong_prefix', dose=dose))
            shuffled = tuple(np.asarray(r.prompt)[rng.permutation(len(r.prompt))].tolist())
            out.append(make_probe(replace(r, prompt=shuffled), 'shuffled_prefix', dose=dose))
            for length in (8, 16, 32):
                out.append(make_probe(replace(r, prompt=r.prompt[-length:]), f'prefix_{length}', dose=dose))
            zero_prompt = (0,) * len(r.prompt)
            out.append(make_probe(replace(r, prompt=zero_prompt), 'no_prefix', dose=dose))
            for hint in (8, 16):
                out.append(make_probe(replace(r, prompt=r.prompt + r.target[:hint], target=r.target[hint:]),
                                      f'true_prefix_hint_{hint}', dose=dose))
            out.append(make_probe(replace(r, prompt=zero_prompt + r.target[:16], target=r.target[16:]),
                                  'no_prefix_hint_16', dose=dose))
            out.append(make_probe(replace(r, prompt=wrong + r.target[:16], target=r.target[16:]),
                                  'wrong_prefix_hint_16', dose=dose))
            out.append(make_probe(r, 'position_shift_16', dose=dose, pad_shift=16))
    for r in fresh:
        out.append(make_probe(r, 'original', category='background'))
        p, length = r.prompt, len(r.target)
        distractors = tuple(np.asarray(p[:-length])[rng.permutation(len(p) - length)].tolist())
        out.append(make_probe(replace(r, prompt=distractors + p[-length:]),
                              'irrelevant_half_shuffled', category='background'))
        # Same token marginal distribution, but changed rule-relevant input.
        relevant = p[-length:]
        new_prompt = p[:-length] + relevant[1:] + relevant[:1]
        out.append(make_probe(replace(r, prompt=new_prompt, target=background_target(new_prompt, length)),
                              'source_rotated_relabelled', category='background'))
        out.append(make_probe(replace(r, prompt=new_prompt), 'source_rotated_old_answer', category='background'))
        out.append(make_probe(r, 'position_shift_16', category='background', pad_shift=16))
    return out


def verify_stream(study, run_dir, candidates):
    """Replay data generation only. No model, forward pass, or training here."""
    expected = load_json(run_dir / 'injection_schedule.json')
    if expected != {str(k): r.eid for k, r in study.injections.items()}:
        raise ValueError('Reconstructed intervention schedule does not match the saved schedule.')
    exposures = load_json(run_dir / 'exposures.json')
    for r in study.audit:
        if exposures.get(r.eid) != int(r.group.split('_')[-1]):
            raise ValueError(f'Saved realized exposure count disagrees for {r.eid}.')
    if exposures.get('background') != study.steps * study.batch - len(study.injections):
        raise ValueError('Saved background presentation count is inconsistent.')
    keys = {records_key(r.prompt) for r in candidates}
    visits = {key: [] for key in keys}
    for step in range(study.steps):
        for record in study.sample(step):
            key = records_key(record.prompt)
            if key in visits:
                visits[key].append(step + 1)
        if (step + 1) % 5000 == 0:
            print(f'Checking fresh-example exclusion: {step + 1}/{study.steps} updates replayed', flush=True)
    return visits


def pack_probe(probe, block_size):
    r = probe['record']
    total = block_size + 1 - probe['pad_shift']
    padding = total - len(r.prompt) - len(r.target)
    if not r.prompt or not r.target or padding < 0:
        raise ValueError('Invalid probe or insufficient context.')
    sequence = (0,) * padding + r.prompt + r.target
    return sequence[:-1], total - len(r.target) - 1


def score(model, probes, batch_size, device, *, generate):
    import torch
    import torch.nn.functional as F
    rows = []
    grouped = defaultdict(list)
    for p in probes:
        grouped[(len(p['record'].target), p['pad_shift'])].append(p)
    model.eval()
    with torch.inference_mode():
        for (length, shift), members in grouped.items():
            for start in range(0, len(members), batch_size):
                chunk = members[start:start + batch_size]
                packed = [pack_probe(p, model.cfg.block_size) for p in chunk]
                offset = packed[0][1]
                x = torch.tensor([r[0] for r in packed], device=device, dtype=torch.long)
                labels = torch.tensor([p['record'].target for p in chunk], device=device, dtype=torch.long)
                # Compute the vocabulary projection only where a suffix is scored.
                logits = model.lm_head(model.hidden_states(x)[:, offset:, :])
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite evaluation logits; no zero scores substituted.')
                losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='none').reshape(-1, length)
                probabilities = logits.softmax(-1)
                mass = probabilities[:, :, 16:272].sum(-1)
                conditional = F.cross_entropy(logits[:, :, 16:272].reshape(-1, 256),
                                             (labels - 16).reshape(-1), reduction='none').reshape(-1, length)
                correct = logits.argmax(-1).eq(labels)
                confidence = probabilities.max(-1).values
                truth_prob = probabilities.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                brier = probabilities.square().sum(-1) - 2 * truth_prob + 1
                values = {'nll': losses.mean(1), 'teacher_forced_accuracy': correct.float().mean(1),
                          'support_mass': mass.mean(1), 'conditional_nll': conditional.mean(1),
                          'confidence': confidence.mean(1), 'brier': brier.mean(1),
                          'first_token_nll': losses[:, 0]}
                arrays = {k: v.cpu().tolist() for k, v in values.items()}
                losses_cpu, correct_cpu = losses.cpu().tolist(), correct.cpu().tolist()
                generated = None
                if generate:
                    prefix = x[:, :offset + 1]
                    generated = model.generate_greedy(prefix, length)[:, -length:].cpu().tolist()
                for j, p in enumerate(chunk):
                    r = p['record']
                    row = {'category': p['category'], 'variant': p['variant'], 'eid': r.eid,
                           'dose': p['dose'], 'scored_tokens': length, 'pad_shift': shift,
                           **{k: a[j] for k, a in arrays.items()},
                           'nll_by_token': losses_cpu[j], 'tf_correct_by_token': correct_cpu[j]}
                    if generated is not None:
                        match = np.asarray(generated[j]) == np.asarray(r.target)
                        row.update(generated_tokens=generated[j], target_tokens=list(r.target),
                                   exact_match=float(match.all()), token_accuracy=float(match.mean()))
                        for size in (1, 4, 8, 16, 32):
                            row[f'em_first_{size}'] = float(match[:size].all()) if size <= length else None
                    rows.append(row)
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['category'], r['variant'], r['dose'], r['scored_tokens'])].append(r)
    out = []
    for (category, variant, dose, length), group in groups.items():
        row = dict(category=category, variant=variant, dose=dose, scored_tokens=length, n=len(group))
        for key in METRICS:
            values = [r[key] for r in group if r.get(key) is not None]
            row[key] = float(np.mean(values)) if values else None
        out.append(row)
    return out


def saved_behavior(rows, parent, study):
    output = []
    schedule = defaultdict(list)
    for slot, record in study.injections.items():
        schedule[record.eid].append(slot)
    for slots in schedule.values():
        slots.sort()
    for checkpoint in rows:
        for group, data in checkpoint['audit'].items():
            for item in data['examples']:
                visits = bisect_left(schedule[item['eid']], checkpoint['step'] * study.batch)
                output.append({'step': checkpoint['step'], 'group': group,
                               'presentations': visits, **item})
    return output


def run_audit(args):
    import pandas as pd
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest, history, paths = inventory(run_dir)
    print('Saved weight snapshots:', ', '.join(str(s) for s, _ in paths), flush=True)
    print(f'{len(history)} behavioral audits; {len(paths)} weight snapshots. These are different inventories.', flush=True)
    if args.list:
        return
    verify_source(manifest)
    parent = import_path('_memorization_audit_parent', REPO / RUN_PATH)
    model_module = import_path('_memorization_audit_model', REPO / MODEL_PATH)
    import torch
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable. Select --device cpu explicitly; do not silently change backend.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable.')
    if args.background_examples < 2 or args.batch_size < 1:
        raise ValueError('At least two background examples and a positive batch size are required.')
    cfg = manifest['suite']
    if (cfg['canary_prefix_tokens'], cfg['canary_suffix_tokens']) != (64, 32):
        raise ValueError('This audit expects the original 64-prefix/32-suffix protocol.')
    batch = int(manifest['input_tokens_per_step']) // int(manifest['source_model']['block_size'])
    study = parent.Study(cfg, manifest['condition'], manifest['stage'], manifest['seed'], batch)
    if study.identity() != manifest['data_sha256']:
        raise ValueError('Reconstructed data identity differs from the run manifest.')
    stored_inventory = load_json(run_dir / 'probe_inventory.json')
    if json.loads(json.dumps([asdict(r) for r in study.audit])) != stored_inventory:
        raise ValueError('Reconstructed canaries differ from the saved inventory.')
    # New outputs only. Never open an existing output folder for overwrite.
    out = (Path(args.output).expanduser() if args.output else
           Path('/tmp') / ('nanogpt_checkpoint_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if out == run_dir or out.is_relative_to(run_dir):
        raise ValueError('Choose a separate output directory, outside the training run.')
    out.mkdir(parents=True, exist_ok=False)
    print(f'Audit output: {out}', flush=True)
    original_hashes = {str(p): sha(p) for p in run_dir.rglob('*') if p.is_file() and p.suffix in ('.pt', '.json', '.jsonl', '.csv')}
    earliest = min(s for s, _ in paths if s > 0)
    early = replay_cohort(parent, study, 0, min(1024, earliest), args.background_examples,
                          args.audit_seed, 'seen_early')
    rng = np.random.default_rng(args.audit_seed + 1)
    fresh = []
    for i in range(args.background_examples):
        prompt = tuple(map(int, rng.integers(16, 272, 64)))
        fresh.append(parent.Record(f'fresh_{i}', 'fresh', prompt, background_target(prompt, 32)))
    if len({r.prompt for r in fresh}) != len(fresh):
        raise ValueError('Fresh cohort contains duplicates.')
    recent = {s: replay_cohort(parent, study, max(0, s - 1024), s, args.background_examples,
                               args.audit_seed + s, 'seen_recent') for s, _ in paths if s > 0}
    probes = final_probes(study.audit, fresh, args.audit_seed + 2)
    fresh_candidates = [p['record'] for p in probes if p['category'] == 'background']
    candidates = early + fresh_candidates + [r for cohort in recent.values() for r in cohort]
    visits = verify_stream(study, run_dir, candidates)
    if any(visits[records_key(r.prompt)] for r in fresh_candidates):
        raise ValueError('A proposed fresh/background-counterfactual input appeared in training.')
    write_json(out / 'protocol.json', {
        'source_run': str(run_dir), 'run_fingerprint': manifest['fingerprint'],
        'checkpoints': [s for s, _ in paths], 'audit_seed': args.audit_seed,
        'background_examples': args.background_examples, 'device': args.device,
        'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': str(np.__version__),
        'script_sha256': sha(Path(__file__)), 'torch_deterministic_algorithms': True,
        'fresh_prompt_exclusion': 'checked against every background record in completed training stream',
        'optimizer_steps': 0, 'new_weightwatcher_calls': 0,
        'background_rule': 'target[j] = 16 + (prefix[-32+j] - 16 + 1) % 256',
        'interpretation': 'post-hoc single-model audit; no replication or causal attribution to a weight spectrum',
    })
    write_json(out / 'probe_inventory.json', [{**{k:v for k,v in p.items() if k != 'record'},
                                             'record': asdict(p['record'])} for p in probes])
    write_json(out / 'background_cohorts.json', {'early': [asdict(r) for r in early],
                                               'fresh': [asdict(r) for r in fresh],
                                               'recent': {str(s): [asdict(r) for r in rs] for s,rs in recent.items()}})
    recorded = saved_behavior(history, parent, study)
    pd.DataFrame(recorded).to_csv(out / 'recorded_canaries.csv', index=False)
    torch.manual_seed(args.audit_seed)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = model_module.GPT(model_module.GPTConfig(**manifest['source_model'])).to(args.device)
    model.requires_grad_(False)
    model.eval()
    warnings, trajectory, all_summary = [], [], []
    original_device = manifest['device'].get('device')
    if original_device != args.device or str(manifest['device'].get('torch')) != str(torch.__version__):
        warnings.append('Evaluation backend or PyTorch version differs from training; inspect reproduction errors before interpreting new scores.')
    by_step = {r['step']:r for r in history}
    for step, path in paths:
        begin = time.monotonic()
        print(f'Evaluating saved weights at step {step} (no training)', flush=True)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('step') != step or checkpoint.get('fingerprint') != manifest['fingerprint']:
            raise ValueError(f'Checkpoint step/fingerprint mismatch: {path}')
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        before = parent.state_digest(model)
        if before != by_step[step]['model_sha256']:
            raise ValueError(f'Model weights do not match the behavioral audit hash: step {step}')
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError(f'Nonfinite saved weights at step {step}')
        core = [make_probe(r, 'true_prefix', dose=int(r.group.split('_')[-1])) for r in study.audit]
        core += [make_probe(r, 'seen_early' if step else 'early_not_yet_seen', category='background') for r in early]
        core += [make_probe(r, 'fresh', category='background') for r in fresh]
        if step:
            core += [make_probe(r, 'seen_recent', category='background') for r in recent[step]]
        scored = score(model, core, args.batch_size, args.device, generate=False)
        saved = {r['eid']:r for group,data in by_step[step]['audit'].items() if group.endswith('/prefix_64') for r in data['examples']}
        errors = [abs(r['nll'] - saved[r['eid']]['nll']) for r in scored if r['category'] == 'canary']
        max_error = max(errors)
        if max_error > 0.005:
            raise ValueError(f'Original canary NLL does not reproduce at step {step}: max error {max_error:.6g}. Investigate backend/source before further interpretation.')
        for row in scored:
            row['step'] = step
            trajectory.append(row)
        rows = aggregate(scored)
        for row in rows:
            row.update(step=step, source_model_sha256=before, max_original_nll_error=max_error)
        all_summary.extend(rows)
        pd.DataFrame(all_summary).to_csv(out / 'checkpoint_scores.csv', index=False)
        with (out / f'teacher_scores_{step:08d}.jsonl').open('w') as handle:
            for row in scored:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
        if step == study.steps:
            final = score(model, probes, args.batch_size, args.device, generate=True)
            true = [r for r in final if r['category'] == 'canary' and r['variant'] == 'true_prefix']
            if any(r['exact_match'] != saved[r['eid']]['exact_match'] for r in true):
                warnings.append('Final greedy exact matches differ from original audit; review backend and score reproduction.')
            pd.DataFrame(aggregate(final)).to_csv(out / 'final_probes.csv', index=False)
            with (out / 'final_probe_details.jsonl').open('w') as handle:
                for row in final:
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
        if parent.state_digest(model) != before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Read-only evaluation invariant failed.')
        print(f'Finished step {step} in {time.monotonic()-begin:.1f}s', flush=True)
    after = {p: sha(p) for p in original_hashes}
    if after != original_hashes:
        raise RuntimeError('An original input changed during audit; results are not accepted.')
    write_json(out / 'source_hashes.json', original_hashes)
    write_report(out, all_summary, final, recorded, warnings)
    write_json(out / 'complete.json', {'status':'complete', 'source_weights_unchanged':True,
                                      'checkpoints':len(paths), 'new_training_updates':0})
    print(f'\nFinished. Read: {out / "report.md"}', flush=True)


def write_report(out, summary, final, recorded, warnings):
    import pandas as pd
    table = pd.DataFrame(summary)
    background = table[table.category == 'background']
    gaps = []
    for step, group in background.groupby('step'):
        fresh = group[group.variant == 'fresh'].iloc[0]
        for variant in ('seen_early', 'seen_recent'):
            seen = group[group.variant == variant]
            if len(seen):
                seen = seen.iloc[0]
                gaps.append({'step':int(step), 'replay_cohort':variant,
                             'fresh_nll_minus_replay_nll':fresh.nll-seen.nll,
                             'replay_accuracy_minus_fresh_accuracy':seen.teacher_forced_accuracy-fresh.teacher_forced_accuracy})
    pd.DataFrame(gaps).to_csv(out / 'background_gaps.csv', index=False)
    scores = {(r['category'],r['variant'],r['eid']):r for r in final}
    paired = []
    for row in final:
        base_variant = 'true_prefix' if row['category'] == 'canary' else 'original'
        # Hint comparisons use the same supplied gold-token count and same scored tail.
        if 'hint_16' in row['variant']:
            base_variant = 'true_prefix_hint_16'
        base = scores.get((row['category'],base_variant,row['eid']))
        if base is None or row['scored_tokens'] != base['scored_tokens'] or row['variant'] == base_variant:
            continue
        paired.append({'category':row['category'], 'variant':row['variant'], 'reference':base_variant,
                       'eid':row['eid'], 'dose':row['dose'], 'scored_tokens':row['scored_tokens'],
                       'perturbed_nll_minus_reference_nll':row['nll']-base['nll'],
                       'perturbed_first_token_nll_minus_reference':row['first_token_nll']-base['first_token_nll'],
                       'reference_exact_minus_perturbed_exact':base['exact_match']-row['exact_match']})
    pd.DataFrame(paired).to_csv(out / 'paired_prompt_effects.csv', index=False)
    lines = ['# Existing AdamW checkpoint audit', '',
             'No new training, optimizer updates, learning-rate changes, checkpoint overwrites, or WeightWatcher fits.', '',
             '## Read these results in order', '',
             '1. `recorded_canaries.csv`: original exact/partial recall and NLL at every recorded audit. Positive lifetime dose is not cumulative exposure at early checkpoints.',
             '2. `checkpoint_scores.csv` and `background_gaps.csv`: actual replay versus disjoint fresh background inputs. A positive fresh-minus-replay NLL gap supports example-specific fitting; inspect accuracy and its time trend as well.',
             '3. `final_probes.csv`: shorter exact prefixes of the generated suffix, prefix ablations, gold-hint-assisted recovery, background rule perturbations, and position shifts.',
             '4. `paired_prompt_effects.csv`: within-target changes, not comparisons between unrelated targets. Hint tests are compared only at the same scored-tail length.', '',
             '## Critical interpretation', '',
             'The background task is a deterministic token transformation of the last 32 prefix tokens. Better fresh-rule performance than random-canary performance can be generalization to the majority task, not overfitting.',
             'A falling fresh-rule loss is not memorization. Poor replay AND fresh performance is not a demonstrated train/test generalization gap. A final loss deterioration alone is not sufficient to distinguish model miscalibration, fitting noise, and forgetting.',
             'Correct recovery after supplying 16 gold suffix tokens is conditional 16-token recovery, not extraction of the original 32-token secret. Compare with wrong-prefix and zero-dose controls.',
             'No-prefix prompts and position shifts are distribution-shift diagnostics. Sensitivity alone does not establish memorization or overfitting. Source rotation is evaluated both with its new correct answer and the deliberately inconsistent old answer.',
             'The replay/fresh audit uses fixed generated examples and validates fresh-input exclusion by reconstructing every completed training input. It depends on the saved sampler and manifest; no missing raw corpus is assumed.',
             'All tests are exploratory and conditional on one trained model. Do not treat tokens, probes, or checkpoints as independent training replications. Absence of greedy extraction does not prove absence of all latent information.',
             'The old canary/behavioral logs have more timestamps than there are saved model files. New forward-pass tests cannot recover unsaved intermediate weights.',
             'Random-label, mixed-label-noise, and conflicting-mapping experiments were NOT present in this training run and cannot be retroactively substituted for its data.', '',
             '## Final full-prefix canaries', '',
             '| Dose | N | NLL | Teacher-forced accuracy | Greedy EM32 | EM first 4 | EM first 8 |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate(final):
        if r['category'] == 'canary' and r['variant'] == 'true_prefix':
            lines.append(f"| {r['dose']} | {r['n']} | {r['nll']:.4f} | {r['teacher_forced_accuracy']:.4f} | {r['exact_match']:.4f} | {r['em_first_4']:.4f} | {r['em_first_8']:.4f} |")
    lines += ['', '## Background reference', '',
              f'Uniform prediction on the known 256-token target support has NLL log(256) = {math.log(256):.6f} and token accuracy 1/256. This is a marginal reference, not the Bayes limit: the background rule is deterministic.',
              '`support_mass` reports probability assigned to valid target tokens. `conditional_nll` renormalizes only for a diagnostic; it does not change model predictions. Full-vocabulary NLL and raw greedy recall remain the primary measurements.', '',
              '## Warnings', ''] + (warnings or ['None. Source/weight checks and original-NLL reproduction checks passed.'])
    (out / 'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Directory containing manifest.json and saved .pt files.')
    parser.add_argument('--device', choices=('mps','cuda','cpu'), default='mps')
    parser.add_argument('--background-examples', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--audit-seed', type=int, default=20260915)
    parser.add_argument('--output', help='New separate output folder; default is readable timestamp under /tmp.')
    parser.add_argument('--list', action='store_true', help='List available saved weights without evaluating them.')
    args = parser.parse_args()
    try:
        run_audit(args)
    except KeyboardInterrupt:
        print('\nAudit stopped; original checkpoints were not overwritten.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
