from __future__ import annotations
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import recall, exposure, prefix_compression, score_continuation, rank_canary
from probes import canary_universe, random_sequence_probes, presentation_schedule
from prepare_plan import resolve, git_blob_sha, write_plan
from monitor import monitor_training_state


class TinyModel(torch.nn.Module):
    """Controlled transition model, not nanoGPT performance evidence."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(block_size=16)

    def forward(self, idx):
        logits = torch.nn.functional.one_hot((idx + 1) % 8, 8).float() * 12
        return logits, None

    def generate_greedy(self, prompts, max_new_tokens):
        ids = prompts.clone()
        for _ in range(max_new_tokens):
            ids = torch.cat((ids, (ids[:, -1:] + 1) % 8), dim=1)
        return ids


def test_exact_and_partial_recall_are_distinct():
    row = recall([1, 8, 3, 4], [1, 2, 3, 4])
    assert not row['sequence_exact_match']
    assert row['free_running_token_match'] == .75
    assert row['longest_exact_prefix'] == 1
    assert recall([1, 2], [1, 2])['sequence_exact_match']


@pytest.mark.parametrize('g,t', [([], []), ([1], [1, 2]), ([[1]], [[1]])])
def test_recall_rejects_malformed(g, t):
    with pytest.raises(ValueError):
        recall(g, t)


def test_exact_exposure():
    row = exposure([8, 3, 5, 9], 1)
    assert row['rank_min'] == row['rank_max'] == 1
    assert row['exposure_bits_lower'] == 2
    assert exposure([8, 3, 5, 9], 3)['exposure_bits_lower'] == 0


def test_exposure_ties_do_not_invent_memorization():
    row = exposure([2, 2, 2, 2], 0)
    assert row['exposure_bits_lower'] == 0
    assert row['exposure_bits_upper'] == 2
    assert row['tie_count'] == 4


@pytest.mark.parametrize('scores,index', [([], 0), ([1, float('nan')], 0), ([1], 1), ([float('inf')], 0)])
def test_exposure_rejects_missing_scores(scores, index):
    with pytest.raises(ValueError):
        exposure(scores, index)


def test_prefix_sweep_fixed_boundary_and_nonmonotone_success():
    observed = []
    def generate(p, n):
        observed.append(p)
        return [9, 9] if len(p) in (1, 3) else [8, 8]
    row = prefix_compression([1, 2, 3], [9, 9], [1, 2, 3], generate)
    assert observed == [[3], [2, 3], [1, 2, 3]]
    assert row['shortest_successful_prefix_in_grid'] == 1
    assert row['prefix_compression_ratio'] == 2


def test_prefix_failure_is_censored():
    row = prefix_compression([1, 2], [5, 6], [1, 2], lambda p, n: [0, 0])
    assert row['search_censored']
    assert row['prefix_compression_ratio'] is None


@pytest.mark.parametrize('grid', [[], [0], [3], [1.5]])
def test_bad_prefix_grid(grid):
    with pytest.raises(ValueError):
        prefix_compression([1, 2], [5], grid, lambda p, n: [5])


def test_suffix_alignment_and_teacher_forcing_distinction():
    model = TinyModel().train()
    good = score_continuation(model, [0, 1], [2, 3])
    assert good['sequence_exact_match'] and good['suffix_nll'] < .01
    bad = score_continuation(model, [0, 1], [2, 5, 6])
    assert bad['teacher_forced_token_accuracy'] == pytest.approx(2 / 3)
    assert bad['free_running_token_match'] == pytest.approx(1 / 3)
    assert bad['longest_exact_prefix'] == 1
    assert model.training


def test_score_preserves_eval_mode_and_rejects_overflow():
    model = TinyModel().eval()
    score_continuation(model, [1], [2])
    assert not model.training
    with pytest.raises(ValueError):
        score_continuation(model, [1] * 16, [2, 3])


def test_candidate_ranking_and_batching():
    model = TinyModel().train()
    candidates = [[2, 3], [4, 3], [2, 5], [4, 5]]
    a = rank_canary(model, [0, 1], candidates, 0, batch_size=1)
    b = rank_canary(model, [0, 1], candidates, 0, batch_size=3)
    assert a['rank_min'] == 1 and a['exposure_bits_lower'] == 2
    assert a['candidate_nll_sum'] == pytest.approx(b['candidate_nll_sum'])
    assert model.training


def test_candidate_duplicates_rejected():
    with pytest.raises(ValueError):
        rank_canary(TinyModel(), [1], [[2], [2]], 0)


def test_canary_universe():
    values = canary_universe(list(range(16)), 3)
    assert len(values) == len({tuple(x) for x in values}) == 4096
    assert values[0] == [0, 0, 0] and values[-1] == [15, 15, 15]


def test_random_probes_are_paired_and_do_not_touch_global_rng():
    np.random.seed(92)
    before = np.random.get_state()
    a = random_sequence_probes(11, per_dose=2)
    b = random_sequence_probes(11, per_dose=2)
    after = np.random.get_state()
    assert a == b
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert len({tuple(x['target']) for x in a}) == len(a)


def test_schedule_counts_full_presentations_and_excludes_zero_dose():
    probes = random_sequence_probes(2, per_dose=2)
    rows = presentation_schedule(probes, 1000, 3)
    counts = Counter(row['probe_id'] for row in rows)
    assert all(counts[p['id']] == p['dose'] for p in probes)
    assert len({row['slot'] for row in rows}) == len(rows)
    assert rows == presentation_schedule(probes, 1000, 3)


def test_schedule_rejects_insufficient_slots():
    with pytest.raises(ValueError):
        presentation_schedule([{'id': 'a', 'dose': 2}], 1, 0)


def config_fixture(tmp_path):
    study = json.loads((ROOT / 'study.json').read_text())
    base = {'protocol': {}, 'dataset': {'train_tokens': 80000000},
            'model': {'block_size': 256},
            'training': {'batch_size': 4, 'grad_accum_steps': 8, 'seeds': study['seeds']},
            'optimizer_profiles': {'adamw': {'family': 'adamw', 'learning_rate': .0006},
                                   'muon': {'family': 'muon', 'matrix_learning_rate': .02}},
            'weightwatcher': {k: study['monitoring'][k] for k in
                             ('fix_fingers', 'ERG', 'randomize', 'min_evals', 'max_fingers', 'require_raw_alpha')}}
    path = tmp_path / study['source']['recipe']
    path.parent.mkdir(parents=True)
    raw = yaml.safe_dump(base).encode()
    path.write_bytes(raw)
    study['source']['recipe_git_blob_sha'] = git_blob_sha(raw)
    study['source']['modules'] = {}
    return study, path


def test_plan_counts_horizon_and_optimizer_inheritance(tmp_path):
    study, _ = config_fixture(tmp_path)
    plan = resolve(tmp_path, study)
    assert len(plan['runs']) == 30
    assert plan['tokens_per_update'] == 8192
    assert plan['schedule_steps'] == 9766
    assert plan['total_steps'] == 39063
    assert plan['actual_target_tokens_per_run'] == 320004096
    assert plan['runs'][0]['baseline_config']['optimizer_profiles']['adamw']['learning_rate'] == .0006
    assert plan['runs'][1]['baseline_config']['optimizer_profiles']['muon']['matrix_learning_rate'] == .02
    assert plan['execution_status'].startswith('plan_only')
    assert plan['fingerprint'] == resolve(tmp_path, study)['fingerprint']


def test_source_drift_fails_closed(tmp_path):
    study, path = config_fixture(tmp_path)
    path.write_text(path.read_text() + '\n# drift\n')
    with pytest.raises(RuntimeError, match='changed'):
        resolve(tmp_path, study)


def test_plan_refuses_implicit_muonclip_substitution(tmp_path):
    study, _ = config_fixture(tmp_path)
    study['optimizers'] = ['adamw', 'muon_clip']
    with pytest.raises(ValueError, match='MuonClip'):
        resolve(tmp_path, study)


def test_plan_refuses_existing_output_directory(tmp_path):
    study, _ = config_fixture(tmp_path)
    plan = resolve(tmp_path, study)
    out = tmp_path / 'output'
    write_plan(plan, out)
    assert len(list((out / 'configs').glob('*.yaml'))) == 30
    with pytest.raises(FileExistsError):
        write_plan(plan, out)


@pytest.mark.parametrize('key,value', [('fix_fingers', False), ('ERG', False), ('randomize', False), ('strict', False)])
def test_monitor_refuses_invalid_settings_before_import(key, value):
    cfg = dict(fix_fingers='clip_xmax', ERG=True, randomize=True, strict=True, enabled=True, require_raw_alpha=True)
    cfg[key] = value
    with pytest.raises(ValueError):
        monitor_training_state(None, '/tmp/not-created', step=0, tokens_seen=0,
                               reference_tokens=80000000, seed=1337, fingerprint='x', ww_config=cfg)
