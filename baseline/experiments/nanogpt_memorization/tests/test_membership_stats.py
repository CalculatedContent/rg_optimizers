"""Tests for the post-hoc membership-likelihood statistics."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import membership_stats as m


def write_audit(path):
    path.mkdir()
    (path / 'complete.json').write_text(json.dumps({'status':'complete','new_training_updates':0}))
    (path / 'protocol.json').write_text(json.dumps({'source_run':'fake','audit_seed':1,'background_examples':4}))
    base = [5.0, 5.1, 4.9, 5.2]
    rows = []
    for i, x in enumerate(base):
        rows.append({'category':'background','variant':'early_not_yet_seen','eid':f's{i}','nll':x})
    for i, x in enumerate(base):
        rows.append({'category':'background','variant':'fresh','eid':f'f{i}','nll':x})
    (path / 'teacher_scores_00000000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    rows = []
    for i, x in enumerate([4.0, 4.1, 3.9, 4.2]):
        rows.append({'category':'background','variant':'seen_early','eid':f's{i}','nll':x})
    for i, x in enumerate([4.9, 5.0, 4.8, 5.1]):
        rows.append({'category':'background','variant':'fresh','eid':f'f{i}','nll':x})
    (path / 'teacher_scores_00001000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_auc_and_holm():
    assert m.auc([1,1,0,0], [4,3,2,1]) == 1.0
    assert m.auc([1,0], [1,1]) == 0.5
    assert np.allclose(m.holm_adjust([.01,.04,.03]), [.03,.06,.06])


def test_end_to_end_detects_baseline_corrected_signal(tmp_path):
    audit = tmp_path / 'audit'
    write_audit(audit)
    out = m.analyze(audit, bootstrap=200, permutations=500, seed=7, output=tmp_path/'out')
    table = pd.read_csv(out / 'membership_stats.csv')
    row = table.iloc[0]
    assert row.baseline_corrected_membership_effect == pytest.approx(.9)
    assert row.auc_baseline_corrected == 1.0
    assert row.p_perm_effect_one_sided < .05
    assert row.p_perm_auc_corrected_one_sided < .05
    text = (out / 'membership_report.md').read_text()
    assert 'single-model' in text
    assert 'Independent training seeds' in text


def test_changed_fixed_cohort_is_rejected(tmp_path):
    audit = tmp_path / 'audit'
    write_audit(audit)
    path = audit / 'teacher_scores_00001000.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]['eid'] = 'different'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='cohort identity'):
        m.analyze(audit, bootstrap=100, permutations=100, seed=1, output=tmp_path/'out')


def test_incomplete_audit_rejected(tmp_path):
    audit = tmp_path / 'audit'
    write_audit(audit)
    (audit / 'complete.json').write_text(json.dumps({'status':'failed','new_training_updates':0}))
    with pytest.raises(ValueError, match='completed read-only'):
        m.analyze(audit, bootstrap=100, permutations=100, seed=1, output=tmp_path/'out')
