"""Read-only checkpoint audit tests. No trained-model performance is claimed."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import audit_checkpoints as a
parent = a.import_path('_test_audit_parent', a.REPO / a.RUN_PATH)
models = a.import_path('_test_audit_models', a.REPO / a.MODEL_PATH)


def cfg():
    value = json.loads((HERE / 'configs/suite.json').read_text())
    value['stages']['full'].update(steps=2, canaries_per_dose=2, doses=[0,1,2], probe_limit=2)
    return value


def test_pack_matches_original_and_short_extraction_does_not_move_target():
    r = parent.Record('x', 'dose_1', tuple(range(16,80)), tuple(range(80,112)))
    for hint in [0,8,16]:
        q = replace(r, prompt=r.prompt+r.target[:hint], target=r.target[hint:])
        x, y = parent.pack([q], 256)
        our_x, offset = a.pack_probe(a.make_probe(q, 'p'),256)
        assert tuple(x[0]) == our_x
        assert tuple(y[0,offset:]) == q.target
        assert offset == 256 - len(q.target)
    short = replace(r, prompt=r.prompt[-8:])
    _, offset = a.pack_probe(a.make_probe(short,'p'),256)
    assert offset == 224


def test_final_probes_preserve_controls_and_update_rule_targets():
    s = parent.Study(cfg(), 'verbatim', 'full', 1337,32)
    r = parent.Record('fresh_0','fresh',tuple(range(16,80)),a.background_target(tuple(range(16,80)),32))
    rows = a.final_probes(s.audit,[r],7)
    by = {(p['category'],p['variant'],p['record'].eid):p for p in rows}
    c = s.audit[0]
    assert by[('canary','wrong_prefix',c.eid)]['record'].prompt != c.prompt
    assert by[('canary','wrong_prefix',c.eid)]['record'].target == c.target
    hinted = by[('canary','true_prefix_hint_16',c.eid)]['record']
    assert hinted.prompt[-16:] == c.target[:16] and hinted.target == c.target[16:]
    perturb = by[('background','source_rotated_relabelled',r.eid)]['record']
    assert perturb.target == a.background_target(perturb.prompt,32)
    assert perturb.target != r.target
    distract = by[('background','irrelevant_half_shuffled',r.eid)]['record']
    assert distract.target == r.target == a.background_target(distract.prompt,32)


def test_real_gpt_scores_match_original_and_never_change_weights():
    torch.set_num_threads(1)
    model = models.GPT(models.GPTConfig(vocab_size=272,n_embd=8))
    model.requires_grad_(False)
    record = parent.Record('x','g',tuple(range(16,80)),tuple(range(80,112)))
    before = parent.state_digest(model)
    expected = parent.evaluate(model,[record],1,'cpu')['g']['examples'][0]
    row = a.score(model,[a.make_probe(record,'true')],1,'cpu',generate=True)[0]
    assert row['nll'] == pytest.approx(expected['nll'],abs=1e-6)
    assert row['teacher_forced_accuracy'] == expected['teacher_forced_accuracy']
    assert row['exact_match'] == expected['exact_match']
    assert row['token_accuracy'] == expected['continuation_token_accuracy']
    assert parent.state_digest(model) == before
    assert all(p.grad is None for p in model.parameters())
    hinted = replace(record,prompt=record.prompt+record.target[:16],target=record.target[16:])
    scored = a.score(model,[a.make_probe(hinted,'hint')],1,'cpu',generate=True)[0]
    assert scored['scored_tokens'] == 16 and scored['em_first_32'] is None


def test_nonfinite_logits_fail_not_zero():
    model = models.GPT(models.GPTConfig(vocab_size=272,n_embd=8))
    with torch.no_grad():
        model.lm_head.weight.fill_(float('nan'))
    r = parent.Record('x','g',(16,17),(18,19))
    with pytest.raises(FloatingPointError):
        a.score(model,[a.make_probe(r,'p')],1,'cpu',generate=False)


def test_no_generation_keeps_extraction_missing():
    model = models.GPT(models.GPTConfig(vocab_size=272,n_embd=8))
    r = parent.Record('x','g',(16,17),(18,19))
    out = a.aggregate(a.score(model,[a.make_probe(r,'p')],1,'cpu',generate=False))[0]
    assert out['exact_match'] is None
    assert out['nll'] is not None


def bundle(path):
    path.mkdir()
    c=cfg(); s=parent.Study(c,'verbatim','full',1337,32)
    model=models.GPT(models.GPTConfig(vocab_size=272,n_embd=8))
    m={'condition':'verbatim','profile':{'family':'adamw'},'stage':'full','seed':1337,
       'recipe':'repository','suite':c,'fingerprint':'fixture_untrained_evaluation_only',
       'runner_sha256':a.sha(a.REPO/a.RUN_PATH),'source_model':asdict(model.cfg),
       'input_tokens_per_step':8192,'data_sha256':s.identity(),
       'device':{'device':'cpu','torch':str(torch.__version__)}}
    a.write_json(path/'manifest.json',m)
    a.write_json(path/'complete.json',dict(status='complete',steps=2,fingerprint=m['fingerprint']))
    a.write_json(path/'probe_inventory.json',[asdict(r) for r in s.audit])
    a.write_json(path/'injection_schedule.json',{str(k):r.eid for k,r in s.injections.items()})
    counts=Counter(r.eid for step in range(2) for r in s.sample(step))
    for r in s.audit:counts.setdefault(r.eid,0)
    a.write_json(path/'exposures.json',dict(counts))
    rows=[]
    for step in [0,1,2]:
        result=parent.evaluate(model,s.probes(final=(step==2)),2,'cpu')
        rows.append({'step':step,'audit':result,'model_sha256':parent.state_digest(model)})
        torch.save({'step':step,'fingerprint':m['fingerprint'],'model':model.state_dict()},path/f'model_step_{step:08d}.pt')
    (path/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return m,s


def test_fresh_collision_checked_against_canaries_too(tmp_path):
    _,s=bundle(tmp_path/'run')
    canary=next(r for r in s.audit if r.group=='dose_1')
    visits=a.verify_stream(s,tmp_path/'run',[canary])
    assert len(visits[a.records_key(canary.prompt)])==1


def test_end_to_end_read_only_with_actual_untrained_gpt(tmp_path):
    torch.set_num_threads(1)
    path=tmp_path/'run';m,s=bundle(path)
    before={str(p):a.sha(p) for p in path.iterdir()}
    args=SimpleNamespace(run_dir=str(path),device='cpu',list=False,background_examples=2,
                         batch_size=2,audit_seed=20260915,output=str(tmp_path/'audit'))
    a.run_audit(args)
    assert before=={str(p):a.sha(p) for p in path.iterdir()}
    report=tmp_path/'audit'
    assert a.load_json(report/'complete.json')['new_training_updates']==0
    assert a.load_json(report/'protocol.json')['new_weightwatcher_calls']==0
    import pandas as pd
    scores=pd.read_csv(report/'checkpoint_scores.csv')
    assert set(scores.step)=={0,1,2}
    final=pd.read_csv(report/'final_probes.csv')
    assert set(final.category)=={'canary','background'}
    assert 'true_prefix_hint_16' in set(final.variant)
    assert (report/'background_gaps.csv').is_file()
    assert (report/'paired_prompt_effects.csv').is_file()
    assert 'not extraction' in (report/'report.md').read_text()
    with pytest.raises(FileExistsError):
        a.run_audit(args)


def test_saved_checkpoint_hash_mismatch_is_rejected(tmp_path):
    path=tmp_path/'run';m,s=bundle(path)
    rows=[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()]
    rows[0]['model_sha256']='bad'
    (path/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    args=SimpleNamespace(run_dir=str(path),device='cpu',list=False,background_examples=2,
                         batch_size=2,audit_seed=20260915,output=str(tmp_path/'audit'))
    with pytest.raises(ValueError,match='hash'):
        a.run_audit(args)


def test_list_does_not_create_outputs(tmp_path):
    path=tmp_path/'run';bundle(path)
    args=SimpleNamespace(run_dir=str(path),list=True)
    a.run_audit(args)
    assert not (path/'analysis').exists()


def test_incomplete_run_is_refused(tmp_path):
    path=tmp_path/'run';bundle(path)
    done=a.load_json(path/'complete.json');done['steps']=1
    a.write_json(path/'complete.json',done)
    with pytest.raises(ValueError,match='completion'):
        a.inventory(path)


def test_zero_dose_never_reconstructed_as_exposed(tmp_path):
    path=tmp_path/'run';_,s=bundle(path)
    _,rows,_=a.inventory(path)
    saved=a.saved_behavior(rows,parent,s)
    assert all(r['presentations']==0 for r in saved if r['group'].startswith('dose_0/'))
    assert max(r['presentations'] for r in saved)==2
