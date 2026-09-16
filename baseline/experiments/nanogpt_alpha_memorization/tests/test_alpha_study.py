from __future__ import annotations
from collections import Counter
import copy,json,math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import am_data as data
import am_metrics as metrics
import am_runtime as runtime
import am_train as training
import am_report_fast as report
import run_study as cli
CFG=json.loads((HERE/'protocol.json').read_text());torch.set_num_threads(1)

def tiny_cfg():
    c=copy.deepcopy(CFG);c.update(steps=4,behavior_every=2,expensive_every=4,checkpoint_every=2,doses=[0,1],long_per_dose=1,short_per_dose=1,rule_audit_limit=2)
    c['model_overrides'].update(n_embd=32);c['exposure_alphabet']=[272,273];c['exposure_length']=2
    return c

def test_protocol_is_exactly_five_adamw_five_muon_and_no_ww():
    jobs=cli.jobs(CFG)
    assert len(jobs)==len(set(jobs))==10
    assert Counter(a for a,s in jobs)=={'adamw':5,'muon':5}
    assert CFG['online_weightwatcher'] is False and 'guard' not in CFG

def test_source_optimizer_profiles_are_unchanged():
    cfg=runtime.baseline(CFG)
    assert cfg['model']['n_head']==1 and cfg['model']['n_embd']==128
    assert cfg['optimizer_profiles']['adamw']['learning_rate']==.0006
    assert cfg['optimizer_profiles']['muon']['matrix_learning_rate']==.02
    assert cfg['optimizer_profiles']['muon']['aux_learning_rate']==.0003

def test_disjoint_rule_splits_and_canary_keys():
    d=data.Dataset(CFG,1337,32);train={r.id for r in d.train};val={r.id for r in d.audit if r.cohort=='validation_clean'};test={r.id for r in d.audit if r.cohort=='test_clean'}
    assert not train&val and not train&test and not val&test and len(train|val|test)==31**2
    assert len({r.prefix for r in d.canaries})==len(d.canaries)==100

def test_actual_presentations_and_withdrawal():
    c=tiny_cfg();d=data.Dataset(c,1337,32);counts=Counter(r.id for i in range(d.steps) for r in d.batch(i))
    assert all(counts[r.id]==r.dose for r in d.canaries)
    assert not any(r.dose>=0 for i in range(d.withdrawal,d.steps) for r in d.batch(i))

def test_pairing_is_fixed_across_optimizer_arms():
    a=data.Dataset(CFG,1337,32);b=data.Dataset(CFG,1337,32);c=data.Dataset(CFG,2027,32)
    assert a.fingerprint==b.fingerprint and a.train==c.train and a.canaries==c.canaries
    assert a.batch(3)!=c.batch(3)

def test_recall_exposure_and_compression_metrics():
    assert metrics.recall([1,9,3],[1,2,3])=={'exact_match':0.,'token_match':2/3,'longest_prefix':1}
    assert metrics.rank_exposure([1,2,3,4],0)['exposure_lower']==2
    r=data.Record('a','long',tuple(range(64)),(8,9));probes=metrics.compression_probes([r],[8,16,32,64])
    assert all(len(p.prefix)+p.offset==64 and p.target==r.target for p in probes)

def test_parent_model_math_attention_and_suffix_alignment():
    source=runtime.baseline(CFG);GPT,GPTConfig,*_=runtime.imports();cfg=tiny_cfg();model=runtime.make_model(source,cfg,4,'cpu').eval()
    native=GPT(model.cfg).eval();native.load_state_dict(model.state_dict());ids=torch.tensor([[1,2,3,4]])
    assert torch.allclose(model(ids)[0],native(ids)[0],rtol=1e-4,atol=2e-6)

def test_all_eight_matrices_and_tied_head_once_for_later_posthoc_ww():
    model=runtime.make_model(runtime.baseline(CFG),CFG,1337,'cpu');names=[n for n,p in runtime.matrices(model)]
    assert len(names)==8 and 'token_embedding.weight' in names and 'position_embedding.weight' in names and 'lm_head.weight' not in names

@pytest.mark.parametrize('arm',['adamw','muon'])
def test_actual_nanogpt_optimizer_forward_backward_update(arm):
    source=runtime.baseline(CFG);model=runtime.make_model(source,CFG,1337,'cpu');*_,make,set_lrs,zero_grad,step=runtime.imports();hs=make(model,source['optimizer_profiles'][arm]);before=runtime.model_hash(model)
    loss=metrics.suffix_losses(model,[data.Record('a','short',(1,18,2,17,3),(18,))])[0].mean();loss.backward();norm,fallback=runtime.safe_clip(model,1.);step(hs)
    assert math.isfinite(norm) and not fallback and before!=runtime.model_hash(model)

def fake_environment(device):return {'device':device,'test_fixture':True}

@pytest.mark.parametrize('arm',['adamw','muon'])
def test_end_to_end_real_cpu_training_without_spectral_calls(monkeypatch,tmp_path,arm):
    monkeypatch.setattr(training,'environment',fake_environment);c=tiny_cfg();result=training.train(c,tmp_path,arm,1337,'cpu')
    assert result['state']=='complete' and result['online_weightwatcher'] is False
    run=tmp_path/arm/'seed_1337';assert not (run/'spectral').exists() and not (run/'control_events.jsonl').exists()
    assert (run/'model_00000000.pt').exists() and (run/'model_00000004.pt').exists()
    saved=torch.load(run/'checkpoint_latest.pt',weights_only=True);d=data.Dataset(c,1337,32)
    assert all(saved['counts'].get(r.id,0)==r.dose for r in d.canaries)
    assert json.loads((run/'behavior'/'step_00000004.json').read_text())['exposure']

def test_pair_report_uses_muon_minus_adamw(monkeypatch,tmp_path):
    monkeypatch.setattr(training,'environment',fake_environment);c=tiny_cfg();training.train(c,tmp_path,'adamw',1337,'cpu');training.train(c,tmp_path,'muon',1337,'cpu')
    dest=report.generate(tmp_path,make_plots=False);paired=pd.read_csv(dest/'tables'/'paired_differences.csv')
    assert len(paired)>0 and set(pd.read_csv(dest/'tables'/'runs.csv').arm)=={'adamw','muon'}

def test_interrupted_resume_matches_uninterrupted_cpu(monkeypatch,tmp_path):
    monkeypatch.setattr(training,'environment',fake_environment);c=tiny_cfg();full=tmp_path/'full';interrupted=tmp_path/'interrupted';training.train(c,full,'adamw',1337,'cpu')
    original=training.audit
    def stop(model,d,cfg,source,run_dir,step,counts):
        if step==2:raise KeyboardInterrupt()
        return original(model,d,cfg,source,run_dir,step,counts)
    monkeypatch.setattr(training,'audit',stop)
    with pytest.raises(KeyboardInterrupt):training.train(c,interrupted,'adamw',1337,'cpu')
    monkeypatch.setattr(training,'audit',original);result=training.train(c,interrupted,'adamw',1337,'cpu',resume=True);assert result['state']=='complete'
    a=torch.load(full/'adamw'/'seed_1337'/'checkpoint_latest.pt',weights_only=True);b=torch.load(interrupted/'adamw'/'seed_1337'/'checkpoint_latest.pt',weights_only=True)
    assert a['model_sha256']==b['model_sha256'] and a['counts']==b['counts']

def test_uncertainty_is_across_seeds_not_probes():
    assert report.interval([3])['ci95'] is None
    rows=pd.DataFrame([{'arm':'a','seed':s,'x':float(s)} for s in range(5) for _ in range(20)]);out=report.summarize(rows,['arm'],['x'])
    assert out.n_seeds.iloc[0]==5 and out['mean'].iloc[0]==2.
