from __future__ import annotations
from collections import Counter
import copy
from dataclasses import asdict,replace
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
import pandas as pd
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import am_data as data
import am_metrics as metrics
import am_runtime as runtime
import am_spectral as spectral
import am_train as training
import am_report as report
import run_study as cli
CFG=json.loads((HERE/'protocol.json').read_text())
torch.set_num_threads(1)


def tiny_cfg():
    c=copy.deepcopy(CFG)
    c.update(steps=4,spectral_every=2,behavior_every=2,expensive_every=4,doses=[0,1],
             long_per_dose=1,short_per_dose=1,rule_audit_limit=2)
    c['model_overrides'].update(n_embd=32)
    c['exposure_alphabet']=[272,273]; c['exposure_length']=2
    return c


def test_source_hashes_match_exact_repository_files():
    cfg=runtime.baseline(CFG)
    assert cfg['model']['n_head']==1 and cfg['model']['n_embd']==128
    assert cfg['optimizer_profiles']['adamw']['learning_rate']==.0006


def test_exactly_ten_jobs_without_outcome_selection():
    jobs=cli.jobs(CFG)
    assert len(jobs)==len(set(jobs))==10
    assert Counter(a for a,s in jobs)=={'adamw':5,'muon_guarded':5}


def test_disjoint_rule_splits_and_canary_keys():
    d=data.Dataset(CFG,1337,32)
    train={r.id for r in d.train}
    val={r.id for r in d.audit if r.cohort=='validation_clean'}
    test={r.id for r in d.audit if r.cohort=='test_clean'}
    assert not train&val and not train&test and not val&test
    assert len(train|val|test)==31**2
    assert len({r.prefix for r in d.canaries})==len(d.canaries)==100
    assert not any(r.cohort=='test_clean' for r in d.probes())
    assert any(r.cohort=='test_clean' for r in d.probes(final=True))


def test_actual_presentations_and_withdrawal():
    c=tiny_cfg(); d=data.Dataset(c,1337,32)
    counts=Counter(r.id for i in range(d.steps) for r in d.batch(i))
    assert all(counts[r.id]==r.dose for r in d.canaries)
    assert not any(r.dose>=0 for i in range(d.withdrawal,d.steps) for r in d.batch(i))
    assert d.planned_counts(1)==Counter(r.id for r in d.batch(0) if r.dose>=0)


def test_pairing_and_fixed_labels_independent_of_audits():
    a=data.Dataset(CFG,1337,32); b=data.Dataset(CFG,1337,32)
    assert a.fingerprint==b.fingerprint
    expected=a.batch(3); a.probes(True); a.batch(5)
    assert expected==a.batch(3)==b.batch(3)
    c=data.Dataset(CFG,2027,32)
    assert a.train==c.train and a.canaries==c.canaries
    assert a.batch(3)!=c.batch(3)


def test_recall_and_exposure_ties():
    assert metrics.recall([1,9,3],[1,2,3])=={'exact_match':0.,'token_match':2/3,'longest_prefix':1}
    with pytest.raises(ValueError): metrics.recall([1],[1,2])
    assert metrics.rank_exposure([1,1,1,1],0)['exposure_lower']==0
    assert metrics.rank_exposure([1,2,3,4],0)['exposure_lower']==2
    with pytest.raises(ValueError): metrics.rank_exposure([1,np.nan],0)


def test_prefix_boundary_and_censoring():
    r=data.Record('a','long',tuple(range(64)),(8,9))
    probes=metrics.compression_probes([r],[8,16,32,64])
    assert all(len(p.prefix)+p.offset==64 for p in probes)
    assert all(p.target==r.target for p in probes)
    rows=[dict(id='a',dose=1,prefix_tokens=8,target_tokens=32,exact_match=0.)]
    assert metrics.compress_summary(rows)[0]['censored']
    assert metrics.compress_summary(rows)[0]['ratio'] is None


def test_parent_model_math_attention_and_suffix_alignment():
    source=runtime.baseline(CFG); GPT,GPTConfig,*_=runtime.imports()
    cfg=tiny_cfg(); model=runtime.make_model(source,cfg,4,'cpu').eval()
    native=GPT(model.cfg).eval(); native.load_state_dict(model.state_dict())
    ids=torch.tensor([[1,2,3,4]])
    assert torch.allclose(model(ids)[0],native(ids)[0],rtol=1e-4,atol=2e-6)
    record=data.Record('x','test',(1,2),(3,4))
    loss,_,_=metrics.suffix_losses(model,[record])
    expected=torch.nn.functional.cross_entropy(model(ids[:,:-1])[0][:,1:,:].reshape(-1,512),ids[:,2:].reshape(-1),reduction='none')
    assert torch.allclose(loss[0],expected)


def test_exact_exposure_prefix_tree_matches_brute_force():
    source=runtime.baseline(CFG); model=runtime.make_model(source,tiny_cfg(),4,'cpu').train()
    r=data.Record('x','short',(1,2),(17,16)); alphabet=[16,17]
    rank,scores=metrics.exposure(model,r,alphabet,2)
    candidates=[(a,b) for a in alphabet for b in alphabet]
    model.eval()
    with torch.no_grad():
        reference=metrics.suffix_losses(model,[replace(r,target=c) for c in candidates])[0].sum(1).numpy()
    assert np.allclose(scores,reference,rtol=1e-6)
    assert rank==dict(metrics.rank_exposure(scores,2),target_index=2)


def test_all_eight_matrices_and_tied_head_once():
    model=runtime.make_model(runtime.baseline(CFG),CFG,1337,'cpu')
    names=[n for n,p in runtime.matrices(model)]
    assert len(names)==8 and 'token_embedding.weight' in names and 'position_embedding.weight' in names
    assert 'lm_head.weight' not in names


@pytest.mark.parametrize('arm',['adamw','muon'])
def test_actual_nanogpt_optimizer_forward_backward_update(arm):
    source=runtime.baseline(CFG); model=runtime.make_model(source,CFG,1337,'cpu')
    *_,make,set_lrs,zero_grad,step=runtime.imports()
    hs=make(model,source['optimizer_profiles'][arm]); training.split_groups(model,hs)
    before=runtime.model_hash(model)
    r=data.Record('a','short',(1,18,2,17,3),(18,))
    loss=metrics.suffix_losses(model,[r])[0].mean(); loss.backward()
    norm,fallback=runtime.safe_clip(model,1.); step(hs)
    assert math.isfinite(norm) and not fallback
    assert before!=runtime.model_hash(model)


def test_safe_gradient_clipping_preserves_bad_entries_and_fixes_norm_overflow():
    m=torch.nn.Linear(2,2,bias=False)
    m.weight.grad=torch.full_like(m.weight,1e30)
    norm,fallback=runtime.safe_clip(m,1.)
    assert fallback and norm==pytest.approx(2e30,rel=1e-6)
    assert float(m.weight.grad.norm())==pytest.approx(1.)
    m.weight.grad.fill_(float('nan'))
    with pytest.raises(FloatingPointError): runtime.safe_clip(m,1.)
    assert torch.isnan(m.weight.grad).all()


def test_ww_contract_all_matrices_clipped_raw_and_rng_preserved(monkeypatch):
    model=runtime.make_model(runtime.baseline(CFG),tiny_cfg(),1337,'cpu')
    before=runtime.model_hash(model); rng=torch.get_rng_state().clone(); npstate=np.random.get_state()
    fake=ModuleType('weightwatcher'); calls=[]
    class WW:
        def __init__(self,model): self.model=model
        def analyze(self,**kwargs):
            calls.append(kwargs); torch.rand(3); np.random.rand(3)
            return pd.DataFrame([dict(longname=k,alpha=2.5,raw_alpha=2.7,D=.1,num_pl_spikes=30)
                                 for k in self.model])
    fake.WeightWatcher=WW; monkeypatch.setitem(sys.modules,'weightwatcher',fake)
    monkeypatch.setattr(spectral.importlib.metadata,'version',lambda n:'0.7.7')
    rows=spectral.measure(model,CFG['weightwatcher'],1337,0)
    assert len(rows)==8 and all(r['alpha_clip_xmax']==2.5 and r['alpha_raw']==2.7 for r in rows)
    assert calls[0]['fix_fingers']=='clip_xmax' and calls[0]['randomize'] is True
    assert runtime.model_hash(model)==before and torch.equal(rng,torch.get_rng_state())
    assert np.array_equal(npstate[1],np.random.get_state()[1])
    assert not spectral.offenders(rows,2.05)
    rows[0]['alpha_raw']=float('nan'); assert len(spectral.offenders(rows,2.05))==1


def fake_environment(device): return {'device':device,'test_fixture':True}


def fake_measure(model,cfg,seed,step):
    return [dict(matrix=n,step=step,alpha_raw=3.,alpha_clip_xmax=3.,D=.1,fit_supported=True,
                 model_sha256=runtime.model_hash(model)) for n,p in runtime.matrices(model)]


def test_end_to_end_real_cpu_models_and_rollback(monkeypatch,tmp_path):
    monkeypatch.setattr(training,'environment',fake_environment)
    monkeypatch.setattr(training,'measure',fake_measure)
    c=tiny_cfg(); a=training.train(c,tmp_path,'adamw',1337,'cpu')
    assert a['state']=='complete'
    calls=Counter()
    def reject_once(model,cfg,seed,step):
        rows=fake_measure(model,cfg,seed,step); calls[step]+=1
        if step==2 and calls[step]==1: rows[0]['alpha_raw']=1.9
        return rows
    monkeypatch.setattr(training,'measure',reject_once)
    b=training.train(c,tmp_path,'muon_guarded',1337,'cpu')
    assert b['state']=='complete' and b['attempted_updates']==6
    run=tmp_path/'muon_guarded'/'seed_1337'
    saved=torch.load(run/'checkpoint_latest.pt',weights_only=True)
    d=data.Dataset(c,1337,32)
    assert all(saved['counts'].get(r.id,0)==r.dose for r in d.canaries)
    assert len(list((run/'rejected_spectra').glob('*.csv')))==1
    assert json.loads((run/'behavior'/'step_00000004.json').read_text())['exposure']
    dest=report.generate(tmp_path,make_plots=False)
    assert (dest/'tables'/'paired_differences.csv').exists()


def test_initial_guard_failure_not_hidden_or_resampled(monkeypatch,tmp_path):
    monkeypatch.setattr(training,'environment',fake_environment)
    def bad(*args):
        rows=fake_measure(*args); rows[0]['alpha_raw']=1.9; return rows
    monkeypatch.setattr(training,'measure',bad)
    result=training.train(tiny_cfg(),tmp_path,'muon_guarded',1337,'cpu')
    assert result['state']=='failed'
    assert not (tmp_path/'muon_guarded'/'seed_1337'/'complete.json').exists()


def test_interrupted_resume_matches_uninterrupted_cpu(monkeypatch,tmp_path):
    monkeypatch.setattr(training,'environment',fake_environment)
    monkeypatch.setattr(training,'measure',fake_measure)
    c=tiny_cfg(); full=tmp_path/'full'; interrupted=tmp_path/'interrupted'
    training.train(c,full,'adamw',1337,'cpu')
    def stop(model,cfg,seed,step):
        if step==4: raise KeyboardInterrupt()
        return fake_measure(model,cfg,seed,step)
    monkeypatch.setattr(training,'measure',stop)
    with pytest.raises(KeyboardInterrupt): training.train(c,interrupted,'adamw',1337,'cpu')
    monkeypatch.setattr(training,'measure',fake_measure)
    result=training.train(c,interrupted,'adamw',1337,'cpu',resume=True)
    assert result['state']=='complete'
    a=torch.load(full/'adamw'/'seed_1337'/'checkpoint_latest.pt',weights_only=True)
    b=torch.load(interrupted/'adamw'/'seed_1337'/'checkpoint_latest.pt',weights_only=True)
    assert a['model_sha256']==b['model_sha256'] and a['counts']==b['counts']


def test_uncertainty_is_across_seeds_not_probes():
    assert report.interval([3])['ci95'] is None
    rows=pd.DataFrame([{'arm':'a','seed':s,'x':float(s)} for s in range(5) for _ in range(20)])
    out=report.summarize(rows,['arm'],['x'])
    assert out.n_seeds.iloc[0]==5 and out['mean'].iloc[0]==2.
    assert report.auc([2,2],[1,1])==1.
    assert report.auc([1,1],[1,1])==.5


def test_plot_output_is_generated_from_explicit_fixture(tmp_path):
    rows=pd.DataFrame([{'arm':a,'step':t,'cohort':'long','metric':'exact_match','dose':64,
                       'mean':v,'ci95':.1,'n_seeds':5} for a,v in [('adamw',.5),('muon_guarded',.3)] for t in [0,100]])
    report.plots(tmp_path,{'behavior_summary':rows})
    assert list((tmp_path/'figures').glob('*.png')) and list((tmp_path/'figures').glob('*.svg'))
