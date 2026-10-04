import copy
import json
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from rg_nanogpt_one_head.model import GPT, GPTConfig, transformer_matrix_items
from rg_nanogpt_one_head.data import write_token_splits
from rg_nanogpt_one_head import gpt2_experiment as g
BASE=Path(__file__).resolve().parents[1]

def config(name='cpu_smoke'):
    return yaml.safe_load((BASE/'configs'/f'gpt2_small_{name}.yaml').read_text())

def data(tmp,c):
    class Encoder:
        n_vocab=64; eot_token=63
        def encode_ordinary(self,text): return list(range(16))
    root=tmp/'data'
    write_token_splits(['document']*100,Encoder(),root,train_tokens=512,val_tokens=128,test_tokens=128,
       dataset_metadata={'dataset_name':c['dataset']['name'],'dataset_config':c['dataset']['config'],
        'dataset_split':'train','dataset_revision':c['dataset']['revision'],'tokenizer':'gpt2'})
    return root

def test_full_architecture():
    c=config('fineweb_adamw_baseline')
    with torch.device('meta'): model=GPT(GPTConfig(**c['model']))
    a=g.architecture(model)
    assert (a['layers'],a['heads'],a['d_model'],a['context'],a['logical_matrices'])==(12,12,768,1024,72)
    assert a['parameters']==124439808 and a['tied']
    items=transformer_matrix_items(model)
    assert len({id(w) for _,_,_,w in items})==72
    assert all(tuple(w.shape)==(768,768) for _,kind,_,w in items if kind in ('W_Q','W_K','W_V','W_O'))

@pytest.mark.parametrize('family',['adamw','muon_clip'])
def test_exact_resume_and_append(tmp_path,family):
    torch.set_num_threads(1)
    c=config(); c['optimizer']=config('fineweb_'+('adamw' if family=='adamw' else 'muonclip')+'_baseline')['optimizer']
    d=data(tmp_path,c)
    g.train(c,d,tmp_path/'full')
    g.train(c,d,tmp_path/'split',stop_after=2)
    before=(tmp_path/'split/metrics/000000002.json').read_bytes()
    g.train(c,d,tmp_path/'split',resume=True)
    def saved(root):
        return torch.load(root/'checkpoints/step_000000004.pt',weights_only=False)
    a,b=saved(tmp_path/'full'),saved(tmp_path/'split')
    for name in a['model']: torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
    assert torch.equal(a['data_rng'],b['data_rng']) and a['tokens_seen']==b['tokens_seen']==64
    from rg_nanogpt_one_head.checkpoints import optimizer_state_sha256
    assert optimizer_state_sha256(a['optimizers'])==optimizer_state_sha256(b['optimizers'])
    assert (tmp_path/'split/metrics/000000002.json').read_bytes()==before
    assert len(list((tmp_path/'split/metrics').glob('*.json')))==3
    with pytest.raises(FileExistsError): g.append_record(tmp_path/'split/metrics/000000002.json',{})
    row=json.loads((tmp_path/'split/metrics/000000004.json').read_text())
    assert row['test_accuracy']+row['test_token_error']==1
    assert row['test_perplexity']==pytest.approx(np.exp(row['test_nll']))
    assert row['train_nll']<json.loads((tmp_path/'split/metrics/000000000.json').read_text())['train_nll']
    assert len(list((tmp_path/'split/checkpoints').glob('step_*.pt')))<=3

def test_schedules_and_batches():
    assert g.due(20,{'logarithmic':True}) and not g.due(21,{'logarithmic':True})
    assert g.due(7,{'steps':[7]}) and g.due(100,{'interval':100})
    x,y=g.batch(np.arange(100),torch.Generator().manual_seed(2),2,8,torch.device('cpu'))
    assert torch.equal(x+1,y)

def test_raw_failure_never_falls_back(tmp_path,monkeypatch):
    c=config(); model=GPT(GPTConfig(**c['model']))
    calls=[]
    class WW:
        def __init__(self,model): self.model=model
        def analyze(self,**kw):
            calls.append(kw)
            return pd.DataFrame([{'name':m['matrix_name'],'alpha':2.5,'raw_alpha':float('nan'), 'rand_distance':.2} for m in self.model.matrix_metadata])
    monkeypatch.setitem(sys.modules,'weightwatcher',types.SimpleNamespace(WeightWatcher=WW))
    before=torch.get_rng_state().clone()
    result=g.measure_ww(model,c,{'step':1},{'test_token_error':.7})
    assert len(result['records'])==12 and calls[0]['randomize'] is True
    assert all(np.isnan(r['alpha_raw']) and r['raw_fit_status']=='failed' and r['alpha_clip_xmax']==2.5 for r in result['records'])
    assert torch.equal(before,torch.get_rng_state())

def test_auxiliary_lr():
    c=config(); c['optimizer']=config('fineweb_muonclip_baseline')['optimizer']
    handles=g.make_handles(GPT(GPTConfig(**c['model'])),c)
    from rg_nanogpt_one_head.muonclip import MuonClip
    assert isinstance(handles[0].optimizer,MuonClip)
    assert handles[0].peak_lr==.02 and handles[1].peak_lr==.0006


def test_interrupted_record_transaction_recovers(tmp_path):
    torch.set_num_threads(1)
    c=config(); d=data(tmp_path,c); out=tmp_path/'transaction'
    g.train(c,d,out,stop_after=2)
    missing=out/'metrics/000000002.json'; expected=missing.read_bytes(); missing.unlink()
    g.train(c,d,out,resume=True,stop_after=2)
    assert missing.read_bytes()==expected
    assert len(list((out/'metrics').glob('*.json')))==2


def test_invalid_gradient_diagnostic_precedes_update(tmp_path):
    model=GPT(GPTConfig(**config()['model']))
    for p in model.parameters(): p.grad=torch.ones_like(p)
    name,p=next(iter(model.named_parameters())); p.grad.view(-1)[0]=float('nan')
    before=p.detach().clone()
    with pytest.raises(RuntimeError,match='BEFORE update 1'):
        g.require_finite_update(model,torch.tensor(float('nan')),[torch.tensor(11.)],tmp_path,1)
    report=json.loads((tmp_path/'nonfinite_diagnostics.json').read_text())
    assert [r['parameter'] for r in report['gradients'] if r['nonfinite_elements']]==[name]
    assert torch.equal(before,p.detach())
