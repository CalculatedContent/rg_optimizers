"""Incremental spectral monitoring, raw alpha semantics and held-out metrics."""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tpu_port import tracking
from tpu_port.ops import language_loss
from tpu_port.gpt import GPT


def test_active_projection_mapping_matches_forward_weights():
    qk=torch.arange(48*256).reshape(48,256,1).expand(-1,-1,768)
    vo=torch.arange(16*768).reshape(16,768,1).expand(-1,-1,768)
    mlp=torch.empty(12,2,2816,768,device='meta')
    got=tracking.projection_matrices({'qk_bank':qk,'vo_bank':vo,'mlp_bank':mlp})
    assert len(got)==50
    assert not any(name.startswith('L07') for name in got)
    assert got['L08_PARALLEL_W_MLP_OUT'].shape==(768,2816)
    model=SimpleNamespace(num_heads=6,head_dim=128,qk_bank=qk,vo_bank=vo,_num_qk_groups=42,
                          attn_v_dim=lambda layer:128 if layer not in (1,8) else 64)
    for layer,(qk_w,v_w,o_w) in GPT._attn_weights(model).items():
        q,k=qk_w.chunk(2)
        for role,expected in zip(('Q','K','V','O'),(q,k,v_w,o_w)):
            torch.testing.assert_close(got[f'L{layer:02d}_W_{role}'],expected,atol=0,rtol=0)


def test_raw_alpha_never_substitutes_clipped_alpha():
    import pandas as pd
    rows=tracking.normalize_rows(pd.DataFrame([
        {'longname':'L00_W_Q','status':'success','alpha':1.8,'raw_alpha':float('nan')},
        {'longname':'L00_W_K','status':'success','alpha':1.9,'raw_alpha':2.7}]),
        ['L00_W_Q','L00_W_K','L00_W_V'],{'step':100})
    assert rows[0]['alpha_raw'] is None and rows[0]['alpha_clip_xmax']==1.8
    assert rows[1]['alpha_raw']==2.7
    assert rows[2]['status']=='not_returned'
    result=tracking.summary(rows,{'step':100})
    assert result['alpha_raw_valid_count']==1 and result['alpha_raw_mean']==2.7


def test_evaluation_loss_accuracy_matches_direct_masked_cross_entropy():
    torch.manual_seed(41)
    x=torch.randn(5,4);w=torch.randn(4,8)
    inputs=torch.tensor([0,1,2,3,4]);targets=torch.tensor([1,2,3,4,5])
    mask=torch.zeros(8,1,dtype=torch.uint8);mask[0,0]=128;mask[2,0]=1
    model=SimpleNamespace(training=False,lm_head=SimpleNamespace(weight=w),prefix_table=torch.arange(8),canon_mask=mask)
    got=language_loss(model,x,inputs,targets,torch.tensor([1.]),torch.tensor([0.]),None,slab=2)
    logits=23*torch.sigmoid((x@w+5)/7.5)
    logits[0,7]=-60;logits[2,0]=-60
    expected=torch.nn.functional.cross_entropy(logits,targets,reduction='none')
    torch.testing.assert_close(got,expected)
    assert model.last_eval_correct.item()==int((logits.argmax(-1)==targets).sum())


def test_real_weightwatcher_worker_drains_and_pairs_snapshot(tmp_path):
    pytest.importorskip('weightwatcher')
    folder=tmp_path/'tracking/snapshots';folder.mkdir(parents=True)
    gen=torch.Generator().manual_seed(42)
    path=folder/'0000100.pt'
    payload={'step':100,'tokens_seen':13107200,'run_id':'test',
        'validation':{'evaluation_tokens':131072,'full_benchmark_evaluation':False,
                      'val_nll':3.4,'val_perplexity':29.9641,'val_token_error':.6,
                      'val_accuracy':.4,'val_error_count':78643,'weight_state':'raw_training'},
        'matrices':{'L00_W_Q':torch.randn(64,64,generator=gen),'L00_W_O':torch.zeros(64,64)}}
    torch.save(payload,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'tracking/TRAINING_DONE').touch()
    assert tracking.watch(tmp_path,time.time()+60)==0
    result=json.loads((tmp_path/'tracking/measurements/0000100.json').read_text())
    assert result['summary']['snapshot_sha256']==digest
    assert result['summary']['val_token_error']==.6
    assert result['summary']['full_benchmark_evaluation'] is False
    assert len(result['layers'])==2 and result['layers'][1]['alpha_raw'] is None
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert (tmp_path/'tracking/layers.csv').exists()
    state=json.loads((tmp_path/'TRACKING_STATUS.json').read_text())
    assert state['status']=='complete' and state['completed']==1 and state['pending']==0


def test_validation_reduces_loss_counts_and_restores_training(monkeypatch,tmp_path):
    pytest.importorskip('torch_xla')
    import torch_xla.core.xla_model as xm
    from tpu_port import runtime
    class Model(torch.nn.Module):
        def forward(self,*args):
            assert not self.training
            self.last_eval_correct=torch.tensor(3)
            return torch.tensor([3.,3.5,3.,3.5])
    def batches(*args):
        yield object();yield object()
    monkeypatch.setattr(xm,'mark_step',lambda **kw:None)
    monkeypatch.setattr(runtime,'distributed_data_generator',batches)
    monkeypatch.setattr(runtime,'device_batch',lambda *args:(None,None,None,()))
    model=Model()
    result=runtime.evaluate(model,SimpleNamespace(world=1),tmp_path,'cpu',
        SimpleNamespace(ws_short=128,ws_long=384),step=100,tokens_seen=13107200,
        batch_tokens=4,batches=2,weight_state='raw_training')
    assert model.training
    assert result['val_nll']==3.25
    assert result['val_accuracy']==.75 and result['val_error_count']==2
    assert result['val_token_error']==.25
    assert result['target_reached'] is False and result['full_benchmark_evaluation'] is False
