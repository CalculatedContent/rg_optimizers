"""Math and CPU transport tests. No CPU result is labelled TPU qualification."""
import importlib.util
import json
from pathlib import Path
import sys
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from tpu_port.ops import window_attention, qk_norm_rope_forward, language_loss
from tpu_port.host_table import HostTable
from tpu_port.ngram_math import adam_beta2_and_wd_mul
from tpu_port.runtime import schedule, PREFLIGHT_STEPS
from tpu_port.optimizer import cascade
from tpu_port.gpt import GPT
from tpu_port.sampled_softmax import candidate_count_at


def test_schedule_and_sampled_variants():
    s=schedule()
    assert s.boundaries==[(0,320),(320,681),(681,1107),(1107,1174),(1174,1194)]
    assert sum((b-a)*stage.batch_size for (a,b),stage in zip(s.boundaries,s.stages))==328663040
    assert [candidate_count_at(s,i) for i in [0,680,681,964,965,1106,1107,1194]]==[10240,10240,14336,14336,24576,24576,0,0]
    assert len(PREFLIGHT_STEPS)==30


def test_bounded_attention_forward_and_backward_matches_dense_mask():
    torch.manual_seed(12)
    q,k,v=[torch.randn(17,3,d,requires_grad=True) for d in [8,8,12]]
    seq=torch.tensor([0,7,17,17])
    got=window_attention(q,k,v,seq,5,.13,query_rows=4)
    pos=torch.arange(17); docs=(pos[:,None]>=seq).sum(-1)
    mask=(pos[None]<=pos[:,None])&(pos[None]>=pos[:,None]-5)&(docs[None]==docs[:,None])
    scores=torch.einsum('thd,shd->hts',q,k)*.13
    expected=torch.einsum('hts,shd->thd',scores.masked_fill(~mask,float('-inf')).softmax(-1),v)
    torch.testing.assert_close(got,expected,atol=1e-6,rtol=1e-5)
    got.square().sum().backward(retain_graph=True)
    grads=[t.grad.clone() for t in (q,k,v)]
    for t in (q,k,v): t.grad=None
    expected.square().sum().backward()
    expected_grads=[t.grad for t in (q,k,v)]
    for a,b in zip(grads,expected_grads): torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('paired,key_offset',[(False,False),(False,True),(True,False)])
def test_rope_matches_explicit_reference_indexing(paired,key_offset):
    torch.manual_seed(31); n,h,d=5,6,128
    qk=torch.randn(n,2*h,d,dtype=torch.bfloat16)
    f1=torch.randn(n,d*(2 if paired else 1),dtype=torch.bfloat16)
    f2=torch.randn_like(f1)
    q,k=qk_norm_rope_forward(qk,f1,f2,h,64,paired,key_offset)
    ref=torch.empty_like(qk)
    for t in range(n):
        for head in range(2*h):
            x=qk[t,head].float(); norm=x*torch.rsqrt(x.square().mean()+torch.finfo(torch.float32).eps)
            lane=(head%h)%2 if paired else 0
            ff1=f1[t,lane*d:(lane+1)*d].float(); ff2=f2[t,lane*d:(lane+1)*d].float()
            y=ff1*norm+ff2*norm.reshape(-1,2).flip(-1).flatten()
            if key_offset and head>=h and t:
                prev=qk[t-1,head].float(); prev=prev*torch.rsqrt(prev.square().mean()+torch.finfo(torch.float32).eps)
                y[64:]=prev[64:]
            ref[t,head]=y.bfloat16()
    shape=(2*n,h//2,d) if paired else (n,h,d)
    torch.testing.assert_close(q,ref[:,:h].reshape(shape),atol=0,rtol=0)
    torch.testing.assert_close(k,ref[:,h:].reshape(shape),atol=0,rtol=0)


def test_lazy_sparse_adam_matches_dense_event_decay():
    table=HostTable(0,1,rows=7,width=4)
    table.shard.copy_(torch.arange(28).view(7,4).bfloat16()/30)
    p=table.shard.clone(); v=torch.zeros(7)
    for event,(step,rows) in enumerate([(1,[1,4]),(3,[2]),(339,[1,2,5]),(343,[0,4])],1):
        ids=torch.tensor(rows); g=(torch.arange(len(rows)*4).view(-1,4).float()/10-.3).bfloat16()
        beta,wd=adam_beta2_and_wd_mul(step); rate=.008*70
        v*=beta; v[ids]+=(1-beta)*g.float().square().mean(-1)
        u=g.float()/(v[ids].sqrt()[:,None]+1e-10)*rate*(1-beta**event)**.5
        current=p[ids].float(); u+=torch.where(u*current>0,current*rate*rate*.005*wd,0)
        p[ids]=(current-u).bfloat16()
        table.apply_rows(step,ids,g,.008,chunk_rows=1)
        torch.testing.assert_close(table.shard,p,atol=0,rtol=0)
        effective=table.exp_avg_sq.clone()
        for e,b in enumerate(table.history[1:],1):
            effective=torch.where(table.last_event<e,effective*b,effective)
        torch.testing.assert_close(effective,v,atol=1e-8,rtol=1e-6)
    assert table.shard.requires_grad is False


def _gloo_worker(rank,folder):
    torch.set_num_threads(1)
    folder=Path(folder)
    dist.init_process_group('gloo',init_method='file://'+str(folder/'init'),rank=rank,world_size=2)
    try:
        table=HostTable(rank,2,rows=16,width=4)
        table.shard.copy_(torch.arange(rank*8,(rank+1)*8)[:,None].expand(8,4).bfloat16())
        requested=torch.tensor([0,3,8,15,3,8])
        ids,cache,slots=table.lookup(requested)
        torch.testing.assert_close(cache[slots],requested[:,None].expand(6,4).bfloat16(),atol=0,rtol=0)
        table.accumulate(ids,slots,torch.full((6,4),float(rank+1),dtype=torch.bfloat16))
        table.update(1,.008)
        torch.save(table.shard,folder/f'{rank}.pt')
    finally: dist.destroy_process_group()


def test_gloo_routes_cross_owner_rows_and_repeated_gradients(tmp_path):
    try:
        mp.spawn(_gloo_worker,args=(str(tmp_path),),nprocs=2,join=True)
    except mp.ProcessRaisedException as error:
        if 'Operation not permitted' in str(error) and not os.environ.get('CI'):
            pytest.skip('Local sandbox blocks Gloo TCP sockets; mandatory in CI/TPU preflight')
        raise
    actual=torch.cat([torch.load(tmp_path/f'{r}.pt',weights_only=True) for r in range(2)])
    expected=HostTable(0,1,rows=16,width=4)
    expected.shard.copy_(torch.arange(16)[:,None].expand(16,4).bfloat16())
    rows=torch.tensor([0,3,8,15]); averaged=torch.tensor([1.5,3.,3.,1.5])[:,None].expand(4,4).bfloat16()
    expected.apply_rows(1,rows,averaged,.008)
    torch.testing.assert_close(actual,expected.shard,atol=0,rtol=0)


def test_full_dense_parameter_shapes_on_meta():
    with torch.device('meta'):
        model=GPT(50257,11,6,128,768,16,ngram_dim=768,world_size=8,device=torch.device('meta'))
    shapes={n:tuple(p.shape) for n,p in model.named_parameters()}
    assert shapes['qk_bank']==(48,256,768)
    assert shapes['vo_bank']==(16,768,768)
    assert shapes['mlp_bank']==(12,2,2816,768)
    assert shapes['value_embeds']==(201216,768)
    assert sum(p.numel() for p in model.parameters())==302792179
    assert model.attn_qk_dim(3)==128 and model.attn_qk_dim(0)==64
    assert model.attn_v_dim(1)==64 and model.attn_v_dim(2)==128


def test_anvil_cascade_rails_and_zero_input():
    gradient=torch.zeros(2,8,4,dtype=torch.bfloat16)
    velocity=torch.zeros(2,2,8,4)
    result=cascade(gradient,velocity,torch.tensor(.9),torch.tensor(.85),torch.tensor(.4385))
    assert torch.count_nonzero(result)==0 and torch.isfinite(result).all()
    cascade(torch.ones_like(gradient),velocity,torch.tensor(.9),torch.tensor(.85),torch.tensor(.4385))
    torch.testing.assert_close(velocity[0],torch.full_like(velocity[0],.15))
    torch.testing.assert_close(velocity[1],torch.full_like(velocity[1],.02))


def test_port_imports_no_cuda_performance_modules():
    import ast
    for path in (HERE/'tpu_port').glob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith(('triton','kernels','track_1_short.perf'))
