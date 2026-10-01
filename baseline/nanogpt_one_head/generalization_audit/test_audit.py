import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import audit
from pinned_model import GPT, GPTConfig

def test_proper_scores_uniform_and_perfect():
    targets=torch.tensor([[0,1],[1,0]])
    r=audit.per_document_scores(torch.zeros(2,2,2),targets,np.array([100,100]))
    assert np.allclose(r['nll'],np.log(2))
    assert np.allclose(r['brier'],.5)
    assert np.allclose(r['mrr'],2/3)  # tied midpoint rank is 1.5
    assert audit.estimate(r,np.arange(2))['ece_pct']==0
    z=torch.full((2,2,2),-20.)
    z.scatter_(-1,targets[...,None],20.)
    r=audit.per_document_scores(z,targets,np.array([100,100]))
    assert np.allclose(r['nll'],0,atol=1e-6)
    assert np.allclose(r['brier'],0,atol=1e-6)
    assert np.allclose(r['mrr'],1)
    assert np.allclose(r['token_error_pct'],0)

def test_document_sampling_reproducible_nonoverlapping():
    t=np.array(sum(([i]*12+[99] for i in range(20)),[]))
    a,docs,starts=audit.document_windows(t,10,9,123,eot=99)
    b,docs2,starts2=audit.document_windows(t,10,9,123,eot=99)
    assert np.array_equal(a,b) and np.array_equal(docs,docs2)
    assert len(set(docs))==10 and not np.any(a==99)
    assert np.array_equal(starts,starts2)

def test_overlap_metrics_and_bootstrap():
    assert audit.lcs_f1([1,2,3],[1,2,3])==1
    assert audit.lcs_f1([1,2],[3,4])==0
    assert audit.repetition3([1]*5)==pytest.approx(2/3)
    r=audit.summarize({'score':np.ones(10)},30,123)['score']
    assert r['value']==r['low']==r['high']==1

def fixture_run(tmp_path):
    run=tmp_path/'run';data=tmp_path/'data'
    (run/'epoch_checkpoints').mkdir(parents=True);(run/'spectral').mkdir();data.mkdir()
    cfg=GPTConfig(vocab_size=50257,block_size=8,n_embd=8)
    from dataclasses import asdict
    config={'model':asdict(cfg),'continuation':{'global_step_offset':100}}
    rng=np.random.default_rng(7)
    files={}
    for split in ['train','test']:
        t=np.array(sum((rng.integers(0,7,14).tolist()+[50256] for _ in range(20)),[]),dtype=np.uint16)
        path=data/(split+'.bin');t.tofile(path);files[split]={'sha256':audit.sha_file(path)}
    manifest={'model':asdict(cfg),'continuation':config['continuation'],'data_metadata':{'files':files}}
    (run/'manifest.json').write_text(json.dumps(manifest))
    rows=[]
    for step in [0,10]:
        torch.manual_seed(step);model=GPT(cfg);state=model.state_dict();digest=audit.state_hash(state)
        ck={'model':state,'model_state_sha256':digest,'step':step,'fingerprint':'test','config':config,'seed':1337}
        torch.save(ck,run/'epoch_checkpoints'/f'model_epoch_0_step_{step:07d}.pt')
        for i,m in enumerate(audit.MATRICES):
            rows.append(dict(step=step,matrix_name='L00_W_'+m,alpha_raw=2.2+i/10+step/100,alpha_clip_xmax=2.1+i/10+step/100,status='success',model_state_sha256=digest,protocol_fingerprint='test',run_seed=1337))
    pd.DataFrame(rows).to_csv(run/'spectral/layers.csv',index=False)
    return run,data,manifest

def test_checkpoint_binding_rejects_wrong_spectrum(tmp_path):
    run,data,manifest=fixture_run(tmp_path)
    rows=pd.read_csv(run/'spectral/layers.csv');rows=rows[rows.step.eq(0)].copy()
    path=next((run/'epoch_checkpoints').glob('*0000000.pt'))
    _,_,step=audit.load_checked(path,rows,manifest);assert step==100
    rows['model_state_sha256']='wrong'
    with pytest.raises(ValueError,match='Spectral/model hash'):audit.load_checked(path,rows,manifest)

def test_end_to_end_fixed_probe_resume(tmp_path,monkeypatch):
    # Exercise real model forward, generation, proper scores, BLEU, aggregation,
    # checkpoint selection and exports, without a network tokenizer download.
    import tiktoken
    class Encoder:
        def decode(self,x):return ' '.join(map(str,x))
    monkeypatch.setattr(tiktoken,'get_encoding',lambda name:Encoder())
    monkeypatch.setattr(audit,'plot',lambda out:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    run,data,manifest=fixture_run(tmp_path)
    args=Namespace(command='run',run_dir=run,data_root=data,output=tmp_path/'output',device='cpu',threads=1,max_checkpoints=2,documents=4,generation_documents=3,batch_size=2,prompt_tokens=3,new_tokens=4,bootstrap=20,probe_seed=111,min_step=1,max_step=1000)
    audit.run(args)
    d=pd.read_csv(args.output/'metrics.csv')
    assert d.global_step.tolist()==[100,110]
    assert d.test_nll.notna().all() and d.test_sentence_bleu.notna().all()
    assert np.allclose(d.test_minus_train_nll,d.test_nll-d.train_nll)
    assert (args.output/'DONE.json').exists()
    before=(args.output/'checkpoints'/'0000000100.json').read_bytes()
    audit.run(args)
    assert before==(args.output/'checkpoints'/'0000000100.json').read_bytes()
    args.probe_seed+=1
    with pytest.raises(ValueError,match='settings/data/code changed'):audit.run(args)
