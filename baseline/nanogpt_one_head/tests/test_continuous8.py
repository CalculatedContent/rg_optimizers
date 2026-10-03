import csv
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from rg_nanogpt_one_head import continuous_support as cs


def test_probe_fixed_document_disjoint_and_rng_independent(tmp_path):
    data=np.tile(np.r_[np.arange(20),63],40).astype(np.uint16)
    a=cs.document_windows(data,16,9,123,eot=63)
    b=cs.document_windows(data,16,9,123,eot=63)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    assert len(set(a[1]))==16 and not np.any(a[0]==63)
    cfg={'training':{'batch_size':4},'model':{'block_size':8},'evaluation':{
        'probe_documents':16,'train_probe_seed':1,'validation_probe_seed':2,'test_probe_seed':3}}
    meta={'eot_token':63,'files':{s:{'sha256':'a'} for s in ('train','val','test')}}
    arrays={s:data for s in ('train','val','test')}
    state=torch.get_rng_state().clone()
    probes=cs.build_document_probes(cfg,arrays,tmp_path,meta)
    assert torch.equal(state,torch.get_rng_state())
    assert sum(y.numel() for x,y in probes[2])==128
    cfg['evaluation']['test_probe_seed']=99
    with pytest.raises(RuntimeError,match='probe changed'):
        cs.build_document_probes(cfg,arrays,tmp_path,meta)


def test_failed_upload_never_publishes_completion_marker(tmp_path,monkeypatch):
    events=[]
    class Sink:
        def file(self,*a):events.append('file');raise OSError('upload failed')
        def json(self,*a):events.append('marker')
    monkeypatch.setattr(cs,'publisher',lambda c:Sink())
    p=tmp_path/'checkpoint_latest.pt';p.write_bytes(b'checkpoint')
    with pytest.raises(OSError,match='upload failed'):
        cs.publish_checkpoint(p,dict(config={},step=4))
    assert events==['file'] and p.exists()


def test_latest_pointer_requires_resume_state(tmp_path,monkeypatch):
    calls=[]
    class Sink:
        def file(self,*a):return {'generation':'1','bytes':10}
        def json(self,value,path):calls.append(path)
    monkeypatch.setattr(cs,'publisher',lambda c:Sink())
    monkeypatch.setattr(cs,'publish_metadata',lambda *a:None)
    p=tmp_path/'checkpoint_latest.pt';p.write_bytes(b'checkpoint')
    payload=dict(config={},step=4,fingerprint='x',model_state_sha256='m',optimizers=[{}])
    cs.publish_checkpoint(p,payload)
    assert 'LATEST_RESUMABLE.json' not in calls
    payload['resume_diagnostics']={'valid':True}
    cs.publish_checkpoint(p,payload)
    assert calls[-1]=='LATEST_RESUMABLE.json'


def test_pair_refuses_changing_layer_subset(tmp_path,monkeypatch):
    monkeypatch.setattr(cs,'plot_pairs',lambda *a:None)
    monkeypatch.setattr(cs,'publish_metadata',lambda *a:None)
    (tmp_path/'fixed_document_probe.json').write_text('{}')
    row=dict(step=2000,tokens_seen=10,elapsed_sec=1,primary_lr=.1,train_loss=2,
             val_loss=3,test_loss=4,train_accuracy=.3,test_accuracy=.2)
    summary={'model_state_sha256':'m','alpha_raw_n':5,'alpha_raw_mean':2.1,
             'alpha_clip_xmax_n':6,'alpha_clip_xmax_mean':2.,'alpha_clip_xmax_min':1.9}
    cs.record_pair({'continuous':{'enabled':True},'model':{'n_layer':1}},tmp_path,row,summary)
    with (tmp_path/'alpha_token_error.csv').open() as f:r=next(csv.DictReader(f))
    assert r['alpha_raw_mean']=='nan' and float(r['test_token_error_pct'])==80
    assert r['alpha_clip_xmax_mean']=='2.0'


def test_guard_rejects_resume_before_tpu_initialization(tmp_path):
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension
    install_muonclip_extension()
    from rg_nanogpt_one_head.engine import run_one
    with pytest.raises(ValueError,match='must start fresh'):
        run_one(cfg={'continuous':{'enabled':True}},data_root=tmp_path,
                results_root=tmp_path,optimizer_name='muon_clip',seed=1,device='tpu',resume=True)


def test_fixed_config_alignment_and_schedule():
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension
    install_muonclip_extension()
    from rg_nanogpt_one_head.config import load_config,epoch_step_map,lr_schedule_steps,optimizer_profile,warmup_steps
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'configs/muonclip_continuous8.yaml')
    assert cfg['model']['n_layer']==12 and cfg['runtime']['tpu_expected_chips']==8
    assert cfg['dataset']['train_tokens']==5_000_000_000 and 'continuation' not in cfg
    assert list(epoch_step_map(cfg))==list(range(0,1_000_001,1000))
    p=optimizer_profile(cfg,'muon_clip')
    assert lr_schedule_steps(cfg,p)==100_000 and warmup_steps(p,100_000)==2000


def test_24h_config_has_aligned_pairs_without_changing_training_dynamics():
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension
    install_muonclip_extension()
    from rg_nanogpt_one_head.config import load_config, epoch_step_map
    root = Path(__file__).resolve().parents[1]/'configs'
    before = load_config(root/'muonclip_continuous8.yaml')
    after = load_config(root/'muonclip_continuous8_24h.yaml')
    for key in ('dataset','model','optimizer_profiles','evaluation','runtime'):
        assert before[key] == after[key]
    assert list(epoch_step_map(after)) == list(range(0,1_000_001,500))
    assert after['training']['checkpoint_interval_steps'] == 500
    assert after['training']['eval_interval_steps'] == 500
    assert after['continuous']['max_wall_hours'] == 23.5
    assert after['continuous']['keep_local_epoch_checkpoints'] == 3


def test_only_uploaded_unchanged_epoch_checkpoints_are_pruned(tmp_path, monkeypatch):
    class Sink:
        fail = False
        def file(self, path, relative):
            if self.fail: raise OSError('upload failed')
            return {'generation':'1', 'bytes':path.stat().st_size}
        def json(self, *args): pass
    sink = Sink()
    monkeypatch.setattr(cs, 'publisher', lambda cfg:sink)
    monkeypatch.setattr(cs, 'publish_metadata', lambda *args:None)
    directory = tmp_path/'epoch_checkpoints'
    directory.mkdir()
    unknown = directory/'unknown.pt'
    unknown.write_bytes(b'not uploaded')
    config = {'continuous':{'keep_local_epoch_checkpoints':2}}
    def publish(step):
        path = directory/f'epoch_{step:06d}.pt'
        path.write_bytes(str(step).encode())
        cs.publish_checkpoint(path, dict(config=config,step=step,fingerprint='f',model_state_sha256='m'))
        return path
    first = publish(1)
    second = publish(2)
    first.write_bytes(b'x')  # Same size, changed contents must survive pruning.
    third = publish(3)
    assert first.exists() and unknown.exists()
    first.write_bytes(b'1')
    fourth = publish(4)
    assert not first.exists() and not second.exists()
    assert third.exists() and fourth.exists() and unknown.exists()
    sink.fail = True
    with pytest.raises(OSError, match='upload failed'): publish(5)
    assert third.exists() and fourth.exists() and (directory/'epoch_000005.pt').exists()


def test_muonclip_checkpoint_preserves_next_updates_lr_optimizer_and_sampler(tmp_path):
    from copy import deepcopy
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension
    install_muonclip_extension()
    from rg_nanogpt_one_head.model import GPT,GPTConfig
    from rg_nanogpt_one_head.config import load_config,optimizer_profile
    from rg_nanogpt_one_head.optimizers import make_optimizer_handles,zero_grad,optimizer_step,set_learning_rates,optimizer_state_dict
    from rg_nanogpt_one_head.checkpoints import save_training_checkpoint,load_training_checkpoint_for_resume,optimizer_state_sha256
    from rg_nanogpt_one_head.evaluation import random_batch,evaluate_probe,fixed_probe
    from rg_nanogpt_one_head.runtime import parameter_snapshot
    torch.set_num_threads(1)
    cfg=deepcopy(load_config(Path(__file__).resolve().parents[1]/'configs/muonclip_reference.yaml'))
    cfg['model'].update(vocab_size=64,block_size=8,n_embd=16,n_head=2,n_layer=2)
    cfg['optimizer_profiles']['muon_clip']['qk_clip_threshold']=0.01
    torch.manual_seed(123)
    net=GPT(GPTConfig(**cfg['model']))
    handles=make_optimizer_handles(net,optimizer_profile(cfg,'muon_clip'))
    gen=torch.Generator().manual_seed(44)
    data=np.tile(np.arange(64,dtype=np.uint16),20)
    def advance(model,opts,generator,start,end):
        for step in range(start,end):
            set_learning_rates(opts,update_index=step,total_steps=10,warmup_steps=2)
            x,y=random_batch(data,batch_size=4,block_size=8,generator=generator)
            zero_grad(opts)
            model(x,y)[1].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
            optimizer_step(opts)
    advance(net,handles,gen,0,3)
    path=tmp_path/'checkpoint.pt'
    save_training_checkpoint(path,model=net,handles=handles,step=3,best_validation_loss=4.,
        best_validation_step=2,elapsed_seconds=1.,fingerprint='test',cfg=cfg,
        optimizer_name='muon_clip',seed=123,train_generator=gen,
        resume_diagnostics={'previous_eval_snapshot':parameter_snapshot(net),
                            'last_grad_pre':1.,'last_grad_post':1.,'last_clipped':False})
    advance(net,handles,gen,3,7)
    resumed=GPT(GPTConfig(**cfg['model']))
    rh=make_optimizer_handles(resumed,optimizer_profile(cfg,'muon_clip'))
    rg=torch.Generator()
    loaded=load_training_checkpoint_for_resume(path,model=resumed,handles=rh,
                        expected_fingerprint='test',train_generator=rg)
    advance(resumed,rh,rg,loaded[0],7)
    for name,value in net.state_dict().items():
        assert torch.equal(value,resumed.state_dict()[name]),name
    assert optimizer_state_sha256(optimizer_state_dict(handles))==optimizer_state_sha256(optimizer_state_dict(rh))
    assert torch.equal(gen.get_state(),rg.get_state())
    probe=fixed_probe(data,batch_size=4,block_size=8,n_batches=2,seed=99)
    assert evaluate_probe(net,probe,torch.device('cpu'))==evaluate_probe(resumed,probe,torch.device('cpu'))


@pytest.mark.parametrize('stop_early',[False,True])
def test_fresh_continuous_engine_writes_fixed_probe_and_pairs(tmp_path,monkeypatch,stop_early):
    from copy import deepcopy
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension
    install_muonclip_extension()
    from rg_nanogpt_one_head.config import load_config
    from rg_nanogpt_one_head.data import write_token_splits
    from rg_nanogpt_one_head.checkpoints import model_state_sha256
    import rg_nanogpt_one_head.train_loop as loop
    import rg_nanogpt_one_head.run_utils as utils
    from rg_nanogpt_one_head.training import run_one
    cfg=deepcopy(load_config(Path(__file__).resolve().parents[1]/'configs/muonclip_continuous8.yaml'))
    cfg['model'].update(vocab_size=64,block_size=8,n_embd=16,n_head=2,n_layer=2)
    cfg['runtime']['tpu_spmd']=False
    cfg['dataset'].update(train_tokens=256,val_tokens=128,test_tokens=128)
    cfg['training'].update(batch_size=2,grad_accum_steps=1,max_steps=4,target_epochs=.25,
                          epoch_interval=.125,eval_interval_steps=2,eval_batches=2,
                          checkpoint_interval_steps=2,min_free_disk_gb=0)
    cfg['continuous']['cloud_required']=False
    cfg['evaluation'].update(probe_documents=4,test_interval_steps=2,bleu_examples=2,
                             bleu_prompt_tokens=3,bleu_continuation_tokens=2,bleu_batch_size=2)
    cfg['optimizer_profiles']['muon_clip']['lr_schedule_steps']=4
    for p in cfg['optimizer_profiles'].values():p.pop('lr_schedule_epochs',None)
    class Encoder:
        n_vocab=64
        eot_token=63
        def encode_ordinary(self,text):return list(range(16))
    data=tmp_path/'data'
    write_token_splits(['doc']*100,Encoder(),data,train_tokens=256,val_tokens=128,test_tokens=128,
        dataset_metadata={'dataset_name':cfg['dataset']['name'],'dataset_config':cfg['dataset']['config'],
                          'dataset_split':'train','dataset_revision':cfg['dataset']['revision'],'tokenizer':'gpt2'})
    def spectrum(model,*a,**kw):
        return {'model_state_sha256':model_state_sha256(model.state_dict()),
                'alpha_raw_n':12,'alpha_raw_mean':3.,'alpha_raw_min':2.5,
                'alpha_clip_xmax_n':12,'alpha_clip_xmax_mean':2.9,'alpha_clip_xmax_min':2.4}
    monkeypatch.setattr(loop,'run_weightwatcher',spectrum)
    monkeypatch.setattr(utils,'evaluate_bleu',lambda *a,**kw:{'bleu':0.})
    if stop_early:
        # A deadline between regular checkpoints must save after the next update,
        # with current finite diagnostics, rather than wait 500 more steps.
        stop=tmp_path/'STOP'
        cfg['training']['stop_file']=str(stop)
        update=loop.optimizer_step
        def request_after_update(*a,**kw):
            result=update(*a,**kw)
            stop.touch()
            return result
        monkeypatch.setattr(loop,'optimizer_step',request_after_update)
        from rg_nanogpt_one_head.continuation import TrainingPaused
        with pytest.raises(TrainingPaused):
            run_one(cfg=cfg,data_root=data,results_root=tmp_path/'results',optimizer_name='muon_clip',
                    seed=2027,device='cpu',resume=False,progress=False)
        checkpoint=torch.load(tmp_path/'results/muon_clip/seed_2027/checkpoint_latest.pt',weights_only=False)
        assert checkpoint['step']==1
        assert checkpoint['seed']==2027
        assert math.isfinite(checkpoint['resume_diagnostics']['last_grad_pre'])
        return
    # Stop at the full step-4 checkpoint: this tests the actual monitoring loop
    # without post-run spectral completion checks against our synthetic spectrum.
    original=loop.save_training_checkpoint
    def save(path,**kw):
        result=original(path,**kw)
        if kw['step']==4:raise RuntimeError('test stopped after saved step 4')
        return result
    monkeypatch.setattr(loop,'save_training_checkpoint',save)
    with pytest.raises(RuntimeError,match='test stopped'):
        run_one(cfg=cfg,data_root=data,results_root=tmp_path/'results',optimizer_name='muon_clip',
                seed=1337,device='cpu',resume=False,progress=False)
    run=tmp_path/'results'/'muon_clip'/'seed_1337'
    with (run/'alpha_token_error.csv').open() as f:rows=list(csv.DictReader(f))
    assert [int(r['step']) for r in rows]==[0,2]
    assert len({r['probe_sha256'] for r in rows})==1
    assert all(r['model_state_sha256'] for r in rows)
    assert (run/'checkpoint_latest.pt').is_file()
    assert (run/'alpha_token_error.png').is_file()
