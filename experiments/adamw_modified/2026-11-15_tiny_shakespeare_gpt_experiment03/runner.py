"""Experiment03: fixed three-arm GPT comparison; no alpha-targeted fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,hashlib,math,copy,contextlib,io,traceback,subprocess,zipfile,datetime
from pathlib import Path
import numpy as np
import torch
import weightwatcher as ww
sys.path.insert(0,str(Path(__file__).resolve().parent/'upstream'))
from model import GPT,GPTConfig
ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'Protocol.json').read_text())
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
ARMS=['adamw','magnitude_graft','sgdm']

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x

def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),indent=2)+'\n');tmp.replace(path)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def model_hash(net):
    h=hashlib.sha256()
    for n,p in net.named_parameters():h.update(n.encode());h.update(p.detach().numpy().tobytes())
    return h.hexdigest()
def norm(x):return torch.linalg.vector_norm(x,dtype=torch.float64)
def eligible(n):return n.startswith('transformer.h.') and n.endswith('.weight') and any(s in n for s in ('.attn.c_attn.','.attn.c_proj.','.mlp.c_fc.','.mlp.c_proj.'))
def make_model(seed):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):return GPT(GPTConfig(**PLAN['model']))
def make_opt(net,arm,mult):
    if arm=='sgdm':return torch.optim.SGD(net.parameters(),lr=PLAN['sgdm_lr']*mult,momentum=.9,weight_decay=0,foreach=False)
    return torch.optim.AdamW(net.parameters(),lr=PLAN['adam_lr']*mult,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
def factor(step):
    if step<=100:return step/100
    return .1+.45*(1+math.cos(math.pi*(step-100)/(PLAN['full_steps']-100)))
def load_data():
    z=np.load(ROOT/'data.npz');return {n:torch.from_numpy(z[n].astype(np.int64)) for n in ['train','validation','test']}
def batch(data,seed,step):
    rng=np.random.default_rng(np.random.SeedSequence([seed,step,913]))
    starts=rng.integers(0,len(data)-PLAN['model']['block_size'],size=PLAN['batch_size'])
    ix=starts[:,None]+np.arange(PLAN['model']['block_size'])[None,:]
    return data[ix].contiguous(),data[ix+1].contiguous(),hashlib.sha256(starts.tobytes()).hexdigest()

@torch.no_grad()
def update(net,opt,arm,mult,step,record=False):
    f=factor(step);decay=1-PLAN['decay_per_step']*f
    before={n:p.clone() for n,p in net.named_parameters() if eligible(n)} if arm=='magnitude_graft' or record else {}
    for g in opt.param_groups:g['lr']=(PLAN['sgdm_lr'] if arm=='sgdm' else PLAN['adam_lr'])*mult*f
    for p in net.parameters():
        if p.ndim==2:p.mul_(decay)
    opt.step();rows=[]
    for n,p in net.named_parameters():
        if not eligible(n):continue
        state=opt.state[p]
        if arm=='sgdm':
            mom=state['momentum_buffer'];target=PLAN['sgdm_lr']*mult*f*norm(mom)
            applied=-PLAN['sgdm_lr']*mult*f*mom;native=None;gain=1.;cosine=None;floor=False
        else:
            count=float(state['step']);mom=state['exp_avg']/.1
            adam=-PLAN['adam_lr']*mult*f*(state['exp_avg']/(1-.9**count))/((state['exp_avg_sq']/(1-.999**count)).sqrt()+1e-8)
            native=norm(adam);target=PLAN['sgdm_lr']*mult*f*norm(mom);floor=bool(native<1e-30)
            gain=target/native.clamp_min(1e-30) if arm=='magnitude_graft' else 1.
            applied=adam*gain
            if arm=='magnitude_graft':p.copy_(before[n]*decay+applied)
            cosine=float((applied.double()*adam.double()).sum()/(norm(applied)*native)) if norm(applied)>0 and native>0 else None
        if record:
            rel=float(abs(norm(applied)/target-1)) if target>1e-20 else None
            if arm=='magnitude_graft' and rel is not None and not floor:assert rel<3e-6
            if arm=='magnitude_graft' and cosine is not None:assert abs(cosine-1)<3e-6
            rows.append({'matrix':n,'gradient_norm':float(norm(p.grad)),'momentum_norm':float(norm(mom)),
              'target_magnitude':float(target),'applied_magnitude':float(norm(applied)),
              'native_adam_magnitude':None if native is None else float(native),'gain':float(gain),
              'direction_cosine':cosine,'denominator_floor':floor,'target_relative_error':rel,
              'weight_norm_before':float(norm(before[n]))})
    return rows

def train_step(net,opt,arm,mult,seed,step,data,record=False):
    net.train();x,y,bh=batch(data,seed,step);opt.zero_grad(set_to_none=True)
    _,loss=net(x,y)
    assert torch.isfinite(loss),'nonfinite training loss'
    loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
    rows=update(net,opt,arm,mult,step,record)
    return {'step':step,'loss':float(loss.detach()),'gradient_norm_before_clip':float(gn),'batch_hash':bh,'updates':rows}

@torch.no_grad()
def evaluate(net,data):
    net.eval();T=PLAN['model']['block_size'];B=PLAN['batch_size'];starts=np.arange(0,len(data)-T,T)
    total=0.;correct=0;n=0;counts=np.zeros(PLAN['model']['vocab_size'],int)
    for st in range(0,len(starts),B):
        ix=starts[st:st+B,None]+np.arange(T)[None,:];x=data[ix].contiguous();y=data[ix+1].contiguous()
        logits,loss=net(x,y);assert torch.isfinite(logits).all()
        pred=logits.argmax(-1);counts+=np.bincount(pred.numpy().ravel(),minlength=len(counts))
        total+=float(loss)*y.numel();correct+=int((pred==y).sum());n+=y.numel()
    return {'loss':total/n,'bits_per_character':total/n/math.log(2),'accuracy':correct/n,'tokens':n,'prediction_counts':counts.tolist()}

@torch.no_grad()
def spectral(net,dest,step):
    holder=torch.nn.ModuleDict();mapping={};spectra={}
    for name,p in net.named_parameters():
        if not eligible(name):continue
        parts=[('physical',p)]
        if '.c_attn.' in name:parts += list(zip(['Q','K','V'],p.chunk(3,dim=0)))
        for suffix,w in parts:
            key=(name+'__'+suffix).replace('.','_');m=torch.nn.Linear(w.shape[1],w.shape[0],bias=False);m.weight.copy_(w)
            holder[key]=m;mapping[key]=(name,suffix);spectra[key]=torch.linalg.svdvals(w.double()).square().numpy()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        df=ww.WeightWatcher(model=holder).analyze(plot=False,randomize=False)
    rows=[]
    for key,(name,part) in mapping.items():
        q=df[df.longname==key];assert len(q)==1,(key,df.longname.tolist())
        z=q.iloc[0];rows.append(clean({'matrix':name,'part':part,**{k:z[k] for k in ['alpha','D','sigma','xmin','xmax','num_pl_spikes','num_evals','spectral_norm','norm','stable_rank','status','warning']},
          'supported':bool(np.isfinite(z.alpha) and z.num_pl_spikes>=20 and z.D<=.2)}))
    dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/f'spectra_{step:06}.npz',**spectra)
    return rows

def save_checkpoint(path,net,opt,spec,step):
    tmp=path.with_suffix('.tmp');torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'spec':spec,'step':step,'torch_rng':torch.get_rng_state(),'model_hash':model_hash(net),'source_sha256':sha(ROOT/'runner.py'),'protocol_sha256':sha(ROOT/'Protocol.json')},tmp);tmp.replace(path)

def run(spec):
    d=ROOT/'runs'/spec['id'];d.mkdir(parents=True,exist_ok=True)
    assert not (d/'started.json').exists(),'Never duplicate a started run'
    write(d/'started.json',{'spec':spec,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    tic=time.time();data=load_data();net=make_model(spec['seed']);opt=make_opt(net,spec['arm'],spec['multiplier'])
    initial=model_hash(net);rows=[];history=[];checkpoints=[]
    for step in range(spec['steps']+1):
        if step:
            assert time.time()<spec['deadline'],'Campaign deadline reached'
            record=step%50==0 or step%PLAN['checkpoint_interval'] in (1,2,3)
            row=train_step(net,opt,spec['arm'],spec['multiplier'],spec['seed'],step,data['train'],record)
            # Flush every step: failure location remains inspectable.
            with (d/'updates.jsonl').open('a') as f:f.write(json.dumps(clean(row))+'\n')
        if step==0 or step%PLAN['checkpoint_interval']==0 or step==spec['steps']:
            assert all(torch.isfinite(p).all() for p in net.parameters())
            cp=d/f'checkpoint_{step:06}.pt';save_checkpoint(cp,net,opt,spec,step);checkpoints.append({'step':step,'path':cp.name,'sha256':sha(cp)})
            # Keep initialization, the nonzero replay state, and the latest state.
            keep={0,PLAN['checkpoint_interval'],step}
            for previous in checkpoints:
                if previous['step'] not in keep:(d/previous['path']).unlink(missing_ok=True)
            checkpoints=[c for c in checkpoints if c['step'] in keep]
            h={'step':step,'validation':evaluate(net,data['validation']),'spectra':spectral(net,d/'spectra',step)}
            history.append(h);write(d/'history.json',history)
            write(d/'status.json',{'spec':spec,'step':step,'elapsed_seconds':time.time()-tic,'validation':h['validation'],'training_active':step<spec['steps']})
            print(spec['id'],step,'val',h['validation']['loss'],flush=True)
    result={'spec':spec,'status':'complete','seconds':time.time()-tic,'initial_hash':initial,'final_hash':model_hash(net),'history':history,'checkpoints':checkpoints}
    if spec['phase']=='confirmatory':result['test']=evaluate(net,data['test'])
    write(d/'done.json',result);verify(spec);return result

def verify(spec):
    d=ROOT/'runs'/spec['id'];done=json.loads((d/'done.json').read_text());data=load_data()
    for c in done['checkpoints']:assert sha(d/c['path'])==c['sha256']
    ck=torch.load(d/done['checkpoints'][0]['path'],weights_only=False);net=make_model(spec['seed']);net.load_state_dict(ck['model'])
    assert model_hash(net)==done['initial_hash']
    opt=make_opt(net,spec['arm'],spec['multiplier']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng'])
    saved=[json.loads(l) for l in (d/'updates.jsonl').read_text().splitlines()]
    assert len(saved)==spec['steps']
    for row in saved:
        _,_,bh=batch(data['train'],spec['seed'],row['step']);assert bh==row['batch_hash']
    replay_origins=[0,PLAN['checkpoint_interval']]
    for origin in replay_origins:
        checkpoint=torch.load(d/f'checkpoint_{origin:06}.pt',weights_only=False)
        net.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer']);torch.set_rng_state(checkpoint['torch_rng'])
        for step in range(origin+1,origin+4):
            r=train_step(net,opt,spec['arm'],spec['multiplier'],spec['seed'],step,data['train'],True)
            assert clean(r)==saved[step-1],('replay',spec['id'],step)
    ck=torch.load(d/done['checkpoints'][-1]['path'],weights_only=False);net.load_state_dict(ck['model']);assert model_hash(net)==done['final_hash']
    measured=evaluate(net,data['validation']);assert measured==done['history'][-1]['validation']
    again=spectral(net,d/'verification_spectra',spec['steps']);assert again==done['history'][-1]['spectra']
    if spec['phase']=='confirmatory':assert evaluate(net,data['test'])==done['test']
    finite=all(torch.isfinite(p).all() for p in net.parameters())
    unique=sum(n>0 for n in measured['prediction_counts'])
    write(d/'Verification.json',{'passed':True,'checkpoint_hashes':len(done['checkpoints']),'initial_hash':done['initial_hash'],
      'three_step_replay_exact':True,'replay_origins':replay_origins,'all_batch_hashes_verified':True,'validation_restoration_exact':True,'spectra_and_alpha_restoration_exact':True,
      'test_restoration_exact':spec['phase']=='confirmatory','finite_parameters':bool(finite),'distinct_predicted_characters':unique,
      'functional_status':'USABLE' if finite and unique>1 else 'FAILED','alpha_used_as_quality_gate':False})
    return done


def preflight():
    # Independent native optimizer and momentum-history comparisons using supplied gradients.
    audit=[]
    for arm in ARMS:
        net=make_model(41000);ref=copy.deepcopy(net);opt=make_opt(net,arm,1);refopt=make_opt(ref,'sgdm' if arm=='sgdm' else 'adamw',1)
        shadow={n:torch.zeros_like(p) for n,p in net.named_parameters() if eligible(n)}
        max_magnitude=0.;max_direction=0.
        for step in range(1,4):
            old={n:p.detach().clone() for n,p in net.named_parameters()}
            for (n,p),(rn,q) in zip(net.named_parameters(),ref.named_parameters()):
                assert n==rn
                p.grad=torch.randn_like(p)*.001;q.grad=p.grad.clone();q.data.copy_(p.data)
                if n in shadow:shadow[n].mul_(.9).add_(p.grad)
            rows=update(net,opt,arm,1,step,True)
            for g in refopt.param_groups:g['lr']=(PLAN['sgdm_lr'] if arm=='sgdm' else PLAN['adam_lr'])*factor(step)
            with torch.no_grad():
                for q in ref.parameters():
                    if q.ndim==2:q.mul_(1-PLAN['decay_per_step']*factor(step))
            refopt.step()
            for (n,p),(rn,q) in zip(net.named_parameters(),ref.named_parameters()):
                if arm!='magnitude_graft' or not eligible(n):assert torch.equal(p,q),('native_or_unmodified',arm,n)
                if arm=='magnitude_graft' and eligible(n):
                    for field in ('step','exp_avg','exp_avg_sq'):assert torch.equal(opt.state[p][field],refopt.state[q][field])
                    state=refopt.state[q];mom=state['exp_avg']/.1
                    assert torch.allclose(mom,shadow[n],rtol=2e-5,atol=2e-9)
                    expected_adam=-PLAN['adam_lr']*factor(step)*(state['exp_avg']/(1-.9**step))/((state['exp_avg_sq']/(1-.999**step)).sqrt()+1e-8)
                    target=PLAN['sgdm_lr']*factor(step)*norm(shadow[n])
                    expected=old[n]*(1-PLAN['decay_per_step']*factor(step))+expected_adam*(target/norm(expected_adam).clamp_min(1e-30))
                    assert torch.allclose(p,expected,rtol=1e-6,atol=1e-8)
            if arm=='magnitude_graft':
                max_magnitude=max(max_magnitude,max(z['target_relative_error'] for z in rows));max_direction=max(max_direction,max(abs(z['direction_cosine']-1) for z in rows))
        audit.append({'arm':arm,'passed':True,'maximum_magnitude_relative_error':max_magnitude,'maximum_direction_cosine_error':max_direction})
    # Negative numerical control: graft SGDM direction to its own magnitude = SGDM.
    v=torch.randn(256,256);u=-.05*v;identity=u*(.05*norm(v)/norm(u));assert torch.allclose(identity,u,rtol=2e-6,atol=1e-8)
    data=load_data();a,b,_=batch(data['train'],41000,1);assert a.shape==b.shape==(PLAN['batch_size'],128)
    forward_checks=[]
    for arm in ARMS:
        net=make_model(41000);opt=make_opt(net,arm,1);tic=time.time()
        first=train_step(net,opt,arm,1,41000,1,data['train'],True)
        state={'model':copy.deepcopy(net.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'rng':torch.get_rng_state()}
        second=train_step(net,opt,arm,1,41000,2,data['train'],True);expected=model_hash(net)
        net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
        replay=train_step(net,opt,arm,1,41000,2,data['train'],True)
        assert clean(replay)==clean(second) and model_hash(net)==expected
        forward_checks.append({'arm':arm,'nonzero_optimizer_state_replay_exact':True,'initial_loss':first['loss'],'seconds_per_step_including_audit':(time.time()-tic)/3})
    net=make_model(41000);assert len([n for n,p in net.named_parameters() if eligible(n)])==8
    sp=spectral(net,ROOT/'preflight_spectra',0);assert len(sp)==14
    write(ROOT/'Preflight.json',{'passed':True,'checks':audit,'forward_and_replay_checks':forward_checks,'sgdm_identity_graft_passed':True,'physical_matrices':8,'spectral_views':14,'parameters':sum(p.numel() for p in net.parameters()),'source_sha256':sha(ROOT/'runner.py'),'data_sha256':sha(ROOT/'data.npz'),'versions':{'torch':torch.__version__,'weightwatcher':ww.__version__,'numpy':np.__version__}})
    print('PREFLIGHT PASSED',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='run':run(json.loads(Path(sys.argv[2]).read_text()))
