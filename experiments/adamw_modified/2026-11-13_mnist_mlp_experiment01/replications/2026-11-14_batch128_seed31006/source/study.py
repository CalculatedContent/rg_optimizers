"""Bounded MNIST-only magnitude-graft experiment; frozen before confirmation runs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, math, time, gzip, hashlib, copy, io, contextlib, argparse
import datetime, platform, shutil, concurrent.futures, multiprocessing
from pathlib import Path
sys.path.insert(0, '/tmp/adamw_shape_deps')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import weightwatcher as ww
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LAYERS = ['fc1', 'fc2', 'fc3']
RATES = [.04, .2, .8]
ARMS = ['adamw', 'magnitude_graft', 'sgdm']
SEEDS = [31001, 31002, 31003, 31004, 31005]
EPOCHS = 120
DATA = None
STAT_NAMES = ['gradient_norm', 'sgdm_momentum_norm', 'adam_update_norm',
              'sgdm_target_norm', 'applied_update_norm', 'gain',
              'weight_norm', 'adam_sgdm_cosine', 'relative_step', 'radial_rate']

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if hasattr(x, 'item'): x = x.item()
    return None if isinstance(x, float) and not np.isfinite(x) else x

def writej(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def weight_hash(net):
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()

def savept(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)

def frobenius(x):
    # Accumulate norms in float64 even when the trained parameters are float32.
    return torch.linalg.vector_norm(x, dtype=torch.float64)

def load_data():
    def get(prefix, n):
        x = np.frombuffer(gzip.decompress((OUT/'data'/f'{prefix}-images-idx3-ubyte.gz').read_bytes()), np.uint8, offset=16).reshape(n, 784).copy()
        y = np.frombuffer(gzip.decompress((OUT/'data'/f'{prefix}-labels-idx1-ubyte.gz').read_bytes()), np.uint8, offset=8).astype('int64')
        return torch.from_numpy(x).float()/255, torch.from_numpy(y)
    x, y = get('train', 60000); xt, yt = get('t10k', 10000)
    split = np.load(OUT/'data/split.npz')
    return x[split['train']], y[split['train']], x[split['validation']], y[split['validation']], xt, yt

def schedule(epoch):
    if epoch <= 5: return epoch/5
    return .1 + .45*(1 + math.cos(math.pi*(epoch-5)/(EPOCHS-5)))

def order(seed, epoch, n=55000):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, 711])).permutation(n)

def specifications():
    # Interleave seeds and conditions; each condition includes all three arms.
    specs = []
    for seed in SEEDS:
        for mult in [.5, 1., 2.]:
            for batch in [128, 256, 512]:
                for arm in ARMS:
                    specs.append(dict(seed=seed, multiplier=mult, batch_size=batch, arm=arm))
    return specs

def run_id(spec):
    return f"b{spec['batch_size']}_lr{spec['multiplier']:g}/{spec['arm']}/seed_{spec['seed']}"

def make_opt(net, spec):
    rates = RATES if spec['arm'] == 'sgdm' else [.001]*3
    groups = [{'params': list(getattr(net, n).parameters()), 'lr': rate*spec['multiplier'], 'layer': n}
              for n, rate in zip(LAYERS, rates)]
    if spec['arm'] == 'sgdm':
        return torch.optim.SGD(groups, momentum=.9, dampening=0., nesterov=False, weight_decay=0., foreach=False)
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)

@torch.no_grad()
def apply_update(net, opt, spec, factor, batch_n, record=False):
    # All arms have the same multiplicative decay per processed example.
    decay = (1 - 1e-5*factor)**(batch_n/256)
    need_old = record or spec['arm'] == 'magnitude_graft'
    old = {n: getattr(net,n).weight.detach().clone() for n in LAYERS} if need_old else {}
    for group, rate in zip(opt.param_groups, RATES if spec['arm']=='sgdm' else [.001]*3):
        group['lr'] = rate*spec['multiplier']*factor
    for p in net.parameters(): p.mul_(decay)
    opt.step()
    rows = []
    if not need_old: return rows
    for n, target_rate, group in zip(LAYERS, RATES, opt.param_groups):
        w = getattr(net,n).weight; state = opt.state[w]
        if spec['arm'] == 'sgdm':
            mom = state['momentum_buffer']
            applied = -target_rate*spec['multiplier']*factor*mom
            native_norm = float('nan'); gain = 1.; cosine = 1.
        else:
            b1,b2 = group['betas']; t = float(state['step'])
            mom = state['exp_avg']/(1-b1)
            mh = state['exp_avg']/(1-b1**t); vh = state['exp_avg_sq']/(1-b2**t)
            adam = -group['lr']*mh/(vh.sqrt()+group['eps'])
            native_norm = frobenius(adam)
            target_norm = target_rate*spec['multiplier']*factor*frobenius(mom)
            gain = target_norm/native_norm.clamp_min(1e-30) if spec['arm']=='magnitude_graft' else 1.
            applied = adam*gain
            if spec['arm']=='magnitude_graft': w.copy_(old[n]*decay+applied)
            cosine = float((adam.double()*(-mom.double())).sum()/(native_norm*frobenius(mom)).clamp_min(1e-30)) if record else 0.
        if record:
            target_norm = target_rate*spec['multiplier']*factor*frobenius(mom)
            rows.append([float(frobenius(w.grad)),float(frobenius(mom)),float(native_norm),float(target_norm),
                         float(frobenius(applied)),float(gain),float(frobenius(old[n])),cosine,
                         float(frobenius(applied)/frobenius(old[n])),float((old[n]*applied).sum()/old[n].square().sum())])
    return rows

def train_step(net, opt, spec, xb, yb, factor, record=False):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(net(xb), yb); loss.backward()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
    return float(loss.detach()), apply_update(net,opt,spec,factor,len(yb),record)

@torch.no_grad()
def evaluate(net, x, y, logits=False):
    net.eval(); total=0.; errors=0; parts=[]
    for st in range(0,len(y),1024):
        z=net(x[st:st+1024]); yy=y[st:st+1024]
        total += float(F.cross_entropy(z,yy,reduction='sum'))
        errors += int((z.argmax(1)!=yy).sum())
        if logits: parts.append(z.numpy())
    row={'loss':total/len(y),'errors':errors,'n':len(y),'accuracy':1-errors/len(y)}
    return (row,np.concatenate(parts)) if logits else row

@torch.no_grad()
def spectra(net, dest, epoch):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        df=ww.WeightWatcher(model=net).analyze(plot=False,randomize=False)
    rows={}; eig={}
    for n in LAYERS:
        w=getattr(net,n).weight; lam=torch.linalg.svdvals(w.double()).square().numpy(); eig[n]=lam
        found=df[df.longname==n]
        if len(found)!=1: raise RuntimeError(f'Expected exactly one WW row: {n}')
        r=found.iloc[0]
        row={k:r[k] for k in ['alpha','D','sigma','xmin','xmax','num_pl_spikes','num_evals','stable_rank','spectral_norm','norm','status','warning']}
        row['eligible_hidden']=n in ['fc1','fc2']
        row['supported']=bool(len(lam)>=50 and np.isfinite(r.alpha) and r.num_pl_spikes>=20 and r.D<=.2)
        row['fixed_rank']={}
        for fraction in [.2,.5]:
            k=max(3,int(len(lam)*fraction)); h=float(np.log(lam[:k]/lam[k-1]).mean())
            row['fixed_rank'][str(fraction)]={'k':k,'h':h,'alpha':1+1/h,'trace_log_residual':h-1}
        rows[n]=clean(row)
    np.savez_compressed(dest/f'spectra/epoch_{epoch:03}.npz',**eig)
    return rows

def freeze_protocol():
    dest=OUT/'Protocol.json'
    if dest.exists():
        p=json.loads(dest.read_text()); assert p['source_sha256']==sha(__file__)
        assert p['specifications']==specifications(); return p
    data=OUT/'data'; data.mkdir(parents=True,exist_ok=True)
    source=ROOT/'recovered/shape_restore/baselines/optimizer_baselines_v1/data'
    for name in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz','split.npz']:
        shutil.copyfile(source/name,data/name)
    p={'name':'Experiment 01: MNIST shallow MLP; magnitude-only AdamW graft',
       'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'user_date':'2026-11-13','source_sha256':sha(__file__),
       'architecture':[784,256,256,10],'activation':'ReLU','epochs':EPOCHS,'warmup_epochs':5,
       'schedule':'5 epoch linear warmup; cosine to 10% peak at epoch 120',
       'seeds':SEEDS,'batch_sizes':[128,256,512],'lr_multipliers':[.5,1.,2.],
       'adam_layer_rates':[.001]*3,'sgdm_magnitude_layer_rates':RATES,
       'momentum':.9,'beta2':.999,'epsilon':1e-8,'arms':ARMS,
       'graft':'Only matrix magnitude changes: D_graft=norm(-eta_S*m/(1-beta1))*D_Adam/norm(D_Adam); native AdamW bias updates; target history belongs to this same graft run.',
       'sgdm':'SGD momentum .9, no Nesterov or dampening, layer rates [.04,.2,.8] times multiplier; same decay as all other arms.',
       'decay':'Every parameter multiplied by (1-1e-5*schedule)^(actual_batch_size/256) each step. Equal cumulative decay per epoch across batch sizes, common across arms and rate multipliers. At B256 equivalent to original AdamW peak LR .001 and WD .01. Effective WD is adjusted when LR changes to isolate update changes.',
       'data':'MNIST original split: 55000 train, 5000 validation, 10000 test; pixels /255; no augmentation.',
       'data_hashes':{q.name:sha(q) for q in data.iterdir()},
       'initialization':'PyTorch Linear reset_parameters; same seed/state across all arms and settings.',
       'paired_order':'SeedSequence([seed,epoch,711]) permutation; same examples and order per epoch; batch sizes only change grouping.',
       'primary_endpoint':'Fixed epoch120, every setting; no alpha targeting, no best-checkpoint or test-based selection.',
       'primary_outcomes':['FC1 alpha','FC2 alpha','mean(abs(hidden_alpha-2))','test accuracy','test cross entropy'],
       'secondary_outcomes':['equal-layer mean alpha','hidden alpha population variance','fraction hidden alpha<2','fixed-rank alpha at 20% and 50%','head alpha exploratory only'],
       'analysis':'Paired differences within seed/setting; all 9 settings reported. Overall effect first average the 9 differences within each seed, then mean and Student t 95% CI over 5 independent seeds. Seed SD, not layer-as-replicate errors. CIs descriptive and not multiplicity-adjusted.',
       'spectra':'WeightWatcher default tail selection, no alpha-targeted cutoff; save raw squared singular values and WW fit fields. Head has only10 eigenvalues and excluded from primary hidden means. Support sensitivity: >=50 eigenvalues, >=20 tail eigenvalues, KS<=.2.',
       'monitors':list(range(0,121,10)),'checkpoints':[0,30,60,90,120],
       'test_evaluation_epochs':[0,30,60,90,120],
       'stat_logging':'First two and each tenth batch of every epoch; save gradients/momentum/update norms and directional/radial summaries for all matrices.',
       'maximum_training_seconds':16200,'operational_limit':'One experiment only. Stop and preserve any partial run if deadline reached; do not start another architecture or dataset.',
       'runtime_estimate_evidence':'60-step timing with old verified updater: per120epochs AdamW679/418/180 seconds at B128/256/512; graft758/398/248. New runner logs less often; six workers planned, with conservative deadline.',
       'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'weightwatcher':ww.__version__},
       'specifications':specifications()}
    writej(dest,p); return p

def preflight():
    global DATA
    freeze_protocol(); DATA=load_data(); x,y,xv,yv,xt,yt=DATA
    assert [len(y),len(yv),len(yt)]==[55000,5000,10000]
    split=np.load(OUT/'data/split.npz')
    assert len(np.intersect1d(split['train'],split['validation']))==0
    audit=[]
    # Independent PyTorch references, with common prescribed gradients.
    for dtype in [torch.float64,torch.float32]:
        for arm in ARMS:
            spec={'arm':arm,'multiplier':1.,'batch_size':128,'seed':31999}
            torch.manual_seed(31999); net=MLP().to(dtype); ref=copy.deepcopy(net)
            opt=make_opt(net,spec); ropt=make_opt(ref,spec)
            max_error=0.; norm_error=0.; direction_error=0.; identity_error=0.
            for t in range(1,9):
                f=schedule(t); decay=(1-1e-5*f)**.5
                old={n:p.detach().clone() for n,p in ref.named_parameters()}
                for p,q in zip(net.parameters(),ref.parameters()):
                    g=torch.randn_like(p);p.grad=g.clone();q.grad=g.clone()
                stats=apply_update(net,opt,spec,f,128,True)
                # AdamW reference implements exact factor through its own weight_decay.
                for group,rate in zip(ropt.param_groups,RATES if arm=='sgdm' else [.001]*3):
                    group['lr']=rate*f
                    if arm!='sgdm': group['weight_decay']=(1-decay)/group['lr']
                if arm=='sgdm':
                    with torch.no_grad():
                        for q in ref.parameters():q.mul_(decay)
                ropt.step()
                if arm=='magnitude_graft':
                    with torch.no_grad():
                        for name,rate,g in zip(LAYERS,RATES,ropt.param_groups):
                            q=getattr(ref,name).weight;state=ropt.state[q]
                            mom=state['exp_avg']/.1
                            adam=-g['lr']*(state['exp_avg']/(1-.9**t))/((state['exp_avg_sq']/(1-.999**t)).sqrt()+1e-8)
                            wanted=rate*f*torch.linalg.vector_norm(mom.double())
                            d=adam/torch.linalg.vector_norm(adam.double())*wanted
                            q.copy_(old[name+'.weight']*decay+d)
                            actual=getattr(net,name).weight-old[name+'.weight']*decay
                            norm_error=max(norm_error,float(abs(torch.linalg.vector_norm(actual.double())/wanted-1)))
                            direction_error=max(direction_error,float(abs(F.cosine_similarity(actual.double().flatten(),adam.double().flatten(),dim=0)-1)))
                for p,q in zip(net.parameters(),ref.parameters()): max_error=max(max_error,float((p-q).abs().max()))
                if arm=='sgdm':
                    for name,rate in zip(LAYERS,RATES):
                        d=-rate*f*opt.state[getattr(net,name).weight]['momentum_buffer']
                        identity_error=max(identity_error,float((d-d/d.norm()*d.norm()).abs().max()))
            tolerance=2e-12 if dtype==torch.float64 else 3e-5
            assert max_error<tolerance,(dtype,arm,max_error)
            assert norm_error<2e-5 and direction_error<2e-5,(dtype,arm,norm_error,direction_error)
            assert identity_error<tolerance
            audit.append({'dtype':str(dtype),'arm':arm,'max_reference_parameter_error':max_error,'max_graft_norm_relative_error':norm_error,'max_graft_direction_cosine_error':direction_error,'sgdm_identity_graft_max_error':identity_error})
    # Check save/load replay, and collect actual short training timings.
    timing=[]; replay=[]
    for arm in ARMS:
        spec={'arm':arm,'multiplier':1.,'batch_size':256,'seed':31999}
        torch.manual_seed(spec['seed']);net=MLP();opt=make_opt(net,spec)
        for i in range(5): train_step(net,opt,spec,x[i*256:(i+1)*256],y[i*256:(i+1)*256],.2)
        ck={'model':copy.deepcopy(net.state_dict()),'opt':copy.deepcopy(opt.state_dict())}
        tic=time.perf_counter()
        for i in range(20): train_step(net,opt,spec,x[i*256:(i+1)*256],y[i*256:(i+1)*256],.2)
        sec=(time.perf_counter()-tic)/20
        net2=MLP();opt2=make_opt(net2,spec);net2.load_state_dict(ck['model']);opt2.load_state_dict(ck['opt'])
        for i in range(20): train_step(net2,opt2,spec,x[i*256:(i+1)*256],y[i*256:(i+1)*256],.2)
        assert weight_hash(net)==weight_hash(net2)
        replay.append({'arm':arm,'20_step_replay_identical':True})
        timing.append({'arm':arm,'batch':256,'seconds_per_step':sec})
    # Tail exponent must be unchanged by uniform scalar weight dilation.
    d=OUT/'preflight';(d/'spectra').mkdir(parents=True,exist_ok=True)
    a=spectra(net,d,0)
    with torch.no_grad():
        for name in LAYERS:getattr(net,name).weight.mul_(3)
    b=spectra(net,d,1)
    alpha_error=max(abs(a[n]['alpha']-b[n]['alpha']) for n in LAYERS)
    assert alpha_error<2e-5
    init_hashes={}
    for seed in SEEDS:
        hs=[]
        for arm in ARMS:
            torch.manual_seed(seed);hs.append(weight_hash(MLP()))
        assert len(set(hs))==1;init_hashes[str(seed)]=hs[0]
    decay_totals={str(b):sum(math.log((1-1e-5*.7)**(min(b,55000-i)/256)) for i in range(0,55000,b)) for b in [128,256,512]}
    assert max(decay_totals.values())-min(decay_totals.values())<1e-12
    result={'passed':True,'reference_update_checks':audit,'resume_replay':replay,'seconds_per_step':timing,
            'radial_scaling_max_alpha_error':alpha_error,'initial_hashes':init_hashes,
            'same_decay_per_epoch_log_factors':decay_totals,'data_split_disjoint':True,
            'source_sha256':sha(__file__)}
    writej(OUT/'Preflight_Audit.json',result);print(json.dumps(result),flush=True)

def train(spec,deadline):
    dest=OUT/'runs'/run_id(spec)
    for p in [dest,dest/'spectra',dest/'checkpoints',dest/'statistics']:p.mkdir(parents=True,exist_ok=True)
    if (dest/'done.json').exists():return json.loads((dest/'done.json').read_text())
    torch.manual_seed(spec['seed']);net=MLP();opt=make_opt(net,spec);initial_hash=weight_hash(net)
    config={**spec,'epochs':EPOCHS,'initial_hash':initial_hash,'source_sha256':sha(__file__)}
    if (dest/'config.json').exists(): assert json.loads((dest/'config.json').read_text())==config
    else:writej(dest/'config.json',config)
    x,y,xv,yv,xt,yt=DATA;tic=time.time();history=[];start=0;status='complete'
    if (dest/'resume.pt').exists():
        ck=torch.load(dest/'resume.pt',weights_only=False);assert ck['config']==config
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng'])
        start=ck['epoch']+1;history=ck['history'];tic=time.time()-ck['elapsed_seconds']
    for epoch in range(start,EPOCHS+1):
        sampled=[];ordhash=None;factor=schedule(epoch)
        if epoch:
            if time.time()>deadline:status='time_limit';break
            net.train();ixs=order(spec['seed'],epoch);ordhash=hashlib.sha256(ixs.tobytes()).hexdigest()
            for j,st in enumerate(range(0,len(y),spec['batch_size'])):
                ix=ixs[st:st+spec['batch_size']];rec=j<2 or j%10==0
                try:loss,stats=train_step(net,opt,spec,x[ix],y[ix],factor,rec)
                except FloatingPointError:
                    status='nonfinite';break
                if rec:sampled.append([j,loss,stats])
            if status!='complete':break
            np.savez_compressed(dest/f'statistics/epoch_{epoch:03}.npz',batch_index=[r[0] for r in sampled],loss=[r[1] for r in sampled],values=np.array([r[2] for r in sampled],np.float32),columns=STAT_NAMES,layers=LAYERS)
        row={'epoch':epoch,'schedule':factor,'batch_order_sha256':ordhash,'elapsed_seconds':time.time()-tic}
        if epoch%10==0:
            row['train']=evaluate(net,x,y);row['validation']=evaluate(net,xv,yv)
            row['spectra']=spectra(net,dest,epoch)
        if epoch%30==0:row['test']=evaluate(net,xt,yt)
        if sampled:
            aa=np.array([r[2] for r in sampled]);row['update_statistics']={n:{k:clean(aa[:,i,j].mean()) for j,k in enumerate(STAT_NAMES)} for i,n in enumerate(LAYERS)}
        history.append(row)
        ck={'model':net.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
            'epoch':epoch,'config':config,'history':history,'elapsed_seconds':time.time()-tic}
        savept(dest/'resume.pt',ck)
        if epoch in [0,30,60,90,120]:savept(dest/f'checkpoints/epoch_{epoch:03}.pt',ck)
        writej(dest/'history.json',history)
        if epoch in [30,60,90,120]:print(run_id(spec),epoch,'alpha',[round(row['spectra'][n]['alpha'],3) for n in LAYERS],'val',round(row['validation']['accuracy'],4),flush=True)
    result={'spec':spec,'status':status,'last_epoch':history[-1]['epoch'] if history else -1,'seconds':time.time()-tic,'initial_hash':initial_hash,'final_weight_hash':weight_hash(net),'final':history[-1] if history else None,'path':str(dest.relative_to(OUT))}
    if status=='complete':
        test,logits=evaluate(net,xt,yt,True)
        assert test==history[-1]['test']
        np.savez_compressed(dest/'test_logits_epoch_120.npz',logits=logits,labels=yt.numpy())
        writej(dest/'done.json',result);(dest/'resume.pt').unlink(missing_ok=True)
    else:
        savept(dest/'stopped_state.pt',{'model':net.state_dict(),'optimizer':opt.state_dict(),'config':config,'status':status})
        writej(dest/'stopped.json',result)
    return result

def run(workers):
    global DATA
    p=freeze_protocol();a=json.loads((OUT/'Preflight_Audit.json').read_text())
    assert a['passed'] and a['source_sha256']==sha(__file__)
    DATA=load_data();start=time.time();deadline=start+p['maximum_training_seconds']
    writej(OUT/'Launch.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':workers,'deadline_epoch_seconds':deadline,'planned':len(specifications()),'next_experiment_authorized':False})
    results=[];failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        pending={pool.submit(train,s,deadline):s for s in specifications()}
        for future in concurrent.futures.as_completed(pending):
            spec=pending[future]
            try:results.append(future.result())
            except Exception as e:failures.append({'spec':spec,'exception':repr(e)})
            counts={k:sum(r['status']==k for r in results) for k in ['complete','time_limit','nonfinite']}
            writej(OUT/'Run_Register.json',{'planned':len(pending),'returned':len(results),'counts':counts,'failures':failures,'elapsed_seconds':time.time()-start})
            print('REGISTER',len(results),counts,'failures',len(failures),flush=True)
    writej(OUT/'Completion.json',{'planned':len(specifications()),'completed':sum(r['status']=='complete' for r in results),'results':[{'spec':r['spec'],'status':r['status']} for r in results],'failures':failures,'seconds':time.time()-start,'next_experiment_started':False})
    print('EXPERIMENT01 STOPPED. NO NEXT EXPERIMENT.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    preflight() if args.preflight else run(args.workers)
