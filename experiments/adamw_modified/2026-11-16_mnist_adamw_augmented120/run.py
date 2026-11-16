"""Two augmented digit-MNIST native AdamW trajectories; every epoch preserved."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, time, copy, hashlib, shutil, urllib.request, traceback
import concurrent.futures as cf
import multiprocessing, platform, inspect
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from schedule import factor, accounting
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PARENT = ROOT.parent / '2026-11-13_mnist_mlp_experiment01'
KERNEL = PARENT / 'frozen'
sys.path.insert(0, str(KERNEL))
import study as s
s.OUT = ROOT
PLAN = json.loads((ROOT / 'Protocol.json').read_text())

def ident(spec):
    return ('augmented' if spec['augmentation'] else 'unaugmented') + '/seed_' + str(spec['seed'])

def small_n(eigenvalues):
    from weightwatcher.WW_powerlaw import WWFit
    fit = object.__new__(WWFit)
    fit.data = np.sort(np.asarray(eigenvalues, float)); fit.N = len(fit.data)
    fit.fit_powerlaw_smallN(k_min=8, lambda_prior=0.)
    return dict(alpha=float(fit.alpha), xmin=float(fit.xmin), sigma=float(fit.sigma),
                D=float(fit.D), n_tail=len(fit.data))

def spectrum(net, d, epoch):
    rows = s.spectra(net, d, epoch)
    lam = np.load(d / f'spectra/epoch_{epoch:03}.npz')['fc3']
    rows['fc3']['small_n'] = small_n(lam)
    rows['fc3']['inference_support'] = '10 correlated eigenvalues; exploratory only'
    return rows  # Native alpha remains alpha; small-N is a separate named estimate.

def same(a, b):
    if isinstance(a, s.torch.Tensor): return isinstance(b, s.torch.Tensor) and s.torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

def checkpoint(d, name, net, opt, spec, epoch, initial):
    s.savept(d / name, dict(model=net.state_dict(), optimizer=opt.state_dict(),
        torch_rng=s.torch.get_rng_state(), spec=spec, epoch=epoch, initial_hash=initial,
        model_hash=s.weight_hash(net), accounting=accounting(epoch),
        kernel_sha256=s.sha(KERNEL/'study.py'), runner_sha256=s.sha(ROOT/'run.py')))

def restore(ck):
    net = s.MLP(); net.load_state_dict(ck['model'])
    opt = s.make_opt(net, ck['spec']); opt.load_state_dict(copy.deepcopy(ck['optimizer']))
    s.torch.set_rng_state(ck['torch_rng'])
    assert s.weight_hash(net) == ck['model_hash'] and same(opt.state_dict(), ck['optimizer'])
    assert all(float(st['step']) == ck['epoch'] * 215 for st in opt.state.values())
    return net, opt

def batch(x, spec, epoch, j):
    if spec['augmentation']: return augment_batch(x, spec['seed'], epoch, j)
    return x, 'unaugmented'

def replay(d, origin, negative=False):
    ck = s.torch.load(d/f'epoch_{origin:04}.pt', weights_only=False)
    sp = ck['spec']; epoch = origin + 1; x, y, *_ = s.DATA
    saved = np.load(d/f'statistics/epoch_{epoch:04}.npz')
    def attempt(wrong=None):
        net, opt = restore(ck)
        if wrong == 'moment':
            for state in opt.state.values(): state['exp_avg'].zero_()
        order = s.order(sp['seed'] + (1 if wrong == 'minibatch' else 0), epoch)
        losses = []; stats = []
        for j in range(2):
            ix = order[j*256:(j+1)*256]; xb, _ = batch(x[ix], sp, epoch, j)
            loss, st = s.train_step(net, opt, sp, xb, y[ix], factor(epoch) * (1.1 if wrong == 'rate' else 1), True)
            losses.append(loss); stats.append(st)
        matches = np.array_equal(np.array(losses), saved['loss'][:2]) and np.array_equal(
            np.array(stats, np.float32), saved['values'][:2], equal_nan=True)
        return matches, net.state_dict(), opt.state_dict()
    a = attempt(); b = attempt()
    assert a[0] and b[0] and same(a[1], b[1]) and same(a[2], b[2]), 'Exact restored replay failed'
    wrongs = []
    if negative:
        for kind in ['minibatch', 'moment', 'rate']:
            bad = attempt(kind); rejected = not bad[0] and not same(a[1], bad[1])
            assert rejected, f'Wrong {kind} control not rejected'
            wrongs.append(dict(kind=kind, rejected=True))
    return dict(origin=origin, exact_recorded_losses_and_statistics=True,
                two_restored_copies_bitwise_model_and_optimizer=True, negative_controls=wrongs)

def native_contract():
    s.torch.manual_seed(33997); net=s.MLP(); ref=copy.deepcopy(net)
    sp=dict(seed=33997,batch_size=256,multiplier=1.,arm='adamw',augmentation=False)
    opt=s.make_opt(net,sp)
    ropt=s.torch.optim.AdamW([{'params':list(getattr(ref,n).parameters()),'lr':.001,'layer':n} for n in s.LAYERS],
                            betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
    x,y,*_=s.DATA; checks=[]
    for i in range(5):
        n=216 if i==4 else 256; xb=x[i*256:i*256+n]; yb=y[i*256:i*256+n]; f=.1 if i>2 else .6
        before={k:p.detach().clone() for k,p in net.named_parameters()}
        ropt.zero_grad(set_to_none=True); loss=s.F.cross_entropy(ref(xb),yb);loss.backward()
        with s.torch.no_grad():
            for p in ref.parameters():p.mul_((1-1e-5*f)**(n/256))
        for group in ropt.param_groups:group['lr']=.001*f
        ropt.step()
        _, stats=s.train_step(net,opt,sp,xb,yb,f,True)
        for (name,p),(_,q) in zip(net.named_parameters(),ref.named_parameters()):
            assert s.torch.equal(p,q) and same(opt.state[p],ropt.state[q])
            movement=float(s.frobenius(p-before[name]));assert movement>0
            checks.append(dict(step=i,parameter=name,nonzero_movement=movement,reference_exact=True))
        assert all(row[5]==1. and row[2]==row[4] for row in stats)
    s.writej(ROOT/'Native_AdamW_Contract.json',dict(passed=True,nonzero_parameter_checks=len(checks),
        reference='Separate torch.optim.AdamW path with explicit historical sample-normalized decay; no graft',records=checks))

def estimator_audit():
    from weightwatcher.WW_powerlaw import WWFit
    source=inspect.getfile(WWFit);assert s.sha(source)==PLAN['fc3_estimator']['source_sha256']
    rng=np.random.default_rng(431);errors=[]
    for alpha in [1.5,2.,3.,5.]:
        for _ in range(25):
            x=1+rng.pareto(alpha-1,10);a=small_n(x);b=small_n(x*1e4);tail=x[x>=a['xmin']]
            err=max(abs(1+(len(tail)-1)/np.log(tail/a['xmin']).sum()-a['alpha']),abs(a['alpha']-b['alpha']))
            assert err<1e-9 and a['n_tail']>=8;errors.append(err)
    s.writej(ROOT/'Estimator_Audit.json',dict(passed=True,controls=100,maximum_error=max(errors),
        source_sha256=s.sha(source),alpha_prior=False,fc3_remains_exploratory=True))

def train(sp, deadline, smoke=False):
    assert sp['arm']=='adamw' and sp['multiplier']==1. and sp['batch_size']==256 and sp['augmentation'] is True
    d=ROOT/('smoke' if smoke else 'runs')/ident(sp);assert not d.exists(),'Do not overwrite or restart a trajectory'
    for sub in ['spectra','statistics']:(d/sub).mkdir(parents=True,exist_ok=True)
    s.torch.manual_seed(sp['seed']);net=s.MLP();opt=s.make_opt(net,sp);initial=s.weight_hash(net)
    x,y,xv,yv,xt,yt=s.DATA;end=2 if smoke else PLAN['fixed_epochs'];history=[];started=time.time();max_fit_error=0.
    for epoch in range(end+1):
        assert time.time()<deadline,'Frozen compute deadline reached'
        tic=time.time();samples=[];order_hash=None;aug_digest=hashlib.sha256();batch_loss=0.
        if epoch:
            net.train();order=s.order(sp['seed'],epoch);order_hash=hashlib.sha256(order.tobytes()).hexdigest()
            for j,start in enumerate(range(0,len(y),256)):
                assert time.time()<deadline,'Frozen compute deadline reached'
                ix=order[start:start+256];xb,ah=batch(x[ix],sp,epoch,j);aug_digest.update(ah.encode());record=True
                loss,stats=s.train_step(net,opt,sp,xb,y[ix],factor(epoch),record);batch_loss+=loss*len(ix)
                if record:
                    assert all(row[5]==1. and row[2]==row[4] for row in stats),'Unexpected graft'
                    samples.append((j,loss,stats))
            np.savez_compressed(d/f'statistics/epoch_{epoch:04}.npz',batch_index=[r[0] for r in samples],
                                loss=[r[1] for r in samples],values=np.array([r[2] for r in samples],np.float32))
        val,vlog=evaluate(net,xv,yv)
        assert all(s.torch.isfinite(p).all() for p in net.parameters())
        monitored=True
        row=dict(epoch=epoch,validation=val,online_training_loss=batch_loss/55000 if epoch else None,
                 batch_order_sha256=order_hash,augmentation_sha256=aug_digest.hexdigest(),**{k:v for k,v in accounting(epoch).items() if k!='epoch'})
        if monitored:
            row['train_unaugmented']=s.evaluate(net,x,y);row['spectra']=spectrum(net,d,epoch)
            max_fit_error=max(max_fit_error,fit_audit(row['spectra'],d/f'spectra/epoch_{epoch:03}.npz'))
        if epoch in PLAN['checkpoints'] or epoch==end:
            checkpoint(d,f'epoch_{epoch:04}.pt',net,opt,sp,epoch,initial)
            np.savez_compressed(d/f'validation_logits_{epoch:04}.npz',logits=vlog,labels=yv.numpy())
        checkpoint(d,'last.pt',net,opt,sp,epoch,initial)
        row['seconds']=time.time()-tic;history.append(row);s.writej(d/'history.json',history)
        if epoch%10==0:print(ident(sp),'epoch',epoch,'val_CE',val['loss'],'val_acc',val['accuracy'],'lr',row['lr'],flush=True)
    shutil.copyfile(d/'last.pt',d/'final.pt')
    replays=[replay(d,o,negative=smoke) for o in ([1] if smoke else [1,60,119])]
    ck=s.torch.load(d/'final.pt',weights_only=False);net,opt=restore(ck)
    restored_val,_=evaluate(net,xv,yv);assert restored_val==history[-1]['validation']
    rd=d/'restored';(rd/'spectra').mkdir(parents=True,exist_ok=True);sp_final=spectrum(net,rd,end)
    a=np.load(d/f'spectra/epoch_{end:03}.npz');b=np.load(rd/f'spectra/epoch_{end:03}.npz')
    assert all(np.array_equal(a[n],b[n]) for n in s.LAYERS) and same(sp_final,history[-1]['spectra'])
    result=dict(spec=sp,endpoint_epoch=end,initial_hash=initial,final_model_hash=ck['model_hash'],
        status='USABLE' if restored_val['accuracy']>=PLAN['functional_accuracy_floor'] and sum(v>0 for v in restored_val['prediction_counts'])>1 else 'FUNCTIONAL_FAILURE',
        validation=restored_val,spectra=sp_final,seconds=time.time()-started,
        verification=dict(passed=True,replays=replays,restored_validation_and_spectra_exact=True,
            maximum_independent_alpha_error=max_fit_error,fixed_endpoint=True,native_update_only=True,
            alpha_selection=False,validation_selection=False,test_selection=False),
        checkpoint_hashes={p.name:s.sha(p) for p in d.glob('*.pt')})
    if not smoke:
        # Unseal test metrics only after all120training epochs for this trajectory have finished.
        results=[]
        for e in PLAN['test_epochs']:
            saved=s.torch.load(d/f'epoch_{e:04}.pt',weights_only=False);test_net,_=restore(saved)
            tr,logits=evaluate(test_net,xt,yt)
            np.savez_compressed(d/f'test_logits_{e:04}.npz',logits=logits,labels=yt.numpy())
            results.append(dict(epoch=e,test=tr,checkpoint_sha256=s.sha(d/f'epoch_{e:04}.pt')))
        result['test_milestones']=results
    s.writej(d/'done.json',result);return result

def guarded(sp,deadline):
    try:return train(sp,deadline)
    except Exception as exc:
        err=dict(spec=sp,status='ERROR',error=repr(exc),traceback=traceback.format_exc())
        s.writej(ROOT/'runs'/ident(sp)/'failure.json',err);return err

def preflight():
    assert not (ROOT/'Launch.json').exists(),'Duplicate launch prohibited'
    now=time.time();s.writej(ROOT/'Launch.json',dict(started_epoch_seconds=now,deadline_epoch_seconds=now+14400,
        planned_runs=2,fixed_epochs=120,github_run_id=os.getenv('GITHUB_RUN_ID')))
    deadline=now+14400;prepare();native_contract();estimator_audit()
    smoke=[]
    for aug in [True]:
        smoke.append(train(dict(seed=33998,batch_size=256,multiplier=1.,arm='adamw',augmentation=aug),deadline,True))
    hs=[json.loads((ROOT/'smoke'/ident(r['spec'])/'history.json').read_text()) for r in smoke]
    initial_hashes={}
    for seed in PLAN['seeds']:
        s.torch.manual_seed(seed);a=s.weight_hash(s.MLP())
        s.torch.manual_seed(seed);b=s.weight_hash(s.MLP())
        assert a==b;initial_hashes[str(seed)]=a
    assert len(set(initial_hashes.values()))==2
    s.writej(ROOT/'Initialization_Audit.json',dict(passed=True,independent_seeds=2,initial_hashes=initial_hashes))
    for row in hs[0][1:]:
        assert row['batch_order_sha256']==hashlib.sha256(s.order(33998,row['epoch']).tobytes()).hexdigest()
    # Same deterministic augmentation implementation, changing actual training inputs.
    x=s.DATA[0][:256];aa,ha=augment_batch(x,33998,1,0);bb,hb=augment_batch(x,33998,1,0)
    assert ha==hb and s.torch.equal(aa,bb) and not s.torch.equal(aa,x)
    u,hu=batch(x,dict(seed=33998,augmentation=False),1,0);assert s.torch.equal(u,x) and hu=='unaugmented'
    # Conservative total serial-work estimate also covers two-worker contention.
    estimate=2*120*max(h[-1]['seconds'] for h in hs)*1.5
    passed=time.time()+estimate<deadline
    s.writej(ROOT/'Preflight.json',dict(passed=passed,smoke_runs=1,excluded_seed=33998,
        reproducible_initialization_and_order=True,augmentation_deterministic_and_nontrivial=True,
        exact_replays=1,wrong_controls_rejected=3,native_nonzero_parameter_checks=30,
        estimated_training_seconds=estimate,remaining_budget_seconds=deadline-time.time(),
        alpha_or_accuracy_not_a_gate=True,results=smoke))
    s.writej(ROOT/'Runtime.json',dict(python=platform.python_version(),platform=platform.platform(),
        torch=s.torch.__version__,numpy=np.__version__,weightwatcher=s.ww.__version__))
    assert passed,'Throughput exceeds frozen budget; preserve evidence without shortening endpoint'
    s.writej(ROOT/'Training_Freeze.json',dict(epoch_seconds=time.time(),protocol_sha256=s.sha(ROOT/'Protocol.json'),
        runner_sha256=s.sha(ROOT/'run.py'),kernel_sha256=s.sha(KERNEL/'study.py'),fixed_epochs=120,planned_runs=2))

def main():
    if '--preflight-only' in sys.argv:preflight();return
    assert '--continue' in sys.argv and not (ROOT/'Run_Register.json').exists(),'No duplicate continuation'
    assert json.loads((ROOT/'Preflight.json').read_text())['passed']
    freeze=json.loads((ROOT/'Training_Freeze.json').read_text())
    assert freeze['runner_sha256']==s.sha(ROOT/'run.py') and freeze['protocol_sha256']==s.sha(ROOT/'Protocol.json')
    prepare();launch=json.loads((ROOT/'Launch.json').read_text());deadline=launch['deadline_epoch_seconds'];results=[]
    with cf.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(guarded,sp,deadline):sp for sp in PLAN['specifications']}
        while futures:
            ready,_=cf.wait(futures,timeout=30,return_when=cf.FIRST_COMPLETED)
            for f in ready:results.append(f.result());futures.pop(f)
            progress=[]
            for sp in PLAN['specifications']:
                path=ROOT/'runs'/ident(sp)/'history.json'
                if path.exists():
                    h=json.loads(path.read_text());progress.append(dict(spec=sp,last_epoch=h[-1]['epoch']))
            s.writej(ROOT/'Run_Register.json',dict(planned=2,completed=sum('test_milestones' in r for r in results),
                returned=len(results),training_active=bool(futures),progress=progress,errors=[r for r in results if r['status']=='ERROR']))
    complete=sum('test_milestones' in r for r in results)
    s.writej(ROOT/'Completion.json',dict(planned=2,completed=complete,results=results,
        seconds=time.time()-launch['started_epoch_seconds'],fixed_epochs=120))
    assert complete==2,'Partial experiment; preserve all available states'

def augment_batch(x,seed,epoch,batch_index):
 rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,batch_index,991]))
 shifts=rng.integers(-2,3,size=(len(x),2));apply=rng.random(len(x))<.5
 shifts[~apply]=0
 image=x.reshape(-1,28,28);out=s.torch.zeros_like(image)
 for dy in range(-2,3):
  for dx in range(-2,3):
   ix=np.flatnonzero((shifts[:,0]==dy)&(shifts[:,1]==dx))
   if len(ix):out[ix,max(0,dy):min(28,28+dy),max(0,dx):min(28,28+dx)]=image[ix,max(0,-dy):min(28,28-dy),max(0,-dx):min(28,28-dx)]
 return out.reshape(-1,784),hashlib.sha256(shifts.tobytes()).hexdigest()

def prepare():
 assert s.sha(KERNEL/'study.py')==PLAN['kernel_sha256']
 (ROOT/'data').mkdir(exist_ok=True)
 provenance=json.loads((ROOT/'Data_Provenance.json').read_text())
 original=json.loads((PARENT/'frozen/Protocol.json').read_text())
 assert provenance['dataset']=='MNIST' and PLAN['dataset']=='MNIST'
 assert provenance['split_sha256']==original['data_hashes']['split.npz']
 for row in provenance['files']:
  assert row['sha256']==original['data_hashes'][row['name']]
  dest=ROOT/'data'/row['name'];local=PARENT/'data'/row['name']
  if not dest.exists():
   if local.exists():shutil.copyfile(local,dest)
   else:dest.write_bytes(urllib.request.urlopen(row['download_url'],timeout=180).read())
  assert s.sha(dest)==row['sha256']
 dest=ROOT/'data/split.npz'
 if not dest.exists():shutil.copyfile(REPO/provenance['split_source_repo_path'],dest)
 assert s.sha(dest)==provenance['split_sha256']
 z=np.load(dest);assert len(np.intersect1d(z['train'],z['validation']))==0 and len(z['train'])==55000 and len(z['validation'])==5000
 s.DATA=s.load_data()
 s.writej(ROOT/'Data_Audit.json',{'passed':True,'dataset':'MNIST','split_sha256':s.sha(dest),'kernel_sha256':s.sha(KERNEL/'study.py'),
   'data_hashes':{f.name:s.sha(f) for f in (ROOT/'data').iterdir()},'original_data_hashes_exact':True})

def evaluate(net,x,y):
 row,logits=s.evaluate(net,x,y,True)
 raw=logits.astype(float);loss=float(np.mean(logsumexp(raw,axis=1)-raw[np.arange(len(y)),y.numpy()]))
 assert abs(loss-row['loss'])<2e-6
 row['independent_cross_entropy_error']=abs(loss-row['loss']);row['prediction_counts']=np.bincount(logits.argmax(1),minlength=10).tolist()
 return row,logits

def fit_audit(rows,filename):
 errors=[];z=np.load(filename)
 for name in s.LAYERS:
  r=rows[name]
  if r['alpha'] is None or len(z[name])<50:continue # Output layer retained but not a supported spectral inference
  x=z[name].astype(float)*r['xmax']/z[name].max();tail=x[(x>=r['xmin']*(1-1e-10))&(x<=r['xmax']*(1+1e-10))]
  if len(tail)<2:continue
  estimate=1+len(tail)/np.log(tail/r['xmin']).sum()
  assert len(tail)==r['num_pl_spikes'] and abs(estimate-r['alpha'])<1e-8
  errors.append(abs(estimate-r['alpha']))
 return max(errors,default=0.)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        s.writej(ROOT/'Failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
