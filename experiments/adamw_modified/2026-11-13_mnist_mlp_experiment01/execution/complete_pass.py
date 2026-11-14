"""Resume the predeclared pass using the unchanged historical training module.

No new settings, seeds, epochs, or optimizer rules are introduced here.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import time,json,hashlib,datetime,concurrent.futures,multiprocessing,platform
from pathlib import Path
import study as s

def validate_restoration():
    p=s.OUT;protocol=json.loads((p/'Protocol.json').read_text())
    assert s.sha(p/'study.py')==protocol['source_sha256']
    for key,value in [('torch',s.torch.__version__),('numpy',s.np.__version__),('weightwatcher',s.ww.__version__)]:
        assert value==protocol['versions'][key],(key,value,protocol['versions'][key])
    for name,digest in protocol['data_hashes'].items():assert s.sha(p/'data'/name)==digest
    s.DATA=s.load_data();x,y,*_=s.DATA;checks=[]
    initial=json.loads((p/'Preflight_Audit.json').read_text())['initial_hashes']
    for seed in s.SEEDS:
        s.torch.manual_seed(seed);assert s.weight_hash(s.MLP())==initial[str(seed)]
    for f in sorted((p/'runs').glob('*/*/seed_*/done.json')):
        d=f.parent;done=json.loads(f.read_text());spec=done['spec']
        ck=s.torch.load(d/'checkpoints/epoch_120.pt',weights_only=False)
        net=s.MLP();net.load_state_dict(ck['model']);assert s.weight_hash(net)==done['final_weight_hash']
        dest=p/'restoration_checks'/s.run_id(spec);(dest/'spectra').mkdir(parents=True,exist_ok=True)
        sp=s.spectra(net,dest,120)
        err=max(abs(sp[n]['alpha']-done['final']['spectra'][n]['alpha']) for n in s.LAYERS);assert err<1e-10
        ck=s.torch.load(d/'checkpoints/epoch_030.pt',weights_only=False)
        net.load_state_dict(ck['model']);opt=s.make_opt(net,spec);opt.load_state_dict(ck['optimizer']);s.torch.set_rng_state(ck['torch_rng'])
        ix=s.order(spec['seed'],31)[:spec['batch_size']]
        loss,stat=s.train_step(net,opt,spec,x[ix],y[ix],s.schedule(31),True)
        saved=s.np.load(d/'statistics/epoch_031.npz')
        actual=s.np.array(stat,s.np.float32);expected=saved['values'][0]
        loss_error=abs(loss-float(saved['loss'][0]))
        assert loss_error<1e-7
        assert s.np.allclose(actual,expected,rtol=1e-5,atol=1e-9,equal_nan=True)
        mask=s.np.isfinite(actual)&s.np.isfinite(expected)
        diff=s.np.abs(actual[mask]-expected[mask])
        checks.append({'spec':spec,'final_hash_match':True,'alpha_error':err,'first_update_replay_exact':bool(s.np.array_equal(actual,expected,equal_nan=True)),
                       'max_update_stat_abs_error':float(diff.max()),'max_update_stat_relative_error':float((diff/s.np.maximum(s.np.abs(expected[mask]),1e-15)).max()),'loss_abs_error':loss_error})
    s.writej(p/'Pass_Restoration_Audit.json',{'passed':True,'checks':checks,'source_sha256':s.sha(p/'study.py'),'versions_match':True,'all_five_initializations_match':True,'all_data_hashes_match':True,
           'platform':platform.platform(),'replay_tolerance':{'loss_abs':1e-7,'statistics_rtol':1e-5,'statistics_atol':1e-9},
           'numerical_note':'Before new training, cross-environment replay showed <=4.77e-7 absolute statistic differences, <=6.99e-6 relative differences and <=1.31e-8 loss differences. Model hashes and initializations match. Tolerance applies to validation only; frozen training code is unchanged.'})
    print('RESTORATION VERIFIED',len(checks),flush=True)

def main():
    validate_restoration();p=s.OUT
    before={str(f.relative_to(p)):s.sha(f) for f in (p/'runs').glob('*/*/seed_*/done.json')}
    prior=json.loads((p/'Run_Register.json').read_text())['elapsed_seconds']+json.loads((p/'Resumption01_Status.json').read_text())['elapsed_seconds']
    budget=max(0,16200-prior);start=time.time();deadline=start+budget
    pending=[spec for spec in s.specifications() if not (p/'runs'/s.run_id(spec)/'done.json').exists()]
    s.writej(p/'Completion_Pass_Launch.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'existing_completed':len(before),'remaining':len(pending),'prior_training_wall_seconds':prior,'remaining_training_budget_seconds':budget,'deadline_epoch_seconds':deadline,'workers':6,'frozen_training_sha256':s.sha(p/'study.py')})
    print('LAUNCHED',len(pending),'remaining runs; budget seconds',budget,flush=True)
    returned=[];failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('fork')) as pool:
        futs={pool.submit(s.train,spec,deadline):spec for spec in pending}
        for f in concurrent.futures.as_completed(futs):
            try:
                r=f.result();returned.append({'spec':r['spec'],'status':r['status'],'last_epoch':r['last_epoch']})
            except Exception as exc:failures.append({'spec':futs[f],'exception':repr(exc)})
            total=len(list((p/'runs').glob('*/*/seed_*/done.json')))
            state={'planned':135,'completed':total,'returned_this_pass':len(returned),'failures':failures,'results':returned,'elapsed_seconds':time.time()-start}
            s.writej(p/'Completion_Pass_Status.json',state)
            print('COMPLETED',total,'/135','elapsed',round(time.time()-start),'failures',len(failures),flush=True)
    assert all(s.sha(p/name)==digest for name,digest in before.items()),'Existing completed result changed'
    s.writej(p/'Completion.json',{'planned':135,'completed':len(list((p/'runs').glob('*/*/seed_*/done.json'))),'results':returned,'failures':failures,'seconds':prior+time.time()-start,'this_pass_seconds':time.time()-start,'prior_completed_unchanged':True,'next_experiment_started':False})
    print('PASS FINISHED',flush=True)

if __name__=='__main__':main()
