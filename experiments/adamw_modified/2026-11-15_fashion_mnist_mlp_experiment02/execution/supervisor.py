"""One bounded 120-run Fashion-MNIST pass with incremental preservation."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import concurrent.futures as cf
import datetime
import json
import multiprocessing
import subprocess
import sys
import time
import traceback
import adapter as a
import preserve

ROOT=a.ROOT;s=a.s


def main():
    p=json.loads((ROOT/'Protocol.json').read_text());pre=json.loads((ROOT/'Preflight_Audit.json').read_text())
    assert pre['passed'] and pre['dataset']=='Fashion-MNIST' and len(p['specifications'])==120
    for name,sha in json.loads((ROOT/'Execution_Source_Manifest.json').read_text()).items():assert s.sha(ROOT/name)==sha
    assert s.sha(ROOT/'study.py')==p['source_sha256'] and s.sha(ROOT/'adapter.py')==p['adapter_sha256']
    for name,sha in p['data_hashes'].items():assert s.sha(ROOT/'data'/name)==sha
    start=time.time();prepared=json.loads((ROOT/'Preparation_Start.json').read_text())['epoch_seconds']
    deadline=min(start+16200,prepared+18000-1800)
    assert deadline>start,'Preparation used the entire allowed training window'
    launch={'experiment':'Experiment02','dataset':'Fashion-MNIST','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'start_epoch_seconds':start,'training_deadline_epoch_seconds':deadline,'total_deadline_epoch_seconds':prepared+18000,
            'workers':6,'planned_runs':120,'source_sha256':p['source_sha256'],'adapter_sha256':p['adapter_sha256'],
            'training_active':True,'next_experiment_started':False,'pid':os.getpid()}
    with (ROOT/'Launch.json').open('x') as f:json.dump(launch,f,indent=2)
    assert not (ROOT/'Completion.json').exists(),'Never overwrite a completed pass'
    s.DATA=s.load_data();a.configure_training(deadline)
    results=[];errors=[];seed_pending={};saved={};save_blocked=False
    def status():
        s.writej(ROOT/'Run_Register.json',{'dataset':'Fashion-MNIST','planned':120,'returned':len(results),
            'completed':sum(r['status']=='complete' for r in results),'results':results,'errors':errors,
            'saved_seed_ids':sorted(saved),'save_blocked':save_blocked,'elapsed_seconds':time.time()-start,
            'training_active':len(results)<120,'deadline_epoch_seconds':deadline})
    with cf.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(a.train_one,spec,deadline):spec for spec in p['specifications']}
        with cf.ThreadPoolExecutor(max_workers=1) as saver:
            while futures:
                done,_=cf.wait(futures,timeout=20,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    spec=futures.pop(future)
                    try:result=future.result()
                    except Exception as exc:
                        result={'spec':spec,'status':'exception','error':repr(exc)};errors.append(result)
                    # Keep register compact; done.json contains all measurements.
                    results.append({k:result[k] for k in ('spec','status','last_epoch','seconds','error') if k in result})
                    print('FINISHED',len(results),s.run_id(spec),result['status'],flush=True)
                for seed in p['seeds']:
                    group=[r for r in results if r['spec']['seed']==seed]
                    if len(group)==24 and all(r['status']=='complete' for r in group) and seed not in seed_pending and not save_blocked:
                        seed_pending[seed]=saver.submit(preserve.save_seed,seed)
                for seed,f in seed_pending.items():
                    if f.done() and seed not in saved:
                        try:
                            saved[seed]=f.result();s.writej(ROOT/f'Seed_{seed}_Saved.json',saved[seed])
                        except Exception as exc:
                            save_blocked=True;saved[seed]={'error':repr(exc)};s.writej(ROOT/f'Seed_{seed}_Save_Failure.json',saved[seed])
                status()
            # Context waits for ordered saves already submitted.
        for seed,f in seed_pending.items():
            if seed not in saved:
                try:saved[seed]=f.result();s.writej(ROOT/f'Seed_{seed}_Saved.json',saved[seed])
                except Exception as exc:save_blocked=True;errors.append({'seed_save':seed,'error':repr(exc)})
    completion={'dataset':'Fashion-MNIST','planned':120,'completed':sum(r['status']=='complete' for r in results),
        'results':results,'errors':errors,'training_seconds':time.time()-start,'training_active':False,
        'saved_seeds':saved,'save_blocked':save_blocked,'source_sha256':p['source_sha256'],'next_experiment_started':False,
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    s.writej(ROOT/'Completion.json',completion)
    print('TRAINING FINISHED',completion['completed'],flush=True)
    if (ROOT/'analyze_and_verify.py').exists():
        with (ROOT/'analysis.log').open('w') as log:
            proc=subprocess.run([sys.executable,str(ROOT/'analyze_and_verify.py')],stdout=log,stderr=subprocess.STDOUT,
                                timeout=max(1,prepared+18000-time.time()))
        if proc.returncode:raise RuntimeError('Postprocessing failed: inspect analysis.log')
    else:s.writej(ROOT/'Postprocessing_Pending.json',{'reason':'Analysis script not yet present','training_active':False})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        s.writej(ROOT/'Experiment02_Supervisor_Failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
