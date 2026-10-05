"""Start/status/stop the fixed 25k plan on the existing TPU; never allocate a node."""
import argparse
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid

PROJECT='tpu-builders-504820'; ZONE='us-west4-a'
NODE='ww-gpt2-validation-48h-20261004-s1337-node'
BASE=Path('/mnt/disks/rg-data/gpt2small'); LATEST=BASE/'MUON_LONG25K_LATEST.json'
MIN_REMAINING=12.5*3600


def run(command,**kwargs): return subprocess.run(command,check=True,text=True,**kwargs)


def timestamp(value):
    value=re.sub(r'(\.\d{6})\d+',r'\1',value)
    return dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def lease_from(node,queue,now):
    if node.get('state')!='READY' or queue.get('state',{}).get('state')!='ACTIVE':
        raise RuntimeError('Existing TPU/queue is not READY/ACTIVE')
    if node.get('acceleratorType')!='v5litepod-8':
        raise RuntimeError('Expected the existing single-host v5litepod-8')
    candidates=[node.get('schedulingConfig',{}).get('terminationTimestamp'),
                queue.get('runDuration',{}).get('terminationTime')]
    for spec in queue.get('tpu',{}).get('nodeSpec',[]):
        if spec.get('nodeId')==NODE:
            candidates.append(spec.get('node',{}).get('schedulingConfig',{}).get('terminationTimestamp'))
    expiries=[timestamp(x) for x in candidates if x]
    if not expiries:
        raise RuntimeError('API did not return an explicit termination time; refusing to infer it from queue creation')
    expiry=min(expiries)
    result={'node':NODE,'queue':queue['name'].rsplit('/',1)[-1],
        'queue_created':queue.get('createTime'),'node_created':node.get('createTime'),
        'max_run_duration':queue.get('runDuration',{}).get('maxRunDuration'),
        'termination_unix':expiry,'termination_utc':dt.datetime.fromtimestamp(expiry,dt.timezone.utc).isoformat(),
        'checked_unix':now,'remaining_hours':(expiry-now)/3600}
    print(json.dumps(result,indent=2),flush=True)
    if expiry-now<=MIN_REMAINING:
        raise RuntimeError('Need more than 12.5 hours remaining; no run started and no allocation changed')
    return result


def live_lease():
    flags=['--project='+PROJECT,'--zone='+ZONE,'--format=json']
    node=json.loads(run(['gcloud','alpha','compute','tpus','tpu-vm','describe',NODE,*flags],capture_output=True).stdout)
    queue_name=node.get('queuedResource','').rsplit('/',1)[-1]
    if not queue_name: raise RuntimeError('Could not determine the node\'s queued resource')
    queue=json.loads(run(['gcloud','alpha','compute','tpus','queued-resources','describe',queue_name,*flags],capture_output=True).stdout)
    return lease_from(node,queue,time.time())


def active(unit):
    r=subprocess.run(['systemctl','show',unit,'--property=ActiveState','--value'],capture_output=True,text=True)
    return r.stdout.strip() in ('active','activating','deactivating','reloading')


def status_remote(tail_metrics=False):
    if not LATEST.exists(): print('No long run launched.'); return
    record=json.loads(LATEST.read_text()); root=Path(record['root'])
    print(json.dumps(record,indent=2),flush=True)
    subprocess.run(['systemctl','--no-pager','--full','status',record['unit']])
    for name in ('RUN_STATUS.json','STARTUP.json','status.json','latest_validation.json',
                 'checkpoint_latest.json','TRACKING_STATUS.json','RESUME_PARITY.json'):
        if (root/name).exists(): print(name+'\n'+(root/name).read_text(),flush=True)
    subprocess.run(['tail','-n','12',str(root/('metrics.jsonl' if tail_metrics else 'run.log'))])


def stop_remote():
    record=json.loads(LATEST.read_text()); root=Path(record['root'])
    (root/'STOP').touch()
    print('Safe stop requested. Trainer will finish the current update, save full state, then drain tracking/backup.')
    print('Watch:',root/'run.log')


def start_remote(commit,lease,request_id,resume=None):
    if os.geteuid()!=0 or not os.path.ismount('/mnt/disks/rg-data'):
        raise RuntimeError('The persistent data disk must already be mounted; root required')
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('Pinned Git commit required')
    if resume:
        resume=Path(resume).resolve()
        if BASE.resolve() not in resume.parents or resume.suffix!='.pt' or not resume.is_file():
            raise RuntimeError('Recovery checkpoint must exist under the persistent experiment directory')
    with (BASE/'port-check-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if LATEST.exists():
            previous=json.loads(LATEST.read_text())
            if previous.get('request_id')==request_id or active(previous['unit']):
                print('Existing launch retained; no duplicate or restart.'); status_remote(); return
        if not -30 <= time.time()-lease['checked_unix'] <= 600:
            raise RuntimeError('Lease check is stale; rerun the launcher')
        if lease['node']!=NODE or lease['termination_unix']-time.time()<=MIN_REMAINING:
            raise RuntimeError('Need more than 12.5 hours remaining on the checked node')
        if shutil.disk_usage(BASE).free < 45*1024**3:
            raise RuntimeError('Need 45 GiB free for remaining shards, spectra and retained checkpoints; nothing deleted')
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
        root=BASE/('muon-long25k-s1337-'+stamp); root.mkdir(); repo=root/'repo'; repo.mkdir()
        run(['git','-C',str(repo),'init','-q'])
        run(['git','-C',str(repo),'remote','add','origin','https://github.com/CalculatedContent/rg_optimizers.git'])
        run(['git','-C',str(repo),'fetch','--depth','1','origin',commit],timeout=180)
        run(['git','-C',str(repo),'checkout','--detach',commit])
        scripts=repo/'baseline/gpt2_small'
        spec=importlib.util.spec_from_file_location('training_guard',scripts/'scripts/run_muonclip.py')
        guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard); guard.assert_idle()
        # Verify reference BEFORE launching a service; never change that directory.
        sys.path.insert(0,str(scripts/'muon_longrun'))
        from common import verify_reference, REFERENCE_NAME, atomic_json
        approved=verify_reference(BASE/REFERENCE_NAME)
        deadline=time.time()+12*3600
        if lease['termination_unix']-deadline<1800:
            raise RuntimeError('Less than 30 minutes lease margin after checkout; no service started')
        unit='rg-muon-long25k-'+stamp+'.service'
        record={'root':str(root),'unit':unit,'commit':commit,'request_id':request_id,
            'node':NODE,'started_unix':time.time(),'service_deadline_unix':deadline,
            'training_deadline_unix':deadline-1800,'steps':25000,'tokens':13107200000,
            'optimizer':'muon','config':approved['manifest']['config'],
            'batch_tokens':524288,'global_microbatch_sequences':64,'accumulation':8,
            'peak_lrs':{'muon':.04,'adam_embedding':.6,'adam_head':.008,'adam_scalar':.04},
            'muon_momentum':{'initial':.85,'final':.95,'ramp_updates':500},
            'newton_schulz':{'steps':5,'coefficients':[3.4445,-4.7750,2.0315]},
            'adam':{'betas':[.9,.95],'eps':1e-8,'weight_decay':0},
            'gradient_clipping':False,'qk_clipping':False,
            'scheduler':{'warmup_updates':0,'cooldown_start':17500,'warmdown_updates':7500,'final_step':25000},
            'reference':approved['reference_run'],'lease':lease,'fresh_initialization':resume is None,
            'resume_source':str(resume) if resume else None,
            'automatic_restart':False,'cloud_uri':'gs://tpu-builders-504820-ww-continuous8/gpt2small/'+root.name}
        atomic_json(root/'launch.json',record); atomic_json(root/'LEASE.json',lease)
        (root/'commit.txt').write_text(commit+'\n')
        env={'PYTHONPATH':str(scripts/'src')+':'+str(scripts.parent/'nanogpt_one_head/src'),
             'PJRT_DEVICE':'TPU','TPU_ACCELERATOR_TYPE':'v5litepod-8',
             'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
             'TOKENIZERS_PARALLELISM':'false'}
        command=['systemd-run','--unit='+unit,'--property=Type=exec','--property=Restart=no',
            '--property=RuntimeMaxSec=43200','--property=TimeoutStopSec=15',
            '--property=KillMode=control-group','--property=StandardOutput=append:'+str(root/'run.log'),
            '--property=StandardError=append:'+str(root/'run.log')]
        command+=['--setenv='+k+'='+v for k,v in env.items()]
        command+=['/mnt/disks/rg-data/continuous8/venv/bin/python','-u',
                  str(scripts/'muon_longrun/long_worker.py'),str(root),str(deadline)]
        if resume: command+=['--resume',str(resume)]
        atomic_json(LATEST,record) # Publish request ID before systemd to make SSH retry idempotent.
        run(command)
        print(json.dumps(record,indent=2),flush=True)
        run(['systemctl','show',unit,'--property=MainPID','--property=ActiveState'])
        print('Muon run: 25,000 total steps, 13.1072B tokens; no target-loss stop.',flush=True)
        print('Peak LR through 17,500; linear warmdown over final 7,500; no LR warmup.',flush=True)
        print('Scalars every 10; full validation every 500 + spectral steps.',flush=True)
        print('WW: 0,100,250,500,750,1000,1500,2000,2500,3000; then 1000, plus 17500/final.',flush=True)
        print('Checkpoints every 2500; latest two rolling + permanent 0/3000/10000/17500/25000.',flush=True)
        print('Estimated ~10–11 hours; 12-hour service cap includes preparation and final backup.',flush=True)
        for action in ('status','metrics','stop'):
            print(f'From your local checkout: python3 baseline/gpt2_small/muon_longrun/launch.py {action}',flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('start','recover','status','metrics','stop'))
    p.add_argument('--checkpoint',type=Path,help='Explicit recovery from this long plan only; never used by start')
    p.add_argument('--on-tpu',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--commit',help=argparse.SUPPRESS); p.add_argument('--lease',help=argparse.SUPPRESS)
    p.add_argument('--request-id',help=argparse.SUPPRESS); a=p.parse_args()
    if (a.action=='recover') != bool(a.checkpoint): p.error('Only recover requires --checkpoint')
    if a.on_tpu:
        if a.action in ('start','recover'): start_remote(a.commit,json.loads(a.lease),a.request_id,a.checkpoint)
        elif a.action=='stop': stop_remote()
        else: status_remote(a.action=='metrics')
        return 0
    command=['sudo','python3','-c',Path(__file__).read_text(),a.action,'--on-tpu']
    if a.checkpoint: command+=['--checkpoint',str(a.checkpoint)]
    if a.action in ('start','recover'):
        repo=Path(__file__).resolve().parents[3]
        if run(['git','-C',str(repo),'status','--porcelain'],capture_output=True).stdout.strip():
            raise RuntimeError('Use a clean checkout of the pushed commit')
        commit=run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True).stdout.strip()
        lease=live_lease()
        command+=['--commit',commit,'--lease',json.dumps(lease),'--request-id',uuid.uuid4().hex]
    return subprocess.run(['gcloud','compute','tpus','tpu-vm','ssh',NODE,'--project='+PROJECT,
                           '--zone='+ZONE,'--worker=0','--command='+shlex.join(command)]).returncode

if __name__=='__main__': raise SystemExit(main())
