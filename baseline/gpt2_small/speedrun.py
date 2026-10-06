"""Main nanoGPT speedrun entry point: plan/start/status/report for paired seeds."""
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

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'muon_speedrun'))
from repeated import plan, write, SUITE_SECONDS
BASE=Path('/mnt/disks/rg-data/gpt2small')
LATEST=BASE/'NANOGPT_SPEEDRUN_SUITE_LATEST.json'
PROJECT='tpu-builders-504820'


def run(command,**kwargs):
    try: return subprocess.run(command,check=True,text=True,**kwargs)
    except subprocess.CalledProcessError as exc:
        if exc.stderr: print(exc.stderr,file=sys.stderr)
        raise


def lease(a, required_seconds=SUITE_SECONDS+1800):
    flags=['--project='+PROJECT,'--zone='+a.zone,'--format=json']
    node=json.loads(run(['gcloud','compute','tpus','tpu-vm','describe',a.node,*flags],capture_output=True).stdout)
    queue_name=node.get('queuedResource','').rsplit('/',1)[-1]
    if not queue_name: raise RuntimeError('No queued-resource lease found')
    queue=json.loads(run(['gcloud','compute','tpus','queued-resources','describe',queue_name,*flags],capture_output=True).stdout)
    if node.get('state')!='READY' or queue.get('state',{}).get('state')!='ACTIVE' or node.get('acceleratorType')!='v5litepod-8':
        raise RuntimeError('Requires an active single-host v5e-8 allocation')
    expiries=[node.get('schedulingConfig',{}).get('terminationTimestamp'),queue.get('runDuration',{}).get('terminationTime')]
    for spec in queue.get('tpu',{}).get('nodeSpec',[]):
        if spec.get('nodeId')==a.node:
            expiries.append(spec.get('node',{}).get('schedulingConfig',{}).get('terminationTimestamp'))
    expiries=[dt.datetime.fromisoformat(re.sub(r'(\.\d{6})\d+',r'\1',s).replace('Z','+00:00')).timestamp() for s in expiries if s]
    if not expiries: raise RuntimeError('No explicit lease expiry; nothing launched')
    result={'node':a.node,'checked_unix':time.time(),'termination_unix':min(expiries),
            'ips':[v['ipAddress'] for v in node.get('networkEndpoints',[]) if v.get('ipAddress')]}
    require_time(result, required_seconds)
    return result


def require_time(checked, required_seconds=SUITE_SECONDS+1800):
    if not -30<=time.time()-checked['checked_unix']<=600:
        raise RuntimeError('Lease check expired; rerun start')
    if checked['termination_unix']-time.time()<required_seconds:
        raise RuntimeError(f'Need at least {int(required_seconds)//3600}h{int(required_seconds)%3600//60:02d}m remaining for the requested caps and lease margin. Existing training untouched.')


def start_here(a):
    checked=json.loads(a.lease); require_time(checked)
    if os.geteuid()!=0 or not os.path.ismount('/mnt/disks/rg-data'):
        raise RuntimeError('Root and the existing mounted data disk are required')
    import urllib.request
    req=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip',headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(req,timeout=5) as response:
        if response.read().decode().strip() not in checked['ips']:
            raise RuntimeError('Guest does not match the checked TPU')
    python=Path('/mnt/disks/rg-data/continuous8/venv/bin/python')
    if not python.is_file(): raise RuntimeError('Existing experiment environment is missing')
    if not re.fullmatch('[0-9a-f]{40}',a.commit): raise ValueError('Pinned commit required')
    if shutil.disk_usage(BASE).free<240*1024**3:
        raise RuntimeError('Need 240 GiB free for six sets of checkpoint/spectral outputs; nothing deleted')
    with (BASE/'port-check-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if LATEST.exists() and json.loads(LATEST.read_text()).get('request_id')==a.request_id:
            print(LATEST.read_text()); return
        spec=importlib.util.spec_from_file_location('suite_guard',HERE/'scripts/run_muonclip.py')
        guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard); guard.assert_idle()
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
        root=BASE/('nanogpt-speedrun-suite-'+stamp); root.mkdir()
        repo=root/'repo'; run(['git','init','-q',str(repo)])
        run(['git','-C',str(repo),'fetch','--depth','1','https://github.com/CalculatedContent/rg_optimizers.git',a.commit],timeout=180)
        run(['git','-C',str(repo),'checkout','--detach','FETCH_HEAD'])
        require_time(checked)
        deadline=time.time()+SUITE_SECONDS
        unit='rg-nanogpt-speedrun-suite-'+stamp+'.service'
        record={'root':str(root),'unit':unit,'commit':a.commit,'request_id':a.request_id,
                'deadline_unix':deadline,'lease':checked,'automatic_restart':False}
        write(root/'PLAN.json',plan()); write(root/'launch.json',record)
        (root/'commit.txt').write_text(a.commit+'\n')
        package=repo/'baseline/gpt2_small'
        env={'PYTHONPATH':str(package/'src')+':'+str(repo/'baseline/nanogpt_one_head/src'),
             'PJRT_DEVICE':'TPU','TPU_ACCELERATOR_TYPE':'v5litepod-8',
             'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
             'TOKENIZERS_PARALLELISM':'false'}
        command=['systemd-run','--unit='+unit,'--property=Type=exec','--property=Restart=no',
                 '--property=RuntimeMaxSec='+str(SUITE_SECONDS),'--property=TimeoutStopSec=15',
                 '--property=KillMode=control-group','--property=StandardOutput=append:'+str(root/'suite.log'),
                 '--property=StandardError=append:'+str(root/'suite.log')]
        command+=['--setenv='+k+'='+v for k,v in env.items()]
        command += [str(python),'-u',str(package/'muon_speedrun/repeated.py'),str(root),str(deadline)]
        write(LATEST,record)
        run(command)
        print(json.dumps(record,indent=2))
        print('Six sequential fresh 19560-update runs; tracking every 250; no target-based early stop.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','start','status','report'))
    p.add_argument('--node'); p.add_argument('--zone',default='us-west4-a')
    p.add_argument('--here',action='store_true'); p.add_argument('--ssh-key-file')
    p.add_argument('--root',type=Path); p.add_argument('--remote',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--lease',help=argparse.SUPPRESS); p.add_argument('--commit',help=argparse.SUPPRESS)
    p.add_argument('--request-id',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.action=='plan': print(json.dumps(plan(),indent=2)); return 0
    if a.remote:
        if a.action=='start': start_here(a)
        else:
            root=a.root or Path(json.loads(LATEST.read_text())['root'])
            if a.action=='report':
                run(['/mnt/disks/rg-data/continuous8/venv/bin/python',
                     str(HERE/'muon_speedrun/repeated.py'),str(root),'--report'])
            else:
                print((root/'launch.json').read_text())
                path=root/'SUITE_STATUS.json'
                if path.exists():
                    print(path.read_text())
                    job=json.loads(path.read_text()).get('job')
                    if job:
                        child=root/(root.name+'-'+job['name'])
                        for name in ('status.json','latest_validation.json','TRACKING_STATUS.json'):
                            if (child/name).exists(): print(name+'\n'+(child/name).read_text())
                run(['tail','-n','15',str(root/'suite.log')])
        return 0
    if not a.node: p.error('--node is required for the existing allocation')
    repo=HERE.parents[1]
    command=['sudo','python3',str(Path(__file__).resolve()),a.action,'--remote']
    if a.root: command+=['--root',str(a.root)]
    if a.action=='start':
        if run(['git','-C',str(repo),'status','--porcelain'],capture_output=True).stdout.strip():
            raise RuntimeError('Use a clean checkout of the pushed commit')
        commit=run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True).stdout.strip()
        checked=lease(a)
        command+=['--commit',commit,'--lease',json.dumps(checked),'--request-id',uuid.uuid4().hex]
    if a.here: return subprocess.run(command).returncode
    # Transfer only this pinned checkout's code via git, not local credentials.
    commit=run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True).stdout.strip()
    script='set -e\nr=$(mktemp -d "$HOME/rg-speedrun-suite.XXXXXX")\n'
    script+='git -C "$r" init -q\ngit -C "$r" fetch --depth 1 https://github.com/CalculatedContent/rg_optimizers.git '+shlex.quote(commit)+'\n'
    script+='git -C "$r" checkout --detach FETCH_HEAD\n'
    script+='sudo python3 "$r/baseline/gpt2_small/speedrun.py" '+shlex.join(command[3:])
    ssh=['gcloud','compute','tpus','tpu-vm','ssh',a.node,'--project='+PROJECT,'--zone='+a.zone,'--worker=0']
    if a.ssh_key_file: ssh+=['--ssh-key-file='+a.ssh_key_file]
    return subprocess.run(ssh+['--command=bash -c '+shlex.quote(script)]).returncode


if __name__=='__main__': raise SystemExit(main())
