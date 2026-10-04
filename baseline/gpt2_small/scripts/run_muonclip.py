"""Launch a fresh continuous MuonClip run on the EXISTING 48-hour TPU."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

PROJECT='tpu-builders-504820'; ZONE='us-west4-a'
QUEUE='ww-gpt2-validation-48h-20261004-s1337'; NODE=QUEUE+'-node'
BASE=Path('/mnt/disks/rg-data/gpt2small'); OLD=BASE/QUEUE
LATEST=BASE/'MUONCLIP_LATEST.json'


def run(args,**kwargs):
    return subprocess.run(args,check=True,text=True,**kwargs)


def active(unit):
    result=subprocess.run(['systemctl','show',unit,'--property=ActiveState','--value'],
                          text=True,capture_output=True,timeout=10)
    return result.stdout.strip() in ('active','activating','deactivating','reloading')


def assert_idle():
    if active('rg-gpt2-validation.service') or active('rg-continuous8.service'):
        raise RuntimeError('An existing training service is active; no training launched.')
    if LATEST.exists() and active(json.loads(LATEST.read_text())['unit']):
        raise RuntimeError('MuonClip is already active. Use status; no second run launched.')
    night=BASE/'PORT_CHECK_LATEST.json'
    if night.exists() and active(json.loads(night.read_text())['unit']):
        raise RuntimeError('Port diagnostic is active; no concurrent training launched.')
    replay=BASE/'MUONCLIP_REPLAY_LATEST.json'
    if replay.exists() and active(json.loads(replay.read_text())['unit']):
        raise RuntimeError('An update replay is active; no concurrent trainer launched.')
    modules={'rg_gpt2_small.replay_update','rg_gpt2_small.experiment','rg_nanogpt_one_head.gpt2_experiment',
             'rg_nanogpt_one_head.continuous_run','rg_nanogpt_one_head.tpu_spmd_check'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: args=set(path.read_bytes().decode().split('\0'))
        except (OSError,UnicodeError): continue
        if args.intersection(modules):
            raise RuntimeError(f'Trainer process {path.parent.name} is active; no concurrent diagnostic launched.')


def launch_remote(commit):
    if os.geteuid()!=0 or not os.path.ismount('/mnt/disks/rg-data'):
        raise RuntimeError('Requires root and the existing mounted data disk.')
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('Expected pinned commit SHA')
    with (BASE/'port-check-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_idle()
        allocation=json.loads((OLD/'allocation.json').read_text())
        allocation_deadline=float(allocation['validation_deadline_unix'])
        if allocation_deadline-time.time()<900:
            raise RuntimeError('Less than 15 minutes remain; training not launched.')
        if not Path('/mnt/disks/rg-data/continuous8/data/train.bin').is_file():
            raise RuntimeError('Preserved FineWeb is missing; no download will be started.')
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
        root=BASE/('muonclip-night-'+stamp); root.mkdir()
        repo=root/'repo'; repo.mkdir()
        run(['git','-C',str(repo),'init','-q'])
        run(['git','-C',str(repo),'remote','add','origin','https://github.com/CalculatedContent/rg_optimizers.git'])
        run(['git','-C',str(repo),'fetch','--depth','1','origin',commit],timeout=180)
        run(['git','-C',str(repo),'checkout','--detach',commit])
        (root/'commit.txt').write_text(commit+'\n')
        deadline=allocation_deadline-600
        if deadline-time.time()<300: raise RuntimeError('Too little time remains after source checkout.')
        unit='rg-gpt2-muonclip-'+stamp+'.service'
        record={'root':str(root),'unit':unit,'commit':commit,'node':NODE,
                'training_deadline_unix':deadline,'service_deadline_unix':allocation_deadline,
                'purpose':'continuous MuonClip with scalar finite guard; per-tensor diagnostic disabled',
                'cloud_uri':'gs://tpu-builders-504820-ww-continuous8/gpt2small/'+root.name}
        (root/'launch.json').write_text(json.dumps(record,indent=2))
        command=['systemd-run','--unit='+unit,'--property=Type=exec','--property=Restart=no',
                 '--property=RuntimeMaxSec='+str(int(allocation_deadline-time.time())),
                 '--property=TimeoutStopSec=30','--property=KillMode=control-group',
                 '--property=StandardOutput=append:'+str(root/'run.log'),
                 '--property=StandardError=append:'+str(root/'run.log'),
                 '/bin/bash',str(repo/'baseline/gpt2_small/scripts/muonclip_worker.sh'),
                 str(root),str(deadline),str(allocation_deadline)]
        run(command)
        temp=LATEST.with_suffix('.tmp'); temp.write_text(json.dumps(record,indent=2)); temp.replace(LATEST)
        print('MuonClip service started:',unit,flush=True)
        print('Log:',root/'run.log',flush=True)
        print('Training cutoff UTC:',dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat(),flush=True)
        print('Checkpoints/token error every 25 updates; raw/clipped alpha every 100; no automatic restart.',flush=True)
        print('Existing TPU allocation, FineWeb and prior outputs retained.',flush=True)


def status_remote():
    if not LATEST.exists():
        print('No overnight MuonClip run has been launched.'); return
    record=json.loads(LATEST.read_text()); root=Path(record['root'])
    print(json.dumps(record,indent=2),flush=True)
    subprocess.run(['systemctl','--no-pager','--full','status',record['unit']],check=False)
    for file in ('RUN_STATUS.json','muonclip/TPU_PORT_FAILURE.json','muonclip/progress.json','muonclip/cloud_checkpoint.json'):
        path=root/file
        if path.is_file(): print('\n'+file+'\n'+path.read_text(),flush=True)
    subprocess.run(['tail','-n','70',str(root/'run.log')],check=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','status'))
    parser.add_argument('--on-tpu',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--commit',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.on_tpu:
        launch_remote(args.commit) if args.action=='start' else status_remote()
        return 0
    commit=None
    if args.action=='start':
        repo=Path(__file__).resolve().parents[3]
        dirty=run(['git','-C',str(repo),'status','--porcelain'],capture_output=True).stdout
        if dirty.strip(): raise RuntimeError('Use a clean checkout of the pushed commit.')
        commit=run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True).stdout.strip()
        queue=json.loads(run(['gcloud','alpha','compute','tpus','queued-resources','describe',QUEUE,
            '--project='+PROJECT,'--zone='+ZONE,'--format=json'],capture_output=True).stdout)
        if queue.get('state',{}).get('state')!='ACTIVE':
            raise RuntimeError('The existing TPU is not ACTIVE; no allocation requested.')
    remote=['sudo','python3','-c',Path(__file__).read_text(),args.action,'--on-tpu']
    if commit: remote+=['--commit',commit]
    return subprocess.run(['gcloud','compute','tpus','tpu-vm','ssh',NODE,
        '--project='+PROJECT,'--zone='+ZONE,'--worker=0','--command='+shlex.join(remote)]).returncode


if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc:
        print('MuonClip launch:',exc,file=sys.stderr); sys.exit(1)
