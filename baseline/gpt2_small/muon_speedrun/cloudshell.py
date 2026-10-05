"""Launch a checkpointed Muon recipe on the existing eight-chip TPU."""
import argparse
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

PROJECT = 'tpu-builders-504820'
ZONE = 'us-west4-a'
QUEUE = 'ww-gpt2-validation-48h-20261004-s1337'
NODE = QUEUE+'-node'
BASE = Path('/mnt/disks/rg-data/gpt2small')
LATEST = BASE/'MUON_SPEEDRUN_LATEST.json'


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def active(unit):
    r = subprocess.run(['systemctl','show',unit,'--property=ActiveState','--value'],
                       capture_output=True,text=True,timeout=10)
    return r.stdout.strip() in ('active','activating','deactivating','reloading')


def status_remote():
    if not LATEST.exists():
        print('No Muon speedrun launched.')
        return
    record = json.loads(LATEST.read_text())
    root = Path(record['root'])
    print(json.dumps(record,indent=2),flush=True)
    subprocess.run(['systemctl','--no-pager','--full','status',record['unit']])
    for name in ('RUN_STATUS.json','status.json','latest_validation.json','checkpoint_latest.json',
                 'PALLAS_DEPENDENCIES.json','ATTENTION_CHECK.json','ATTENTION_CHECK_FAILURE.json',
                 'TRACKING_CONFIG.json','TRACKING_STATUS.json'):
        if (root/name).exists():
            print(name+'\n'+(root/name).read_text(),flush=True)
    subprocess.run(['tail','-n','15',str(root/'run.log')])


def stop_current():
    if not LATEST.exists():
        return
    record = json.loads(LATEST.read_text())
    unit = record['unit']
    if not re.fullmatch(r'rg-muon-speedrun-\d{8}-\d{6}\.service', unit):
        raise RuntimeError('Unexpected service name; refusing to stop it')
    print('Stopping previous Muon speedrun: '+unit, flush=True)
    run(['systemctl','stop',unit], timeout=60)
    if active(unit):
        raise RuntimeError('Previous service is still active; new run not started')


def start_remote(commit, hours=3, optimizer='muon', microbatch=64, attention='flash', replace_current=False,
                 launch_id=None):
    if os.geteuid() != 0 or not os.path.ismount('/mnt/disks/rg-data'):
        raise RuntimeError('Requires the existing mounted disk and root')
    if not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Expected pinned commit')
    with (BASE/'port-check-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if LATEST.exists() and launch_id and json.loads(LATEST.read_text()).get('launch_id') == launch_id:
            print('This launch request was already handled; SSH retry will not restart it.')
            status_remote()
            return
        if not replace_current and LATEST.exists() and active(json.loads(LATEST.read_text())['unit']):
            print('A speedrun is already active; no duplicate launched.')
            status_remote()
            return
        allocation = json.loads((BASE/QUEUE/'allocation.json').read_text())
        deadline = min(time.time()+hours*3600,float(allocation['validation_deadline_unix'])-30)
        if deadline-time.time() < 2400:
            raise RuntimeError('Less than 40 minutes remain on this allocation')
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
        root = BASE/('muon-speedrun-'+optimizer+'-'+stamp)
        root.mkdir()
        repo = root/'repo'
        repo.mkdir()
        run(['git','-C',str(repo),'init','-q'])
        run(['git','-C',str(repo),'remote','add','origin','https://github.com/CalculatedContent/rg_optimizers.git'])
        run(['git','-C',str(repo),'fetch','--depth','1','origin',commit],timeout=120)
        run(['git','-C',str(repo),'checkout','--detach',commit])
        if replace_current:
            stop_current()
        guard = repo/'baseline/gpt2_small/scripts/run_muonclip.py'
        spec = importlib.util.spec_from_file_location('training_guard',guard)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.assert_idle()
        unit = 'rg-muon-speedrun-'+stamp+'.service'
        base = repo/'baseline/gpt2_small'
        env = {'PYTHONPATH':str(base/'src')+':'+str(base.parent/'nanogpt_one_head/src'),
               'PJRT_DEVICE':'TPU','TPU_ACCELERATOR_TYPE':'v5litepod-8',
               'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
               'TOKENIZERS_PARALLELISM':'false'}
        record = {'root':str(root),'unit':unit,'commit':commit,'optimizer':optimizer,
                  'launch_id':launch_id,
                  'started_unix':time.time(),'deadline_unix':deadline,'hours_cap':hours,
                  'target_val_nll':3.28,'checkpoint_interval':125,'microbatch':microbatch,
                  'weightwatcher_interval':125, 'validation_token_error':True,
                  'cloud_uri':'gs://tpu-builders-504820-ww-continuous8/gpt2small/'+root.name}
        (root/'launch.json').write_text(json.dumps(record,indent=2))
        (root/'commit.txt').write_text(commit+'\n')
        command = ['systemd-run','--unit='+unit,'--property=Type=exec','--property=Restart=no',
                   '--property=RuntimeMaxSec='+str(int(deadline-time.time())-5),
                   '--property=TimeoutStopSec=5','--property=KillMode=control-group',
                   '--property=StandardOutput=append:'+str(root/'run.log'),
                   '--property=StandardError=append:'+str(root/'run.log')]
        command += ['--setenv='+k+'='+v for k,v in env.items()]
        command += ['/mnt/disks/rg-data/continuous8/venv/bin/python','-u',
                    str(base/'muon_speedrun/worker.py'),str(root),str(deadline),
                    '--optimizer',optimizer,'--microbatch',str(microbatch),'--attention',attention]
        run(command)
        temp = LATEST.with_suffix('.tmp')
        temp.write_text(json.dumps(record,indent=2))
        temp.replace(LATEST)
        print('Started '+optimizer+' recipe: '+unit,flush=True)
        print('Log: '+str(root/'run.log'),flush=True)
        print('3,000 updates, stops at full-validation NLL <= 3.28; checkpoints every 125.',flush=True)
        print('Paired validation token error and raw/clipped WeightWatcher alpha every 125 updates and final.',flush=True)
        print('Hard cutoff UTC: '+dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat(),flush=True)
        print('No automatic restart or new TPU allocation. Three hours is a cap, not an ETA.',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('start','status'))
    p.add_argument('--hours',type=float,default=3)
    p.add_argument('--optimizer',choices=('muon','adam'),default='muon')
    p.add_argument('--microbatch',type=int,choices=(32,64,128),default=64)
    p.add_argument('--attention',choices=('auto','flash','math'),default='flash')
    p.add_argument('--replace-current',action='store_true',help='Stop the previous speedrun and start from initialization')
    p.add_argument('--on-tpu',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--commit',help=argparse.SUPPRESS)
    p.add_argument('--launch-id',help=argparse.SUPPRESS)
    a = p.parse_args()
    if not 1 <= a.hours <= 12:
        raise ValueError('Hours must be between 1 and 12, bounded by existing allocation')
    if a.on_tpu:
        if a.action == 'start':
            start_remote(a.commit,a.hours,a.optimizer,a.microbatch,a.attention,a.replace_current,a.launch_id)
        else:
            status_remote()
        return 0
    command = ['sudo','python3','-c',Path(__file__).read_text(),a.action,'--on-tpu',
               '--hours',str(a.hours),'--optimizer',a.optimizer,'--microbatch',str(a.microbatch),
               '--attention',a.attention]
    if a.replace_current:
        command += ['--replace-current']
    if a.action == 'start':
        repo = Path(__file__).resolve().parents[3]
        if run(['git','-C',str(repo),'status','--porcelain'],capture_output=True).stdout.strip():
            raise RuntimeError('Launch from a clean checkout of the pushed commit')
        commit = run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True).stdout.strip()
        command += ['--commit',commit,'--launch-id',uuid.uuid4().hex]
    return subprocess.run(['gcloud','compute','tpus','tpu-vm','ssh',NODE,
        '--project='+PROJECT,'--zone='+ZONE,'--worker=0','--command='+shlex.join(command)]).returncode


if __name__ == '__main__':
    sys.exit(main())
