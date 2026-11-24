"""Run the four authorized frozen-recipe confirmations and retain resumable progress."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SEEDS=(1341,1342,1343,1344)
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def read(path):return json.loads(path.read_text())
def status(workers,started):
    rows=[]
    for seed,proc in workers.items():
        folder=ROOT/'runs'/f'seed_{seed}'
        progress=folder/'confirmation/rate_0.4/graft'/f'seed_{seed}'/'progress.json'
        state=folder/'worker_state.json'
        row=dict(seed=seed,pid=proc.pid,exit_code=proc.poll(),step=0)
        if progress.exists():row.update(read(progress))
        if state.exists():row['worker_status']=read(state)['status']
        rows.append(row)
    complete=sum(x.get('worker_status')=='completed' and x['exit_code']==0 for x in rows)
    return dict(started_at=started,updated_at=utc(),supervisor_pid=os.getpid(),
                status='training' if any(x['exit_code'] is None for x in rows) else ('training_complete' if complete==4 else 'failed'),
                existing_completed_seed=1340,new_completed_seeds=complete,target_updates=20000,workers=rows)

def main():
    lock=(ROOT/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(ROOT/'preflight.json')['status']=='passed'
    started=utc();workers={};handles=[]
    (ROOT/'logs').mkdir(exist_ok=True)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    for seed in SEEDS:
        log=(ROOT/'logs'/f'seed_{seed}.log').open('a')
        handles.append(log)
        workers[seed]=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_additional_seed.py'),'--seed',str(seed)],
                                      cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    while True:
        record=status(workers,started);write(ROOT/'status.json',record)
        if record['status']!='training':break
        time.sleep(15)
    for h in handles:h.close()
    if record['status']=='training_complete':
        script=ROOT/'analyze_results.py'
        if script.exists():
            p=subprocess.run([sys.executable,str(script),'--root',str(ROOT)],cwd=ROOT,capture_output=True,text=True)
            (ROOT/'analysis_run.log').write_text(p.stdout+p.stderr)
            record.update(analysis_returncode=p.returncode,status='complete' if p.returncode==0 else 'analysis_failed')
        else:record.update(status='analysis_pending')
        record['finished_at']=utc();write(ROOT/'status.json',record)
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
