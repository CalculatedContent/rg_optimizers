"""Bounded setup, one continuous trainer, async CPU WW, persistent scientific output."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import HERE, SHORT, REFERENCE_NAME, atomic_json, verify_reference
from worker import bounded, finish_tracking


def cloud_backup(root):
    from rg_nanogpt_one_head.continuous_support import CloudPublisher
    publisher=CloudPublisher('gs://tpu-builders-504820-ww-continuous8/gpt2small/'+root.name)
    receipts=[]
    for folder in (root,root/'tracking'):
        files=folder.iterdir() if folder==root else folder.rglob('*')
        for path in sorted(files):
            if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
                publisher.snapshot_text_file(path,path.relative_to(root).as_posix())
    for path in sorted((root/'checkpoints').glob('step_*.pt')):
        receipts.append(publisher.file(path,path.relative_to(root).as_posix()))
    result={'status':'verified','checkpoints':receipts,'unix_time':time.time()}
    publisher.json(result,'CLOUD_BACKUP_VERIFIED.json'); atomic_json(root/'CLOUD_BACKUP_VERIFIED.json',result)


def main():
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('deadline',type=float)
    p.add_argument('--backup-only',action='store_true'); p.add_argument('--resume',type=Path)
    a=p.parse_args(); root=a.root
    if a.backup_only: cloud_backup(root); return 0
    train_deadline=a.deadline-1800
    state={'status':'preparing','automatic_restart':False,'deadline_unix':a.deadline,
           'training_deadline_unix':train_deadline,'steps':25000}
    atomic_json(root/'RUN_STATUS.json',state); tracker=None
    def phase(command,seconds,label):
        if (root/'STOP').exists(): raise RuntimeError('Stop requested during setup')
        result=bounded(command,min(seconds,train_deadline-time.time()),root,label)
        if result['exit_code']!=0: raise RuntimeError(label+' failed: '+str(result))
    try:
        approved=verify_reference(root.parent/REFERENCE_NAME)
        logs=root.parent/REFERENCE_NAME/'metrics.jsonl'
        if logs.exists():
            import statistics
            durations=[r['seconds'] for line in logs.read_text().splitlines()
                       if (r:=json.loads(line)).get('kind')=='train' and r.get('step',0)>=100]
            if durations: approved['median_update_seconds']=statistics.median(durations)
        atomic_json(root/'REFERENCE.json',approved)
        phase([sys.executable,str(SHORT/'pallas_dependencies.py'),str(root)],600,'pinned Pallas overlay')
        os.environ['PYTHONPATH']=str(root/'pallas-deps')+os.pathsep+os.environ.get('PYTHONPATH','')
        phase([sys.executable,'-u',str(HERE/'train_long.py'),'prepare','--root',str(root),
               '--deadline',str(min(time.time()+1800,train_deadline-600))],1800,'verify full benchmark corpus')
        phase([sys.executable,'-u',str(SHORT/'run.py'),'attention-check','--legacy-attention-check','--root',str(root),
               '--microbatch','64','--deadline',str(time.time()+300)],300,'8-chip flash attention check')
        env={**os.environ,'PJRT_DEVICE':'CPU','CUDA_VISIBLE_DEVICES':'',
             'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        tracker=subprocess.Popen([sys.executable,'-u',str(HERE/'track_long.py'),str(root),
                                  str(a.deadline-1200)],env=env,start_new_session=True)
        state['status']='training'; atomic_json(root/'RUN_STATUS.json',state)
        command=[sys.executable,'-u',str(HERE/'train_long.py'),'train','--root',str(root),
                 '--deadline',str(train_deadline)]
        if a.resume: command+=['--resume',str(a.resume)]
        result=bounded(command,train_deadline-time.time(),root,'25,000 Muon updates',watch=True)
        state.update(result)
        if result['exit_code']!=0: raise RuntimeError('Continuous trainer failed; see FAILURE.json/run.log')
        state.update(json.loads((root/'status.json').read_text()))
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
    finally:
        if tracker is not None:
            state['tracking']=finish_tracking(tracker,root,min(time.time()+600,a.deadline-1200))
    atomic_json(root/'RUN_STATUS.json',state)
    backup=bounded([sys.executable,'-u',__file__,str(root),str(a.deadline),'--backup-only'],
                   max(0,a.deadline-time.time()-20),root,'retained results cloud backup')
    state['backup']=backup; atomic_json(root/'RUN_STATUS.json',state)
    print(json.dumps(state),flush=True)
    return 0 if state.get('status')=='schedule_complete' and backup['exit_code']==0 and state.get('tracking',{}).get('status')=='complete' else 1

if __name__=='__main__': raise SystemExit(main())
