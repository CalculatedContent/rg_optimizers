"""Bounded setup, optional flash check, one fresh training run, verified backup."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(root, name, value):
    temp = root/(name+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(root/name)


def bounded(command, seconds, root, label, watch=False):
    if seconds <= 0:
        return {'exit_code':None, 'timed_out':True, 'phase':label}
    print('START '+label, flush=True)
    child = subprocess.Popen(command, start_new_session=True)
    end = time.monotonic()+seconds
    try:
        while True:
            remaining = end-time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, seconds)
            try:
                return {'exit_code':child.wait(timeout=min(30, remaining)), 'timed_out':False, 'phase':label}
            except subprocess.TimeoutExpired:
                print('WAIT '+label+': '+str(int(remaining))+'s to phase cutoff', flush=True)
                status = root/'status.json'
                if watch and status.exists() and time.time()-status.stat().st_mtime > 900:
                    raise RuntimeError('No training progress recorded for 15 minutes')
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)
        return {'exit_code':child.returncode, 'timed_out':True, 'phase':label, 'error':str(exc)}


def backup(root):
    from rg_nanogpt_one_head.continuous_support import CloudPublisher
    publisher = CloudPublisher('gs://tpu-builders-504820-ww-continuous8/gpt2small/'+root.name)
    receipts = []
    # Upload small scientific tables first. Retain immutable spectral weights on /mnt;
    # current full-state checkpoints keep their existing cloud backup behavior.
    for path in sorted((root/'tracking').rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.csv'):
            receipts.append(publisher.file(path, path.relative_to(root).as_posix()))
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.pt', '.txt', '.log'):
            # Logs may still grow as supervisor/upload output is appended.
            if path.suffix == '.log':
                publisher.snapshot_text_file(path, path.name)
            else:
                receipts.append(publisher.file(path, path.name))
    publisher.json({'files':receipts, 'status':'verified'}, 'CLOUD_BACKUP_VERIFIED.json')
    write(root, 'CLOUD_BACKUP_VERIFIED.json', {'files':receipts, 'status':'verified'})
    print('Cloud backup verified: '+root.name, flush=True)


def start_tracking(root, deadline):
    # CPU-only child; never import WeightWatcher in the training process.
    env = {**os.environ, 'PJRT_DEVICE':'CPU', 'CUDA_VISIBLE_DEVICES':'',
           'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
    command = [sys.executable, '-u', str(Path(__file__).with_name('tracking.py'))]
    subprocess.run(command+['check','--root',str(root)], env=env, check=True,
                   timeout=min(120, max(1, deadline-time.time())))
    return subprocess.Popen(command+['watch','--root',str(root),'--deadline',str(deadline)],
                            env=env, start_new_session=True)


def finish_tracking(child, root, deadline):
    from tracking import tracking_status
    folder = root/'tracking'
    folder.mkdir(exist_ok=True)
    (folder/'TRAINING_DONE').touch()
    try:
        while child.poll() is None and time.time() < deadline:
            try:
                child.wait(timeout=min(30, max(.1, deadline-time.time())))
            except subprocess.TimeoutExpired:
                print('WAIT CPU WeightWatcher: draining saved snapshots', flush=True)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
    finally:
        state = tracking_status(root, 'complete' if child.returncode == 0 else 'incomplete',
                                exit_code=child.returncode)
    if state['pending'] or state['failed']:
        state = tracking_status(root, 'incomplete', exit_code=child.returncode)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('deadline', type=float)
    p.add_argument('--optimizer', choices=('muon','adam','adamw'), default='muon')
    p.add_argument('--microbatch', type=int, choices=(32,64,128), default=64)
    p.add_argument('--attention', choices=('auto','flash','math'), default='flash')
    p.add_argument('--backup-only', action='store_true')
    a = p.parse_args()
    if a.backup_only:
        backup(a.root)
        return 0
    here = Path(__file__).resolve().parent
    run = {'status':'preparing', 'target_met':False, 'automatic_restart':False,
           'deadline_unix':a.deadline, 'optimizer':a.optimizer}
    write(a.root, 'RUN_STATUS.json', run)
    common = [sys.executable, '-u', str(here/'run.py')]
    args = ['--root',str(a.root),'--microbatch',str(a.microbatch)]
    train_deadline = a.deadline-600
    tracker = None
    try:
        if a.attention == 'math' and a.microbatch > 64:
            raise RuntimeError('Mathematical attention with microbatch 128 exceeded this TPU memory; use <=64')
        if a.attention != 'math':
            installed = bounded([sys.executable,'-u',str(here/'pallas_dependencies.py'),str(a.root)],
                                min(600,train_deadline-time.time()-300),a.root,'pinned Pallas dependencies')
            if installed['exit_code'] != 0:
                write(a.root,'PALLAS_DEPENDENCY_FAILURE.json',installed)
                raise RuntimeError('Pallas dependency setup failed; training not started')
            os.environ['PYTHONPATH'] = str(a.root/'pallas-deps')+os.pathsep+os.environ.get('PYTHONPATH','')
        prep_deadline = min(time.time()+900, train_deadline-300)
        result = bounded(common+['prepare',*args,'--deadline',str(prep_deadline)],
                         prep_deadline-time.time(), a.root, 'benchmark data')
        if result['exit_code'] != 0:
            raise RuntimeError('Data preparation failed: '+str(result))
        attention = 'math'
        if a.attention != 'math':
            check_deadline = min(time.time()+300, train_deadline-300)
            checked = bounded(common+['attention-check',*args,'--deadline',str(check_deadline)],
                              check_deadline-time.time(), a.root, 'TPU flash attention forward/backward')
            if checked['exit_code'] == 0:
                attention = 'flash'
            else:
                write(a.root, 'ATTENTION_CHECK_FAILURE.json', checked)
                raise RuntimeError('TPU flash attention failed validation; no automatic mathematical-attention fallback')
        tracker = start_tracking(a.root, train_deadline)
        run.update(status='training', attention=attention)
        write(a.root, 'RUN_STATUS.json', run)
        result = bounded(common+['train',*args,'--deadline',str(train_deadline),
                                 '--attention',attention,'--optimizer',a.optimizer],
                         train_deadline-time.time(), a.root, '3,000-update '+a.optimizer+' recipe', watch=True)
        run.update(result)
        if result['exit_code'] != 0:
            run['status'] = 'failed_or_timed_out'
            failure = a.root/'FAILURE.json'
            if failure.exists():
                run['error'] = json.loads(failure.read_text()).get('error','')[:2000]
        else:
            status = json.loads((a.root/'status.json').read_text())
            run.update(status=status['status'], target_met=status['target_met'], step=status['step'])
    except Exception as exc:
        run.update(status='failed', error=str(exc))
    finally:
        if tracker is not None:
            run['tracking'] = finish_tracking(tracker, a.root, train_deadline)
    if run['status'] in ('failed','failed_or_timed_out'):
        previous = {}
        if (a.root/'status.json').exists():
            previous = json.loads((a.root/'status.json').read_text())
        write(a.root,'status.json',{'status':run['status'],'last_recorded_step':previous.get('step'),
                                  'error':run.get('error'), 'target_met':False})
    write(a.root, 'RUN_STATUS.json', run)
    backed = bounded([sys.executable,'-u',__file__,str(a.root),str(a.deadline),'--backup-only'],
                     min(590, a.deadline-time.time()-10), a.root, 'cloud backup')
    run['backup'] = backed
    write(a.root, 'RUN_STATUS.json', run)
    print(json.dumps(run), flush=True)
    return 0 if (run['status'] in ('target_reached','schedule_complete_target_not_met')
                 and run.get('tracking', {}).get('status') == 'complete') else 1


if __name__ == '__main__':
    sys.exit(main())
