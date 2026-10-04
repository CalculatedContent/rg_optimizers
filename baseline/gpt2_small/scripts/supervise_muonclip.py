"""Supervise one continuous trainer; stop on failure/stall, never restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def write(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)


def prepare_config(template, run_id):
    cfg=yaml.safe_load(Path(template).read_text())
    cfg.update(run_id=run_id, validation_tensor_checks=False, validation_gradient_checks=False,
               finite_update_guard=True, progress_reporting=True, benchmark_sync_every_step=True,
               synchronized_finite_checks=True, checkpoint_before_evaluation=True,
               cloud_checkpoints=True, metrics_interval=25, metrics_steps=[1,2,4], milestones=[0])
    cfg['ww'].update(enabled=True, interval=100, steps=[25], logarithmic=False)
    return cfg


def stop(child):
    if child.poll() is not None: return
    child.terminate()
    try: child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        child.kill(); child.wait(timeout=20)


def watch(child, output, deadline, grace_deadline, stall_seconds=1800):
    last_change=time.monotonic(); previous=None; requested=False
    while True:
        rc=child.poll()
        if rc is not None: return rc
        now=time.time()
        if now>=deadline and not requested:
            (output/'STOP').touch(); requested=True
            print('Allocation cutoff reached; requesting final save.',flush=True)
        if now>=grace_deadline:
            raise RuntimeError('Trainer did not exit within allocation cutoff grace period.')
        path=output/'progress.json'
        if path.exists():
            current=path.read_text()  # atomic file replacement by trainer
            if current!=previous:
                previous=current; last_change=time.monotonic()
        if time.monotonic()-last_change>=stall_seconds:
            raise RuntimeError('No training stage completed/changed for 30 minutes; stopping stalled run.')
        try: return child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            stage=json.loads(previous) if previous else {'stage':'starting','completed_step':0}
            print('[muonclip-watch] '+json.dumps(stage),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path); parser.add_argument('deadline',type=float)
    parser.add_argument('service_deadline',type=float)
    args=parser.parse_args(); root=args.root; output=root/'muonclip'
    base=Path(__file__).resolve().parents[1]
    child=None
    report={'status':'starting','long_run_started':False,'training_deadline_unix':args.deadline,
            'per_tensor_diagnostic':'scalar_flags_no_stack', 'scalar_finite_guard':True,
            'synchronized_optimizer_stages':True, 'checkpoint_before_evaluation':True,
            'automatic_restart':False}
    try:
        if args.deadline<=time.time()+60: raise RuntimeError('Too little allocation time remaining.')
        output.mkdir()  # Refuse reuse/overwrite of any previous run.
        cfg=prepare_config(base/'configs/gpt2_small_fineweb_muonclip_long_ww.yaml',root.name)
        with (root/'config.yaml').open('x') as f: yaml.safe_dump(cfg,f,sort_keys=False)
        from rg_nanogpt_one_head.continuous_support import CloudPublisher
        sink=CloudPublisher(os.environ['RG_GPT2_GCS_URI'])
        sink.claim({'run_id':root.name,'source_commit':os.environ['RG_GPT2_SOURCE_COMMIT']})
        for name in ('config.yaml','commit.txt','launch.json'):
            sink.file(root/name,name)
        print('Cloud upload/checksum verification passed; starting one fresh MuonClip process.',flush=True)
        cmd=[sys.executable,'-X','faulthandler','-u','-m','rg_gpt2_small.experiment',
             '--config',str(root/'config.yaml'),'--data-root','/mnt/disks/rg-data/continuous8/data',
             '--output',str(output),'--device','tpu','--allow-long-run',
             '--deadline-unix',str(args.deadline)]
        child=subprocess.Popen(cmd)
        report.update(status='running',long_run_started=True,pid=child.pid)
        write(root/'RUN_STATUS.json',report)
        rc=watch(child,output,args.deadline,min(args.deadline+300,args.service_deadline-300))
        report['child_exit_code']=rc
        if rc: raise RuntimeError(f'MuonClip exited with code {rc}; inspect run.log.')
        state=json.loads((output/'status.json').read_text())
        report.update(status='completed' if state['completed'] else 'stopped',training=state)
        return 0
    except Exception as exc:
        if child is not None: stop(child)
        report.update(status='failed_or_incomplete',error=str(exc))
        failure=output/'TPU_PORT_FAILURE.json'
        if output.exists() and not failure.exists():
            path=output/'progress.json'
            write(failure,{'attribution':'unconfirmed','exception':str(exc),
                          'child_exit_code':child.returncode if child is not None else None,
                          'last_progress':json.loads(path.read_text()) if path.exists() else None})
        print(json.dumps(report),flush=True)
        return 1
    finally:
        write(root/'RUN_STATUS.json',report)


if __name__=='__main__': sys.exit(main())
