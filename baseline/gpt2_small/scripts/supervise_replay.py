"""Bounded, single-update child. No automatic long run or restart."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from validation_watchdog import wait_for_child


def preserve_child_failure(root, error, report):
    """Native aborts/timeouts cannot execute the child's Python exception handler."""
    from rg_gpt2_small.port_debug import write
    output=root/'diagnostic'; path=output/'TPU_PORT_FAILURE.json'
    if path.exists(): return
    stage=output/'diagnostics/current_stage.json'
    write(path,{'status':'process_failed_or_timed_out','attribution':'unconfirmed',
                'exception':str(error),'child_exit_code':report.get('child_exit_code'),
                'last_stage':json.loads(stage.read_text()) if stage.exists() else None,
                'evidence':['../run.log','diagnostics/','REPLAY_SOURCE.json','../commit.txt']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path); parser.add_argument('deadline',type=float)
    parser.add_argument('checkpoint',type=Path)
    args=parser.parse_args(); root=args.root
    report={'status':'starting','deadline_unix':args.deadline,'automatic_long_run':False}
    try:
        if args.deadline<=time.time()+60: raise RuntimeError('Insufficient time for replay.')
        from rg_nanogpt_one_head.continuous_support import CloudPublisher
        sink=CloudPublisher('gs://tpu-builders-504820-ww-continuous8/gpt2small/'+root.name)
        sink.claim({'run_id':root.name,'source_checkpoint':str(args.checkpoint)})
        for name in ('commit.txt','launch.json'): sink.file(root/name,name)
        cmd=[sys.executable,'-X','faulthandler','-u','-m','rg_gpt2_small.replay_update',
             '--checkpoint',str(args.checkpoint),'--data-root','/mnt/disks/rg-data/continuous8/data',
             '--output',str(root/'diagnostic'),'--device','tpu']
        child=subprocess.Popen(cmd)
        report.update(status='running',pid=child.pid)
        (root/'PROBE_STATUS.json').write_text(json.dumps(report,indent=2))
        rc=wait_for_child(child,'MuonClip update-2 replay',args.deadline)
        report['child_exit_code']=rc
        if rc: raise RuntimeError(f'Replay exited with code {rc}; see diagnostic/FIRST_INVALID.json and run.log.')
        state=json.loads((root/'diagnostic/REPLAY_STATUS.json').read_text())
        if state['status']!='one_update_passed': raise RuntimeError('Replay completion not confirmed.')
        report.update(status='one_update_passed')
        return 0
    except Exception as exc:
        report.update(status='failed_or_incomplete',error=str(exc))
        preserve_child_failure(root,exc,report)
        print(json.dumps(report),flush=True)
        return 1
    finally:
        (root/'PROBE_STATUS.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': sys.exit(main())
