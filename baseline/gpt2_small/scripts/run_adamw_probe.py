"""Four updates only, with an external deadline even if XLA never returns."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from validation_watchdog import wait_for_child


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path); parser.add_argument('deadline',type=float)
    args=parser.parse_args(); root=args.root
    report={'status':'running','long_run_started':False,'training_deadline_unix':args.deadline,
            'purpose':'TPU-port numerical debugging; not a performance benchmark'}
    status=root/'PROBE_STATUS.json'
    def persist():
        tmp=status.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)); tmp.replace(status)
    persist()
    try:
        if args.deadline<=time.time(): raise RuntimeError('Diagnostic training window expired before launch')
        cmd=[sys.executable,'-X','faulthandler','-u','-m','rg_gpt2_small.experiment','--config',str(root/'config.yaml'),
             '--data-root','/mnt/disks/rg-data/continuous8/data','--output',str(root/'adamw'),
             '--device','tpu','--stop-after','4','--deadline-unix',str(args.deadline)]
        child=subprocess.Popen(cmd)
        rc=wait_for_child(child,'AdamW port diagnostic, four updates',args.deadline)
        report['child_exit_code']=rc
        if rc: raise RuntimeError(f'AdamW diagnostic exited with code {rc}; inspect run.log and TPU_PORT_FAILURE.json')
        state=json.loads((root/'adamw/status.json').read_text())
        if state['step']!=4: raise RuntimeError('Stopped before completing four updates')
        report.update(status='four_updates_completed',completed_updates=4,
                      interpretation='Diagnostic checks passed only; full validation and long-run stability remain unproven.')
        print(json.dumps(report),flush=True)
        return 0
    except Exception as exc:
        report.update(status='failed_or_incomplete',error=str(exc))
        failure=root/'adamw/TPU_PORT_FAILURE.json'
        if not failure.exists():
            stage=root/'adamw/diagnostics/current_stage.json'
            failure.parent.mkdir(parents=True,exist_ok=True)
            failure.write_text(json.dumps({
                'status':'process_failed_or_timed_out','attribution':'unconfirmed',
                'exception':str(exc),'child_exit_code':report.get('child_exit_code'),
                'last_stage':json.loads(stage.read_text()) if stage.exists() else None,
                'evidence':['../run.log','diagnostics/','../config.yaml','../commit.txt']},indent=2))
        print(json.dumps(report),flush=True)
        return 1
    finally:
        persist()


if __name__=='__main__': sys.exit(main())
