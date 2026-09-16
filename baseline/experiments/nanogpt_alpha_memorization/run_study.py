#!/usr/bin/env python3
"""One command: five AdamW then five ordinary Muon memorization runs on MPS, no online WW."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
LATEST=Path('/tmp/nanogpt_alpha_memorization_latest.txt')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

def timestamp(): return datetime.now().strftime('%Y%m%d_%H%M%S')
def jobs(cfg): return [(arm,seed) for arm in cfg['arms'] for seed in cfg['seeds']]

def resolve_root(value=None,create=False):
    if value: root=Path(value).expanduser().resolve()
    elif create:
        root=(Path('/tmp')/f'nanogpt_alpha_memorization_{timestamp()}').resolve()
        if root.exists(): raise ValueError('Timestamp collision; rerun in one second.')
    else:
        if not LATEST.exists(): raise ValueError('No latest study. Supply --root with its printed path.')
        root=Path(LATEST.read_text().strip()).resolve()
    if create and not any(root.is_relative_to(Path(p).resolve()) and root!=Path(p).resolve() for p in ('/tmp','/private/tmp')):
        raise ValueError('Use a dedicated results directory under /tmp.')
    return root

def preflight(cfg,root,device):
    import copy
    from am_train import train
    c=copy.deepcopy(cfg); c.update(steps=2,behavior_every=1,expensive_every=2,checkpoint_every=1,doses=[0,1],long_per_dose=1,short_per_dose=1,rule_audit_limit=2)
    for arm in c['arms']:
        result=train(c,root/'preflight',arm,1337,device)
        if result['state']!='complete': raise RuntimeError('Preflight failed: '+json.dumps(result))

def export_review(root):
    import nbformat
    from nbclient import NotebookClient
    from am_report_fast import generate
    report=generate(root); out=HERE/'review'/f'{root.name}_{timestamp()}'; out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(report,out/'report'); shutil.copy2(root/'protocol.json',out/'protocol.json')
    template=HERE/'notebooks'/'01_Memorization_Results.ipynb'
    notebook=nbformat.read(template,as_version=4); notebook.cells.insert(0,nbformat.v4.new_code_cell(f'STUDY_ROOT = {str(root)!r}'))
    NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(HERE)}}).execute(); nbformat.write(notebook,out/template.name)
    for path in root.glob('*/seed_*/manifest.json'):
        target=out/'manifests'/path.relative_to(root); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    print(f'Review bundle ready to git add (no weights, no credentials): {out}',flush=True); return out

def stream(command,log):
    with log.open('a') as f:
        child=subprocess.Popen(command,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in child.stdout: print(line,end='',flush=True); f.write(line); f.flush()
            return child.wait()
        except BaseException:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','plan','report','export','worker'],nargs='?',default='run')
    parser.add_argument('--root'); parser.add_argument('--device',choices=['mps','cpu','cuda'],default='mps'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--arm',choices=['adamw','muon']); parser.add_argument('--seed',type=int); parser.add_argument('--steps',type=int); parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv); cfg=json.loads((HERE/'protocol.json').read_text())
    if args.steps is not None: cfg['steps']=args.steps
    if cfg['steps']<2: parser.error('At least two updates required.')
    try:
        if args.command=='plan':
            for arm,seed in jobs(cfg): print(f'{arm:8s} seed={seed} updates={cfg["steps"]}')
            return 0
        root=resolve_root(args.root,create=args.command=='run' and not args.resume)
        if args.command in ('report','export'):
            if args.command=='export': export_review(root)
            else:
                from am_report_fast import generate
                print(generate(root,make_plots=not args.no_plots)/'summary.md')
            return 0
        if args.command=='worker':
            from am_train import train
            cfg=json.loads((root/'protocol.json').read_text())
            if args.arm is None or args.seed not in cfg['seeds']: raise ValueError('Worker needs a declared arm and seed.')
            result=train(cfg,root,args.arm,args.seed,args.device,args.resume); print(json.dumps(result),flush=True); return 0 if result['state']=='complete' else 1
        import fcntl
        if not args.resume:
            root.mkdir(parents=True,exist_ok=False); (root/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
        else:
            saved=json.loads((root/'protocol.json').read_text())
            if saved!=cfg: raise ValueError('Resume protocol changed; repeat the original options.')
        with (root/'.queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); LATEST.write_text(str(root)+'\n'); (root/'logs').mkdir(exist_ok=True)
            os.environ.setdefault('MPLCONFIGDIR',str(root/'cache'/'matplotlib'))
            print(f'Results: {root}\n10 planned runs: AdamW x5 first, then Muon x5. Online WeightWatcher OFF.',flush=True)
            if not args.resume: preflight(cfg,root,args.device)
            statuses=[]
            for arm,seed in jobs(cfg):
                command=[sys.executable,'-u',str(HERE/'run_study.py'),'worker','--root',str(root),'--device',args.device,'--arm',arm,'--seed',str(seed)]
                if args.resume: command+=['--resume']
                code=stream(command,root/'logs'/f'{arm}_{seed}.log'); statuses.append({'arm':arm,'seed':seed,'returncode':code})
                (root/'queue_status.json').write_text(json.dumps(statuses,indent=2)+'\n')
                from am_report_fast import generate
                try: generate(root,make_plots=False)
                except (ValueError,OSError) as exc: print(f'Report warning: {exc}',flush=True)
            from am_report_fast import generate
            print(generate(root,make_plots=not args.no_plots)/'summary.md'); return int(any(r['returncode'] for r in statuses))
    except KeyboardInterrupt:
        print('\nStopped. Results and safe checkpoints retained.',file=sys.stderr); return 130
    except (OSError,ValueError,RuntimeError) as exc:
        print(f'Error: {exc}',file=sys.stderr); return 1

if __name__=='__main__': raise SystemExit(main())
