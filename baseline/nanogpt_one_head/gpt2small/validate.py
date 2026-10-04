"""Bounded validation only. Called on the existing TPU after clean shutdown."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

p=argparse.ArgumentParser()
p.add_argument('--root',required=True); p.add_argument('--data',required=True)
p.add_argument('--deadline',type=float,required=True)
a=p.parse_args(); root=Path(a.root); base=Path(__file__).resolve().parents[1]
root.mkdir(parents=True,exist_ok=True)
for folder in ('logs','metrics','ww_metrics','checkpoints','summaries','configs'): (root/folder).mkdir(exist_ok=True)
configs={}
for opt in ('adamw','muonclip'):
    cfg=yaml.safe_load((base/'configs'/f'gpt2_small_fineweb_{opt}_baseline.yaml').read_text())
    cfg['run_id']=root.name+'_'+opt
    # Only control stop points here. Keep the planned LR/token horizon immutable across resume.
    cfg['metrics_interval']=4
    cfg['benchmark_sync_every_step']=True
    if opt=='muonclip': cfg['ww']['steps']=[4,25]
    path=root/'configs'/f'{opt}.yaml'; path.write_text(yaml.safe_dump(cfg,sort_keys=False)); configs[opt]=path
report={'status':'running','deadline_unix':a.deadline,'long_run_started':False,'phases':[]}

def persist():
    (root/'summaries/validation_report.json').write_text(json.dumps(report,indent=2))

def history(run):
    return {str(x.relative_to(run)):hashlib.sha256(x.read_bytes()).hexdigest()
            for folder in ('metrics','ww_metrics') for x in (run/folder).glob('*.json')}

try:
    for stop in (4,25):
        for opt in ('adamw','muonclip'):
            # Reserve five minutes for final checkpoint and backup; never start after the cutoff.
            if time.time()>a.deadline-300: raise RuntimeError('Insufficient allocation time for next phase; no new allocation requested')
            run=root/opt; old=history(run)
            cmd=[sys.executable,'-u','-m','rg_nanogpt_one_head.gpt2_experiment',
                 '--config',str(configs[opt]),'--data-root',a.data,'--output',str(run),
                 '--device','tpu','--stop-after',str(stop),'--deadline-unix',str(a.deadline-300)]
            if stop>4: cmd.append('--resume')
            print('START',opt,'through step',stop,flush=True)
            with (root/'logs'/f'{opt}_{stop}.log').open('x') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            state=json.loads((run/'status.json').read_text())
            if state['step']!=stop: raise RuntimeError(f'{opt} stopped before requested step {stop}')
            now=history(run)
            if any(now.get(k)!=v for k,v in old.items()): raise RuntimeError('Resume changed previous scientific history')
            rows=[json.loads(x.read_text()) for x in sorted((run/'metrics').glob('*.json'))]
            initial,latest=rows[0],rows[-1]
            if latest['train_nll']>=initial['train_nll']: raise RuntimeError(f'{opt} training probe NLL did not decrease; inspect before proceeding')
            if latest['val_nll']>initial['val_nll']+1: raise RuntimeError(f'{opt} validation NLL increased substantially')
            ww=[]
            for x in (run/'ww_metrics').glob('*.json'):
                measured=json.loads(x.read_text()); rec=measured['records']
                if len(rec)!=72 or len({r['matrix_name'] for r in rec})!=72: raise RuntimeError('WW matrix inventory failure')
                if any(sum(r['matrix_type']==kind for r in rec)!=12 for kind in ('Q','K','V','O','MLP_IN','MLP_OUT')): raise RuntimeError('WW type counts incorrect')
                ww.append({'file':x.name,'seconds':measured['seconds'],
                           'raw_success':sum(r['raw_fit_status']=='success' for r in rec),
                           'clipped_success':sum(r['clipped_fit_status']=='success' for r in rec),
                           'null_success':sum(r['null_status']=='success' for r in rec)})
            if opt=='muonclip' and (not ww or any(x['null_success']!=72 for x in ww)):
                raise RuntimeError('Randomized WW control incomplete; see retained records')
            report['phases'].append({'optimizer':opt,'step':stop,'metrics':latest,'ww':ww,
                                    'resume_history_unchanged':bool(old)})
            persist(); print(json.dumps(report['phases'][-1]),flush=True)
            subprocess.run([sys.executable,str(base/'gpt2small/analyze.py'),str(run)],check=True)
    report['status']='short_validation_completed'
    report['interpretation']='25 updates demonstrate functionality and initial direction only; not reproduction of a NanoGPT speedrun benchmark or proof of long-run stability.'
except Exception as exc:
    report['status']='failed_or_incomplete'; report['error']=str(exc); persist(); raise
finally:
    persist()
