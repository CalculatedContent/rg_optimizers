"""Freeze ten seeds and run them sequentially; recover checkpointed interruptions."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from common import ARCHITECTURE, INPUT_SHA, MODEL_SHA, SOURCE, sha, source_hashes, write_json

def admission():
    root=Path('/sys/fs/cgroup')
    if not (root/'memory.stat').exists(): return True
    limit=(root/'memory.max').read_text().strip()
    if limit=='max': return True
    stat=dict((k,int(v)) for k,v in (line.split() for line in (root/'memory.stat').read_text().splitlines()))
    return int(limit)-stat['anon']-stat['kernel']>=950*2**20

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(1338,1348)))
    p.add_argument('--prepare-only',action='store_true'); a=p.parse_args()
    root=a.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    lock=(root/'batch.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert len(a.seeds)==len(set(a.seeds))==10 and 1337 not in a.seeds
    assert sha(a.data)==INPUT_SHA and sha(SOURCE/'vendor/model.py')==MODEL_SHA
    packages={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','weightwatcher','powerlaw')}
    assert packages['weightwatcher']=='0.7.7'
    manifest=dict(schema=1,study='ten fresh seeds of the short character GPT pilot',seeds=a.seeds,
                  source_sha256=source_hashes(),input_sha256=INPUT_SHA,model_sha256=MODEL_SHA,
                  architecture=ARCHITECTURE,trainable_parameters=804096,batch_size=12,threads=1,
                  split='Fixed contiguous 90/5/5; all 65 corpus characters define vocabulary',
                  optimizer=dict(name='AdamW',betas=[.9,.99],weight_decay=.1,grad_clip=1.),
                  schedule='Original first 2000 updates: 100-step warmup to .001, cosine to .0001. Thereafter .0001, one plateau-triggered halving to .00005.',
                  stopping='Full validation every500 from step2000; 3 checks without cumulative .005-nat improvement trigger LR halving; repeat plateau stops. Cap20000 is not convergence.',
                  checkpoint_every=250,max_steps=20000,selection='Lowest full validation loss from step2000; endpoint reported separately',
                  alpha='Raw default WW0.7.7; all18 unique matrices; packed QKV; tied token/output weights counted once; no alpha selection.',
                  test_policy='Same fixed text holdout as the pilot, already inspected. Ten seeds and all settings frozen before training. No test at step2000; final endpoint and validation-selected test scores are descriptive and cannot trigger retuning.',
                  variation='Only model/minibatch/random seeds; no hyperparameter search',versions=packages)
    path=root/'manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest,'Frozen batch manifest mismatch'
    else: write_json(path,manifest)
    if a.prepare_only: print(json.dumps(manifest)); return
    for seed in a.seeds:
        out=root/'runs'/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists(): continue
        while not admission():
            write_json(root/'status.json',dict(state='waiting_for_memory',seed=seed,utc=datetime.now(timezone.utc).isoformat()))
            time.sleep(30)
        write_json(root/'status.json',dict(state='running',seed=seed,completed=sum((root/'runs'/f'seed_{s}'/'result.json').exists() for s in a.seeds),planned=10,utc=datetime.now(timezone.utc).isoformat()))
        with (out/'training.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(SOURCE/'train.py'),'--output',str(out),'--data',str(a.data.resolve()),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
            code=child.wait()
        if code:
            write_json(root/'status.json',dict(state='needs_resume',seed=seed,exit_code=code,utc=datetime.now(timezone.utc).isoformat()))
            raise SystemExit(code)
        subprocess.run([sys.executable,str(SOURCE/'summarize.py'),'--output',str(root)],check=True)
    write_json(root/'status.json',dict(state='complete',completed=10,planned=10,utc=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__': main()
