"""Two-process CPU/Gloo integration of the TPU scheduler; no TPU emulation.

Uses real model updates, optimizer checkpoints and evaluation on synthetic data.
Only alpha fitting is stubbed: this test verifies orchestration, not spectral fits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from unittest.mock import patch


def worker(root, simulate=False):
    import comparison as c
    import tpu_parallel
    from test_audit import tiny_design,tiny_data
    def prepare(path,data):
        c.freeze(path/'design.json',tiny_design())
    alpha=dict(mean_alpha=None,below_two=0,successful_fits=0,unique_matrices=18,
               method='TEST STUB: no spectral fitting')
    args=argparse.Namespace(cores=2,output=root,data=root/'synthetic',preflight=False)
    with ExitStack() as stack:
        if simulate:
            import torch.distributed as dist
            rank=int(os.environ['RANK']);counter=0
            def barrier():
                nonlocal counter
                directory=root.parent/'barriers';directory.mkdir(exist_ok=True)
                (directory/f'{counter}_{rank}').touch()
                deadline=time.monotonic()+90
                while not all((directory/f'{counter}_{r}').exists() for r in range(2)):
                    if time.monotonic()>deadline:raise TimeoutError('Simulated barrier timeout')
                    time.sleep(.01)
                counter+=1
            for name,replacement in [('init_process_group',lambda *a,**k:None),('destroy_process_group',lambda:None),('get_rank',lambda:rank),('get_world_size',lambda:2),('barrier',barrier)]:
                stack.enter_context(patch.object(dist,name,replacement))
        stack.enter_context(patch.object(c,'prepare',side_effect=prepare))
        stack.enter_context(patch.object(c.base,'load_data',side_effect=tiny_data))
        stack.enter_context(patch.object(c.base,'sampled_loss',side_effect=lambda m,d:c.base.full_eval(m,d)['loss_nats_per_character']))
        stack.enter_context(patch.object(c.base,'spectrum',return_value=alpha))
        return tpu_parallel.work(args,backend='gloo')


def execute(cmd,env,root,simulate):
    if not simulate:
        subprocess.run(cmd,env=env,check=True,timeout=180)
        return
    import shutil
    shutil.rmtree(root.parent/'barriers',ignore_errors=True)
    children=[]
    try:
        for rank in range(2):
            rank_env=dict(env,RANK=str(rank),WORLD_SIZE='2')
            children.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(root),'--simulate'],env=rank_env))
        deadline=time.monotonic()+180
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Simulated rank failed')
            if time.monotonic()>deadline:raise TimeoutError('Simulated integration timed out')
            time.sleep(.1)
        assert all(p.returncode==0 for p in children)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--simulate',action='store_true');a=p.parse_args()
    if a.worker:return worker(a.worker,a.simulate)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)/'study'
        cmd=[sys.executable,'-m','torch.distributed.run','--rdzv-backend=c10d','--rdzv-endpoint=127.0.0.1:0','--local-addr=127.0.0.1','--nproc_per_node=2',
             '--max_restarts=0',str(Path(__file__).resolve()),'--worker',str(root)]
        env=os.environ.copy();env['SHORT_GPT_DEVICE']='cpu';env['GLOO_SOCKET_IFNAME']='lo'
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]='1'
        execute(cmd,env,root,a.simulate)
        assert len(list((root/'calibration').glob('*/summary_2.json')))==12
        assert len(list((root/'calibration').glob('*/summary_3.json')))==4
        assert len(list((root/'confirmation').glob('*/result.json')))==4
        finals=json.loads((root/'finalists.json').read_text())
        assert all(len(v)==2 for v in finals['groups'].values())
        records=list(root.glob('**/last.pt'))+[root/'recipes.json',root/'finalists.json']
        before={str(p):p.read_bytes() for p in records}
        execute(cmd,env,root,a.simulate)
        assert all(p.read_bytes()==before[str(p)] for p in records),'Resume rewrote completed work'
        print('PASS: two-process CPU orchestration, all three stages, completed-job resume, frozen selections; communication=' + ('FILE-BARRIER SIMULATION' if a.simulate else 'Gloo'))
    return 0

if __name__=='__main__':raise SystemExit(main())
