"""One independent trial per TPU process, up to eight active local devices."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys


def assigned(jobs, rank, cores):
    if not 1 <= cores <= 8:
        raise ValueError('cores must be between 1 and 8')
    return [job for index, job in enumerate(jobs) if index % cores == rank]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cores', type=int, choices=range(1, 9), default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--preflight', action='store_true', help='Test every active device without training the study')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return p


def launch(a):
    # Configure topology before spawning; never initialize accelerator tensors here.
    import torch
    import torch_tpu  # noqa: F401
    torch.tpu.distributed.set_tpu_launch_env()
    world = int(os.environ['WORLD_SIZE'])
    nodes = int(os.environ.get('NNODES', '1'))
    node_rank = int(os.environ.get('NODE_RANK', '0'))
    if nodes != 1 or node_rank != 0 or not a.cores <= world <= 8:
        raise RuntimeError('This launcher requires a single-host slice with cores <= local WORLD_SIZE <= 8')
    env = os.environ.copy()
    env['SHORT_GPT_DEVICE'] = 'tpu'
    command = [sys.executable, '-m', 'torch.distributed.run', '--rdzv-backend=c10d','--rdzv-endpoint=127.0.0.1:0','--local-addr=127.0.0.1',
               f'--nproc_per_node={world}', '--max_restarts=0', str(Path(__file__).resolve()),
               '--worker', '--cores', str(a.cores), '--output', str(a.output.resolve()),
               '--data', str(a.data.resolve())]
    if a.preflight:
        command.append('--preflight')
    return subprocess.call(command, env=env)


def work(a, backend="tpu_dist"):
    from datetime import timedelta
    import math
    import signal
    import time
    import unittest
    import torch
    if backend == 'tpu_dist':
        import torch_tpu  # noqa: F401
    elif backend != 'gloo':
        raise ValueError('Only tpu_dist or the CPU integration-test backend gloo is supported')
    import torch.distributed as dist
    import comparison as c
    import runtime
    runtime.DEVICE = 'tpu' if backend == 'tpu_dist' else 'cpu'
    root, data = a.output.resolve(), a.data.resolve()
    root.mkdir(parents=True, exist_ok=True)
    held = []
    if int(os.environ['RANK']) == 0:
        held = [c.lock(root/'batch.lock'), c.lock(root.parent/'short_gpt_training.lock')]
    # Official TorchTPU order: topology environment -> process group -> local device.
    dist.init_process_group(backend, timeout=timedelta(hours=24))
    rank, world = dist.get_rank(), dist.get_world_size()
    if not a.cores <= world <= 8:
        raise RuntimeError('Invalid worker/core count')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime.initialize()
    start = time.perf_counter()
    try:
        if rank == 0:
            c.write(root/'parallel_launch.json', dict(cores=a.cores,processes=world,utc=c.utc(),
                    mode='independent trials, one per active device',backend=backend,preflight=a.preflight,
                    topology=os.environ.get('TORCH_TPU_TOPOLOGY')))
        dist.barrier()
        if a.preflight:
            if rank < a.cores:
                from test_portable import PortableTests
                result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PortableTests))
                c.write(root/f'preflight_rank{rank}.json',dict(rank=rank,passed=result.wasSuccessful(),
                        seconds=time.perf_counter()-start,environment=runtime.environment(),utc=c.utc()))
                if not result.wasSuccessful():
                    raise RuntimeError(f'Preflight failed on rank {rank}')
            dist.barrier()
            return 0
        if rank == 0:
            c.prepare(root,data)
        dist.barrier()
        d = c.verify(root)
        signal.signal(signal.SIGTERM,c.stop_signal)
        signal.signal(signal.SIGINT,c.stop_signal)

        def stage(jobs):
            for phase, recipe, seed, target in assigned(jobs,rank,a.cores):
                if c.STOP or (root/'PAUSE').exists():
                    raise RuntimeError('Paused: resume from the saved checkpoints')
                out = root/phase/f'{recipe["id"]}_seed{seed}'
                out.mkdir(parents=True,exist_ok=True)
                c.freeze(out/'job.json',dict(phase=phase,recipe=recipe,seed=seed))
                done = out/('result.json' if phase=='confirmation' else f'summary_{target}.json')
                if done.exists() or (out/'numerical_failure.json').exists():
                    continue
                c.write(root/f'rank{rank}_status.json',dict(job=str(out),target=target,utc=c.utc()))
                code = c.worker(root,data,out/'job.json',target)
                if code:
                    raise RuntimeError(f'Worker rank {rank} needs resume: exit {code}')
                gc.collect()
            dist.barrier()  # Complete every trial before a selection transition.

        seed = d['calibration_seed']
        stage([('calibration',r,seed,d['screen_steps']) for r in d['candidates']])
        if rank == 0:
            groups = {name:sorted([r for r in d['candidates'] if r['optimizer']==name],
                      key=lambda r:c.ranking(root,r,seed,d['screen_steps']))[:d['finalists_per_optimizer']]
                      for name in ['AdamW','SGDM']}
            assert all(math.isfinite(c.ranking(root,r,seed,d['screen_steps'])[0]) for g in groups.values() for r in g)
            c.freeze(root/'finalists.json',dict(screen_steps=d['screen_steps'],groups=groups))
        dist.barrier()
        groups = c.read(root/'finalists.json')['groups']
        stage([('calibration',groups[name][i],seed,d['max_steps'])
               for i in range(d['finalists_per_optimizer']) for name in ['AdamW','SGDM']])
        if rank == 0:
            winners = {name:min(group,key=lambda r:c.ranking(root,r,seed,d['max_steps'])) for name,group in groups.items()}
            assert all(math.isfinite(c.ranking(root,r,seed,d['max_steps'])[0]) for r in winners.values())
            c.freeze(root/'recipes.json',dict(design_sha256=c.sha(root/'design.json'),selected=winners,
                calibration_evidence={name:dict(candidate=r['id'],ranking=c.ranking(root,r,seed,d['max_steps'])) for name,r in winners.items()},
                confirmation_seeds=d['confirmation_seeds'],selection='Validation only; frozen before confirmation test evaluation'))
        dist.barrier()
        winners = c.read(root/'recipes.json')['selected']
        stage([('confirmation',winners[name],seed,d['max_steps'])
               for i,seed in enumerate(d['confirmation_seeds'])
               for name in (['AdamW','SGDM'] if i%2==0 else ['SGDM','AdamW'])])
        if rank == 0:
            c.summarize(root)
            c.write(root/'status.json',dict(state='training_finished_analysis_pending',
                completed_confirmation=c.completed(root),planned_confirmation=10,
                numerical_failures=len(list((root/'confirmation').glob('*/numerical_failure.json'))),
                launch_wall_seconds=time.perf_counter()-start,utc=c.utc()))
        dist.barrier()
        return 0
    finally:
        # On failure torchrun terminates sibling workers; checkpoints remain resumable.
        for handle in held:
            handle.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parser().parse_args()
    raise SystemExit(work(args) if args.worker else launch(args))
