"""TPU correctness port launcher. No CUDA reference modifications or hardware provisioning."""
import argparse
import datetime
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time

from capacity import HERE, GIB, load_contract, sha256, verify_reference


def source_fingerprint():
    paths=[HERE/'experiment.py',HERE/'capacity.py',HERE/'port_contract.json']
    paths+=sorted((HERE/'tpu_port').glob('*.py'))
    paths+=[HERE/'requirements-tpu.txt']
    return {p.relative_to(HERE).as_posix():sha256(p) for p in paths}


def available_host_memory():
    # MemAvailable avoids counting the page cache as permanently occupied.
    fields={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    available=int(fields['MemAvailable'].split()[0])*1024
    # Respect a container's memory limit, when present.
    for base in [Path('/sys/fs/cgroup')]:
        maximum=base/'memory.max'; current=base/'memory.current'
        if maximum.exists() and current.exists() and maximum.read_text().strip()!='max':
            available=min(available,max(0,int(maximum.read_text())-int(current.read_text())))
    return available


def data_manifest(root):
    folder=Path(root)/'data/fineweb10B'
    expected=['fineweb_val_000000.bin']+[f'fineweb_train_{i:06d}.bin' for i in range(1,10)]
    actual={p.name for p in folder.glob('fineweb_*.bin')}
    if actual!=set(expected):
        raise RuntimeError('Expected exactly the pinned downloader\'s validation shard and nine training shards; use prepare in a clean data root')
    result={}
    import struct
    for name in expected:
        path=folder/name
        with path.open('rb') as handle: header=handle.read(1024)
        if len(header)!=1024: raise RuntimeError('Truncated shard header: '+name)
        magic,version,tokens=struct.unpack_from('<iii',header)
        if (magic,version)!=(20240520,1) or tokens<=0 or path.stat().st_size!=1024+2*tokens:
            raise RuntimeError('Invalid FineWeb shard: '+name)
        if name.startswith('fineweb_val') and tokens<10485761:
            raise RuntimeError('Validation shard too short')
        result[name]={'bytes':path.stat().st_size,'tokens':tokens,'sha256':sha256(path)}
    return result


def reference_module():
    path=HERE.parent/load_contract()['reference_folder']/'experiment.py'
    spec=importlib.util.spec_from_file_location('pinned_cuda_reference',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def runtime_environment():
    for variable in ('XLA_USE_SPMD','XLA_AUTO_SPMD','XLA_USE_BF16','XLA_DOWNCAST_BF16','NUM_SCHEDULED_ITERATIONS','TRAIN_SEED'):
        if variable in os.environ:
            raise RuntimeError('Unset '+variable+'; this port fixes execution, precision and schedule explicitly')
    if os.environ.get('PJRT_DEVICE','TPU')!='TPU':
        raise RuntimeError('Production/preflight requires PJRT_DEVICE=TPU')
    os.environ['PJRT_DEVICE']='TPU'
    os.environ.setdefault('OMP_NUM_THREADS','4')
    torch_version=importlib.metadata.version('torch')
    xla_version=importlib.metadata.version('torch_xla')
    versions=[torch_version.split('+')[0],xla_version.split('+')[0]]
    if versions!=['2.9.0','2.9.0']:
        raise RuntimeError('Use matching torch==2.9.0 and torch_xla==2.9.0: '+repr(versions))
    return {'torch':torch_version,'torch_xla':xla_version,
            'python':sys.version,'host_available_bytes':available_host_memory()}


def tracking_command(action,root):
    return [sys.executable,'-m','tpu_port.tracking',action,'--root',str(root)]


def tracking_environment():
    env=os.environ.copy()
    env.update(PJRT_DEVICE='CPU',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    return env


def probe_subprocess(cmd,root):
    # Poll failure reports even when compilation emits no stdout. On an OOM,
    # stop the entire launch process group, including ranks blocked in a collective.
    with (root/'console.log').open('w') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                for path in root.glob('probe-rank-*.json'):
                    report=json.loads(path.read_text())
                    if report.get('status')=='failed':
                        raise RuntimeError(f"Probe rank {report['rank']} failed at {report['failed_stage']}: {report['exception']}")
                try: process.wait(timeout=1)
                except subprocess.TimeoutExpired: pass
            return process.returncode
        finally:
            # Kill remaining descendants too if a rank or the launch parent failed.
            if process.poll() is None or process.returncode:
                try: os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()


def launch(action,data_root,results_root,receipt=None):
    verify_reference()
    results_root=Path(results_root).resolve(); results_root.mkdir(parents=True,exist_ok=True)
    needed=(160 if action=='run' else 2 if action=='preflight' else 0)*GIB
    if shutil.disk_usage(results_root).free<needed:
        raise RuntimeError(f'Need at least {needed//GIB} GiB free results disk; no TPU allocation made')
    environment=runtime_environment()
    if action not in ('check','probe') and environment['host_available_bytes']<192*GIB:
        raise RuntimeError('Need at least 192 GiB available host RAM for full table, data and CPU buffers; no allocation made')
    data=None if action in ('check','probe') else data_manifest(data_root)
    fingerprint=source_fingerprint()
    if action=='run':
        if receipt is None:
            # Preflight is a real fresh process run, not a fabricated ready flag.
            checked=launch('preflight',data_root,results_root)
            receipt=checked/'PREFLIGHT_COMPLETE.json'
        receipt=Path(receipt).resolve()
        passed=json.loads(receipt.read_text())
        previous=json.loads((receipt.parent/'run_manifest.json').read_text())
        tracking=json.loads((receipt.parent/'TRACKING_STATUS.json').read_text())
        if (passed['kind']!='preflight' or passed['training_steps']!=30 or
                previous['status']!='complete' or tracking['status']!='complete' or
                previous['source_files']!=fingerprint or previous['data']!=data or
                previous['environment']['torch']!=environment['torch'] or
                previous['environment']['torch_xla']!=environment['torch_xla']):
            raise RuntimeError('Preflight receipt does not qualify this source/data/environment')
    root=results_root/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+action+'-'+secrets.token_hex(4))
    root.mkdir()
    options={'action':action,'data_root':str(Path(data_root).resolve()),'output':str(root),
             'reference':str(HERE.parent/load_contract()['reference_folder']),
             'seed':secrets.randbits(31)}
    manifest={'mode':'tpu-port','target':'v5litepod-8','environment':environment,
              'source_files':fingerprint,'data':data,'options':options,
              'preflight_receipt':str(receipt) if receipt else None,
              'status':'started','convergence_verified':False}
    (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    tracking_process=None; tracking_log=None
    try:
        if action not in ('check','probe'):
            # CPU-only subprocess: analysis cannot mutate training tensors or consume training RNG.
            subprocess.run(tracking_command('check',root),cwd=HERE,env=tracking_environment(),check=True)
            tracking_log=(root/'weightwatcher.log').open('w')
            tracking_process=subprocess.Popen(tracking_command('watch',root),cwd=HERE,
                env=tracking_environment(),stdout=tracking_log,stderr=subprocess.STDOUT)
        # Execute in a fresh interpreter: libtpu must not be initialized in the spawn parent.
        cmd=[sys.executable,str(HERE/'experiment.py'),'_worker-launch','--options',json.dumps(options)]
        if action=='probe':
            code=probe_subprocess(cmd,root)
        else:
            with (root/'console.log').open('w') as log:
                process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                try:
                    for line in process.stdout:
                        print(line,end='',flush=True); log.write(line); log.flush()
                        if tracking_process is not None and tracking_process.poll() is not None:
                            raise RuntimeError('WeightWatcher worker exited early; inspect '+str(root/'weightwatcher.log'))
                    code=process.wait()
                except BaseException:
                    process.terminate(); process.wait(); raise
        if code: raise RuntimeError('TPU subprocess failed; inspect '+str(root/'console.log'))
        if action=='probe':
            from tpu_port.probe_report import summarize
            if summarize(root,final=True)['status']!='complete':
                raise RuntimeError('Probe did not complete all eight ranks, including full-size evaluation')
        if tracking_process is not None:
            (root/'tracking/TRAINING_DONE').touch()
            while tracking_process.poll() is None:
                print('Waiting for queued WeightWatcher snapshots; see '+str(root/'TRACKING_STATUS.json'),flush=True)
                try: tracking_process.wait(timeout=30)
                except subprocess.TimeoutExpired: pass
            if tracking_process.returncode:
                raise RuntimeError('WeightWatcher analysis failed; inspect '+str(root/'TRACKING_STATUS.json'))
            tracking=json.loads((root/'TRACKING_STATUS.json').read_text())
            if tracking['status']!='complete' or tracking['completed']!=tracking['snapshots'] or not tracking['completed']:
                raise RuntimeError('WeightWatcher snapshot coverage incomplete')
            manifest['weightwatcher_complete']=True
        manifest['status']='complete'
        if action=='run':
            result=json.loads((root/'FINAL_RESULT.json').read_text())
            result['weightwatcher_complete']=True
            (root/'FINAL_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
            manifest['convergence_verified']=result['target_reached'] and result['weights_complete']
        (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Results:',root,flush=True)
        return root
    except BaseException as error:
        manifest['status']='failed'; manifest['error']=str(error)
        (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        raise
    finally:
        if action=='probe':
            from tpu_port.probe_report import summarize
            summarize(root,final=True,exception=manifest.get('error'))
        if tracking_process is not None and tracking_process.poll() is None:
            tracking_process.terminate(); tracking_process.wait()
        if tracking_log is not None: tracking_log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','verify','prepare','check','probe','preflight','run','cpu-smoke','_worker-launch'])
    parser.add_argument('--data-root',type=Path,default=HERE/'cache')
    parser.add_argument('--results-root',type=Path,default=HERE/'results')
    parser.add_argument('--preflight-receipt',type=Path)
    parser.add_argument('--options',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.action=='_worker-launch':
        import torch_xla
        from tpu_port.runtime import worker
        torch_xla.launch(worker,args=(json.loads(args.options),))
    elif args.action=='verify': print(json.dumps(verify_reference(),indent=2))
    elif args.action=='plan': print(json.dumps(load_contract(),indent=2))
    elif args.action=='prepare': reference_module().prepare(args.data_root.resolve())
    elif args.action=='cpu-smoke':
        from tpu_port.smoke import smoke
        print(json.dumps(smoke(),indent=2))
    else: launch(args.action,args.data_root,args.results_root,args.preflight_receipt)


if __name__=='__main__': main()
