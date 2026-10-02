#!/usr/bin/env python3
"""Submit bounded v5e-8 experiments; prepare data and train on the TPU VM."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

PROJECT = 'tpu-builders-504820'
ZONE = 'us-west4-a'
RUN_PREFIX = 'ww-continuous8-pilot-20261002'
BUCKET = PROJECT + '-ww-continuous8'
SA_ID = 'rg-continuous-tpu'
SA = SA_ID + '@' + PROJECT + '.iam.gserviceaccount.com'
OLD_PREFIXES = ('ww-long-', 'ww-mem2-', 'ww-mem-', 'ww-v6e16-', 'ww-continuous8-20261002')
OLD_DISKS = {'ww-full-data-20260929', 'ww-continuous8-20261002-data'}
_LAUNCH_RECORD = None


def boot_id():
    path = Path('/proc/sys/kernel/random/boot_id')
    return path.read_text().strip() if path.exists() else 'unknown'


def launch_record(**changes):
    """Small durable progress record in HOME; never put the corpus in HOME."""
    global _LAUNCH_RECORD
    if _LAUNCH_RECORD is None:
        _LAUNCH_RECORD = dict(pid=os.getpid(), boot_id=boot_id(),
                              started_utc=datetime.now(timezone.utc).isoformat())
    _LAUNCH_RECORD.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
    path = Path.home()/'continuous8-launch.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(_LAUNCH_RECORD, indent=2)+'\n')
    temp.replace(path)
    if 'phase' in changes:
        print('[launch] '+changes['phase'], flush=True)


def show_local_launch():
    path = Path.home()/'continuous8-launch.json'
    if not path.exists():
        print('No persistent launch record. Earlier launcher versions did not save one.')
        return
    record = json.loads(path.read_text())
    alive = False
    if record.get('boot_id') == boot_id() and isinstance(record.get('pid'),int):
        try:
            os.kill(record['pid'], 0)
            cmdline = Path(f"/proc/{record['pid']}/cmdline").read_bytes()
            alive = b'cloudshell.py' in cmdline and b'launch' in cmdline
        except (ProcessLookupError, FileNotFoundError):
            pass
    if record.get('status') == 'running' and not alive:
        record['observed_status'] = 'INTERRUPTED: recorded launcher process is no longer present'
    print(json.dumps(record, indent=2))


def check_environment():
    """Read-only diagnosis: no installs, cleanup, resource creation, or launch."""
    print('READ-ONLY CHECK. No TPUs or other resources will be created/deleted.', flush=True)
    show_local_launch()
    root = Path(__file__).resolve().parents[3]
    print('Python:', sys.version.split()[0])
    print('Source:', subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip())
    dirty = subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    print('Checkout:', 'DIRTY (launch will refuse)' if dirty else 'clean')
    print('Data preparation runs on the TPU VM and its persistent disk, not Cloud Shell.')
    data = Path(tempfile.gettempdir())/'rg-continuous8-cloudshell/data'
    print('Local preparation:', 'absent' if not data.exists() else str(data))
    if data.exists():
        for p in sorted(data.iterdir()):
            if p.is_file(): print(f'  {p.name}: {p.stat().st_size:,} bytes')
    errors = []
    checks = [
        ('TPU requests', ('alpha','compute','tpus','queued-resources','list','--zone='+ZONE)),
        ('TPU VMs', ('compute','tpus','tpu-vm','list','--zone='+ZONE)),
        ('Experiment bucket', ('storage','buckets','describe','gs://'+BUCKET)),
        ('Experiment service account', ('iam','service-accounts','describe',SA)),
    ]
    for label,args in checks:
        print(label+':',flush=True)
        try:
            value = inventory(*args)
            if isinstance(value,list):
                print(json.dumps([{'name':short(x),'state':x.get('state')} for x in value],indent=2))
            else:
                print(json.dumps({k:value[k] for k in ('name','email','location','disabled') if k in value}))
        except subprocess.CalledProcessError:
            errors.append(label)
            print('CHECK FAILED: '+label+' (see gcloud error above)',flush=True)
    print('Persistent log:', Path.home()/'continuous8-launch.log')
    print('Read-only checks finished; creation permissions and data preparation are not validated.')
    if errors:
        print('Failed checks:', ', '.join(errors))


def gc(*args, capture=False):
    return subprocess.run(['gcloud', *args, '--project='+PROJECT], check=True,
                          text=True, stdout=subprocess.PIPE if capture else None).stdout


def inventory(*args):
    return json.loads(gc(*args, '--format=json', capture=True))


def short(resource):
    return resource['name'].rsplit('/', 1)[-1]


def old(name):
    return name.startswith(OLD_PREFIXES)


def cleanup():
    # Snapshot inventory before deleting resources; record exact targets.
    plan = {'queues': [], 'nodes': [], 'disks': [], 'snapshots': []}
    for zone in ('us-west4-a', 'us-east5-a'):
        for q in inventory('alpha', 'compute', 'tpus', 'queued-resources', 'list', '--zone='+zone):
            if old(short(q)):
                detail = inventory('alpha', 'compute', 'tpus', 'queued-resources', 'describe', short(q), '--zone='+zone)
                specs = detail.get('tpu', {}).get('nodeSpec', [])
                if any(not old(s.get('nodeId', '')) for s in specs):
                    raise RuntimeError('Old queue includes unrecognized nodes; refusing deletion')
                plan['queues'].append((zone, short(q)))
        for n in inventory('compute', 'tpus', 'tpu-vm', 'list', '--zone='+zone):
            if old(short(n)):
                plan['nodes'].append((zone, short(n)))
                for d in n.get('dataDisks', []):
                    source = d['sourceDisk']
                    if f'projects/{PROJECT}/zones/{zone}/disks/' not in source:
                        raise RuntimeError('Unexpected old disk project/zone')
                    OLD_DISKS.add(source.rsplit('/', 1)[-1])
    for d in inventory('compute', 'disks', 'list'):
        if short(d) in OLD_DISKS or old(short(d)):
            OLD_DISKS.add(short(d))
            plan['disks'].append((d['zone'].rsplit('/',1)[-1], short(d)))
    for s in inventory('compute', 'snapshots', 'list'):
        if s.get('sourceDisk','').rsplit('/',1)[-1] in OLD_DISKS or short(s).startswith('ww-long-saved-'):
            plan['snapshots'].append(short(s))
    audit = Path.home() / 'continuous8-cleanup.json'
    audit.write_text(json.dumps(plan, indent=2))
    print('Deleting these earlier experiment resources:', json.dumps(plan, indent=2), flush=True)
    for zone, name in plan['queues']:
        gc('alpha','compute','tpus','queued-resources','delete', name, '--zone='+zone, '--force', '--quiet')
    # Refresh; queue deletion usually already removed its node.
    for zone in ('us-west4-a', 'us-east5-a'):
        for n in inventory('compute','tpus','tpu-vm','list','--zone='+zone):
            if old(short(n)):
                gc('compute','tpus','tpu-vm','delete',short(n),'--zone='+zone,'--quiet')
    for zone, name in plan['disks']:
        gc('compute','disks','delete',name,'--zone='+zone,'--quiet')
    for name in plan['snapshots']:
        gc('compute','snapshots','delete',name,'--quiet')
    # Successful inventories are required; access errors cannot masquerade as absence.
    for zone in ('us-west4-a', 'us-east5-a'):
        for resource in ('queued-resources','tpu-vm'):
            remaining = inventory('alpha','compute','tpus',resource,'list','--zone='+zone)
            if any(old(short(x)) for x in remaining):
                raise RuntimeError('Earlier experiment TPU/queue still exists')
    print('Earlier targeted TPU experiments and their discovered disks/snapshots removed.', flush=True)


def status():
    show_local_launch()
    requests = inventory('alpha','compute','tpus','queued-resources','list','--zone='+ZONE)
    if not any(short(item).startswith(RUN_PREFIX) for item in requests):
        print('NO TPU REQUEST for this pilot. It is not training or waiting for TPU capacity.')
        print('This launcher prepares data on the TPU VM: no request means the new pipeline has not started.')
        print('Log: '+str(Path.home()/'continuous8-launch.log'))
        return
    for item in requests:
        if not short(item).startswith(RUN_PREFIX):
            continue
        name = short(item)
        detail = inventory('alpha','compute','tpus','queued-resources','describe',name,'--zone='+ZONE)
        print(json.dumps(detail, indent=2))
        print(f'Cloud results: gs://{BUCKET}/runs/{name}/')
        if detail.get('state',{}).get('state') == 'ACTIVE':
            gc('compute','tpus','tpu-vm','ssh',name+'-node','--zone='+ZONE,'--worker=0',
               '--command=systemctl --no-pager status rg-continuous8.service; tail -n 35 /mnt/disks/rg-data/continuous8/run.log')


def provision(machines, hours):
    launch_record(status='running',phase='checking source and existing requests', machines=machines, hours=hours)
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    if dirty:
        raise RuntimeError('Provision from a clean checked-in source tree')
    requests = inventory('alpha','compute','tpus','queued-resources','list','--zone='+ZONE)
    if any(short(q).startswith(RUN_PREFIX) for q in requests):
        launch_record(status='existing_request',phase='existing TPU request; no new allocation')
        print('Existing pilot request retained; no additional allocation or restart.')
        status()
        return
    launch_record(phase='enabling required APIs')
    gc('services','enable','tpu.googleapis.com','compute.googleapis.com','storage.googleapis.com','iam.googleapis.com')
    launch_record(phase='checking or creating experiment bucket')
    buckets = inventory('storage','buckets','list')
    if not any(x.get('name','').removeprefix('gs://').rstrip('/')==BUCKET for x in buckets):
        gc('storage','buckets','create','gs://'+BUCKET,'--location=us-west4','--uniform-bucket-level-access')
    launch_record(phase='checking or creating experiment service account')
    accounts = inventory('iam','service-accounts','list')
    if not any(x.get('email')==SA for x in accounts):
        gc('iam','service-accounts','create',SA_ID,'--display-name=Continuous MuonClip TPU storage')
    launch_record(phase='granting experiment bucket access')
    gc('storage','buckets','add-iam-policy-binding','gs://'+BUCKET,
       '--member=serviceAccount:'+SA,'--role=roles/storage.objectAdmin')
    seeds = (1337, 2027)[:machines]
    runs = [(seed, f'{RUN_PREFIX}-s{seed}') for seed in seeds]
    launch_record(phase='checking for previous experiment disks')
    disks = inventory('compute','disks','list')
    if any(short(d).startswith(RUN_PREFIX) for d in disks):
        raise RuntimeError('Dedicated run disk exists without its queue. Keep it intact and inspect the previous attempt before retrying.')

    # Only submit from Cloud Shell. All long work is performed by the TPU VM's
    # systemd service on its attached persistent disk, independent of this shell.
    data_uri = f'gs://{BUCKET}/corpora/{RUN_PREFIX}'
    print(f'Plan: {machines} machine(s), {hours}h maximum each; TPU compute ${machines*hours*8*0.60:.2f} plus storage.',flush=True)
    print('Allocation time includes software setup, on-VM data preparation, preflight and training.', flush=True)

    template = Path(__file__).with_name('startup.sh').read_text()
    state = dict(project=PROJECT,zone=ZONE,commit=commit,hours=hours,machines=machines,
                 data_uri=data_uri,runs=[])
    (Path.home()/'continuous8-resources.json').write_text(json.dumps(state,indent=2))
    for seed, queue in runs:
        node, disk = queue+'-node', queue+'-data'
        uri = f'gs://{BUCKET}/runs/{queue}'
        launch_record(phase='creating run disk for seed '+str(seed))
        gc('compute','disks','create',disk,'--zone='+ZONE,'--size=200GB','--type=pd-balanced',
           '--labels=experiment=continuous8')
        startup = (template.replace('__COMMIT__',commit).replace('__GCS_URI__',uri)
                   .replace('__DATA_URI__',data_uri+'-s'+str(seed)).replace('__SEED__',str(seed))
                   .replace('__STOP_HOURS__',str(hours-0.5)))
        state['runs'].append(dict(seed=seed,queue=queue,node=node,disk=disk,gcs_uri=uri))
        (Path.home()/'continuous8-resources.json').write_text(json.dumps(state,indent=2))
        with tempfile.NamedTemporaryFile(mode='w',suffix='.sh') as f:
            f.write(startup); f.flush()
            launch_record(phase='submitting TPU request '+queue)
            gc('alpha','compute','tpus','queued-resources','create',queue,
               '--zone='+ZONE,'--node-id='+node,'--accelerator-type=v5litepod-8',
               '--runtime-version=v2-alpha-tpuv5-lite','--provisioning-model=flex-start',
               f'--max-run-duration={hours}h','--valid-until-duration=24h',
               '--service-account='+SA,'--scopes=https://www.googleapis.com/auth/cloud-platform',
               '--data-disk=source=projects/'+PROJECT+'/zones/'+ZONE+'/disks/'+disk+',mode=read-write',
               '--metadata-from-file=startup-script='+f.name,'--labels=experiment=continuous8',
               '--quiet','--async')
        print('Submitted '+queue+'; cloud results: '+uri,flush=True)
    launch_record(status='submitted',phase='TPU requests submitted; waiting for capacity/setup')
    print('Setup and preflight start automatically when each machine is allocated.')
    print('Inspect with: python3 baseline/nanogpt_one_head/continuous8/cloudshell.py status')
    print('Run settings: '+str(Path.home()/'continuous8-resources.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch','status','check'])
    parser.add_argument('--delete-old-experiments', action='store_true')
    parser.add_argument('--machines', type=int, choices=[1,2], default=1)
    parser.add_argument('--hours', type=int, choices=[4,6], default=6)
    args = parser.parse_args()
    if args.action == 'check':
        check_environment()
    elif args.action == 'status':
        status()
    else:
        if (args.machines, args.hours) not in ((1,6),(2,4)):
            parser.error('Supported budgets: --machines 1 --hours 6 or --machines 2 --hours 4')
        try:
            launch_record(status='running',phase='starting launcher',machines=args.machines,hours=args.hours)
            if args.delete_old_experiments:
                launch_record(phase='removing previously authorized old experiment resources')
                cleanup()
            provision(args.machines, args.hours)
        except BaseException as exc:
            launch_record(status='failed',error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
            print('LAUNCH FAILED. See '+str(Path.home()/'continuous8-launch.json'),file=sys.stderr,flush=True)
            raise SystemExit(1)

if __name__ == '__main__':
    main()
