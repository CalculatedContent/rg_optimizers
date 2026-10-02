#!/usr/bin/env python3
"""Run in Cloud Shell: scoped old-experiment cleanup and one v5e-8 request."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

PROJECT = 'tpu-builders-504820'
ZONE = 'us-west4-a'
QUEUE = 'ww-continuous8-20261002'
NODE = QUEUE + '-node'
DISK = QUEUE + '-data'
BUCKET = PROJECT + '-ww-continuous8'
SA_ID = 'rg-continuous-tpu'
SA = SA_ID + '@' + PROJECT + '.iam.gserviceaccount.com'
OLD_PREFIXES = ('ww-long-', 'ww-mem2-', 'ww-mem-', 'ww-v6e16-')
OLD_DISKS = {'ww-full-data-20260929'}


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
        if short(d) in OLD_DISKS:
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
    detail = inventory('alpha','compute','tpus','queued-resources','describe',QUEUE,'--zone='+ZONE)
    print(json.dumps(detail, indent=2))
    print(f'Cloud results: gs://{BUCKET}/runs/{QUEUE}/')
    if detail.get('state',{}).get('state') == 'ACTIVE':
        gc('compute','tpus','tpu-vm','ssh',NODE,'--zone='+ZONE,'--worker=0',
           '--command=systemctl --no-pager status rg-continuous8.service; tail -n 35 /mnt/disks/rg-data/continuous8/run.log')


def provision():
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    if dirty:
        raise RuntimeError('Provision from a clean checked-in source tree')
    requests = inventory('alpha','compute','tpus','queued-resources','list','--zone='+ZONE)
    if any(short(q)==QUEUE for q in requests):
        print('Existing continuous-run request retained; no second run or restart.')
        status()
        return
    gc('services','enable','tpu.googleapis.com','compute.googleapis.com','storage.googleapis.com','iam.googleapis.com')
    buckets = inventory('storage','buckets','list')
    if not any(x.get('name','').removeprefix('gs://').rstrip('/')==BUCKET for x in buckets):
        gc('storage','buckets','create','gs://'+BUCKET,'--location=us-west4','--uniform-bucket-level-access')
    accounts = inventory('iam','service-accounts','list')
    if not any(x.get('email')==SA for x in accounts):
        gc('iam','service-accounts','create',SA_ID,'--display-name=Continuous MuonClip TPU storage')
    gc('storage','buckets','add-iam-policy-binding','gs://'+BUCKET,
       '--member=serviceAccount:'+SA,'--role=roles/storage.objectAdmin')
    disks = inventory('compute','disks','list')
    if any(short(d)==DISK for d in disks):
        raise RuntimeError('Dedicated run disk exists without its queue. Keep it intact and inspect the previous attempt before retrying.')
    gc('compute','disks','create',DISK,'--zone='+ZONE,'--size=500GB','--type=pd-balanced',
       '--labels=experiment=continuous8')
    uri = f'gs://{BUCKET}/runs/{QUEUE}'
    template = Path(__file__).with_name('startup.sh').read_text()
    startup = template.replace('__COMMIT__',commit).replace('__GCS_URI__',uri)
    state = dict(project=PROJECT,zone=ZONE,queue=QUEUE,node=NODE,disk=DISK,gcs_uri=uri,commit=commit)
    (Path.home()/'continuous8-resources.json').write_text(json.dumps(state,indent=2))
    with tempfile.NamedTemporaryFile(mode='w',suffix='.sh') as f:
        f.write(startup); f.flush()
        gc('alpha','compute','tpus','queued-resources','create',QUEUE,
           '--zone='+ZONE,'--node-id='+NODE,'--accelerator-type=v5litepod-8',
           '--runtime-version=v2-alpha-tpuv5-lite','--provisioning-model=flex-start',
           '--max-run-duration=7d','--valid-until-duration=24h',
           '--service-account='+SA,'--scopes=https://www.googleapis.com/auth/cloud-platform',
           '--data-disk=source=projects/'+PROJECT+'/zones/'+ZONE+'/disks/'+DISK+',mode=read-write',
           '--metadata-from-file=startup-script='+f.name,'--labels=experiment=continuous8',
           '--quiet','--async')
    print('Submitted one continuous8 request. It starts setup and preflight automatically when allocated.')
    print('Inspect with: python3 baseline/nanogpt_one_head/continuous8/cloudshell.py status')
    print('Run settings: '+str(Path.home()/'continuous8-resources.json'))
    print('Cloud results: '+uri)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch','status'])
    parser.add_argument('--delete-old-experiments', action='store_true')
    args = parser.parse_args()
    if args.action == 'status':
        status()
    else:
        if args.delete_old_experiments:
            cleanup()
        provision()

if __name__ == '__main__':
    main()
