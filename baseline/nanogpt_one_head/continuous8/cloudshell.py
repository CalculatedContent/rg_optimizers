#!/usr/bin/env python3
"""Cloud Shell: bounded v5e-8 experiments with data prepared before allocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT = 'tpu-builders-504820'
ZONE = 'us-west4-a'
RUN_PREFIX = 'ww-continuous8-pilot-20261002'
BUCKET = PROJECT + '-ww-continuous8'
SA_ID = 'rg-continuous-tpu'
SA = SA_ID + '@' + PROJECT + '.iam.gserviceaccount.com'
OLD_PREFIXES = ('ww-long-', 'ww-mem2-', 'ww-mem-', 'ww-v6e16-', 'ww-continuous8-20261002')
OLD_DISKS = {'ww-full-data-20260929', 'ww-continuous8-20261002-data'}


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
    requests = inventory('alpha','compute','tpus','queued-resources','list','--zone='+ZONE)
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
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    if dirty:
        raise RuntimeError('Provision from a clean checked-in source tree')
    requests = inventory('alpha','compute','tpus','queued-resources','list','--zone='+ZONE)
    if any(short(q).startswith(RUN_PREFIX) for q in requests):
        print('Existing pilot request retained; no additional allocation or restart.')
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
    seeds = (1337, 2027)[:machines]
    runs = [(seed, f'{RUN_PREFIX}-s{seed}') for seed in seeds]
    disks = inventory('compute','disks','list')
    if any(short(d).startswith(RUN_PREFIX) for d in disks):
        raise RuntimeError('Dedicated run disk exists without its queue. Keep it intact and inspect the previous attempt before retrying.')

    # Tokenization happens once on Cloud Shell CPU, outside the TPU budget.
    temp = Path(tempfile.gettempdir())/'rg-continuous8-cloudshell'
    temp.mkdir(exist_ok=True)
    py = temp/'venv/bin/python'
    if not py.exists():
        subprocess.run([sys.executable,'-m','venv',str(temp/'venv')], check=True)
    subprocess.run([str(py),'-m','pip','install','--disable-pip-version-check','--no-cache-dir',
                    'numpy==1.26.4','datasets==3.3.2','tiktoken==0.9.0','PyYAML==6.0.2'], check=True)
    exp = Path(__file__).resolve().parents[1]
    data_uri = f'gs://{BUCKET}/corpora/{RUN_PREFIX}'
    print(f'Plan: {machines} machine(s), {hours}h maximum each; TPU compute ${machines*hours*8*0.60:.2f} plus storage.',flush=True)
    subprocess.run([str(py),str(Path(__file__).with_name('prepare_cloud_data.py')),
                    '--config',str(exp/'configs/muonclip_continuous8.yaml'),
                    '--output-dir',str(temp/'data'),'--gcs-uri',data_uri],check=True)

    template = Path(__file__).with_name('startup.sh').read_text()
    state = dict(project=PROJECT,zone=ZONE,commit=commit,hours=hours,machines=machines,
                 data_uri=data_uri,runs=[])
    (Path.home()/'continuous8-resources.json').write_text(json.dumps(state,indent=2))
    for seed, queue in runs:
        node, disk = queue+'-node', queue+'-data'
        uri = f'gs://{BUCKET}/runs/{queue}'
        gc('compute','disks','create',disk,'--zone='+ZONE,'--size=200GB','--type=pd-balanced',
           '--labels=experiment=continuous8')
        startup = (template.replace('__COMMIT__',commit).replace('__GCS_URI__',uri)
                   .replace('__DATA_URI__',data_uri).replace('__SEED__',str(seed))
                   .replace('__STOP_HOURS__',str(hours-0.5)))
        state['runs'].append(dict(seed=seed,queue=queue,node=node,disk=disk,gcs_uri=uri))
        (Path.home()/'continuous8-resources.json').write_text(json.dumps(state,indent=2))
        with tempfile.NamedTemporaryFile(mode='w',suffix='.sh') as f:
            f.write(startup); f.flush()
            gc('alpha','compute','tpus','queued-resources','create',queue,
               '--zone='+ZONE,'--node-id='+node,'--accelerator-type=v5litepod-8',
               '--runtime-version=v2-alpha-tpuv5-lite','--provisioning-model=flex-start',
               f'--max-run-duration={hours}h','--valid-until-duration=24h',
               '--service-account='+SA,'--scopes=https://www.googleapis.com/auth/cloud-platform',
               '--data-disk=source=projects/'+PROJECT+'/zones/'+ZONE+'/disks/'+disk+',mode=read-write',
               '--metadata-from-file=startup-script='+f.name,'--labels=experiment=continuous8',
               '--quiet','--async')
        print('Submitted '+queue+'; cloud results: '+uri,flush=True)
    print('Setup and preflight start automatically when each machine is allocated.')
    print('Inspect with: python3 baseline/nanogpt_one_head/continuous8/cloudshell.py status')
    print('Run settings: '+str(Path.home()/'continuous8-resources.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch','status'])
    parser.add_argument('--delete-old-experiments', action='store_true')
    parser.add_argument('--machines', type=int, choices=[1,2], default=1)
    parser.add_argument('--hours', type=int, choices=[4,6], default=6)
    args = parser.parse_args()
    if args.action == 'status':
        status()
    else:
        if (args.machines, args.hours) not in ((1,6),(2,4)):
            parser.error('Supported budgets: --machines 1 --hours 6 or --machines 2 --hours 4')
        if args.delete_old_experiments:
            cleanup()
        provision(args.machines, args.hours)

if __name__ == '__main__':
    main()
