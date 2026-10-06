#!/usr/bin/env python3
"""Replace experiment TPUs in the two used zones with one 24-hour v5e-8.

Retain every data disk and bucket; reuse the existing pilot data disk. This is a
fresh scientific run in a new directory/archive, never a checkpoint continuation.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from cloudshell import PROJECT, ZONE, BUCKET, SA, gc, inventory, short

ZONES = ('us-west4-a', 'us-east5-a')
QUEUE = 'ww-continuous8-24h-20261003-s1337'
NODE = QUEUE+'-node'
DISK = 'ww-continuous8-pilot-20261002-s1337-data'
OLD_NODE = 'ww-continuous8-pilot-20261002-s1337-node'
BASE = '/mnt/disks/rg-data/continuous8-24h-20261003-s1337'
SHARED = '/mnt/disks/rg-data/continuous8'
URI = f'gs://{BUCKET}/runs/{QUEUE}'
DATA_URI = f'gs://{BUCKET}/corpora/ww-continuous8-pilot-20261002-s1337'


def make_startup(commit):
    source = Path(__file__).with_name('startup.sh').read_text()
    source = source.replace('BASE=/mnt/disks/rg-data/continuous8', 'BASE='+BASE)
    # Reusing a known disk must never format a missing/incorrect filesystem.
    start = source.index('if [ -z "$TYPE" ]; then')
    end = source.index('mkdir -p /mnt/disks/rg-data', start)
    source = source[:start] + '''if [ "$TYPE" != ext4 ]; then
  echo 'Expected the existing ext4 data disk; refusing to format.' >&2
  exit 1
fi
''' + source[end:]
    source = source.replace('Environment=RG_CONTINUOUS_SEED=__SEED__',
        'Environment=RG_CONTINUOUS_SEED=__SEED__\n'
        'Environment=RG_CONTINUOUS_BASE=$BASE\n'
        'Environment=RG_CONTINUOUS_SHARED_BASE='+SHARED+'\n'
        'Environment=RG_CONTINUOUS_CONFIG=$BASE/repo/baseline/nanogpt_one_head/configs/muonclip_continuous8_24h.yaml')
    return (source.replace('__COMMIT__', commit).replace('__GCS_URI__', URI)
            .replace('__DATA_URI__', DATA_URI).replace('__SEED__', '1337')
            .replace('__STOP_HOURS__', '23.5'))


def save_record(record, **changes):
    record.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
    path = Path.home()/'continuous8-24h-resources.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2)+'\n')
    tmp.replace(path)
    if 'phase' in changes:
        print('[24h] '+changes['phase'], flush=True)


def main():
    record = dict(project=PROJECT, zone=ZONE, queue=QUEUE, node=NODE, disk=DISK,
                  base=BASE, gcs_uri=URI, hours=24, machines=1, seed=1337)
    try:
        # Complete all read-only checks and build the startup script first.
        root = Path(__file__).resolve().parents[3]
        commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
            raise RuntimeError('Use a clean checkout so the allocated VM receives the same code.')
        save_record(record, commit=commit, phase='checking existing resources', status='preparing')
        queues = {z:inventory('alpha','compute','tpus','queued-resources','list','--zone='+z) for z in ZONES}
        if any(short(q)==QUEUE for q in queues[ZONE]):
            save_record(record, status='existing', phase='24-hour request already exists; no deletion or duplicate launch')
            print(json.dumps(inventory('alpha','compute','tpus','queued-resources','describe',QUEUE,
                                       '--zone='+ZONE),indent=2))
            return
        nodes = {z:inventory('compute','tpus','tpu-vm','list','--zone='+z) for z in ZONES}
        disk = inventory('compute','disks','describe',DISK,'--zone='+ZONE)
        inventory('storage','buckets','describe','gs://'+BUCKET)
        inventory('iam','service-accounts','describe',SA)
        if disk.get('zone','').rsplit('/',1)[-1] != ZONE:
            raise RuntimeError('Existing data disk is in the wrong zone.')
        save_record(record, old_queues=queues, old_nodes=nodes, preserved_disk=disk)
        startup = make_startup(commit)
        subprocess.run(['bash','-n'],input=startup,text=True,check=True)
        startup_path = Path.home()/'continuous8-24h-startup.sh'
        startup_path.write_text(startup)
        print('One v5e-8, 24-hour allocation; compute cap $115.20 plus storage.',flush=True)
        print('Stop requested at 23.5h from boot; setup is included. Measurements and checkpoints every 500 updates.',flush=True)
        print('Removing TPU queues/VMs in us-west4-a and us-east5-a; retaining all data disks and buckets.',flush=True)
        if any(short(n)==OLD_NODE for n in nodes[ZONE]):
            save_record(record, phase='stopping the old worker and flushing the persistent disk')
            command = '''sudo bash -ec '
test -d /mnt/disks/rg-data/continuous8
mountpoint -q /mnt/disks/rg-data
if ! timeout 120 systemctl stop rg-continuous8.service; then
  systemctl kill --kill-who=all --signal=SIGKILL rg-continuous8.service
fi
sync
' '''
            gc('compute','tpus','tpu-vm','ssh',OLD_NODE,'--zone='+ZONE,'--worker=0','--command='+command)
        save_record(record, phase='deleting old TPU requests and their nodes', status='replacing')
        for zone, items in queues.items():
            for item in items:
                gc('alpha','compute','tpus','queued-resources','delete',short(item),
                   '--zone='+zone,'--force','--quiet')
        for zone in ZONES:
            for item in inventory('compute','tpus','tpu-vm','list','--zone='+zone):
                gc('compute','tpus','tpu-vm','delete',short(item),'--zone='+zone,'--quiet')
        for zone in ZONES:
            for kind in ('queued-resources','tpu-vm'):
                if inventory('alpha','compute','tpus',kind,'list','--zone='+zone):
                    raise RuntimeError(f'{zone}: {kind} still present; no new allocation submitted.')
        save_record(record, phase='checking that the preserved disk is detached')
        for attempt in range(13):
            disk = inventory('compute','disks','describe',DISK,'--zone='+ZONE)
            if not disk.get('users'):
                break
            if attempt == 12:
                raise RuntimeError('Data disk is still attached; preserved, but no new request submitted.')
            time.sleep(5)
        save_record(record, phase='submitting the new 24-hour request')
        gc('alpha','compute','tpus','queued-resources','create',QUEUE,
           '--zone='+ZONE,'--node-id='+NODE,'--accelerator-type=v5litepod-8',
           '--runtime-version=v2-alpha-tpuv5-lite','--provisioning-model=flex-start',
           '--max-run-duration=24h','--valid-until-duration=24h',
           '--service-account='+SA,'--scopes=https://www.googleapis.com/auth/cloud-platform',
           '--data-disk=source=projects/'+PROJECT+'/zones/'+ZONE+'/disks/'+DISK+',mode=read-write',
           '--metadata-from-file=startup-script='+str(startup_path),
           '--labels=experiment=continuous8-24h','--quiet','--async')
        save_record(record, status='submitted', phase='request submitted; setup starts automatically when allocated')
        print('Node: '+NODE+'\nLog: '+BASE+'/run.log\nCloud results: '+URI,flush=True)
        print('Record: '+str(Path.home()/'continuous8-24h-resources.json'),flush=True)
    except BaseException as exc:
        save_record(record, status='failed', error=f'{type(exc).__name__}: {exc}')
        print('STOPPED: '+str(exc)+'\nData disks and cloud archives were retained.',file=sys.stderr,flush=True)
        raise


if __name__ == '__main__':
    main()
