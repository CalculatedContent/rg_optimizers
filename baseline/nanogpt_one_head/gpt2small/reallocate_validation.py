#!/usr/bin/env python3
"""Replace the expiring TPU, retain its corpus disk, run bounded validation only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

PROJECT = 'tpu-builders-504820'
ZONE = 'us-west4-a'
BUCKET = PROJECT + '-ww-continuous8'
SA = 'rg-continuous-tpu@' + PROJECT + '.iam.gserviceaccount.com'
OLD_QUEUE = 'ww-continuous8-24h-20261003-s1337'
OLD_NODE = OLD_QUEUE + '-node'
QUEUE = 'ww-gpt2-validation-20261004-s1337'
NODE = QUEUE + '-node'
DISK = 'ww-continuous8-pilot-20261002-s1337-data'
DISK_PATH = f'projects/{PROJECT}/zones/{ZONE}/disks/{DISK}'
ROOT = '/mnt/disks/rg-data/gpt2small/' + QUEUE
HOURS = 4


def gc(*args, capture=False):
    return subprocess.run(['gcloud', *args, '--project=' + PROJECT], check=True,
                          text=True, stdout=subprocess.PIPE if capture else None).stdout


def inventory(*args):
    return json.loads(gc(*args, '--format=json', capture=True))


def named(items, name):
    return next((x for x in items if x['name'].rsplit('/', 1)[-1] == name), None)


def queues():
    return inventory('alpha', 'compute', 'tpus', 'queued-resources', 'list', '--zone=' + ZONE)


def nodes():
    return inventory('compute', 'tpus', 'tpu-vm', 'list', '--zone=' + ZONE)


def save(**changes):
    path = Path.home() / 'gpt2-validation-resources.json'
    record = json.loads(path.read_text()) if path.exists() else {}
    record.update(project=PROJECT, zone=ZONE, queue=QUEUE, node=NODE, disk=DISK,
                  root=ROOT, hours=HOURS, updated_utc=datetime.now(timezone.utc).isoformat(), **changes)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(record, indent=2) + '\n'); temp.replace(path)
    if 'phase' in changes:
        print('[validation] ' + changes['phase'], flush=True)


def make_startup(commit):
    text = Path(__file__).with_name('replacement_startup.sh').read_text()
    return text.replace('__COMMIT__', commit).replace('__ROOT__', ROOT).replace('__HOURS__', str(HOURS))


def status():
    request = named(queues(), QUEUE)
    if request is None:
        print('No replacement request exists.'); return
    detail = inventory('alpha', 'compute', 'tpus', 'queued-resources', 'describe', QUEUE, '--zone=' + ZONE)
    state = detail.get('state', {}).get('state')
    print('TPU request:', state, flush=True)
    print('Cloud outputs: gs://' + BUCKET + '/gpt2small/' + QUEUE, flush=True)
    if state == 'ACTIVE':
        command = ('sudo systemctl --no-pager --full status rg-gpt2-validation.service || true; '
                   f'sudo tail -n 35 {ROOT}/startup.log {ROOT}/run.log; '
                   f'if sudo test -f {ROOT}/summaries/validation_report.json; then '
                   f'sudo cat {ROOT}/summaries/validation_report.json; fi')
        gc('compute', 'tpus', 'tpu-vm', 'ssh', NODE, '--zone=' + ZONE, '--worker=0', '--command=' + command)


def launch():
    # Check for our fixed replacement name before any destructive action.
    current_queues = queues(); current_nodes = nodes()
    if named(current_queues, QUEUE) or named(current_nodes, NODE):
        print('Replacement already exists; no deletion, duplicate allocation or validation restart.'); return
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Launch from a clean checkout.')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    disk = inventory('compute', 'disks', 'describe', DISK, '--zone=' + ZONE)
    if disk['zone'].rsplit('/', 1)[-1] != ZONE:
        raise RuntimeError('Unexpected disk zone.')
    old_queue = named(current_queues, OLD_QUEUE)
    if old_queue:
        detail = inventory('alpha', 'compute', 'tpus', 'queued-resources', 'describe', OLD_QUEUE, '--zone=' + ZONE)
        specs = detail.get('tpu', {}).get('nodeSpec', [])
        if len(specs) != 1 or specs[0].get('nodeId') != OLD_NODE:
            raise RuntimeError('Old request has unexpected nodes; refusing deletion.')
    old_node = named(current_nodes, OLD_NODE)
    if old_node:
        detail = inventory('compute', 'tpus', 'tpu-vm', 'describe', OLD_NODE, '--zone=' + ZONE)
        sources = [d.get('sourceDisk', '').removeprefix('https://www.googleapis.com/compute/v1/')
                   for d in detail.get('dataDisks', [])]
        if sources != [DISK_PATH]:
            raise RuntimeError('Old TPU does not have exactly the expected preserved data disk.')
    elif disk.get('users') and not old_queue:
        raise RuntimeError('Data disk is in use by another resource; no changes made.')
    source = make_startup(commit)
    subprocess.run(['bash', '-n'], input=source, text=True, check=True)
    startup = Path.home() / 'gpt2-validation-startup.sh'; startup.write_text(source)
    save(commit=commit, preserved_disk=disk, phase='replacing only the expired/expiring experiment TPU')
    print('One v5e-8, at most 4 hours including setup. Estimated compute $19.20 plus storage.', flush=True)
    print('Keeping the existing disk, FineWeb, environments and cloud archives. Short validation only.', flush=True)
    if old_node and detail.get('state') == 'READY':
        # Do not terminate a newly started diagnostic or any unrelated workload.
        command = """sudo bash -se <<'CHECK'
if pgrep -af '[p]ython.*(gpt2_experiment|gpt2small/validate.py|continuous_run)'; then
  echo 'A trainer is still running; replacement aborted.' >&2
  exit 1
fi
sync
CHECK"""
        try:
            gc('compute', 'tpus', 'tpu-vm', 'ssh', OLD_NODE, '--zone=' + ZONE, '--worker=0', '--command=' + command)
        except subprocess.CalledProcessError:
            # Expiry can race this read-only check. Proceed only if no READY old VM remains.
            remaining = named(nodes(), OLD_NODE)
            if remaining and remaining.get('state') == 'READY':
                raise
    if old_queue:
        gc('alpha', 'compute', 'tpus', 'queued-resources', 'delete', OLD_QUEUE,
           '--zone=' + ZONE, '--force', '--quiet')
    if named(nodes(), OLD_NODE):
        gc('compute', 'tpus', 'tpu-vm', 'delete', OLD_NODE, '--zone=' + ZONE, '--quiet')
    if named(queues(), OLD_QUEUE) or named(nodes(), OLD_NODE):
        raise RuntimeError('Old TPU still present; no replacement submitted.')
    save(phase='waiting for the preserved data disk to detach')
    for attempt in range(25):
        if not inventory('compute', 'disks', 'describe', DISK, '--zone=' + ZONE).get('users'):
            break
        if attempt == 24:
            raise RuntimeError('Disk still attached; preserved, no replacement submitted. Rerun later.')
        time.sleep(5)
    save(phase='submitting four-hour validation allocation')
    gc('alpha', 'compute', 'tpus', 'queued-resources', 'create', QUEUE, '--zone=' + ZONE,
       '--node-id=' + NODE, '--accelerator-type=v5litepod-8', '--runtime-version=v2-alpha-tpuv5-lite',
       '--provisioning-model=flex-start', '--max-run-duration=4h', '--valid-until-duration=4h',
       '--service-account=' + SA, '--scopes=https://www.googleapis.com/auth/cloud-platform',
       '--data-disk=source=' + DISK_PATH + ',mode=read-write',
       '--metadata-from-file=startup-script=' + str(startup),
       '--labels=experiment=gpt2-validation', '--quiet', '--async')
    save(phase='submitted; validation starts automatically when capacity is allocated')
    print('Node: ' + NODE + '\nLog: ' + ROOT + '/run.log', flush=True)
    print('Check: python3 baseline/nanogpt_one_head/gpt2small/reallocate_validation.py status', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch', 'status'], nargs='?', default='launch')
    args = parser.parse_args()
    if args.action == 'status':
        status()
    else:
        try:
            launch()
        except BaseException as exc:
            save(phase='stopped; disk and corpus retained', error=str(exc))
            raise


if __name__ == '__main__':
    main()
