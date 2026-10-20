"""Launch one bounded run on the existing TPU. Never allocates or extends a lease."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import urllib.request

PROJECT = 'tpu-builders-504820'
MOUNT = Path('/mnt/disks/rg-data')
BASE = MOUNT / 'distilbert_scratch_torchtpu'
LATEST = BASE / 'LATEST.json'
FOLDER = 'baseline/experiments/distilbert_scratch_torchtpu'


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def checked_lease(node, zone):
    flags = ['--project=' + PROJECT, '--zone=' + zone, '--format=json']
    spec = json.loads(run(['gcloud', 'compute', 'tpus', 'tpu-vm', 'describe', node, *flags], capture_output=True).stdout)
    queue_name = spec.get('queuedResource', '').rsplit('/', 1)[-1]
    if not queue_name:
        raise RuntimeError('Explicit existing queued-resource lease required')
    queue = json.loads(run(['gcloud', 'alpha', 'compute', 'tpus', 'queued-resources', 'describe', queue_name, *flags], capture_output=True).stdout)
    if spec.get('state') != 'READY' or spec.get('acceleratorType') != 'v5litepod-8' or queue.get('state', {}).get('state') != 'ACTIVE':
        raise RuntimeError('Requires an active single-host v5e-8 allocation')
    expiries = [spec.get('schedulingConfig', {}).get('terminationTimestamp'), queue.get('runDuration', {}).get('terminationTime')]
    for entry in queue.get('tpu', {}).get('nodeSpec', []):
        if entry.get('nodeId') == node:
            expiries.append(entry.get('node', {}).get('schedulingConfig', {}).get('terminationTimestamp'))
    parsed = [dt.datetime.fromisoformat(re.sub(r'(\.\d{6})\d+', r'\1', v).replace('Z', '+00:00')).timestamp() for v in expiries if v]
    if not parsed:
        raise RuntimeError('No explicit lease expiry found')
    return {'node': node, 'checked_unix': time.time(), 'termination_unix': min(parsed),
            'ips': [v['ipAddress'] for v in spec.get('networkEndpoints', []) if v.get('ipAddress')]}


def bounded_deadline(checked, hours):
    if not 0 < hours <= 6:
        raise ValueError('Budget must be greater than zero and at most six hours')
    if not -30 <= time.time() - checked['checked_unix'] <= 600:
        raise RuntimeError('Lease check expired; retry start')
    deadline = min(time.time() + hours * 3600, checked['termination_unix'] - 300)
    if deadline - time.time() < 1800:
        raise RuntimeError('Less than 30 minutes remain; nothing launched')
    return deadline


def discover_node(zone):
    specs = json.loads(run(['gcloud', 'compute', 'tpus', 'tpu-vm', 'list', '--project=' + PROJECT,
                           '--zone=' + zone, '--format=json'], capture_output=True).stdout)
    eligible = []
    reasons = []
    for spec in specs:
        if spec.get('state') != 'READY' or spec.get('acceleratorType') != 'v5litepod-8':
            continue
        node = spec['name'].rsplit('/', 1)[-1]
        try:
            checked = checked_lease(node, zone)
            bounded_deadline(checked, 5.5)
            eligible.append(node)
        except Exception as exc:
            reasons.append(node + ': ' + str(exc))
    if len(eligible) != 1:
        raise RuntimeError('Expected one eligible existing v5e-8 TPU. Specify --node explicitly if several exist. '
                           + json.dumps({'eligible': eligible, 'excluded': reasons}))
    return eligible[0]


def active(unit):
    result = subprocess.run(['systemctl', 'is-active', unit], capture_output=True, text=True)
    return result.stdout.strip() in ('active', 'activating', 'deactivating', 'reloading')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def start_remote(a):
    if os.geteuid() != 0 or not os.path.ismount(MOUNT):
        raise RuntimeError('Root and the existing mounted /mnt/disks/rg-data are required')
    if not re.fullmatch('[a-f0-9]{40}', a.commit):
        raise ValueError('A pinned 40-character source commit is required')
    checked = json.loads(a.lease)
    deadline = bounded_deadline(checked, a.hours)
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip',
                                    headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.read().decode().strip() not in checked['ips']:
            raise RuntimeError('Guest does not match checked TPU')
    python = next((shutil.which(p) for p in ('python3.12', 'python3.13') if shutil.which(p)), None)
    if not python:
        raise RuntimeError('TorchTPU requires Python >=3.12. Install python3.12 and python3.12-venv first.')
    if shutil.disk_usage(MOUNT).free < 30 * 1024**3:
        raise RuntimeError('Need at least 30 GiB free; no data deleted')
    BASE.mkdir(exist_ok=True)
    # Use the same launch lock as the existing GPT-2 controllers.
    lockpath = MOUNT / 'gpt2small/port-check-launch.lock'
    lockpath.parent.mkdir(exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if LATEST.exists() and active(json.loads(LATEST.read_text())['unit']):
            raise RuntimeError('DistilBERT is already active; use status')
        # Cover newer RG services as well as the legacy process guard.
        units = run(['systemctl', 'list-units', '--type=service', '--state=running,activating', '--no-legend', '--plain', 'rg-*'], capture_output=True).stdout
        if units.strip():
            raise RuntimeError('An RG service is already active; nothing stopped or launched:\n' + units)
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
        root = BASE / ('initial-' + stamp)
        root.mkdir()
        repo = root / 'repo'
        run(['git', 'init', '-q', str(repo)])
        run(['git', '-C', str(repo), 'fetch', '--depth', '1', 'https://github.com/CalculatedContent/rg_optimizers.git', a.commit], timeout=180)
        run(['git', '-C', str(repo), 'checkout', '--detach', 'FETCH_HEAD'])
        devices = [p for p in Path('/dev').glob('accel*') if p.is_char_device()]
        devices += [p for p in Path('/dev/accel').glob('accel*') if p.is_char_device()]
        if not devices or not shutil.which('fuser'):
            raise RuntimeError('Cannot verify TPU device occupancy (/dev/accel* and fuser required)')
        usage = subprocess.run(['fuser', *map(str, devices)], capture_output=True, text=True)
        if usage.returncode == 0 or usage.stdout.strip():
            raise RuntimeError('TPU devices are in use; existing processes untouched')
        if usage.returncode != 1 or usage.stderr.strip():
            raise RuntimeError('Could not establish that TPU devices are idle: ' + usage.stderr)
        if a.resume_run:
            resume = Path(a.resume_run).resolve()
            if not resume.is_relative_to(BASE.resolve()) or not (resume / 'latest_checkpoint.json').is_file():
                raise RuntimeError('Resume path must be an existing committed run under ' + str(BASE))
            original = json.loads((resume.parent / 'launch.json').read_text())
            if original['commit'] != a.commit:
                raise RuntimeError('Resume with the original source commit')
        bounded_deadline(checked, a.hours)  # Recheck freshness without extending the original deadline.
        unit = 'rg-distilbert-torchtpu-' + stamp + '.service'
        record = {'root': str(root), 'unit': unit, 'commit': a.commit, 'lease': checked,
                  'deadline_unix': deadline, 'maximum_hours': a.hours, 'resume_run': a.resume_run,
                  'backend': 'native TorchTPU', 'pretrained_weights': False,
                  'automatic_restart': False, 'experiment_count': 1,
                  'storage': str(MOUNT), 'test_accessed_at_launch': False,
                  'test_monitoring_enabled': True, 'test_used_for_selection': False,
                  'test_policy': 'MLM monitoring only; no test-based selection'}
        write(root / 'launch.json', record)
        env = {'TORCH_TPU_DEFER_AND_FUSE': '1',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'HF_HOME': str(BASE / 'hf'),
               'PYTHONUNBUFFERED': '1'}
        command = ['systemd-run', '--unit=' + unit, '--property=Type=exec', '--property=Restart=no',
                   '--property=RuntimeMaxSec=' + str(int(deadline - time.time())), '--property=TimeoutStopSec=30',
                   '--property=KillMode=control-group', '--property=StandardOutput=append:' + str(root / 'run.log'),
                   '--property=StandardError=append:' + str(root / 'run.log')]
        command += ['--setenv=' + key + '=' + value for key, value in env.items()]
        command += [str(python), '-u', str(repo / FOLDER / 'supervisor.py'), '--root', str(root), '--deadline', str(deadline)]
        if a.resume_run:
            command += ['--resume-run', a.resume_run]
        run(command)
        write(LATEST, record)
        print(json.dumps(record, indent=2))


def status_remote():
    record = json.loads(LATEST.read_text())
    print(json.dumps(record, indent=2))
    root = Path(record['root'])
    runroot = Path(record['resume_run']) if record.get('resume_run') else root / 'initial'
    for path in (root / 'preflight.json', root / 'ERROR.json', root / 'RESULT.json', runroot / 'runtime.json', runroot / 'status.json'):
        if path.exists():
            print(path.name + '\n' + path.read_text())
    subprocess.run(['systemctl', 'status', record['unit'], '--no-pager', '-n', '8'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('start', 'status'))
    p.add_argument('--node', help='Existing node; defaults to discovering the sole eligible v5e-8 in --zone')
    p.add_argument('--zone', default='us-west4-a')
    p.add_argument('--hours', type=float, default=5.5)
    p.add_argument('--resume-run')
    p.add_argument('--commit')
    p.add_argument('--remote', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--lease', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.remote:
        return start_remote(a) if a.action == 'start' else status_remote()
    here = Path(__file__).resolve()
    repo = here.parents[3]
    if not a.node:
        a.node = discover_node(a.zone)
    command = ['sudo', 'python3', '-c', here.read_text(), a.action, '--remote', '--hours', str(a.hours)]
    if a.action == 'start':
        if run(['git', '-C', str(repo), 'status', '--porcelain'], capture_output=True).stdout.strip():
            raise RuntimeError('Launch from a clean source checkout')
        head = run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True).stdout.strip()
        if a.commit and a.commit != head:
            raise RuntimeError('--commit must equal the clean checkout HEAD')
        lease = checked_lease(a.node, a.zone)
        bounded_deadline(lease, a.hours)
        command += ['--commit', head, '--lease', json.dumps(lease)]
        if a.resume_run:
            command += ['--resume-run', a.resume_run]
    run(['gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', a.node, '--project=' + PROJECT,
         '--zone=' + a.zone, '--worker=0', '--command=' + shlex.join(command)])


if __name__ == '__main__':
    main()
