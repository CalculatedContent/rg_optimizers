"""Capture a live validation stall, optionally stop its service; keep TPU/data.

Run from Cloud Shell. The script sends itself over SSH without changing the
running checkout or Python environment. No GCP allocation/storage mutations.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

PROJECT = 'tpu-builders-504820'
ZONE = 'us-west4-a'
RUN = 'ww-gpt2-validation-48h-20261004-s1337'
ROOT = Path('/mnt/disks/rg-data/gpt2small') / RUN
SERVICE = 'rg-gpt2-validation.service'
PYTHON = '/mnt/disks/rg-data/continuous8/venv/bin/python'


def service_info():
    result = subprocess.run(
        ['systemctl', 'show', SERVICE, '--property=ExecStart,ActiveState,SubState,MainPID,ControlGroup'],
        capture_output=True, text=True, check=True, timeout=15)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def verify_target(info, root):
    expected = [str(root / ('repo/baseline/' + path)) for path in (
        'gpt2_small/scripts/replacement_worker.sh', 'nanogpt_one_head/gpt2small/replacement_worker.sh')]
    if not any(path in info.get('ExecStart', '') for path in expected):
        raise RuntimeError('Service belongs to another run; refusing to stop it.')


def training_pids(root, proc=Path('/proc')):
    found = []
    for folder in proc.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            args = (folder / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if not {'rg_gpt2_small.experiment', 'rg_nanogpt_one_head.gpt2_experiment'}.intersection(args) or '--output' not in args:
                continue
            output = Path(args[args.index('--output') + 1])
            if output.parent == root and output.name in ('adamw', 'muonclip'):
                found.append(int(folder.name))
        except (OSError, UnicodeError, IndexError):
            continue
    return sorted(found)


def capture_command(destination, args, timeout=20):
    with destination.open('w') as out:
        try:
            result = subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
            out.write(f'\n[exit code {result.returncode}]\n')
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            out.write(f'\n[capture unavailable: {exc}]\n')
            return False


def stop_service(root, report):
    verify_target(service_info(), root)
    # Queue the stop so the client's timeout cannot leave an unobserved stop job.
    subprocess.run(['systemctl', 'stop', '--no-block', SERVICE], check=True, timeout=15)
    for attempt in range(30):
        state = service_info()
        if state.get('ActiveState') in ('inactive', 'failed') and state.get('MainPID') == '0':
            report['service_stopped'] = True
            return
        if attempt in (0, 10, 20):
            print('Waiting for validation service to stop...', flush=True)
        time.sleep(2)
    # The blocked update may ignore TERM. Only this verified service is targeted.
    verify_target(service_info(), root)
    report['forced_service_kill'] = True
    subprocess.run(['systemctl', 'kill', '--kill-who=all', '--signal=KILL', SERVICE],
                   check=True, timeout=15)
    for _ in range(10):
        state = service_info()
        if state.get('ActiveState') in ('inactive', 'failed') and state.get('MainPID') == '0':
            report['service_stopped'] = True
            return
        time.sleep(1)
    raise RuntimeError('Stop is not confirmed; inspect service-after.txt. No TPU was deleted.')


def on_tpu(stop=False):
    if os.geteuid() != 0:
        raise RuntimeError('Run the remote capture as root.')
    if not os.path.ismount('/mnt/disks/rg-data') or not ROOT.is_dir():
        raise RuntimeError('Expected mounted disk/run directory is missing.')
    info = service_info()
    verify_target(info, ROOT)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    dest = ROOT / 'diagnostics' / ('stall-' + stamp)
    dest.mkdir(parents=True)
    pids = training_pids(ROOT)
    report = {'captured_at_utc': stamp, 'run': str(ROOT), 'training_pids': pids,
              'service_before': info, 'stop_requested': stop, 'service_stopped': False}
    print('Saving diagnostics:', dest, flush=True)
    try:
        capture_command(dest / 'memory.txt', ['free', '-h'])
        capture_command(dest / 'disk.txt', ['df', '-h', '/mnt/disks/rg-data'])
        capture_command(dest / 'journal.txt', ['journalctl', '-u', SERVICE, '-n', '80', '--no-pager'])
        for source in (ROOT / 'run.log', ROOT / 'validation.log', ROOT / 'logs/adamw_4.log',
                       ROOT / 'adamw/status.json', ROOT / 'adamw/checkpoints/latest.json',
                       ROOT / 'adamw/logs/xla_compile_metrics_after_step_2.txt'):
            name = str(source.relative_to(ROOT)).replace('/', '_')
            capture_command(dest / name, ['tail', '-n', '150', str(source)])
        for pid in pids:
            for name in ('status', 'stat', 'io', 'wchan', 'stack', 'cgroup'):
                capture_command(dest / f'pid-{pid}-{name}.txt', ['cat', f'/proc/{pid}/{name}'])
            capture_command(dest / f'pid-{pid}-cpu-before.txt',
                            ['ps', '-p', str(pid), '-o', 'pid,stat,etime,time,pcpu,rss,wchan:32'])
        if pids:
            # Install only the profiler binary in a temporary directory; no training dependencies change.
            with tempfile.TemporaryDirectory(prefix='rg-stall-profiler-') as tools:
                installed = capture_command(dest / 'profiler-install.txt', [PYTHON, '-m', 'pip', 'install',
                    '--target', tools, '--no-deps', '--no-cache-dir', '--only-binary=:all:',
                    '--disable-pip-version-check', '--progress-bar', 'off', '--retries', '1',
                    '--timeout', '15', 'py-spy==0.4.1'], timeout=60)
                if installed:
                    for pid in pids:
                        print('Capturing Python and native stacks for PID', pid, flush=True)
                        spy = str(Path(tools) / 'bin/py-spy')
                        capture_command(dest / f'pid-{pid}-python-stack.txt',
                                        [spy, 'dump', '--pid', str(pid)], timeout=15)
                        capture_command(dest / f'pid-{pid}-native-stack.txt',
                                        [spy, 'dump', '--native', '--pid', str(pid)], timeout=20)
                else:
                    print('Profiler unavailable; process/log diagnostics retained.', flush=True)
            time.sleep(3)
            for pid in pids:
                capture_command(dest / f'pid-{pid}-cpu-after.txt',
                                ['ps', '-p', str(pid), '-o', 'pid,stat,etime,time,pcpu,rss,wchan:32'])
                capture_command(dest / f'pid-{pid}-threads.txt',
                                ['ps', '-L', '-p', str(pid), '-o', 'pid,tid,stat,time,pcpu,wchan:32,comm'])
        if stop:
            print('Stopping validation; the unfinished update may be lost.', flush=True)
            stop_service(ROOT, report)
    finally:
        capture_command(dest / 'service-after.txt', ['systemctl', 'status', SERVICE, '--no-pager', '--full'])
        (dest / 'capture.json').write_text(json.dumps(report, indent=2))
        os.sync()
        for filename in ['adamw_checkpoints_latest.json'] + [
                f'pid-{pid}-{part}.txt' for pid in pids for part in ('cpu-before', 'cpu-after', 'python-stack')]:
            print('\n' + filename, flush=True)
            print('\n'.join((dest / filename).read_text().splitlines()[:65])
                  if (dest / filename).is_file() else 'Not captured.', flush=True)
        print('\nFull diagnostics saved on persistent disk:', dest, flush=True)
    if report['service_stopped']:
        print('Validation stopped. TPU allocation, FineWeb and existing checkpoints retained.', flush=True)
    elif not stop:
        print('Read-only capture complete; validation was not stopped.', flush=True)


def show_last():
    """Read the saved capture without touching the trainer, allocation or data."""
    captures = sorted((ROOT / 'diagnostics').glob('stall-*'))
    if not captures:
        raise RuntimeError('No saved stall capture found.')
    dest = captures[-1]
    print('Saved diagnostics:', dest, flush=True)
    for path in [dest/'capture.json', dest/'service-after.txt',
                 dest/'adamw_logs_xla_compile_metrics_after_step_2.txt',
                 *sorted(dest.glob('pid-*-native-stack.txt'))]:
        print('\n===', path.name, '===', flush=True)
        if not path.is_file():
            print('Not captured.', flush=True)
            continue
        lines = path.read_text(errors='replace').splitlines()
        print('\n'.join(lines[:120]), flush=True)
        if 'native-stack' in path.name:
            keywords = ('Compile', 'compile', 'Execute', 'execute', 'Hlo', 'hlo', 'llvm', 'xla::')
            matches = [line for line in lines[120:] if any(word in line for word in keywords)]
            print('\nAdditional compiler/execution frames:\n' + '\n'.join(matches[:100]), flush=True)
    print('\n=== Current validation service ===', flush=True)
    print(json.dumps(service_info(), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--on-tpu', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--stop', action='store_true', help='Stop only the validation service after capture')
    parser.add_argument('--show-last', action='store_true', help='Read the saved diagnostic; do not capture or stop')
    args = parser.parse_args()
    if args.stop and args.show_last:
        parser.error('--stop and --show-last cannot be combined')
    if args.on_tpu:
        show_last() if args.show_last else on_tpu(args.stop)
        return 0
    remote = ['sudo', 'python3', '-c', Path(__file__).read_text(), '--on-tpu']
    if args.stop:
        remote.append('--stop')
    if args.show_last:
        remote.append('--show-last')
    # Do not alter the running checkout or ask it to fetch/checkout another commit.
    return subprocess.run(['gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', RUN + '-node',
        '--project=' + PROJECT, '--zone=' + ZONE, '--worker=0', '--command=' + shlex.join(remote)]).returncode


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('Diagnostic failed:', exc, file=sys.stderr)
        sys.exit(1)
