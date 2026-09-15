#!/usr/bin/env python3
"""Run the two-model study or analyze its saved results. No shell setup needed."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LATEST = Path('/tmp/nanogpt_memorization_latest.txt')


def stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def exclusive(path: Path):
    import fcntl
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another launcher is using {path.parent}.') from exc
        yield


def choose_root(args, *, create: bool) -> Path:
    if args.root:
        root = Path(args.root).expanduser().resolve()
    elif args.latest or args.command == 'analyze' or args.resume:
        if not LATEST.is_file():
            raise ValueError('No saved results location. Use --root /tmp/your_results_directory.')
        root = Path(LATEST.read_text().strip()).resolve()
    else:
        root = Path('/tmp') / f'nanogpt_memorization_{stamp()}'
        number = 2
        while root.exists():
            root = Path('/tmp') / f'nanogpt_memorization_{stamp()}_{number:02d}'
            number += 1
    root = root.resolve()
    if args.command == 'run':
        if not any(root.is_relative_to(Path(p).resolve()) and root != Path(p).resolve()
                   for p in ('/tmp', '/private/tmp')):
            raise ValueError('Training results must be in a dedicated directory under /tmp.')
        if args.resume and not root.is_dir():
            raise ValueError('Cannot resume a results directory that does not exist.')
        if create:
            # A generated timestamp directory must be new, even across a race.
            root.mkdir(parents=True, exist_ok=bool(args.root or args.latest or args.resume))
    elif not root.is_dir():
        raise ValueError(f'Results directory does not exist: {root}')
    return root


def commands(args, cfg: dict, root: Path) -> list[tuple[str, list[str]]]:
    optimizers = cfg['optimizers'] if args.optimizer == 'both' else [args.optimizer]
    stages = ['smoke'] if args.stage == 'smoke' else ['smoke', args.stage]
    jobs = []
    for stage in stages:
        for optimizer in optimizers:
            label = f'{stage}/{args.recipe}/verbatim/{optimizer}/seed_{args.seed}'
            command = [sys.executable, '-u', str(HERE / 'run.py'), 'run',
                       '--stage', stage, '--condition', 'verbatim',
                       '--optimizer', optimizer, '--seed', str(args.seed),
                       '--device', args.device, '--root', str(root), '--recipe', args.recipe]
            # Reuse only previously successful integration checks, never an
            # existing scientific run unless the user explicitly asks to resume.
            if args.resume or (stage == 'smoke' and (root / label / 'complete.json').is_file()):
                command.append('--resume')
            jobs.append((label, command))
    return jobs


def stream_job(command: list[str], log: Path) -> int:
    """Stream one child to terminal and disk; preserve its return code."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as handle:
        handle.write('\nCOMMAND ' + shlex.join(command) + '\n')
        handle.flush()
        child = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 encoding='utf-8', errors='replace')
        try:
            assert child.stdout is not None
            for line in child.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
            return child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        finally:
            if child.stdout:
                child.stdout.close()


def verify_completion(run_dir: Path, expected_steps: int) -> None:
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    done = json.loads((run_dir / 'complete.json').read_text())
    if (done.get('status') != 'complete' or done.get('steps') != expected_steps
            or not manifest.get('fingerprint')
            or done.get('fingerprint') != manifest['fingerprint']):
        raise RuntimeError(f'No valid completion record: {run_dir}')


def report(root: Path, *, plots: bool = True) -> Path:
    from analyze import analyze
    destination = analyze(root, plots=plots)
    print(f'\nAnalysis: {destination / "summary.md"}', flush=True)
    return destination


def launch(args, cfg: dict, root: Path) -> int:
    jobs = commands(args, cfg, root)
    count = len(cfg['optimizers']) if args.optimizer == 'both' else 1
    print(f'Results: {root}')
    print(f'{count} {args.stage} verbatim run(s), seed {args.seed}. No other conditions are scheduled.')
    if args.dry_run:
        for _, command in jobs:
            print(shlex.join(command))
        return 0
    with exclusive(root / '.study.lock'):
        for label, command in jobs:
            if (root / label / 'manifest.json').exists() and '--resume' not in command:
                raise ValueError(f'Run already exists: {label}. Use --resume, or a new --root.')
        atomic_text(LATEST, str(root) + '\n')
        invocation = {'created_at': datetime.now().isoformat(), 'root': str(root),
                      'jobs': [{'run': label, 'command': command} for label, command in jobs]}
        atomic_text(root / 'logs' / f'plan_{stamp()}_{os.getpid()}.json',
                    json.dumps(invocation, indent=2) + '\n')
        for label, command in jobs:
            print(f'\nSTART {label}', flush=True)
            log = root / 'logs' / (label.replace('/', '__') + '.log')
            status = root / label / 'launcher_status.json'
            atomic_text(status, json.dumps({'state': 'running', 'log': str(log)}) + '\n')
            try:
                code = stream_job(command, log)
            except KeyboardInterrupt:
                atomic_text(status, json.dumps({'state': 'interrupted', 'log': str(log)}) + '\n')
                print('\nStopped. Existing results and checkpoints have been kept.')
                return 130
            if code == 0:
                try:
                    verify_completion(root / label, int(cfg['stages'][label.split('/')[0]]['steps']))
                except (OSError, ValueError, RuntimeError) as exc:
                    print(f'Completion check failed: {exc}')
                    code = 1
            atomic_text(status, json.dumps({'state': 'succeeded' if code == 0 else 'failed',
                                           'returncode': code, 'log': str(log)}) + '\n')
            if code:
                print(f'\nRun failed. Log: {log}\nNo retries or later runs were started.')
                print('The failure does not by itself identify an optimizer or backend cause.')
                try:
                    report(root, plots=False)
                except (OSError, ValueError, RuntimeError) as exc:
                    print(f'Analysis unavailable: {exc}')
                return code if code > 0 else 1
            if not label.startswith('smoke/'):
                report(root)
        if args.stage == 'smoke':
            print('Smoke checks completed. They are not memorization measurements.')
        else:
            print('Requested study completed. No 80-run campaign was started.')
    return 0


def main(argv=None) -> int:
    cfg = json.loads((HERE / 'configs/suite.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Default: verbatim, seed 1337, AdamW then Muon.')
    run.add_argument('--optimizer', choices=['adamw', 'muon', 'both'], default='both')
    run.add_argument('--device', choices=['mps', 'cuda', 'cpu'], default='mps')
    run.add_argument('--seed', type=int, choices=cfg['seeds'], default=1337)
    run.add_argument('--stage', choices=cfg['stages'], default='full')
    run.add_argument('--recipe', choices=['repository', 'shared_aux_decay'], default='repository')
    run.add_argument('--resume', action='store_true', help='Resume only with the original training fingerprint.')
    run.add_argument('--dry-run', action='store_true', help='Print commands; create nothing and start nothing.')
    analysis = sub.add_parser('analyze', help='Analyze saved data, including an incomplete or running study.')
    analysis.add_argument('--no-plots', action='store_true')
    for child in (run, analysis):
        group = child.add_mutually_exclusive_group()
        group.add_argument('--root', help='Existing or new results directory.')
        group.add_argument('--latest', action='store_true', help='Use the last location saved by this launcher.')
    args = parser.parse_args(argv)
    # Keep choose_root independent of subcommand-specific attributes.
    if args.command == 'analyze':
        args.resume = False
    try:
        root = choose_root(args, create=args.command == 'run' and not args.dry_run)
        if args.command == 'run':
            return launch(args, cfg, root)
        report(root, plots=not args.no_plots)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nStopped. No results were deleted.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
