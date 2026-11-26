"""Restore/publish only this campaign's artifacts, using a separate clean checkout."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

REPOSITORY = 'CalculatedContent/rg_optimizers'
PREFIX = 'california_housing_benchmark/results/2026-11-26_published_mlp'


def git(args, *, cwd, env=None, check=True):
    return subprocess.run(['git', *args], cwd=cwd, env=env, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def auth_env():
    env = os.environ.copy()
    token = env.get('GH_TOKEN') or env.get('GITHUB_TOKEN')
    if token:
        value = base64.b64encode(('x-access-token:' + token).encode()).decode()
        env['GIT_CONFIG_COUNT'] = '1'
        env['GIT_CONFIG_KEY_0'] = 'http.https://github.com/.extraheader'
        env['GIT_CONFIG_VALUE_0'] = 'AUTHORIZATION: basic ' + value
    return env


def valid_paths(paths):
    result = []
    for name in paths:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not (str(path) == PREFIX or str(path).startswith(PREFIX + '/')):
            raise ValueError(f'Publication path outside authorized campaign: {name}')
        result.append(str(path))
    return result


def check_monotonic(local, remote):
    if not remote.exists() or not local.exists():
        return
    for result in local.rglob('result.json') if local.is_dir() else []:
        other = remote / result.relative_to(local)
        if not other.exists():
            continue
        a, b = json.loads(result.read_text()), json.loads(other.read_text())
        for key in ['optimizer', 'seed', 'phase', 'config', 'plan_sha256', 'data_manifest_sha256', 'source_bundle_sha256']:
            if a.get(key) != b.get(key):
                raise RuntimeError(f'Will not overwrite a different run: {result} ({key})')
        for key in ['frozen_selection_sha256', 'source_manifest_sha256']:
            if a.get(key) is not None and b.get(key) is not None and a[key] != b[key]:
                raise RuntimeError(f'Will not overwrite a different frozen campaign: {result} ({key})')
        if b.get('completed_updates', 0) > a.get('completed_updates', 0):
            raise RuntimeError(f'Will not overwrite newer archived progress: {result}')
        if b.get('status') == 'complete' and a.get('status') != 'complete':
            raise RuntimeError(f'Will not replace a complete run with an incomplete state: {result}')


def ignore_archived_files(directory, names):
    ignored = set(shutil.ignore_patterns('.run.lock', '*.tmp', '__pycache__')(directory, names))
    result_path = Path(directory) / 'result.json'
    trial_path = Path(directory).parent / 'trial.json'
    if result_path.exists() and trial_path.exists():
        result = json.loads(result_path.read_text())
        trial = json.loads(trial_path.read_text())
        if (trial.get('status') == 'complete' and result.get('phase') == 'calibration'
                and result.get('status') == 'complete'):
            ignored.update(name for name in names if name.endswith('.pt'))
    return ignored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['restore', 'publish'])
    parser.add_argument('--branch', required=True)
    parser.add_argument('--paths', nargs='+', required=True)
    args = parser.parse_args()
    paths = valid_paths(args.paths)
    source = Path.cwd()
    env = auth_env()
    if args.operation == 'publish' and not (env.get('GH_TOKEN') or env.get('GITHUB_TOKEN')):
        raise RuntimeError('Repository publication requires the workflow token')
    with tempfile.TemporaryDirectory(prefix='california-publish-') as tmp:
        work = Path(tmp)
        git(['init', '-q'], cwd=work)
        git(['remote', 'add', 'origin', f'https://github.com/{REPOSITORY}.git'], cwd=work)
        git(['config', 'user.name', 'github-actions[bot]'], cwd=work)
        git(['config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com'], cwd=work)
        for attempt in range(8):
            git(['fetch', '--depth=1', '--filter=blob:none', 'origin', args.branch], cwd=work, env=env)
            git(['sparse-checkout', 'init', '--no-cone'], cwd=work)
            git(['sparse-checkout', 'set', '--no-cone', *['/' + p for p in paths]], cwd=work)
            git(['checkout', '-B', 'archive', 'FETCH_HEAD'], cwd=work, env=env)
            if args.operation == 'restore':
                for name in paths:
                    src, dest = work / name, source / name
                    if src.is_dir():
                        shutil.copytree(src, dest, dirs_exist_ok=True)
                    elif src.is_file():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest)
                print('Restored matching campaign paths from branch (when present).', flush=True)
                return
            existing = []
            for name in paths:
                src, dest = source / name, work / name
                if not src.exists():
                    continue
                check_monotonic(src, dest)
                if src.is_dir():
                    shutil.copytree(src, dest, dirs_exist_ok=True,
                                    ignore=ignore_archived_files)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                existing.append(name)
            if not existing:
                print('No campaign outputs to publish.', flush=True)
                return
            git(['add', '-f', '--', *existing], cwd=work)
            if git(['diff', '--cached', '--quiet'], cwd=work, check=False).returncode == 0:
                print('Campaign outputs already archived.', flush=True)
                return
            git(['commit', '-q', '-m', 'Archive published California Housing benchmark results and checkpoints [skip ci]'], cwd=work)
            result = git(['push', 'origin', f'HEAD:refs/heads/{args.branch}'], cwd=work, env=env, check=False)
            if result.returncode == 0:
                print('Archived campaign outputs in the repository.', flush=True)
                return
            # Another seed can advance the same branch; rebuild on its new head.
            # Do not force-push, delete other files, or expose auth/header output.
            if attempt == 7:
                raise RuntimeError('Could not publish after 8 branch-update attempts; retained Actions artifact remains available')
            git(['reset', '--hard', 'HEAD'], cwd=work)
            time.sleep(min(2 * (attempt + 1), 12))


if __name__ == '__main__':
    main()
