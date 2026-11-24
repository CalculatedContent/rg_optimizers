"""Finish the replay seed and monitor the five current-runtime confirmations."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SEEDS = (1340, 1341, 1342, 1343, 1344)

def utc():
    return datetime.now(timezone.utc).isoformat()

def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def read(path):
    return json.loads(path.read_text())

def main():
    lock = (ROOT / 'cohort.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = utc()
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    with (ROOT / 'logs/seed_1340.log').open('a') as log:
        process = subprocess.Popen(
            [sys.executable, '-u', str(ROOT / 'run_additional_seed.py'), '--seed', '1340'],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        while True:
            rows = []
            for seed in SEEDS:
                folder = ROOT / 'runs' / f'seed_{seed}'
                job = folder / 'confirmation/rate_0.4/graft' / f'seed_{seed}'
                state_path = folder / 'worker_state.json'
                progress_path = job / 'progress.json'
                row = dict(seed=seed, status='pending', step=0)
                if state_path.exists():
                    row.update(read(state_path))
                if progress_path.exists():
                    row.update(read(progress_path))
                row['result_exists'] = (job / 'result.json').exists()
                rows.append(row)
            if process.poll() not in (None, 0):
                rows[0]['status'] = 'failed'
                rows[0]['exit_code'] = process.returncode
            complete = sum(row['status'] == 'completed' and row['result_exists'] for row in rows)
            failed = any(row['status'] == 'failed' for row in rows)
            record = dict(
                started_at=started, updated_at=utc(), target_updates=20000,
                cohort='five_current_runtime_seeds', seeds=list(SEEDS),
                completed_seeds=complete, workers=rows,
                status='failed' if failed else ('training_complete' if complete == 5 else 'training'),
                archived_seed1340='retained separately; excluded from cohort statistics')
            write(ROOT / 'cohort_status.json', record)
            if failed or complete == 5:
                break
            time.sleep(15)
        if complete == 5:
            result = subprocess.run(
                [sys.executable, str(ROOT / 'analyze_results.py'), '--root', str(ROOT)],
                cwd=ROOT, capture_output=True, text=True)
            (ROOT / 'cohort_analysis.log').write_text(result.stdout + result.stderr)
            record.update(analysis_returncode=result.returncode,
                          status='complete' if result.returncode == 0 else 'analysis_failed',
                          finished_at=utc())
            write(ROOT / 'cohort_status.json', record)
        print(json.dumps(record, indent=2), flush=True)

if __name__ == '__main__':
    main()
