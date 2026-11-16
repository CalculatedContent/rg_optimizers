"""Preserve completed and interrupted outcomes even if analysis fails."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'Protocol.json').read_text())
out = ROOT / 'archives'
out.mkdir(exist_ok=True)
for seed in PLAN['independent_seeds']:
    files = [f for d in (ROOT / 'runs').glob(f'*/*/seed_{seed}') for f in d.rglob('*')
             if f.is_file() and f.suffix != '.tmp']
    files += [f for d in (ROOT / 'restoration').glob(f'*/*/seed_{seed}') for f in d.rglob('*') if f.is_file()]
    if not files:
        continue
    files += [f for pattern in ('*.json', '*.py', '*.txt', '*.md') for f in ROOT.glob(pattern)]
    target = out / f'Fashion_NoAug120_Seed_{seed}.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_STORED) as z:
        for f in sorted(set(files)):
            z.write(f, str(f.relative_to(ROOT)))
    h = hashlib.sha256()
    with target.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    (out / (target.name + '.json')).write_text(json.dumps({'sha256': h.hexdigest(), 'bytes': target.stat().st_size,
        'seed': seed, 'all_available_outcomes_retained': True}, indent=2) + '\n')
    print('Preserved', seed, len(files), 'files', flush=True)
