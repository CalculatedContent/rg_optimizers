"""Package every available endpoint, failed trajectory and excluded smoke state."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'Protocol.json').read_text())
OUT=ROOT/'archives';OUT.mkdir(exist_ok=True)
for seed in [PLAN['smoke_seed']]+PLAN['seeds']:
    branch='smoke' if seed==PLAN['smoke_seed'] else 'runs'
    files=[p for d in (ROOT/branch).glob(f'*/seed_{seed}') for p in d.rglob('*') if p.is_file() and p.suffix!='.tmp']
    if not files:continue
    files += [p for pattern in ['*.json','*.py','*.md','*.txt'] for p in ROOT.glob(pattern)]
    name=f'Fashion_AdamW_Paired120_Seed_{seed}.zip';dest=OUT/name
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_STORED) as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(ROOT)))
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    (OUT/(name+'.json')).write_text(json.dumps(dict(seed=seed,sha256=h.hexdigest(),bytes=dest.stat().st_size,
        all_available_states_preserved=True,smoke_excluded=branch=='smoke'),indent=2)+'\n')
    print('Preserved',name,len(files),'files',flush=True)
