"""Run as root on the known VM: graceful stop, compact archive, narrow checkpoint cleanup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
MOUNT=Path('/mnt/disks/rg-data')
OLD=MOUNT/'continuous8-24h-20261003-s1337'
DATA=MOUNT/'continuous8/data'
ROOT=Path(sys.argv[1])
if ROOT.resolve() == MOUNT/'gpt2small' or not ROOT.resolve().is_relative_to(MOUNT/'gpt2small'):
    raise SystemExit('Unexpected new output root; no cleanup')
if OLD.resolve() != OLD:
    raise SystemExit('Unexpected old output symlink; no cleanup')
if not os.path.ismount(MOUNT): raise SystemExit('Persistent disk is not mounted; no cleanup')
if DATA.is_symlink() or not DATA.resolve().is_relative_to(MOUNT): raise SystemExit('Unexpected corpus path')
meta=json.loads((DATA/'meta.json').read_text())
if meta.get('tokenizer')!='gpt2' or not meta.get('document_disjoint_splits'): raise SystemExit('Corpus identity/isolation check failed')
for split in ('train','val','test'):
    f=DATA/f'{split}.bin'
    if f.stat().st_size!=2*meta['splits'][split]: raise SystemExit('Corpus file size mismatch')
ROOT.mkdir(parents=True,exist_ok=True)
subprocess.run(['df','-h',str(MOUNT)],check=True)
subprocess.run(['du','-sh',*[str(p) for p in MOUNT.iterdir()]],check=True)
subprocess.run(['systemctl','--no-pager','--full','status','rg-continuous8.service'],check=False)
claim=json.loads((OLD/'results/CONTINUOUS_STARTED.json').read_text())
(ROOT/'old_allocation.json').write_text(json.dumps(claim,indent=2))
(ROOT/'data_inventory.json').write_text(json.dumps({'path':str(DATA),'bytes':sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file()),'token_shards':3,'metadata':meta},indent=2))
(OLD/'results/STOP').touch()
# Let the old trainer finish its current update and final checkpoint; never SIGKILL it here.
end=time.monotonic()+600
while True:
    running=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try: args=p.read_bytes().split(b'\0')
        except (OSError,ProcessLookupError): continue
        if b'rg_nanogpt_one_head.continuous_run' in args: running.append(p.parent.name)
    if not running: break
    if time.monotonic()>end: raise SystemExit('Graceful stop timed out; no files deleted')
    time.sleep(5)
subprocess.run(['systemctl','stop','rg-continuous8.service'],check=True)
os.sync()
run=OLD/'results/muon_clip/seed_1337'
if run.resolve() != run or not (run/'manifest.json').is_file(): raise SystemExit('Cannot identify old run outputs; no cleanup')
archive=ROOT/'old_scientific_results.tgz'
if archive.exists(): raise SystemExit('Archive already exists; refusing repeated cleanup')
files=[p for p in OLD.rglob('*') if p.is_file() and not p.is_symlink()
       and 'repo' not in p.relative_to(OLD).parts
       and p.suffix in ('.csv','.json','.yaml','.log','.txt')]
with tarfile.open(archive,'w:gz') as tar:
    for p in files: tar.add(p,arcname=str(p.relative_to(OLD)),recursive=False)
# Verify every archived file before deleting any model outputs.
with tarfile.open(archive) as tar:
    for p in files:
        member=tar.extractfile(str(p.relative_to(OLD)))
        if member is None or hashlib.sha256(member.read()).digest()!=hashlib.sha256(p.read_bytes()).digest():
            raise SystemExit('Archive verification failed; no deletion')
removed=[]
for p in run.glob('*.pt'):
    if p.is_symlink(): raise SystemExit('Unexpected checkpoint symlink')
    if p.name.startswith(('checkpoint_','model_epoch_')):
        removed.append({'path':str(p),'bytes':p.stat().st_size}); p.unlink()
(ROOT/'cleanup.json').write_text(json.dumps({'archive':str(archive),'removed':removed,'preserved_data':str(DATA)},indent=2))
os.sync(); subprocess.run(['df','-h',str(MOUNT)],check=True)
print('Old training stopped; corpus protected; compact archive verified; old model checkpoint files removed.',flush=True)
