"""Preserve all completed, failed and partial model states."""
from pathlib import Path
import zipfile,hashlib,json
R=Path(__file__).resolve().parent
(R/'archives').mkdir(exist_ok=True)
for seed in [32998,32999,32001,32002,32003,32004,32005]:
 files=[]
 for kind in ['smoke','pilot','runs']:
  for d in (R/kind).glob(f'**/seed_{seed}'):
   files.extend(p for p in d.rglob('*') if p.is_file())
 if not files:continue
 dest=R/'archives'/f'Fashion_Scale_Seed_{seed}.zip';tmp=dest.with_suffix('.tmp')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for p in files:z.write(p,str(p.relative_to(R)))
 tmp.replace(dest)
 (dest.with_suffix('.zip.json')).write_text(json.dumps({'file':dest.name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'files':len(files)},indent=2)+'\n')
