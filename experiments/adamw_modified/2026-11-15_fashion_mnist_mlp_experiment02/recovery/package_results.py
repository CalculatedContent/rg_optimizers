"""Archive every newly available output, including interrupted states."""
from pathlib import Path
import zipfile,json,hashlib
h=Path(__file__).resolve().parent;w=h/'run_work';out=h/'archives';out.mkdir(exist_ok=True)
for seed in (31003,31004,31005):
 files=[p for d in (w/'runs').glob(f'*/*/seed_{seed}') for p in d.rglob('*') if p.is_file() and p.suffix!='.tmp'] if (w/'runs').exists() else []
 if not files:continue
 files += [p for p in w.glob('*.json') if p.is_file()]+[p for p in w.glob('*.py') if p.is_file()]+[p for p in (w/'data').glob('*') if p.is_file()]
 archive=out/f'Experiment02_Recovery_Seed_{seed}.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
  for p in sorted(set(files)):z.write(p,str(p.relative_to(w)))
  for name in ['Recovery_Plan.json','Recovery_File_Manifest.json']:
   if (h/name).exists():z.write(h/name,name)
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 meta={'file':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
 (out/(archive.name+'.json')).write_text(json.dumps(meta,indent=2)+'\n');print(meta,flush=True)
