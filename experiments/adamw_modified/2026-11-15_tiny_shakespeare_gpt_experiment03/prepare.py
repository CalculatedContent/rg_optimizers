"""Hash-checked public data preparation; no training."""
from pathlib import Path
import json,hashlib,urllib.request
import numpy as np
p=Path(__file__).resolve().parent;meta=json.loads((p/'Provenance.json').read_text());plan=json.loads((p/'Protocol.json').read_text())
for name in ('model.py','LICENSE'):
 assert hashlib.sha256((p/'upstream'/name).read_bytes()).hexdigest()==meta[name]['sha256']
q=p/'input.txt'
if not q.exists():q.write_bytes(urllib.request.urlopen(meta['input.txt']['url'],timeout=180).read())
raw=q.read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['input.txt']['sha256']
chars=plan['split']['characters'];lookup={c:i for i,c in enumerate(chars)};a=np.array([lookup[c] for c in raw.decode('utf-8')],dtype=np.uint16)
assert sorted(set(raw.decode('utf-8')))==chars
parts={n:a[slice(*plan['split'][n])] for n in ('train','validation','test')}
assert sum(len(v) for v in parts.values())==len(a)
np.savez_compressed(p/'data.npz',**parts)
(p/'Data_Audit.json').write_text(json.dumps({'passed':True,'input_sha256':meta['input.txt']['sha256'],'array_hashes':{n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in parts.items()},'lengths':{n:len(v) for n,v in parts.items()}},indent=2)+'\n')
