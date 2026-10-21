"""Post-hoc default WW fits; every unique matrix, output alias explicit."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import io, json, time, hashlib, importlib.metadata, contextlib
from pathlib import Path
import torch
torch.set_num_threads(1)
import weightwatcher as ww
assert importlib.metadata.version('weightwatcher') == '0.7.7'
root=Path(__file__).resolve().parent
blob=(root/'convergence/last.pt').read_bytes()
ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
rows=[]; start=time.perf_counter()
for name,w in ck['model'].items():
    if w.ndim != 2 or name=='lm_head.weight': continue
    aliases=[name]
    if name=='transformer.wte.weight':
        assert torch.equal(w,ck['model']['lm_head.weight'])
        aliases.append('lm_head.weight (tied output projection)')
    m=torch.nn.Sequential(torch.nn.Linear(w.shape[1],w.shape[0],bias=False))
    with torch.no_grad(): m[0].weight.copy_(w)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        frame=ww.WeightWatcher(m).analyze()
    records=json.loads(frame.to_json(orient='records'))
    rows.append(dict(matrix=name,aliases=aliases,shape=list(w.shape),fits=records))
    print(name,[(r.get('alpha'),r.get('status')) for r in records],flush=True)
valid=[r['fits'][0]['alpha'] for r in rows if len(r['fits'])==1 and r['fits'][0].get('status')=='success' and r['fits'][0].get('alpha') is not None]
out=dict(step=ck['steps'],checkpoint_sha256=hashlib.sha256(blob).hexdigest(),weightwatcher_version='0.7.7',method='Default analyze() independently on each actual 2D tensor represented as a Linear matrix; packed QKV retained. Tied token/output tensor counted once. LayerNorm vectors not matrices.',unique_matrix_count=len(rows),successful_fits=len(valid),mean_alpha=sum(valid)/len(valid),below_two=sum(a<2 for a in valid),analysis_seconds=time.perf_counter()-start,matrices=rows)
(root/'convergence/alpha.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='matrices'}))
