"""Create verified checkpoint shards and a current resume snapshot for archival."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import torch
from common import INPUT_SHA, sha, write_json

def pack(path,entries):
    integrity={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in entries.items()}
    tmp=path.with_suffix('.tmp.zip')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for n,b in entries.items():z.writestr(n,b)
        z.writestr('integrity.json',json.dumps(integrity,indent=2))
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
        for n,d in integrity.items():assert hashlib.sha256(z.read(n)).hexdigest()==d['sha256']
    tmp.replace(path)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();manifest=json.loads((root/'manifest.json').read_text()); entries={};shards=[];active=[]
    for path in root.iterdir():
        if path.is_file() and path.suffix in ('.json','.txt','.png'):entries[path.name]=path.read_bytes()
    assert sha(a.data)==INPUT_SHA;entries['input.txt']=a.data.read_bytes()
    for seed in manifest['seeds']:
        run=root/'runs'/f'seed_{seed}'
        if not (run/'last.pt').exists():continue
        # Atomic training writes mean opening once gets one complete checkpoint.
        b=(run/'last.pt').read_bytes();ck=torch.load(io.BytesIO(b),weights_only=False,map_location='cpu')
        assert ck['config']['seed']==seed
        assert all(k in ck for k in ['optimizer','torch_rng','train_rng','numpy_rng','python_rng','best_state'])
        prefix=f'runs/seed_{seed}/';payload={prefix+'last.pt':b}
        for path in run.iterdir():
            if path.suffix in ('.json','.log','.txt'):payload[prefix+path.name]=path.read_bytes()
        if (run/'result.json').exists():
            result=json.loads(payload[prefix+'result.json']);assert hashlib.sha256(b).hexdigest()==result['checkpoint_sha256']
            shard=root/f'Short_GPT_Seed_{seed}.zip'
            if not shard.exists():info=pack(shard,payload)
            else:info=dict(path=str(shard),bytes=shard.stat().st_size,sha256=sha(shard))
            shards.append(info|dict(seed=seed));entries[prefix+'result.json']=payload[prefix+'result.json']
        else:
            entries.update(payload);active.append(dict(seed=seed,step=ck['step']))
    entries['snapshot.json']=json.dumps(dict(active=active,shards=shards)).encode()
    resume=pack(root/'Short_GPT_Ten_Seeds_Resume.zip',entries)
    write_json(root/'package_index.json',dict(resume=resume,shards=shards,active=active))
    print(json.dumps(dict(resume=resume,shards=shards,active=active)))

if __name__=='__main__':main()
