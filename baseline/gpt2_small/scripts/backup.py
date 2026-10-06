"""Upload validation artifacts using object permissions; no bucket-metadata read."""
import argparse
import json
from pathlib import Path
from rg_nanogpt_one_head.continuous_support import CloudPublisher
p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args()
root=Path(a.root).resolve()
if not root.is_dir(): raise SystemExit('Output directory missing')
publisher=CloudPublisher('gs://tpu-builders-504820-ww-continuous8/gpt2small/'+root.name)
receipts=[]
for path in sorted(root.rglob('*')):
    relative=path.relative_to(root)
    if path.is_symlink() or not path.is_file() or 'repo' in relative.parts or path.suffix=='.tmp': continue
    print('Uploading',relative,flush=True)
    receipts.append(publisher.file(path,relative.as_posix()))
receipt=root/'CLOUD_BACKUP_VERIFIED.json'
receipt.write_text(json.dumps({'files':receipts,'method':'object upload with CRC32C and object-size verification'},indent=2))
publisher.file(receipt,receipt.name)
print('Cloud backup verified:',root.name,flush=True)
