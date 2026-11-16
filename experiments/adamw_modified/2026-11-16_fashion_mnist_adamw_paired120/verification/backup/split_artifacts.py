"""Archive-only: verify original artifact bytes and split; never train."""
import os,hashlib,json,subprocess
from pathlib import Path
SOURCES=[(33001,14226930099,876712398,"673aa3cf6ea358c7cfa43122a15274d2e85e0145dbee4b5c7a5d43188c6cfaa4"),(33002,14226668097,879658821,"a7921742285578eb73cba9887e0f2f0b150e0b6d304337a223448a955c630515")]
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
out=Path("preserved_parts");out.mkdir(exist_ok=False);records=[]
for seed,aid,size,expected in SOURCES:
 original=out/f"original_{aid}.zip"
 subprocess.run(["curl","--fail","--location","--silent","--show-error","--header","Authorization: Bearer "+os.environ["GH_TOKEN"],"--header","Accept: application/vnd.github+json",f"https://api.github.com/repos/CalculatedContent/rg_optimizers/actions/artifacts/{aid}/zip","--output",str(original)],check=True)
 assert original.stat().st_size==size and digest(original)==expected,"Original artifact hash mismatch"
 parts=[];rebuilt=hashlib.sha256()
 with original.open("rb") as f:
  i=0
  while True:
   block=f.read(450*1024*1024)
   if not block:break
   i+=1;p=out/f"Fashion_AdamW_Paired120_Seed_{seed}.artifact.zip.part{i:02}"
   p.write_bytes(block);rebuilt.update(block)
   parts.append(dict(file=p.name,bytes=len(block),sha256=digest(p)))
 assert rebuilt.hexdigest()==expected and sum(x["bytes"] for x in parts)==size
 records.append(dict(seed=seed,original_artifact_id=aid,original_bytes=size,original_sha256=expected,parts=parts))
 original.unlink()
(out/"Reconstruction_Manifest.json").write_text(json.dumps(dict(source_run=43864280628,no_training=True,records=records,instructions="Concatenate each seed's parts in listed order to reconstruct the exact original GitHub artifact ZIP; verify original SHA256, then unzip the artifact to obtain the inner seed ZIP and its metadata."),indent=2)+"\n")
print("Verified and split both original seed archives; no training.")
