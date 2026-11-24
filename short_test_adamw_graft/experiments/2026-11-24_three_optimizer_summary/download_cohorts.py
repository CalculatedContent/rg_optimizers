#!/usr/bin/env python3
"""Retrieve the pinned GitHub primary cohorts; never launch training."""
import concurrent.futures, hashlib, json, pathlib, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
COHORTS=json.loads((ROOT/'data/cohorts.json').read_text())
REPO='CalculatedContent/rg_optimizers'
def get(url):
    with urllib.request.urlopen(url, timeout=90) as r: return r.read()
jobs=[]
for c in COHORTS:
    arm=c['arm']; base=c['exp']; commit=c['commit']
    tree=json.loads(get(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1"))
    path=ROOT/'data/raw'/arm/'git_tree.json'; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(tree,indent=2))
    select=[]
    for e in tree['tree']:
        p=e['path']
        if not p.startswith(base+'/') or e['type']!='blob': continue
        rel=p[len(base)+1:]
        # Training results, provenance, analysis outputs and reproducibility instructions.
        if rel.endswith('.zip') or rel.endswith('.pt') or rel.startswith('existing_seed1340/') or rel.startswith('frozen_source/'): continue
        if '/checkpoint_' in rel or rel.endswith('.png') or rel.endswith('.pdf'): continue
        if rel.startswith('runs/'):
            if any(rel.startswith(f'runs/seed_{s}/') for s in range(1340,1345)):
                select.append(e)
        elif rel.startswith('analysis/') or rel in ('README.md','ANALYSIS_README.md','plan.json','cohort_status.json','replay.py','analyze_results.py','complete_cohort.py','archive_seed.py','run_additional_seed.py'):
            select.append(e)
    for e in select:
        rel=e['path'][len(base)+1:]
        jobs.append((c,e,ROOT/'data/raw'/arm/rel))
def download(job):
    c,e,dest=job
    url=f"https://raw.githubusercontent.com/{REPO}/{c['commit']}/{e['path']}"
    b=get(url); expected=e['sha']
    actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    assert actual==expected,(str(dest),actual,expected)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    return dict(arm=c['arm'],commit=c['commit'],repository_path=e['path'],local_path=str(dest.relative_to(ROOT)),bytes=len(b),git_blob_sha=actual,sha256=hashlib.sha256(b).hexdigest(),url=url)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    manifest=list(pool.map(download,jobs))
(ROOT/'data/source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f"Retrieved and Git-blob-verified {len(manifest)} files, {sum(m['bytes'] for m in manifest):,} bytes.")

