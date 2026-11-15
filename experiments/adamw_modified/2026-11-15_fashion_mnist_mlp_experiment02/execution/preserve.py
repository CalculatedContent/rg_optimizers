"""Save immutable source/data and completed seed blocks, once per archive."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ARCHIVES=ROOT.parent/'archives'
HELPER=ROOT.parents[1]/'fashion_exec_helpers/library_upload.py'


def archive_and_save(name, files):
    ARCHIVES.mkdir(exist_ok=True)
    archive=ARCHIVES/name
    receipt=ARCHIVES/(name+'.receipt.json')
    attempt=ARCHIVES/(name+'.attempt.json')
    if receipt.exists():
        result=json.loads(receipt.read_text())
        if result.get('results') and all(r.get('status')=='succeeded' and r.get('local_metadata_applied') for r in result['results']):
            return {'archive':str(archive),'receipt':result,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
        raise RuntimeError('Existing failed or uncertain save must be resolved before retry: '+str(receipt))
    if attempt.exists():raise RuntimeError('Uncertain prior upload; do not retry automatically: '+str(attempt))
    if not archive.exists():
        partial=archive.with_suffix('.zip.part')
        assert not partial.exists(), 'Resolve unfinished archive before retrying'
        with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for path in sorted(set(Path(p) for p in files)):
                assert path.is_file()
                z.write(path,path.relative_to(ROOT).as_posix())
        partial.replace(archive)
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    with attempt.open('x') as f:json.dump({'path':str(archive),'sha256':sha,'bytes':archive.stat().st_size},f)
    request={'uploads':[{'local_path':str(archive),'purpose':'create_library_file','library_artifact_type':'other'}]}
    result=subprocess.run([sys.executable,str(HELPER)],input=json.dumps(request),text=True,capture_output=True,timeout=600)
    receipt.write_text(result.stdout)
    (ARCHIVES/(name+'.upload.log')).write_text(result.stderr)
    if result.returncode:raise RuntimeError('Upload failed or uncertain; inspect '+str(receipt))
    data=json.loads(result.stdout)
    assert len(data['results'])==1 and data['results'][0]['status']=='succeeded' and data['results'][0]['local_metadata_applied']
    return {'archive':str(archive),'receipt':data,'sha256':sha,'bytes':archive.stat().st_size}


def source_files():
    names=['study.py','adapter.py','prepare.py','supervisor.py','preserve.py','Protocol.json','Protocol_Plan.json',
           'Parent_Protocol.json','Data_Provenance.json','Preflight_Audit.json','Preparation_Start.json','Execution_Source_Manifest.json']
    return [ROOT/n for n in names]+list((ROOT/'data').iterdir())


def save_source():return archive_and_save('Experiment02_FashionMNIST_Source_And_Data.zip',source_files())


def save_seed(seed):
    plan=json.loads((ROOT/'Protocol.json').read_text())
    specs=[s for s in plan['specifications'] if s['seed']==seed]
    files=source_files()
    for s in specs:
        d=ROOT/'runs'/f"b{s['batch_size']}_lr{s['multiplier']:g}"/s['arm']/f'seed_{seed}'
        assert (d/'done.json').exists() or (d/'stopped.json').exists()
        files.extend(p for p in d.rglob('*') if p.is_file() and p.suffix!='.tmp')
    return archive_and_save(f'Experiment02_FashionMNIST_Seed_{seed}.zip',files)


if __name__=='__main__':
    result=save_source() if len(sys.argv)==1 else save_seed(int(sys.argv[1]))
    print(json.dumps(result))
