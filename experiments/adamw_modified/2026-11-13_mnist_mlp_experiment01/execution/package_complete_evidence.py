"""Package common inputs, analysis and verification after the full pass ends."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parents[1] / 'artifacts/pass_complete_2026-11-14'


def main():
    v = json.loads((ROOT / 'Final_Verification.json').read_text())
    assert v['passed'] and v['completed_runs'] == 135
    target = DEST / 'Experiment01_Complete135_Analysis_and_Data.zip'
    assert not target.exists(), 'Resolve an existing archive before replacing it.'
    files = []
    for suffix in ('*.py', '*.json', '*.csv', '*.png'):
        files.extend(ROOT.glob(suffix))
    files = [f for f in files if f.name not in ('safeguard_and_finalize.py', 'verify_completed.py')]
    files.extend(f for f in (ROOT / 'data').iterdir() if f.is_file())
    files.extend(f for f in (ROOT / 'verification_pass').rglob('*') if f.is_file())
    for name in ('completion_pass.log', 'analysis.log', 'verification.log', 'plot_pass.log'):
        f = ROOT / name
        if f.exists():
            files.append(f)
    members = []
    with zipfile.ZipFile(target.with_suffix('.zip.part'), 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as z:
        for f in sorted(set(files)):
            name = str(f.relative_to(ROOT))
            z.write(f, name)
            members.append({'path': name, 'bytes': f.stat().st_size,
                            'sha256': hashlib.sha256(f.read_bytes()).hexdigest()})
        z.writestr('RESTORE.md', '# Restore Experiment 01\n\nExtract this common archive and all five seed checkpoint archives into the same empty directory. This yields study.py, Protocol.json, data/ and runs/. Check the frozen source hash and Checkpoint_Manifest.json before analysis. Completion.json and Final_Verification.json are authoritative final status; earlier launch/register files are historical provenance. See Completion_Environment.json for package versions. No training is needed to recreate the plots: run analyze.py, verify_pass.py and plot_pass.py in the restored directory. Exact replay depends on the numerical runtime; the original 14 cross-runtime checks are documented separately.\n')
        z.writestr('Common_Archive_Manifest.json', json.dumps(members, indent=2) + '\n')
    target.with_suffix('.zip.part').replace(target)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    row = {'local_path': str(target), 'bytes': target.stat().st_size,
           'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
           'zip_crc_passed': True, 'member_count': len(members) + 2}
    (DEST / 'Common_Archive_Index.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row))


if __name__ == '__main__':
    main()
