"""Restore only the five exact completed parent artifacts; never train."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'Protocol.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    incoming = ROOT / 'incoming'
    states = ROOT / 'parent_states'
    assert not states.exists(), 'Parent extraction already exists: do not duplicate'
    incoming.mkdir(exist_ok=True)
    states.mkdir()
    receipts = []
    for row in PLAN['parent_artifacts']:
        endpoint = f"repos/CalculatedContent/rg_optimizers/actions/artifacts/{row['artifact_id']}"
        metadata = json.loads(subprocess.check_output(['gh', 'api', endpoint], timeout=120))
        assert metadata['name'] == row['name'] and not metadata['expired']
        assert metadata['workflow_run']['id'] == PLAN['parent_run_id']
        outer = incoming / f"{row['artifact_id']}.zip"
        with outer.open('wb') as f:
            subprocess.run(['gh', 'api', endpoint + '/zip'], stdout=f, check=True, timeout=300)
        assert sha(outer) == row['outer_sha256'], 'Parent artifact bytes changed'
        inner = incoming / row['inner_name']
        with zipfile.ZipFile(outer) as z:
            names = {Path(n).name: n for n in z.namelist()}
            inner.write_bytes(z.read(names[row['inner_name']]))
            inner_meta = json.loads(z.read(names[row['inner_name'] + '.json']))
        assert inner.stat().st_size == inner_meta['bytes'] and sha(inner) == inner_meta['sha256']
        count = 0
        with zipfile.ZipFile(inner) as z:
            for info in z.infolist():
                parts = Path(info.filename).parts
                if info.is_dir() or len(parts) < 5 or parts[0] != 'runs':
                    continue
                assert parts[3] == f"seed_{row['seed']}" and '..' not in parts
                target = states.joinpath(*parts)
                assert not target.exists(), 'Parent observations must not be overwritten'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
                count += 1
        done = list((states / 'runs').glob(f"*/*/seed_{row['seed']}/done.json"))
        assert len(done) == 24
        checkpoint_hashes = {}
        for f in done:
            result = json.loads(f.read_text())
            assert result['endpoint_epoch'] == 20 and result['spec']['seed'] == row['seed']
            for name, expected in result['checkpoint_hashes'].items():
                assert sha(f.parent / name) == expected
            checkpoint_hashes[str(f.parent.relative_to(states))] = sha(f.parent / 'final.pt')
        receipts.append({**row, 'verified': True, 'inner_sha256': inner_meta['sha256'],
                         'extracted_files': count, 'completed_parent_runs': len(done),
                         'final_checkpoint_sha256': checkpoint_hashes})
        (ROOT / 'Parent_Restore_Receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
        print('Restored and hash-verified seed', row['seed'], 'runs', len(done), flush=True)
        outer.unlink()
        inner.unlink()
    assert len(receipts) == 5


if __name__ == '__main__':
    main()
