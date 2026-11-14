"""Package finished seed blocks without starting training or an upload service."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parents[1] / 'artifacts' / 'pass_complete_2026-11-14'
FROZEN = '1bfd7032e00ad87d729670197925fb40e19bc802a8eb716db2571e56b538d3d8'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    assert sha(ROOT / 'study.py') == FROZEN
    DEST.mkdir(parents=True, exist_ok=True)
    index = []
    for seed in range(31001, 31006):
        done = sorted((ROOT / 'runs').glob(f'*/*/seed_{seed}/done.json'))
        if len(done) != 27:
            continue
        for path in done:
            obj = json.loads(path.read_text())
            assert obj['status'] == 'complete' and obj['last_epoch'] == 120
            assert obj['spec']['seed'] == seed
        archive = DEST / f'Experiment01_Seed{seed}_27runs_Checkpoints.zip'
        if not archive.exists():
            temp = archive.with_suffix('.zip.part')
            with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
                for path in done:
                    for f in sorted(path.parent.rglob('*')):
                        if f.is_file() and f.suffix not in ('.tmp', '.part'):
                            z.write(f, str(f.relative_to(ROOT)))
                for name in ('Protocol.json', 'study.py', 'Preflight_Audit.json',
                             'Preflight_Amendment.json', 'Pass_Restoration_Audit.json',
                             'Completion_Pass_Launch.json'):
                    z.write(ROOT / name, name)
            temp.replace(archive)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            checkpoints = [n for n in z.namelist() if '/checkpoints/' in n and n.endswith('.pt')]
            completions = [n for n in z.namelist() if n.endswith('/done.json')]
            assert len(checkpoints) == 135 and len(completions) == 27
        row = {'seed': seed, 'completed_runs': 27, 'checkpoints': 135,
               'local_path': str(archive), 'bytes': archive.stat().st_size,
               'sha256': sha(archive), 'zip_crc_passed': True}
        index.append(row)
        print(json.dumps(row), flush=True)
    (DEST / 'Seed_Archive_Index.json').write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    main()
