"""Check the frozen source manifest before any experiment work."""
import hashlib
import json
from pathlib import Path


def main():
    package = Path(__file__).resolve().parent
    root = package.parent
    manifest = json.loads((package / 'SOURCE_SHA256.json').read_text())
    for name, expected in manifest['files'].items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Frozen source hash mismatch: {name}')
    print(f'Verified {len(manifest["files"])} frozen source files.')


if __name__ == '__main__':
    main()
