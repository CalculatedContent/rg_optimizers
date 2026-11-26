"""Verify the frozen source before calibration, training, or analysis."""
import hashlib
import json
from pathlib import Path

def main():
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / 'SOURCE_SHA256.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((package.parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Frozen source hash mismatch: {name}')
    print(f'Verified {len(manifest["files"])} frozen source files.')

if __name__ == '__main__':
    main()
