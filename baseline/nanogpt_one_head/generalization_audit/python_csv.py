"""Read audit CSVs from byte snapshots with pandas' Python parser."""
import io
import os
from pathlib import Path
import pandas as pd


def snapshot_bytes(path):
    data = Path(path).read_bytes()
    # Audit/training writers terminate each row with a newline. Drop a live
    # writer's unfinished final record rather than trying to parse it.
    if data and not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n') + 1]
    return data


def install():
    if getattr(pd.read_csv, '_audit_python_csv', False):
        return
    original = pd.read_csv

    def read_csv(source, *args, **kwargs):
        kwargs['engine'] = 'python'
        if isinstance(source, (str, os.PathLike)):
            source = io.BytesIO(snapshot_bytes(source))
        return original(source, *args, **kwargs)

    read_csv._audit_python_csv = True
    pd.read_csv = read_csv


if __name__ == '__main__':
    import runpy
    import sys
    install()
    scorer = Path(__file__).with_name('audit.py')
    sys.argv[0] = str(scorer)
    runpy.run_path(str(scorer), run_name='__main__')
