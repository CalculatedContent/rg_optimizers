"""Audit CSV reader: stdlib tokenization and explicit NumPy/Pandas dtypes.

Pandas read_csv(engine='python') still uses native type inference. This module
bypasses both its CSV parsers and that inference, while preserving audit.py.
"""
import csv
import io
import os
from pathlib import Path
import numpy as np
import pandas as pd

NA = {'', 'nan', 'NaN', 'NAN', 'NA', 'N/A', 'NULL', 'null', 'None', '<NA>'}
TEXT_COLUMNS = {'model_state_sha256', 'protocol_fingerprint', 'matrix_name',
                'matrix_type', 'status', 'metric', 'variant', 'period',
                'name', 'longname', 'layer_type', 'warning', 'finger_policy',
                'primary_alpha_variant', 'checkpoint_path'}


def snapshot_bytes(path):
    data = Path(path).read_bytes()
    if data and not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n') + 1]
    return data


def typed_column(name, values):
    present = [v for v in values if v not in NA]
    if not present:
        a = np.full(len(values), np.nan, dtype=np.float64)
        return pd.Series(a, dtype=a.dtype, copy=False)
    if name not in TEXT_COLUMNS:
        if present and len(present) == len(values) and all(v in {'True', 'False'} for v in present):
            a = np.array([v == 'True' for v in values], dtype=np.bool_)
            return pd.Series(a, dtype=a.dtype, copy=False)
        if present and len(present) == len(values):
            try:
                a = np.array([int(v) for v in values], dtype=np.int64)
                return pd.Series(a, dtype=a.dtype, copy=False)
            except (ValueError, OverflowError):
                pass
        try:
            a = np.array([float('nan') if v in NA else float(v) for v in values], dtype=np.float64)
            return pd.Series(a, dtype=a.dtype, copy=False)
        except ValueError:
            pass
    a = np.array([float('nan') if v in NA else v for v in values], dtype=object)
    return pd.Series(a, dtype=object, copy=False)


def read_csv(source, *args, **kwargs):
    if args or kwargs:
        raise TypeError('Audit stdlib CSV reader supports plain read_csv(source) only')
    content = snapshot_bytes(source) if isinstance(source, (str, os.PathLike)) else source.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8-sig')
    rows = csv.reader(io.StringIO(content, newline=''))
    header = next(rows, None)
    if not header or len(set(header)) != len(header):
        raise ValueError(f'Empty or duplicate CSV header: {source}')
    columns = [[] for _ in header]
    for number, row in enumerate(rows, 2):
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f'CSV row {number}: expected {len(header)} fields, got {len(row)} in {source}')
        for dest, value in zip(columns, row):
            dest.append(value)
    return pd.DataFrame({name: typed_column(name, values) for name, values in zip(header, columns)}, copy=False)


def install():
    pd.read_csv = read_csv


if __name__ == '__main__':
    import runpy
    import sys
    install()
    scorer = Path(__file__).with_name('audit.py')
    sys.argv[0] = str(scorer)
    runpy.run_path(str(scorer), run_name='__main__')
