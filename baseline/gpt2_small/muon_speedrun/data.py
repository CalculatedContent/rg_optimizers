"""Reuse the pinned FineWeb benchmark cache and its SHA256 downloader."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys

REFERENCE = Path(__file__).resolve().parents[1]/'speedrun30'
sys.path.insert(0, str(REFERENCE))
spec = importlib.util.spec_from_file_location('fineweb_reference', REFERENCE/'train.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
FineWeb = reference.FineWeb
TrainStream = reference.TrainStream
write_json = reference.write_json


def required_shards(source, microbatch=128, updates=3000):
    remaining = updates * 524288
    needed = ['fineweb_val_000000.bin']
    count = microbatch * 1024
    for name, info in sorted(source.manifest['files'].items()):
        if '_train_' not in name:
            continue
        tokens = (info['size']-1024)//2
        remaining -= ((tokens-1)//count)*count
        needed.append(name)
        if remaining <= 0:
            return needed
    raise RuntimeError('Pinned corpus too small for requested schedule')


def prepare(cache, deadline, root, microbatch):
    source = FineWeb(cache, deadline)
    needed = required_shards(source, microbatch)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, _ in zip(needed, pool.map(source.array, needed)):
            print('Verified benchmark shard: '+name, flush=True)
    write_json(Path(root)/'data_receipts.json', source.receipts)
