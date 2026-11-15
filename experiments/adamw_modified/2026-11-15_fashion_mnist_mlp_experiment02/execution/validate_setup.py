"""Validate the Fashion-MNIST plan and optional local data; never train/download.

Default: python execution/validate_setup.py
Optional data check: python execution/validate_setup.py --check-data data
"""
import argparse
import csv
import gzip
import hashlib
import json
import struct
from pathlib import Path


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(root, data_dir=None):
    plan = json.loads((root / 'protocol/Protocol_Plan.json').read_text())
    parent = json.loads((root / 'frozen/Parent_Protocol.json').read_text())
    status = json.loads((root / 'results/STATUS.json').read_text())
    assert plan['dataset'] == 'Fashion-MNIST' and plan['architecture'] == [784, 256, 256, 10]
    assert plan['planned_runs'] == 120 and plan['seeds'] == [31001, 31002, 31003, 31004, 31005]
    expected = [s for s in parent['specifications'] if not (s['batch_size'] == 128 and s['multiplier'] == 2)]
    assert plan['specifications'] == expected and len(expected) == 120
    for field in ('architecture','activation','epochs','seeds','arms','adam_layer_rates',
                  'sgdm_magnitude_layer_rates','momentum','beta2','epsilon','graft','sgdm','decay',
                  'warmup_epochs','schedule','initialization','paired_order','spectra','monitors',
                  'checkpoints','test_evaluation_epochs','stat_logging'):
        assert plan[field] == parent[field], field
    reference_hash = digest(root / 'frozen/optimizer_reference.py')
    assert reference_hash == plan['parent_training_source_sha256'] == parent['source_sha256']
    assert plan['data']['split']['sha256'] == parent['data_hashes']['split.npz']
    with (root / 'configs/run_matrix.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 120 and len({r['run_id'] for r in rows}) == 120
    for row, spec in zip(rows, expected):
        assert int(row['seed']) == spec['seed'] and int(row['batch_size']) == spec['batch_size']
        assert float(row['lr_multiplier']) == spec['multiplier'] and row['arm'] == spec['arm']
        assert int(row['epochs']) == 120 and row['status'] == 'planned'
        assert row['run_id'] == f"b{spec['batch_size']}_lr{spec['multiplier']:g}/{spec['arm']}/seed_{spec['seed']}"
    assert status['planned_runs'] == 120
    if status['status'] == 'planned_not_started':
        assert status['completed_runs'] == 0 and not status['training_active'] and not status['training_launched']
        assert status['usable_models'] is None and status['failed_models'] is None
    assert root.name == Path(plan['experiment_directory']).name
    assert root.name != plan['parent_experiment']
    result = {'setup_valid': True, 'experiment': 'Experiment02', 'dataset': 'Fashion-MNIST',
              'planned_runs': 120, 'paired_seeds': 5, 'settings': 8, 'arms': 3,
              'architecture': plan['architecture'], 'optimizer_reference_sha256': reference_hash,
              'reference_and_parent_settings_match': True, 'run_matrix_matches_plan': True,
              'data_checked': False, 'training_performed': False, 'runtime_preflight_performed': False,
              'execution_adapter_ready': False, 'setup_only': True}
    if data_dir is not None:
        directory = Path(data_dir).resolve()
        if directory != (root / 'data').resolve():
            raise ValueError('Stage data in this experiment/data directory; never write into the MNIST sibling.')
        measured = {}
        for f in plan['data']['files']:
            path = directory / f['name']
            if not path.is_file():
                raise FileNotFoundError(f"Missing Fashion-MNIST file: {path}")
            assert digest(path, 'md5') == f['md5'], f"Wrong dataset/checksum: {f['name']}"
            raw = gzip.decompress(path.read_bytes())
            if f['kind'] == 'images':
                assert struct.unpack('>IIII', raw[:16]) == (2051, f['count'], 28, 28)
                assert len(raw) == 16 + f['count'] * 784
            else:
                assert struct.unpack('>II', raw[:8]) == (2049, f['count'])
                assert len(raw) == 8 + f['count'] and max(raw[8:]) <= 9
            measured[f['name']] = digest(path)
        split = directory / 'split.npz'
        assert digest(split) == plan['data']['split']['sha256'], 'Split must be byte-identical to Experiment 01'
        import numpy as np
        with np.load(split, allow_pickle=False) as z:
            train, validation = z['train'], z['validation']
            assert train.shape == (55000,) and validation.shape == (5000,)
            assert np.issubdtype(train.dtype, np.integer) and np.issubdtype(validation.dtype, np.integer)
            joined = np.concatenate([train, validation])
            assert np.array_equal(np.sort(joined), np.arange(60000))
        measured['split.npz'] = digest(split)
        result.update(data_checked=True, data_sha256=measured)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(Path(__file__).resolve().parents[1], args.check_data), indent=2))
