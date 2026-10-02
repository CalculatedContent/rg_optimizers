"""Opt-in continuous-run probes, paired measurements and durable publication.

No training RNG is consumed here. GCS writes are synchronous: a failed backup
raises instead of allowing an apparently healthy but unprotected long run.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def document_windows(data, count, width, seed, eot=50256):
    # Scan in bounded chunks: do not allocate a 5 GB boolean corpus mask.
    ends = []
    chunk = 4_000_000
    for offset in range(0, len(data), chunk):
        ends.extend((np.flatnonzero(data[offset:offset+chunk] == eot) + offset).tolist())
    starts = np.asarray([0, *[e + 1 for e in ends]], dtype=np.int64)
    ends = np.asarray([*ends, len(data)], dtype=np.int64)
    eligible = np.flatnonzero(ends - starts >= width)
    if len(eligible) < count:
        raise ValueError(f'Need {count} eligible documents, found {len(eligible)}')
    rng = np.random.default_rng(seed)
    docs = rng.choice(eligible, count, replace=False)
    offsets = np.asarray([rng.integers(starts[i], ends[i]-width+1) for i in docs])
    windows = np.stack([np.asarray(data[o:o+width], dtype=np.int64) for o in offsets])
    if np.any(windows == eot):
        raise RuntimeError('Probe crossed a document boundary')
    return windows, docs, offsets


def build_document_probes(cfg, arrays, run_dir, metadata):
    ev = cfg['evaluation']
    count = int(ev['probe_documents'])
    batch = int(cfg['training']['batch_size'])
    width = int(cfg['model']['block_size']) + 1
    if count % batch:
        raise ValueError('probe_documents must divide into full evaluation batches')
    records, probes = {}, []
    for split, key in [('train','train_probe_seed'), ('val','validation_probe_seed'), ('test','test_probe_seed')]:
        windows, docs, offsets = document_windows(arrays[split], count, width, int(ev[key]), int(metadata['eot_token']))
        records[split] = dict(document_ids=docs.tolist(), token_offsets=offsets.tolist(),
                              windows_sha256=hashlib.sha256(windows.tobytes()).hexdigest(),
                              data_sha256=metadata['files'][split]['sha256'])
        probes.append([(torch.from_numpy(windows[i:i+batch,:-1].copy()),
                        torch.from_numpy(windows[i:i+batch,1:].copy()))
                       for i in range(0, count, batch)])
    record = dict(schema_version=1, documents=count, context=width-1, splits=records,
                  metric='100 * incorrect argmax next-token predictions / scored tokens; teacher forced',
                  tokens_per_split=count*(width-1))
    path = Path(run_dir)/'fixed_document_probe.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise RuntimeError('Fixed document probe changed')
    atomic_json(path, record)
    return probes


class CloudPublisher:
    def __init__(self, uri):
        if not uri.startswith('gs://') or '/' not in uri[5:]:
            raise ValueError('Expected gs://bucket/run-prefix')
        from google.cloud import storage
        bucket, self.prefix = uri[5:].split('/', 1)
        self.prefix = self.prefix.strip('/')
        if not self.prefix:
            raise ValueError('An experiment-specific bucket prefix is required')
        self.bucket = storage.Client().bucket(bucket)
        self.metadata_cache = {}

    def file(self, path, relative):
        blob = self.bucket.blob(self.prefix + '/' + relative)
        blob.upload_from_filename(str(path), checksum='crc32c', timeout=600)
        blob.reload()
        if int(blob.size) != Path(path).stat().st_size:
            raise RuntimeError('Cloud object size differs from local artifact')
        return {'object': blob.name, 'generation': str(blob.generation),
                'bytes': int(blob.size), 'crc32c': blob.crc32c}

    def claim(self, value):
        # A deleted/recreated VM must never silently overwrite an earlier archive.
        self.bucket.blob(self.prefix + '/RUN_CLAIM.json').upload_from_string(
            json.dumps(value), content_type='application/json', checksum='crc32c',
            if_generation_match=0, timeout=120)

    def snapshot_text_file(self, path, relative):
        # Runtime libraries may append to the log during upload. Upload a fixed
        # prefix, never compare its length to a subsequently growing live file.
        with Path(path).open('rb') as f:
            content = f.read(os.fstat(f.fileno()).st_size)
        self.bucket.blob(self.prefix + '/' + relative).upload_from_string(
            content, content_type='text/plain', checksum='crc32c', timeout=120)

    def json(self, value, relative):
        self.bucket.blob(self.prefix + '/' + relative).upload_from_string(
            json.dumps(value, indent=2, allow_nan=False), content_type='application/json',
            checksum='crc32c', timeout=120)


_PUBLISHERS = {}
def publisher(cfg):
    if not cfg.get('continuous', {}).get('enabled'):
        return None
    uri = os.environ.get('RG_CONTINUOUS_GCS_URI', '')
    if not uri:
        if cfg['continuous'].get('cloud_required', True):
            raise RuntimeError('RG_CONTINUOUS_GCS_URI is required')
        return None
    if uri not in _PUBLISHERS:
        _PUBLISHERS[uri] = CloudPublisher(uri)
    return _PUBLISHERS[uri]


def publish_metadata(cfg, run_dir):
    sink = publisher(cfg)
    if sink is None:
        return
    root = Path(run_dir)
    # Called on the training thread, after writers flush, with no concurrent writer.
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in {'.json', '.csv', '.yaml', '.png', '.pdf'}:
            stat = path.stat()
            identity = (stat.st_mtime_ns, stat.st_size)
            if sink.metadata_cache.get(str(path)) != identity:
                sink.file(path, 'results/' + path.relative_to(root).as_posix())
                sink.metadata_cache[str(path)] = identity
    log = os.environ.get('RG_CONTINUOUS_RUN_LOG')
    if log and Path(log).is_file():
        sink.snapshot_text_file(log, 'run.log')


def publish_checkpoint(path, payload):
    cfg = payload.get('config', {})
    sink = publisher(cfg)
    if sink is None:
        return
    path = Path(path)
    step = int(payload['step'])
    relative = f'checkpoints/step_{step:09d}/{path.name}'
    receipt = sink.file(path, relative)
    receipt.update(step=step, sha256=sha_file(path), fingerprint=payload['fingerprint'],
                   model_state_sha256=payload['model_state_sha256'],
                   resumable=bool(payload.get('optimizers')) and
                             (step == 0 or payload.get('resume_diagnostics') is not None))
    # A completion marker is published only AFTER the complete verified upload.
    sink.json(receipt, relative + '.complete.json')
    if receipt['resumable'] and path.name in {'checkpoint_latest.pt', 'checkpoint_final.pt', 'checkpoint_initial.pt'}:
        sink.json(receipt, 'LATEST_RESUMABLE.json')
    root = path.parent.parent if path.parent.name == 'epoch_checkpoints' else path.parent
    publish_metadata(cfg, root)
    print(f'[continuous-backup] saved step={step} file={path.name}', flush=True)


def record_pair(cfg, run_dir, row, summary):
    if not cfg.get('continuous', {}).get('enabled'):
        return
    count = 6 * int(cfg['model']['n_layer'])
    pair = {k: row[k] for k in ('step', 'tokens_seen', 'elapsed_sec', 'primary_lr', 'train_loss', 'val_loss', 'test_loss')}
    pair.update(train_token_error_pct=100*(1-row['train_accuracy']),
                test_token_error_pct=100*(1-row['test_accuracy']),
                model_state_sha256=summary['model_state_sha256'],
                probe_sha256=sha_file(Path(run_dir)/'fixed_document_probe.json'),
                expected_matrices=count)
    for kind in ('raw', 'clip_xmax'):
        n = int(summary.get(f'alpha_{kind}_n', 0))
        pair[f'alpha_{kind}_n'] = n
        for stat in ('mean', 'min'):
            # Never silently average a changing subset of layers.
            pair[f'alpha_{kind}_{stat}'] = summary.get(f'alpha_{kind}_{stat}', float('nan')) if n == count else float('nan')
    path = Path(run_dir)/'alpha_token_error.csv'
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pair))
        if not exists:
            writer.writeheader()
        writer.writerow(pair)
        f.flush()
        os.fsync(f.fileno())
    plot_pairs(path)
    publish_metadata(cfg, run_dir)


def plot_pairs(path):
    # No pandas CSV parser: the previous environment segfaulted in that path.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with Path(path).open() as f:
        rows = [r for r in csv.DictReader(f) if int(r['step']) > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key in zip(axes, ('alpha_raw_mean', 'alpha_raw_min')):
        pairs = [(float(r[key]), float(r['test_token_error_pct']), int(r['step'])) for r in rows]
        pairs = np.asarray([p for p in pairs if all(math.isfinite(v) for v in p)], dtype=float).reshape(-1, 3)
        if len(pairs):
            x, y, steps = pairs.T
            dots = ax.scatter(x, y, c=steps, s=18, cmap='viridis')
            fig.colorbar(dots, ax=ax, label='Training step')
            if len(pairs) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                line = np.array([x.min(), x.max()])
                ax.plot(line, slope*line+intercept, color='black', lw=1)
                ax.set_title(f"Pearson r={np.corrcoef(x,y)[0,1]:.3f}; n={len(x)}")
        ax.set_xlabel(key.replace('_', ' '))
        ax.set_ylabel('Fixed-test token error (%)')
    fig.suptitle('Continuous MuonClip — no detrending; step zero excluded')
    fig.savefig(Path(path).with_suffix('.png'), dpi=160)
    plt.close(fig)


def record_token_errors(cfg, run_dir, row):
    if not cfg.get('continuous', {}).get('enabled'):
        return
    result = {'step': row['step'], 'tokens_seen': row['tokens_seen'],
              'train_token_error_pct': 100*(1-row['train_accuracy']),
              'val_token_error_pct': 100*(1-row['val_accuracy']),
              'test_token_error_pct': 100*(1-row['test_accuracy']),
              'probe_sha256': sha_file(Path(run_dir)/'fixed_document_probe.json')}
    path = Path(run_dir)/'token_error.csv'
    exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(result))
        if not exists: w.writeheader()
        w.writerow(result)
    publish_metadata(cfg, run_dir)
