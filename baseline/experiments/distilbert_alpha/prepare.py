"""Download only official IMDb training data; freeze resolved revisions and split."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from common import atomic_json, build_model, config_dict, digest, file_hash, read_config, seed_all, stratified_split


def prepare(root, c):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    identity = {k: config_dict(c)[k] for k in ('model', 'dataset', 'model_revision', 'dataset_revision', 'split_seed', 'validation_per_class', 'max_length', 'synthetic')}
    manifest = root / 'prepared.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['identity'] != identity:
            raise ValueError('Prepared data identity mismatch; choose a new data directory')
        for name, sha in old['files'].items():
            if file_hash(root / name) != sha:
                raise ValueError('Prepared data changed: ' + name)
        return old
    if c.synthetic:
        rng = np.random.default_rng(44)
        labels = np.tile(np.array([0, 1], dtype=np.int64), 32)
        ids = rng.integers(1, 64, (len(labels), c.max_length), dtype=np.int64)
        masks = np.ones_like(ids)
        resolved = {'model': 'synthetic-random', 'dataset': 'synthetic'}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoTokenizer
        api = HfApi()
        model_sha = api.model_info(c.model, revision=c.model_revision).sha
        data_sha = api.dataset_info(c.dataset, revision=c.dataset_revision).sha
        if model_sha != c.model_revision or data_sha != c.dataset_revision:
            raise RuntimeError('Resolved revisions differ from the frozen protocol')
        resolved = {'model': model_sha, 'dataset': data_sha}
        # Explicit allowlist excludes the IMDb test and unsupervised shards.
        source = snapshot_download(c.dataset, repo_type='dataset', revision=data_sha,
                                   allow_patterns=['plain_text/train-*.parquet'], cache_dir=str(root / 'hf'))
        shards = sorted(str(p) for p in Path(source).glob('plain_text/train-*.parquet'))
        if not shards:
            raise RuntimeError('No official training shards found; test is never used as fallback')
        data = load_dataset('parquet', data_files={'train': shards}, split='train', cache_dir=str(root / 'hf'))
        labels = np.array(data['label'], dtype=np.int64)
        if len(labels) != 25000 or set(labels) != {0, 1}:
            raise ValueError('Unexpected IMDb training set')
        tokenizer = AutoTokenizer.from_pretrained(c.model, revision=model_sha, cache_dir=str(root / 'hf'))
        tokenizer.save_pretrained(root / 'model')
        from transformers import DistilBertModel
        # Save the pretrained encoder; each seed reproducibly initializes the classification head.
        encoder = DistilBertModel.from_pretrained(c.model, revision=model_sha, cache_dir=str(root / 'hf'))
        encoder.save_pretrained(root / 'model')
        encoded = tokenizer(data['text'], padding='max_length', truncation=True, max_length=c.max_length, return_tensors='np')
        ids, masks = encoded['input_ids'], encoded['attention_mask']
    train, val = stratified_split(labels, c.split_seed, c.validation_per_class)
    np.savez(root / 'split.npz', train=train, validation=val)
    np.savez(root / 'tokens.npz', input_ids=ids, attention_mask=masks, labels=labels)
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in [root / 'tokens.npz', root / 'split.npz']}
    for p in sorted((root / 'model').glob('*')):
        if p.is_file():
            files[p.relative_to(root).as_posix()] = file_hash(p)
    out = {'identity': identity, 'resolved_revisions': resolved, 'files': files,
           'train_examples': len(train), 'validation_examples': len(val), 'test_accessed': False,
           'split_sha256': files['split.npz'], 'synthetic_engineering_only': c.synthetic,
           'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy')}}
    atomic_json(manifest, out)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.data, read_config(a.config)), indent=2))
