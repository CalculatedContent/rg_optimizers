"""Pack WikiText-103 within each official split and freeze upstream/data hashes."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import atomic_json, file_hash

TOKENIZER = 'distilbert/distilbert-base-uncased'
TOKENIZER_REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
DATASET = 'Salesforce/wikitext'
DATASET_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def prepare(root, length=128, dataset_revision=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Preparation requires an empty destination; existing data preserved')
    root.mkdir(parents=True, exist_ok=True)
    revision = HfApi().dataset_info(DATASET, revision=dataset_revision or DATASET_REVISION).sha
    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    tok.save_pretrained(root / 'tokenizer')
    ds = load_dataset(DATASET, 'wikitext-103-raw-v1', revision=revision)
    meta = dict(dataset=DATASET, dataset_config='wikitext-103-raw-v1', dataset_revision=revision,
                tokenizer=TOKENIZER, tokenizer_revision=TOKENIZER_REVISION,
                sequence_length=length, vocab_size=len(tok), mask_id=tok.mask_token_id,
                special_ids=tok.all_special_ids, synthetic=False, splits={},
                packing=f'[CLS] + {length - 2} content tokens + [SEP]; separator between nonempty lines; no cross-split packing')
    for split in ('train', 'validation', 'test'):
        raw = root / (split + '.bin')
        buf, rows, text_tokens = [], 0, 0
        with raw.open('wb') as f:
            for batch in ds[split].iter(batch_size=1024):
                text = [s for s in batch['text'] if s.strip()]
                if not text:
                    continue
                for ids in tok(text, add_special_tokens=False, truncation=False)['input_ids']:
                    if not ids:
                        continue
                    text_tokens += len(ids)
                    buf.extend(ids + [tok.sep_token_id])
                end = len(buf) // (length - 2) * (length - 2)
                if end:
                    mid = np.asarray(buf[:end], dtype=np.int32).reshape(-1, length - 2)
                    blocks = np.empty((len(mid), length), dtype=np.int32)
                    blocks[:, 0], blocks[:, -1] = tok.cls_token_id, tok.sep_token_id
                    blocks[:, 1:-1] = mid
                    # Exclude any all-special block before freezing the dataset.
                    blocks = blocks[(~np.isin(blocks, tok.all_special_ids)).any(axis=1)]
                    blocks.tofile(f)
                    rows += len(blocks)
                    buf = buf[end:]
        if not rows:
            raise RuntimeError('No prepared rows in ' + split)
        meta['splits'][split] = dict(rows=rows, text_tokens=text_tokens,
                                    dropped_remainder=len(buf), file=raw.name, sha256=file_hash(raw))
    atomic_json(root / 'prepared.json', meta)
    return meta


def synthetic(root, length=16):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    meta = dict(sequence_length=length, vocab_size=128, mask_id=4,
                special_ids=[0, 1, 2, 3, 4], synthetic=True, splits={})
    for split, count in [('train', 19), ('validation', 11), ('test', 13)]:
        ids = np.tile(np.arange(length) % 7 + 5, (count, 1)).astype(np.int32)
        ids[:, 0], ids[:, -1] = 1, 2
        path = root / (split + '.bin')
        ids.tofile(path)
        meta['splits'][split] = dict(rows=count, file=path.name, sha256=file_hash(path))
    atomic_json(root / 'prepared.json', meta)
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--dataset-revision')
    a = p.parse_args()
    if not 8 <= a.length <= 512:
        p.error('length must be 8..512')
    print(json.dumps(prepare(a.output, a.length, a.dataset_revision), indent=2))
