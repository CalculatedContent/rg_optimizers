"""Official IDX files and the experiment's fixed 55,000 / 5,000 split."""
import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

SPLIT_SEED = 20261015
FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
URLS = {
    "MNIST": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "FashionMNIST": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}
MD5 = {
    "MNIST": ("f68b3c2dcbeaaa9fbdd348bbdeb94873", "d53e105ee54ea40749a09fcbcd1e9432",
              "9fb629c4189551a2d022fa330f9573f3", "ec29112dd5afa0611ce80d1b7f02629c"),
    "FashionMNIST": ("8d4fb7e6c68d591d4c3dfef9ec88bf0d", "25c81989df183df01b3e8a0aad5dffbe",
                     "bef4ecab320f06d8554ea6380940ec79", "bb300cfdad3c16e7a12a480ee83cd310"),
}


def read_idx(root, dataset, index):
    path = Path(root) / dataset / FILES[index]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raw = path.read_bytes()
    else:
        with urllib.request.urlopen(URLS[dataset] + FILES[index], timeout=120) as response:
            raw = response.read()
    if hashlib.md5(raw).hexdigest() != MD5[dataset][index]:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    if not path.exists():
        path.write_bytes(raw)
    raw = gzip.decompress(raw)
    magic, count = struct.unpack(">II", raw[:8])
    if index % 2 == 0:
        if magic != 2051 or struct.unpack(">II", raw[8:16]) != (28, 28):
            raise ValueError(f"Invalid image header: {path}")
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 784).copy()
    if magic != 2049:
        raise ValueError(f"Invalid label header: {path}")
    return np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)


def split_indices(labels):
    rng = np.random.default_rng(SPLIT_SEED)
    train, validation = [], []
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(labels == label))
        validation.extend(ids[:500])
        train.extend(ids[500:])
    return np.asarray(train), np.asarray(validation)


def load_data(root, dataset, test=False):
    """Search reads only the official training files; test is explicitly opt-in."""
    first = 2 if test else 0
    x = torch.tensor(read_idx(root, dataset, first), dtype=torch.float32) / 255
    y = torch.tensor(read_idx(root, dataset, first + 1), dtype=torch.long)
    if test:
        return x, y
    train, validation = split_indices(y.numpy())
    return x, y, train, validation
