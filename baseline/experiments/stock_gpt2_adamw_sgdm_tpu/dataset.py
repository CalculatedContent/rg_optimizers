"""Pinned FineWeb tokens and a batch-independent, resumable sequential stream."""
from __future__ import annotations

# Keep direct-script entry points inside this experiment's package namespace.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]))
    __package__ = "baseline.experiments.stock_gpt2_adamw_sgdm_tpu"

import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import torch

from .checkpoint import sha256_file, write_json

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "gpt2_small/speedrun30/data_manifest.json"
MANIFEST_SHA256 = "028141b267e9a6c99af8beea42bed58fe1d7418bdf034d630f1792aeb6a20e19"
VAL_NAME = "fineweb_val_000000.bin"


def load_manifest():
    if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256:
        raise RuntimeError("Pinned FineWeb data manifest changed")
    return json.loads(MANIFEST_PATH.read_text())


def required_names(manifest, budget_tokens):
    names = [VAL_NAME]
    available = 0
    for name, item in sorted(manifest["files"].items()):
        if "_train_" not in name:
            continue
        names.append(name)
        available += (item["size"] - 1024) // 2
        if available >= budget_tokens + 1:
            break
    if len(names) < 2:
        raise ValueError("No training shards in pinned manifest")
    return names


def read_shard(path, *, expected_size=None):
    path = Path(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError("Incorrect shard size: " + str(path))
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<i4", count=256)
    if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
        raise ValueError("Expected GPT-2 uint16 llm.c shard: " + str(path))
    count = int(header[2])
    if count < 2 or 1024 + count * 2 != path.stat().st_size:
        raise ValueError("Incorrect shard token count: " + str(path))
    return np.memmap(path, mode="r", dtype="<u2", offset=1024, shape=(count,))


def prepare(data_dir, budget_tokens, *, download=False):
    """Hash each required existing file; optionally download missing/bad files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    receipts = {}
    for name in required_names(manifest, budget_tokens):
        expected = manifest["files"][name]
        path = data_dir / name
        valid = path.exists() and path.stat().st_size == expected["size"] and sha256_file(path) == expected["sha256"]
        if not valid:
            if not download:
                raise RuntimeError(f"Missing/unverified shard {name}; run prepare_data.py first")
            url = f'https://huggingface.co/datasets/{manifest["repo"]}/resolve/{manifest["revision"]}/{name}'
            temporary = path.with_suffix(".partial")
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    handle.write(chunk)
                    digest.update(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                raise RuntimeError("Downloaded shard failed verification: " + name)
            os.replace(temporary, path)
        read_shard(path, expected_size=expected["size"])
        receipts[name] = expected
    result = {"manifest_sha256": MANIFEST_SHA256, "repo": manifest["repo"],
              "revision": manifest["revision"], "files": receipts,
              "training_order": "sorted sequential shards; continuous tokens; no dropped batch tails"}
    write_json(data_dir / "DATA_VERIFIED.json", result)
    return result


class TokenStream:
    """Continuous corpus stream; changing microbatch size never changes its prefix.

    Cursor points at the next input token. A batch of N targets consumes N input
    positions and peeks one following target. Shard boundaries never drop tokens.
    """
    def __init__(self, arrays, *, names=None, shard=0, position=0, consumed=0):
        if not arrays or any(len(item) < 2 for item in arrays):
            raise ValueError("A token stream needs nonempty shards")
        self.arrays = arrays
        self.names = names or [str(index) for index in range(len(arrays))]
        self.shard, self.position, self.consumed = shard, position, consumed
        self.load_state_dict(self.state_dict())

    def state_dict(self):
        return {"shard": self.shard, "position": self.position, "consumed": self.consumed,
                "names": list(self.names)}

    def load_state_dict(self, state):
        if state["names"] != list(self.names):
            raise ValueError("Data shard inventory differs from checkpoint")
        shard, position, consumed = (state[name] for name in ("shard", "position", "consumed"))
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (shard, position, consumed)):
            raise ValueError("Invalid data cursor types")
        if not 0 <= shard < len(self.arrays) or not 0 <= position < len(self.arrays[shard]) or consumed < 0:
            raise ValueError("Invalid data cursor")
        cycle = sum(len(item) for item in self.arrays)
        absolute = sum(len(item) for item in self.arrays[:shard]) + position
        if consumed % cycle != absolute:
            raise ValueError("Data cursor disagrees with consumed token count")
        self.shard, self.position, self.consumed = shard, position, consumed

    def _take(self, count):
        chunks = []
        while count:
            array = self.arrays[self.shard]
            amount = min(count, len(array) - self.position)
            chunks.append(np.asarray(array[self.position:self.position + amount], dtype=np.int64))
            self.position += amount
            self.consumed += amount
            count -= amount
            if self.position == len(array):
                self.shard = (self.shard + 1) % len(self.arrays)
                self.position = 0
        return np.concatenate(chunks)

    def next_batch(self, batch, context):
        count = batch * context
        inputs = self._take(count)
        next_token = int(self.arrays[self.shard][self.position])
        targets = np.concatenate((inputs[1:], np.array([next_token], dtype=np.int64)))
        return torch.from_numpy(inputs).view(batch, context), torch.from_numpy(targets).view(batch, context)


def build_data(config, data_dir):
    if config.mode == "smoke":
        generator = np.random.default_rng(config.data_seed)
        train = generator.integers(0, config.vocab_size, size=4099, dtype=np.int64)
        val = generator.integers(0, config.vocab_size,
                                 size=config.validation_offset + config.validation_tokens + 1, dtype=np.int64)
        return TokenStream([train], names=["synthetic-smoke-only"]), val, {"synthetic": True, "data_seed": config.data_seed}
    receipts = prepare(data_dir, config.budget_tokens, download=False)
    names = [name for name in receipts["files"] if "_train_" in name]
    arrays = [read_shard(Path(data_dir) / name) for name in names]
    val = read_shard(Path(data_dir) / VAL_NAME)
    if config.validation_offset + config.validation_tokens + 1 > len(val):
        raise ValueError("Validation slice exceeds verified shard")
    return TokenStream(arrays, names=names), val, receipts
