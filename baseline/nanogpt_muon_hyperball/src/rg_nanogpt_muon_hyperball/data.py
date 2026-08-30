from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Protocol

import numpy as np

from .config import load_config, roots

TOKEN_DTYPE = np.dtype(np.uint16)
SPLIT_NAMES = ("train", "val", "test")


class Encoder(Protocol):
    n_vocab: int
    eot_token: int

    def encode_ordinary(self, text: str) -> list[int]: ...


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_document(text: str, encoder: Encoder) -> np.ndarray:
    tokens = encoder.encode_ordinary(text)
    tokens.append(int(encoder.eot_token))
    if tokens and max(tokens) > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("token id exceeds uint16 storage capacity")
    return np.asarray(tokens, dtype=TOKEN_DTYPE)


def write_token_splits(
    texts: Iterable[str],
    encoder: Encoder,
    output_dir: str | Path,
    *,
    train_tokens: int,
    val_tokens: int,
    test_tokens: int,
    dataset_metadata: dict[str, object] | None = None,
    progress_every_documents: int = 2_000,
) -> dict[str, object]:
    """Write exact, document-disjoint splits without loading the corpus in RAM."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "train": int(train_tokens),
        "val": int(val_tokens),
        "test": int(test_tokens),
    }
    if any(value <= 0 for value in targets.values()):
        raise ValueError("all split sizes must be positive")

    partial = {name: output_dir / f"{name}.bin.partial" for name in targets}
    final = {name: output_dir / f"{name}.bin" for name in targets}
    for path in partial.values():
        path.unlink(missing_ok=True)

    handles = {name: path.open("wb") for name, path in partial.items()}
    written = {name: 0 for name in targets}
    document_counts = {name: 0 for name in targets}
    split_names = list(targets)
    split_index = 0
    documents = 0
    started = time.monotonic()

    try:
        for text in texts:
            if split_index >= len(split_names):
                break
            documents += 1
            encoded = _encode_document(str(text), encoder)
            split = split_names[split_index]
            remaining = targets[split] - written[split]
            take = min(remaining, len(encoded))
            if take:
                encoded[:take].tofile(handles[split])
                written[split] += int(take)
                document_counts[split] += 1
            # Never carry a document remainder into the next split. This is the
            # invariant that makes train/validation/test document-disjoint.
            if written[split] == targets[split]:
                split_index += 1
            if progress_every_documents and documents % int(progress_every_documents) == 0:
                total = sum(written.values())
                required = sum(targets.values())
                elapsed = time.monotonic() - started
                rate = total / max(elapsed, 1e-9)
                print(
                    f"[one-head-data] documents={documents:,} "
                    f"tokens={total:,}/{required:,} "
                    f"({100 * total / max(required, 1):.1f}%) "
                    f"rate={rate:,.0f} tok/s",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    if written != targets:
        for path in partial.values():
            path.unlink(missing_ok=True)
        raise RuntimeError(
            f"stream ended before exact splits were filled: {written} != {targets}"
        )

    for name in split_names:
        os.replace(partial[name], final[name])

    metadata: dict[str, object] = {
        "schema_version": 2,
        "tokenizer": "gpt2",
        "vocab_size": int(encoder.n_vocab),
        "eot_token": int(encoder.eot_token),
        "dtype": TOKEN_DTYPE.name,
        "splits": written,
        "split_document_counts": document_counts,
        "document_disjoint_splits": True,
        "documents_consumed": int(documents),
        "files": {
            name: {
                "path": final[name].name,
                "sha256": _sha256(final[name]),
                "bytes": int(final[name].stat().st_size),
            }
            for name in split_names
        },
    }
    if dataset_metadata:
        metadata.update(dataset_metadata)
    temporary = output_dir / "meta.json.tmp"
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(output_dir / "meta.json")
    return metadata


def _validate_file_identity(
    output_dir: Path,
    metadata: dict[str, object],
    expected_splits: dict[str, int],
) -> None:
    file_metadata = metadata.get("files")
    if not isinstance(file_metadata, dict):
        raise RuntimeError(
            "prepared corpus metadata does not contain file hashes; "
            "re-run data preparation with --force"
        )

    for split in SPLIT_NAMES:
        record = file_metadata.get(split)
        if not isinstance(record, dict):
            raise RuntimeError(f"prepared corpus metadata is missing files.{split}")
        path = output_dir / str(record.get("path", f"{split}.bin"))
        if path.name != f"{split}.bin" or not path.is_file():
            raise RuntimeError(f"prepared {split} file path is invalid: {path}")

        expected_bytes = expected_splits[split] * TOKEN_DTYPE.itemsize
        recorded_bytes = int(record.get("bytes", -1))
        actual_bytes = int(path.stat().st_size)
        if recorded_bytes != expected_bytes or actual_bytes != expected_bytes:
            raise RuntimeError(
                f"prepared {split} byte size mismatch: "
                f"metadata={recorded_bytes}, actual={actual_bytes}, "
                f"expected={expected_bytes}"
            )

        recorded_hash = str(record.get("sha256", ""))
        actual_hash = _sha256(path)
        if not recorded_hash or actual_hash != recorded_hash:
            raise RuntimeError(
                f"prepared {split} SHA-256 mismatch; the cached corpus is corrupt "
                "or was modified. Re-run data preparation with --force."
            )


def validate_prepared_data(
    output_dir: str | Path,
    cfg: dict,
) -> dict[str, object]:
    """Validate dataset identity, exact sizes, and every persisted file hash."""

    output_dir = Path(output_dir)
    metadata_path = output_dir / "meta.json"
    required = [
        metadata_path,
        *(output_dir / f"{split}.bin" for split in SPLIT_NAMES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "prepared data are incomplete: " + ", ".join(missing)
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_splits = {
        "train": int(cfg["dataset"]["train_tokens"]),
        "val": int(cfg["dataset"]["val_tokens"]),
        "test": int(cfg["dataset"]["test_tokens"]),
    }
    if metadata.get("splits") != expected_splits:
        raise RuntimeError(
            "prepared split sizes do not match config: "
            f"{metadata.get('splits')} != {expected_splits}"
        )
    if metadata.get("dataset_name") != cfg["dataset"]["name"]:
        raise RuntimeError("prepared dataset identity does not match config")
    if metadata.get("dataset_config") != cfg["dataset"]["config"]:
        raise RuntimeError("prepared dataset configuration does not match config")
    if metadata.get("dataset_revision") != cfg["dataset"]["revision"]:
        raise RuntimeError("prepared dataset revision does not match config")
    if metadata.get("tokenizer") != "gpt2":
        raise RuntimeError("prepared tokenizer must be GPT-2 BPE")
    if metadata.get("dtype") != TOKEN_DTYPE.name:
        raise RuntimeError("prepared token dtype must be uint16")
    if metadata.get("document_disjoint_splits") is not True:
        raise RuntimeError("prepared splits are not marked document-disjoint")

    _validate_file_identity(output_dir, metadata, expected_splits)
    return metadata


def prepare_fineweb_edu(
    cfg: dict,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if not force:
        try:
            metadata = validate_prepared_data(output_dir, cfg)
            print(f"[one-head-data] reusing verified data at {output_dir}")
            return metadata
        except FileNotFoundError:
            pass

    try:
        import tiktoken
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "data preparation requires datasets and tiktoken; "
            "install dependencies into the active conda environment with "
            "`python -m pip install -e .`"
        ) from exc

    dataset_cfg = cfg["dataset"]
    print(
        "[one-head-data] streaming pinned FineWeb-Edu; "
        "the first run requires internet access",
        flush=True,
    )
    stream = load_dataset(
        str(dataset_cfg["name"]),
        name=str(dataset_cfg["config"]),
        split=str(dataset_cfg.get("split", "train")),
        revision=str(dataset_cfg["revision"]),
        streaming=True,
    )
    encoder = tiktoken.get_encoding("gpt2")
    metadata = write_token_splits(
        (row["text"] for row in stream),
        encoder,
        output_dir,
        train_tokens=int(dataset_cfg["train_tokens"]),
        val_tokens=int(dataset_cfg["val_tokens"]),
        test_tokens=int(dataset_cfg["test_tokens"]),
        dataset_metadata={
            "dataset_name": str(dataset_cfg["name"]),
            "dataset_config": str(dataset_cfg["config"]),
            "dataset_split": str(dataset_cfg.get("split", "train")),
            "dataset_revision": str(dataset_cfg["revision"]),
        },
    )
    validate_prepared_data(output_dir, cfg)
    print(f"[one-head-data] complete and verified: {output_dir}")
    return metadata


def load_memmaps(
    output_dir: str | Path,
    cfg: dict,
) -> tuple[dict[str, object], dict[str, np.memmap]]:
    output_dir = Path(output_dir)
    metadata = validate_prepared_data(output_dir, cfg)
    arrays = {
        split: np.memmap(
            output_dir / f"{split}.bin",
            dtype=TOKEN_DTYPE,
            mode="r",
        )
        for split in SPLIT_NAMES
    }
    return metadata, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned FineWeb-Edu one-head baseline corpus"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output_dir) if args.output_dir else roots()["data"]
    prepare_fineweb_edu(cfg, output, force=args.force)


if __name__ == "__main__":
    main()
