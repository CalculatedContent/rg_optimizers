"""Prepare the published California split and exact noisy-quantile recipe.

The transformation follows Yandex's Apache-2.0 implementation at upstream
commit c34e33152db9eea8ed4b2d742b9bef5628d71fb5, lib/data.py. Its unmodified
source and license are retained in reference/upstream. All raw arrays are the
authors' arrays, downloaded from a pinned archive and individually hashed.

Recovery note: the original local row-index JSON was lost with its runtime.
Only that ancillary lookup is now derived by exact row matching against
sklearn's checksum-verified canonical data. The author split and preprocessing
remain unchanged. No random split is generated.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import QuantileTransformer

REFERENCE = Path(__file__).resolve().parent / "reference"
SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}
SPLIT_COUNTS = {"train": 13209, "validation": 3303, "test": 4128}
PREPROCESSING_SEED = 0
PREFIX_BYTES = 4 * 1024 * 1024
VERIFIED_PREPARED_NPZ_SHA256 = "d546ac115ae7bd9e49a20adae63c949db6052a225d45294a58dabdd516ee1314"
VERIFIED_FEATURE_HASHES = {
    "train": "942a9caf7aaefcb5b27881c98b73b4126d6f473fcb5fd38c835e43d387e91ff6",
    "validation": "f2a6e4ced40a417067ec92ee9bf8784c14715199ab825b0bd9b3e891e5f21862",
    "test": "b870b45891bcb08d9f97e41244351479e86e4e512a7c5b10080042b45e57b5b4",
}
CANONICAL_SOURCE = {
    "url": "https://ndownloader.figshare.com/files/5976036",
    "sha256": "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
    "loader": "sklearn.datasets.fetch_california_housing, scikit-learn 1.7.0",
    "purpose": "Original row identity only; not used to create splits or fit preprocessing",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.dtype.str.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def archive_specification():
    return json.loads((REFERENCE / "dataset_archive_manifest.json").read_text())


def extract_verified_prefix(prefix, specification):
    """Extract only explicitly named, hashed files; never unpack arbitrary paths."""
    result = {}
    for item in specification["files"]:
        offset, size = int(item["tar_offset"]), int(item["size"])
        if offset + 512 + size > len(prefix):
            raise ValueError("Downloaded prefix does not contain every California file")
        header = prefix[offset:offset + 512]
        name = header[:100].split(b"\0", 1)[0].decode("utf-8")
        actual_size = int(header[124:136].strip(b"\0 ") or b"0", 8)
        if name != item["path"] or actual_size != size or header[156:157] != b"0":
            raise ValueError("Pinned tar file header differs from reference")
        payload = prefix[offset + 512:offset + 512 + size]
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"Raw author file checksum mismatch: {name}")
        basename = Path(name).name
        if name != f"california/{basename}":
            raise ValueError("Unexpected archive file path")
        result[basename] = payload
    return result


def download_prefix(specification):
    """Read at most 4 MiB plus one byte; reject a server ignoring Range."""
    request = urllib.request.Request(
        specification["url"], headers={"Range": f"bytes=0-{PREFIX_BYTES - 1}"})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206:
            raise RuntimeError("Archive server ignored Range; refusing the full archive download")
        expected_range = f"bytes 0-{PREFIX_BYTES - 1}/{specification['archive_total_bytes']}"
        if response.headers.get("Content-Range") != expected_range:
            raise RuntimeError("Archive server returned an unexpected byte range")
        prefix = response.read(PREFIX_BYTES + 1)
    if len(prefix) != PREFIX_BYTES:
        raise RuntimeError("Archive prefix has an unexpected length")
    return prefix


def ensure_raw_data(data_dir, source_dir=None):
    """Reuse intact author files, import exact local copies, or fetch one prefix."""
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    specification = archive_specification()
    missing = []
    for item in specification["files"]:
        name = Path(item["path"]).name
        path = raw_dir / name
        if path.exists():
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Existing raw array changed: {path}")
        else:
            missing.append(name)
    if not missing:
        return raw_dir, specification
    if source_dir is not None:
        source_dir = Path(source_dir)
        payloads = {}
        for item in specification["files"]:
            name = Path(item["path"]).name
            payload = (source_dir / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Imported author file checksum mismatch: {name}")
            payloads[name] = payload
    else:
        payloads = extract_verified_prefix(download_prefix(specification), specification)
    for name in missing:
        destination = raw_dir / name
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(payloads[name])
        os.replace(temporary, destination)
    return raw_dir, specification


def transform_author_arrays(x_parts, y_parts):
    """Match upstream float32 operations and its fixed preprocessing seed exactly."""
    if any(value.dtype != np.float32 for value in (*x_parts.values(), *y_parts.values())):
        raise ValueError("The published arrays must be float32")
    n_quantiles = max(min(x_parts["train"].shape[0] // 30, 1000), 10)
    normalizer = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=1_000_000_000,
        random_state=PREPROCESSING_SEED,
    )
    fitting_inputs = x_parts["train"] + np.random.RandomState(PREPROCESSING_SEED).normal(
        0.0, 1e-5, x_parts["train"].shape).astype(x_parts["train"].dtype)
    normalizer.fit(fitting_inputs)
    transformed = {key: np.nan_to_num(normalizer.transform(value))
                   for key, value in x_parts.items()}
    mask = np.array([len(np.unique(column)) > 1 for column in transformed["train"].T])
    transformed = {key: value[:, mask].astype(np.float32)
                   for key, value in transformed.items()}
    if mask.shape != (8,) or not mask.all():
        raise ValueError("The frozen California input columns unexpectedly changed")
    # These float32 reductions and arithmetic intentionally match the paper.
    y_mean, y_scale = float(y_parts["train"].mean()), float(y_parts["train"].std())
    standardized = {key: (value - y_mean) / y_scale for key, value in y_parts.items()}
    metadata = {"kind": "noisy-quantile", "fit_on": "train only",
                "preprocessing_seed": PREPROCESSING_SEED, "noise_std": 1e-5,
                "noise_dtype": "float32", "noise_use": "normalizer fit only",
                "n_quantiles": n_quantiles, "output_distribution": "normal",
                "subsample": 1_000_000_000, "label_reduction_dtype": "float32",
                "y_mean": y_mean, "y_scale": y_scale, "label_ddof": 0}
    return transformed, standardized, metadata


def match_original_indices(x_parts, y_parts, canonical_x, canonical_y):
    """Recover row identity by all feature/target bytes; refuse ambiguity."""
    canonical_x = np.asarray(canonical_x, dtype=np.float32)
    canonical_y = np.asarray(canonical_y, dtype=np.float32)
    if canonical_x.ndim != 2 or canonical_y.shape != (len(canonical_x),):
        raise ValueError("Unexpected canonical data shape")
    rows = np.column_stack((canonical_x, canonical_y))
    lookup = {}
    for index, row in enumerate(rows):
        key = row.tobytes()
        if key in lookup:
            raise ValueError("Canonical row identity is not unique")
        lookup[key] = index
    result = {}
    for split in x_parts:
        if x_parts[split].shape[1:] != canonical_x.shape[1:]:
            raise ValueError("Feature columns differ from canonical data")
        rows = np.column_stack((x_parts[split], y_parts[split])).astype(np.float32)
        try:
            result[split] = np.asarray([lookup[row.tobytes()] for row in rows], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("An author row does not match canonical float32 data") from exc
    all_indices = np.concatenate(list(result.values()))
    if not np.array_equal(np.sort(all_indices), np.arange(len(canonical_x))):
        raise ValueError("Author splits do not form a complete disjoint partition")
    return result


def recover_original_indices(x_parts, y_parts, data_dir):
    canonical_x, canonical_y = fetch_california_housing(
        data_home=str(Path(data_dir) / "canonical_source"), return_X_y=True)
    if canonical_x.shape != (20640, 8):
        raise ValueError("Unexpected canonical California data shape")
    indices = match_original_indices(x_parts, y_parts, canonical_x, canonical_y)
    atomic_json(Path(data_dir) / "original_row_indices.json",
                {SPLIT_NAMES[split]: values.tolist() for split, values in indices.items()})
    return indices


def prepare_data(data_dir, source_dir=None):
    """Return a verified manifest, creating the prepared dataset if absent."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive, manifest_path = data_dir / "prepared.npz", data_dir / "manifest.json"
    raw_dir, specification = ensure_raw_data(data_dir, source_dir)
    preprocessing_versions = {"numpy": np.__version__, "scikit-learn": sklearn.__version__}
    if archive.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if sha256(archive) != manifest["prepared_npz_sha256"]:
            raise ValueError("Prepared data checksum mismatch")
        if manifest["preprocessing_versions"] != preprocessing_versions:
            raise ValueError("Prepared data belongs to a different preprocessing environment")
        if manifest["preparation_source_sha256"] != sha256(__file__):
            raise ValueError("Prepared data belongs to a different preparation source")
        return manifest
    if archive.exists() or manifest_path.exists():
        raise ValueError("Partial prepared dataset exists; inspect before replacing it")
    x_parts, y_parts = {}, {}
    for split, original_name in SPLIT_NAMES.items():
        x_parts[split] = np.load(raw_dir / f"X_num_{original_name}.npy", allow_pickle=False)
        y_parts[split] = np.load(raw_dir / f"Y_{original_name}.npy", allow_pickle=False)
        if x_parts[split].shape != (SPLIT_COUNTS[split], 8) or y_parts[split].shape != (SPLIT_COUNTS[split],):
            raise ValueError(f"Unexpected author split shape: {split}")
    x_scaled, y_scaled, preprocessing = transform_author_arrays(x_parts, y_parts)
    for split, expected in VERIFIED_FEATURE_HASHES.items():
        if array_hash(x_scaled[split]) != expected:
            raise ValueError(f"Prepared features differ from pre-recovery verified arrays: {split}")
    if (preprocessing["y_mean"] != 2.0749101638793945
            or preprocessing["y_scale"] != 1.1605902910232544):
        raise ValueError("Target scaling differs from pre-recovery verified float32 reductions")
    index_arrays = recover_original_indices(x_parts, y_parts, data_dir)
    arrays = {"y_mean": np.asarray(preprocessing["y_mean"], dtype=np.float64),
              "y_scale": np.asarray(preprocessing["y_scale"], dtype=np.float64)}
    for split in SPLIT_NAMES:
        arrays[f"x_{split}"] = x_scaled[split]
        arrays[f"y_{split}"] = y_scaled[split]
        arrays[f"y_{split}_original"] = y_parts[split]
        arrays[f"indices_{split}"] = index_arrays[split]
    temporary = archive.with_name(archive.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    if sha256(temporary) != VERIFIED_PREPARED_NPZ_SHA256:
        raise ValueError("Recovered prepared archive differs from the verified original; inspect before training")
    os.replace(temporary, archive)
    manifest = {
        "schema_version": 1, "dataset": "California Housing",
        "dataset_source": "Authors' published TabM/2026 optimizer-benchmark arrays",
        "source_repository": "https://github.com/yandex-research/tabular-dl-optimizers",
        "source_commit": "c34e33152db9eea8ed4b2d742b9bef5628d71fb5",
        "source_archive_url": specification["url"],
        "huggingface_repository_commit": specification["huggingface_repository_commit"],
        "archive_hash_note": "Only the 4 MiB prefix is downloaded; individual array SHA256 hashes are verified.",
        "raw_file_sha256": {Path(item["path"]).name: item["sha256"] for item in specification["files"]},
        "target_unit": "100000 USD", "n_examples": 20640, "n_features": 8,
        "split": {"kind": "fixed author-provided arrays", "counts": SPLIT_COUNTS,
                  "indices_sha256": {key: array_hash(value) for key, value in index_arrays.items()},
                  "indices_mapping_note": "Exact raw float32 feature/target rows matched to sklearn canonical row order; all 20640 original rows occur once.",
                  "canonical_source": CANONICAL_SOURCE},
        "normalization": preprocessing, "preprocessing_versions": preprocessing_versions,
        "preparation_source_sha256": sha256(__file__),
        "prepared_array_sha256": {key: array_hash(value) for key, value in arrays.items()},
        "prepared_npz_sha256": sha256(archive),
        "recovery_note": "Original indices regenerated by exact row identity; prepared archive matches the previously verified SHA256.",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def load_data(data_dir):
    manifest = prepare_data(data_dir)
    with np.load(Path(data_dir) / "prepared.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return arrays, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    manifest = prepare_data(args.data_dir, args.source_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
