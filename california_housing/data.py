"""Download, split and standardize California Housing without test leakage."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SPLIT_SEED = 20261126
SPLIT_COUNTS = {"train": 16512, "validation": 2064, "test": 2064}


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.dtype.str.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def prepare_data(data_dir):
    """Return the preparation manifest; reuse only an intact existing archive."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "prepared.npz"
    manifest_path = data_dir / "manifest.json"
    if archive.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != manifest["prepared_npz_sha256"]:
            raise ValueError("Prepared data archive differs from its manifest")
        if manifest["split"]["seed"] != SPLIT_SEED:
            raise ValueError("Existing data uses another split")
        return manifest
    if archive.exists() or manifest_path.exists():
        raise ValueError("Partial data preparation exists; inspect it before replacing it")

    import sklearn
    from sklearn.datasets import fetch_california_housing
    from sklearn.datasets._california_housing import ARCHIVE

    dataset = fetch_california_housing(data_home=str(data_dir / "sklearn_cache"))
    x = np.asarray(dataset.data, dtype=np.float64)
    y = np.asarray(dataset.target, dtype=np.float64)
    if x.shape != (20640, 8) or y.shape != (20640,):
        raise ValueError("Unexpected California Housing data dimensions")
    indices = np.random.default_rng(SPLIT_SEED).permutation(len(y))
    splits = {"train": indices[:16512], "validation": indices[16512:18576],
              "test": indices[18576:]}
    train_idx = splits["train"]
    x_mean, x_scale = x[train_idx].mean(0), x[train_idx].std(0, ddof=0)
    y_mean, y_scale = float(y[train_idx].mean()), float(y[train_idx].std(ddof=0))
    if np.any(x_scale <= 0) or y_scale <= 0:
        raise ValueError("A training scaler has zero variance")
    arrays = {"x_mean": x_mean, "x_scale": x_scale,
              "y_mean": np.array(y_mean), "y_scale": np.array(y_scale)}
    for name, idx in splits.items():
        arrays[f"x_{name}"] = ((x[idx] - x_mean) / x_scale).astype(np.float32)
        arrays[f"y_{name}"] = ((y[idx] - y_mean) / y_scale).astype(np.float32)
        arrays[f"y_{name}_original"] = y[idx].copy()
        arrays[f"indices_{name}"] = idx
    temporary = archive.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, archive)
    manifest = {
        "schema_version": 1, "dataset": "California Housing",
        "source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html",
        "source_archive_url": ARCHIVE.url,
        "source_archive_expected_sha256": ARCHIVE.checksum,
        "source_archive_check": "scikit-learn fetcher verifies the archive checksum on download",
        "sklearn_version": sklearn.__version__,
        "raw_array_sha256": {"features_float64": _array_hash(x), "target_float64": _array_hash(y)},
        "feature_names": list(dataset.feature_names), "n_examples": len(y), "n_features": 8,
        "target_unit": "100000 USD", "split": {"seed": SPLIT_SEED,
        "generator": "numpy.random.default_rng / PCG64", "fractions": [0.8, 0.1, 0.1],
        "counts": SPLIT_COUNTS, "indices_sha256": {k: _array_hash(v) for k, v in splits.items()}},
        "normalization": {"fit_on": "train only", "ddof": 0, "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(), "y_mean": y_mean, "y_scale": y_scale,
        "output_dtype": "float32"},
        "prepared_npz_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def load_data(data_dir):
    data_dir = Path(data_dir)
    manifest = prepare_data(data_dir)
    with np.load(data_dir / "prepared.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return arrays, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.data_dir), indent=2, sort_keys=True))
