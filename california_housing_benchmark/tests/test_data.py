"""Check upstream numerical parity, held-out isolation, and exact data recovery."""
import ast
from dataclasses import dataclass
import enum
import hashlib
from pathlib import Path

import numpy as np
import pytest
import sklearn
import sklearn.preprocessing

from california_housing_benchmark import data


def upstream_functions():
    source = data.REFERENCE / "upstream/lib/data.py"
    module = ast.parse(source.read_text())
    keep = {"NumPolicy", "transform_num", "RegressionLabelStats", "standardize_labels"}
    selected = [node for node in module.body if getattr(node, "name", None) in keep]
    assert {node.name for node in selected} == keep
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"__name__": __name__, "np": np, "sklearn": sklearn, "enum": enum,
                 "dataclass": dataclass, "PartKey": str}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["transform_num"], namespace["standardize_labels"]


def example_arrays():
    rng = np.random.default_rng(931)
    x, y = {}, {}
    for split, n in (("train", 600), ("validation", 63), ("test", 77)):
        x[split] = rng.normal(size=(n, 8)).astype(np.float32)
        y[split] = (2.0 + 0.2 * x[split][:, 0] + rng.normal(0, .1, n)).astype(np.float32)
    return x, y


def test_preprocessing_matches_extracted_upstream_functions_exactly():
    x, y = example_arrays()
    official_x, official_y = upstream_functions()
    expected_x = official_x(x, "noisy-quantile", 0)
    expected_y, stats = official_y(y)
    actual_x, actual_y, metadata = data.transform_author_arrays(x, y)
    for split in x:
        np.testing.assert_array_equal(actual_x[split], expected_x[split])
        np.testing.assert_array_equal(actual_y[split], expected_y[split])
    assert metadata["y_mean"] == stats.mean
    assert metadata["y_scale"] == stats.std
    assert all(value.dtype == np.float32 for value in actual_x.values())
    assert all(value.dtype == np.float32 for value in actual_y.values())


def test_held_out_data_never_changes_training_fit_or_label_scaler():
    x, y = example_arrays()
    first_x, first_y, first_metadata = data.transform_author_arrays(x, y)
    changed_x = {k: v.copy() for k, v in x.items()}
    changed_y = {k: v.copy() for k, v in y.items()}
    for split in ("validation", "test"):
        changed_x[split] *= 1e8
        changed_y[split] += 1e6
    second_x, second_y, second_metadata = data.transform_author_arrays(changed_x, changed_y)
    np.testing.assert_array_equal(first_x["train"], second_x["train"])
    np.testing.assert_array_equal(first_y["train"], second_y["train"])
    assert first_metadata == second_metadata
    assert np.isfinite(second_x["test"]).all()
    assert abs(second_x["test"]).max() < 6


def test_tar_payload_and_header_are_both_verified():
    payload = b"exact bytes"
    prefix = bytearray(1024)
    name = b"california/example.npy"
    prefix[:len(name)] = name
    prefix[124:136] = f"{len(payload):011o}\0".encode()
    prefix[156:157] = b"0"
    prefix[512:512 + len(payload)] = payload
    spec = {"files": [{"path": name.decode(), "tar_offset": 0, "size": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest()}]}
    assert data.extract_verified_prefix(bytes(prefix), spec) == {"example.npy": payload}
    corrupt = bytearray(prefix)
    corrupt[512] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        data.extract_verified_prefix(bytes(corrupt), spec)
    wrong_name = bytearray(prefix)
    wrong_name[0] = ord("X")
    with pytest.raises(ValueError, match="header"):
        data.extract_verified_prefix(bytes(wrong_name), spec)


def test_ignored_range_is_rejected_before_reading_body(monkeypatch):
    class Response:
        status = 200
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, *args):
            raise AssertionError("Must not read a full archive")
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="ignored Range"):
        data.download_prefix({"url": "https://example.invalid/archive", "archive_total_bytes": 10000000})


def test_original_indices_recovered_exactly_and_ambiguity_refused():
    canonical_x = (np.arange(48).reshape(6, 8) / 10).astype(np.float32)
    canonical_y = (np.arange(6) / 3).astype(np.float32)
    selection = {"train": np.array([4, 1, 5]), "validation": np.array([0]),
                 "test": np.array([2, 3])}
    x = {k: canonical_x[v] for k, v in selection.items()}
    y = {k: canonical_y[v] for k, v in selection.items()}
    recovered = data.match_original_indices(x, y, canonical_x, canonical_y)
    for split in selection:
        np.testing.assert_array_equal(recovered[split], selection[split])
    duplicate_x, duplicate_y = canonical_x.copy(), canonical_y.copy()
    duplicate_x[1], duplicate_y[1] = duplicate_x[0], duplicate_y[0]
    with pytest.raises(ValueError, match="not unique"):
        data.match_original_indices(x, y, duplicate_x, duplicate_y)
    x["test"] = x["test"][[0, 0]]
    y["test"] = y["test"][[0, 0]]
    with pytest.raises(ValueError, match="complete disjoint partition"):
        data.match_original_indices(x, y, canonical_x, canonical_y)
