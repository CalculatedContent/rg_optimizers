"""Dependency-light checks for resumable weight-quotient notebook tables."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd


BASELINE_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = (
    BASELINE_ROOT
    / "experiments"
    / "mnist_mlp3_tangent_rg"
    / "scripts"
    / "build_notebooks.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WeightQuotientResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_module(
            "weight_quotient_resume_builder", BUILDER_PATH
        )
        cls.weight_quotients = load_module(
            "weight_quotient_resume_math",
            BASELINE_ROOT / "rg_baselines" / "tangent_rg" / "weight_quotients.py",
        )
        cls.weightwatcher_fit = load_module(
            "weight_quotient_resume_ww",
            BASELINE_ROOT / "rg_baselines" / "tangent_rg" / "weightwatcher_fit.py",
        )

    def helper_namespace(self, *, ww_min_evals: int = 8) -> dict[str, object]:
        namespace: dict[str, object] = {
            "hashlib": hashlib,
            "copy": copy,
            "inspect": inspect,
            "json": json,
            "re": re,
            "Path": Path,
            "np": np,
            "pd": pd,
            "weight_quotients": self.weight_quotients,
            "weightwatcher_fit": self.weightwatcher_fit,
            "analyze_weightwatcher_dual": (
                self.weightwatcher_fit.analyze_weightwatcher_dual
            ),
            "validate_weightwatcher_measurement": (
                self.weightwatcher_fit.validate_weightwatcher_measurement
            ),
            "RESUME_PARTIAL_RESULTS": True,
            "WW_MIN_EVALS": int(ww_min_evals),
            "WW_MAX_EVALS": None,
            "WW_MAX_FINGERS": 10,
            "WW_SVD_METHOD": "accurate",
        }
        exec(self.builder.WEIGHT_QUOTIENT_HELPERS, namespace)
        namespace["WEIGHT_QUOTIENT_ANALYSIS_CODE_SHA256"] = "current-code"
        return namespace

    def test_only_complete_matching_groups_resume(self) -> None:
        namespace = self.helper_namespace()
        changed_settings = self.helper_namespace(ww_min_evals=9)
        self.assertNotEqual(
            namespace["WEIGHT_QUOTIENT_ANALYSIS_SETTINGS_SHA256"],
            changed_settings["WEIGHT_QUOTIENT_ANALYSIS_SETTINGS_SHA256"],
        )
        context = {
            "optimizer": "muon",
            "seed": 1337,
            "epoch": 1000,
            "global_step": 430_000,
            "protocol_fingerprint": "run-fingerprint",
            "source_artifact_kind": "verified_final_100_tail_cache",
            "analysis_settings_sha256": namespace[
                "WEIGHT_QUOTIENT_ANALYSIS_SETTINGS_SHA256"
            ],
        }
        contexts = pd.DataFrame([context])
        common = {
            **context,
            "method": "gram_ridge",
            "profile_id": "ridge-half",
            "parameter_profile": '{"tau_fraction": 0.5}',
            "analysis_code_sha256": "current-code",
        }
        fits = pd.DataFrame(
            [
                {
                    **common,
                    "layer": layer,
                    "fit_variant": variant,
                    "transformed_spectrum_sha256": "same-spectrum",
                }
                for layer in namespace["EXPECTED_WEIGHT_LAYERS"]
                for variant in ("raw", "clip_xmax")
            ]
        )
        operators = pd.DataFrame(
            [
                {**common, "layer": layer}
                for layer in namespace["EXPECTED_WEIGHT_LAYERS"]
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            fit_path = Path(directory) / "fits.csv"
            operator_path = Path(directory) / "operators.csv"
            fits.to_csv(fit_path, index=False)
            operators.to_csv(operator_path, index=False)
            loaded_fits, loaded_operators, completed = namespace[
                "load_resumable_tables"
            ](
                fit_path=fit_path,
                secondary_path=operator_path,
                contexts=contexts,
                method="gram_ridge",
                profiles=[{"profile_id": "ridge-half", "tau_fraction": 0.5}],
            )
            self.assertEqual(len(loaded_fits), 6)
            self.assertEqual(len(loaded_operators), 3)
            self.assertEqual(len(completed), 1)

            stale = fits.copy()
            stale["analysis_code_sha256"] = "old-code"
            stale.to_csv(fit_path, index=False)
            loaded_fits, loaded_operators, completed = namespace[
                "load_resumable_tables"
            ](
                fit_path=fit_path,
                secondary_path=operator_path,
                contexts=contexts,
                method="gram_ridge",
                profiles=[{"profile_id": "ridge-half", "tau_fraction": 0.5}],
            )
            self.assertTrue(loaded_fits.empty)
            self.assertTrue(loaded_operators.empty)
            self.assertFalse(completed)

            fits.to_csv(fit_path, index=False)
            operators.to_csv(operator_path, index=False)
            changed_contexts = contexts.copy()
            changed_contexts["analysis_settings_sha256"] = "changed-ww-settings"
            loaded_fits, loaded_operators, completed = namespace[
                "load_resumable_tables"
            ](
                fit_path=fit_path,
                secondary_path=operator_path,
                contexts=changed_contexts,
                method="gram_ridge",
                profiles=[{"profile_id": "ridge-half", "tau_fraction": 0.5}],
            )
            self.assertTrue(loaded_fits.empty)
            self.assertTrue(loaded_operators.empty)
            self.assertFalse(completed)

    def test_transformed_matrix_dual_fit_and_resume_integration(self) -> None:
        namespace = self.helper_namespace()

        class Parameter:
            dtype = "float32"

        class Model:
            def __init__(self, matrices):
                self.matrices = {
                    key: np.asarray(value, dtype=np.float32).copy()
                    for key, value in matrices.items()
                }

            def named_parameters(self):
                return [(key, Parameter()) for key in self.matrices]

        def model_esd(model, layer):
            singular = np.linalg.svd(model.matrices[layer], compute_uv=False)
            values = singular**2
            return np.sort(values[np.isfinite(values) & (values > 0.0)])

        analysis_calls = {"count": 0}

        def fake_analyze(model, **_kwargs):
            analysis_calls["count"] += 1
            esds = {
                layer: model_esd(model, layer)
                for layer in namespace["EXPECTED_WEIGHT_LAYERS"]
            }
            details = pd.DataFrame(
                [
                    {
                        "layer": layer,
                        "fit_variant": variant,
                        "fit_ok": True,
                        "alpha": 2.2,
                        "ks_D": 0.1,
                        "n_tail": len(esds[layer]),
                        "detX_num": len(esds[layer]),
                        "tail_decades": 1.0,
                    }
                    for layer in namespace["EXPECTED_WEIGHT_LAYERS"]
                    for variant in ("raw", "clip_xmax")
                ]
            )
            return SimpleNamespace(details=details, esds=esds)

        def fake_validate(*_args, **_kwargs):
            return SimpleNamespace(
                structural_errors=(),
                primary_fit_failures=(),
                raw_audit_warnings=(),
            )

        rng = np.random.default_rng(7)
        matrices = {
            "fc1.weight": rng.normal(size=(12, 16)),
            "fc2.weight": rng.normal(size=(12, 12)),
            "fc3.weight": rng.normal(size=(10, 12)),
        }
        namespace.update(
            {
                "analyze_weightwatcher_dual": fake_analyze,
                "validate_weightwatcher_measurement": fake_validate,
                "WEIGHT_QUOTIENT_ANALYSIS_CODE_SHA256": "current-code",
                "WW_MIN_EVALS": 2,
                "WW_MAX_EVALS": None,
                "WW_MAX_FINGERS": 10,
                "WW_SVD_METHOD": "accurate",
                "METHOD_PROFILE_IDS": {},
                "METHOD_FIT_FRAMES": {},
                "METHOD_OPERATOR_FRAMES": {},
                "checkpoint_model": lambda _path: Model(matrices),
                "matrix_from_model": (
                    lambda model, layer: model.matrices[layer].astype(float)
                ),
                "replace_model_matrix": (
                    lambda model, layer, matrix: model.matrices.__setitem__(
                        layer, np.asarray(matrix, dtype=np.float32)
                    )
                ),
                "model_layer_esd": model_esd,
            }
        )
        context = {
            "optimizer": "muon",
            "seed": 1337,
            "epoch": 1000,
            "global_step": 430_000,
            "checkpoint_path": "synthetic",
            "protocol_fingerprint": "run-fingerprint",
            "checkpoint_index": 0,
            "checkpoint_count": 1,
            "is_anchor": True,
            "is_final": True,
            "source_artifact_kind": "verified_final_100_tail_cache",
            "analysis_code_sha256": "current-code",
            "analysis_settings_sha256": namespace[
                "WEIGHT_QUOTIENT_ANALYSIS_SETTINGS_SHA256"
            ],
            "analysis_settings_json": namespace[
                "WEIGHT_QUOTIENT_ANALYSIS_SETTINGS_JSON"
            ],
        }
        contexts = pd.DataFrame([context])
        profile = [{"profile_id": "ridge-half", "tau_fraction": 0.5}]
        with tempfile.TemporaryDirectory() as directory:
            namespace["QUOTIENT_ANALYSIS_DIR"] = Path(directory)
            raw_fits, computed_midpoints = namespace[
                "run_raw_weightwatcher_controls"
            ](contexts)
            self.assertEqual(len(raw_fits), 6)
            self.assertEqual(len(computed_midpoints), 3)
            self.assertEqual(analysis_calls["count"], 1)
            resumed_raw, resumed_midpoints = namespace[
                "run_raw_weightwatcher_controls"
            ](contexts)
            self.assertEqual(len(resumed_raw), 6)
            self.assertEqual(len(resumed_midpoints), 3)
            self.assertEqual(analysis_calls["count"], 1)

            fits, operators = namespace["run_quotient_method"](
                "gram_ridge",
                profile,
                contexts=contexts,
                midpoints=computed_midpoints,
            )
            self.assertEqual(len(fits), 6)
            self.assertEqual(len(operators), 3)
            self.assertEqual(set(operators["layer"]), set(namespace["EXPECTED_WEIGHT_LAYERS"]))
            self.assertEqual(analysis_calls["count"], 2)

            namespace["METHOD_PROFILE_IDS"] = {}
            namespace["METHOD_FIT_FRAMES"] = {}
            namespace["METHOD_OPERATOR_FRAMES"] = {}
            resumed_fits, resumed_operators = namespace["run_quotient_method"](
                "gram_ridge",
                profile,
                contexts=contexts,
                midpoints=computed_midpoints,
            )
            self.assertEqual(len(resumed_fits), 6)
            self.assertEqual(len(resumed_operators), 3)
            self.assertEqual(analysis_calls["count"], 2)


if __name__ == "__main__":
    unittest.main()
