from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PACKAGE_ROOT.parent
EXPERIMENT_ROOT = (
    BASELINE_ROOT
    / "experiments"
    / "nanogpt_one_head_2026_08_21_baseline"
)
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "baseline.yaml"
CAMPAIGN_PATH = EXPERIMENT_ROOT / "campaign.yaml"
RUNNER_PATH = EXPERIMENT_ROOT / "scripts" / "run_experiment.py"
REPORT_PATH = EXPERIMENT_ROOT / "scripts" / "build_report.py"
NOTEBOOK_PATH = (
    EXPERIMENT_ROOT
    / "notebooks"
    / "01_Performance_and_Spectra.ipynb"
)
DOCTOR_SMOKE_PATH = (
    PACKAGE_ROOT / "src" / "rg_nanogpt_one_head" / "doctor_smoke.py"
)

sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from rg_nanogpt_one_head.config import (  # noqa: E402
    epoch_step_map,
    max_steps,
    optimizer_profile,
    tokens_per_step,
)
from rg_nanogpt_one_head.model import GPT, GPTConfig  # noqa: E402
from rg_nanogpt_one_head.optimizers import make_optimizer_handles  # noqa: E402
from rg_nanogpt_one_head.spectral import run_weightwatcher  # noqa: E402


EXPECTED_ARMS = ["adamw", "muon_clip"]
EXPECTED_SEEDS = [1337, 2027, 4099, 31415, 271828]


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "campaign_run_experiment_20260821",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_report() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "campaign_build_report_20260821",
        REPORT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _campaign_document() -> dict:
    document = yaml.safe_load(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _baseline_config_document() -> dict:
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_dated_campaign_has_the_frozen_two_by_five_design_and_grid():
    runner = _load_runner()
    report = _load_report()
    assert list(runner.CANONICAL_OPTIMIZERS) == EXPECTED_ARMS
    assert list(runner.CANONICAL_SEEDS) == EXPECTED_SEEDS
    assert runner.EXPECTED_REPLICATES == 10
    assert list(report.OPTIMIZERS) == EXPECTED_ARMS
    assert list(report.SEEDS) == EXPECTED_SEEDS

    campaign_document = _campaign_document()
    campaign = campaign_document["campaign"]
    assert campaign["id"] == "nanogpt_one_head_2026_08_21_baseline_v3"
    assert str(campaign["prepared_date"]) == "2026-08-22"
    assert campaign["config"] == "configs/baseline.yaml"
    assert campaign["optimizers"] == EXPECTED_ARMS
    assert campaign["seeds"] == EXPECTED_SEEDS
    assert campaign["require_complete_replicates"] == 10
    assert campaign["require_clean_git"] is True
    assert campaign["require_tmp_root"] is True
    assert campaign["primary_checkpoint_policy"] == (
        "minimum_validation_probe_nll"
    )
    assert campaign["protected_test_policy"] == (
        "held_out_posthoc_never_selects"
    )

    cfg = _baseline_config_document()
    assert cfg["training"]["seeds"] == EXPECTED_SEEDS
    assert cfg["dataset"] == {
        "name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "revision": "593b3a867298afb8ce42625a270ef20ddcad28f9",
        "tokenizer": "gpt2",
        "train_tokens": 80_000_000,
        "val_tokens": 1_000_000,
        "test_tokens": 1_000_000,
    }
    assert cfg["model"] == {
        "vocab_size": 50_257,
        "block_size": 256,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 128,
        "dropout": 0.0,
        "bias": False,
        "tie_weights": True,
    }
    assert cfg["training"] == {
        "seeds": EXPECTED_SEEDS,
        "batch_size": 4,
        "grad_accum_steps": 8,
        "target_epochs": 4.0,
        "epoch_interval": 0.25,
        "eval_interval_steps": 500,
        "eval_batches": 64,
        "checkpoint_interval_steps": 500,
        "grad_clip": 1.0,
    }

    expected_profile_fields = {
        "adamw": {
            "family": "adamw",
            "learning_rate": 0.0006,
            "min_learning_rate": 0.00006,
            "warmup_fraction": 0.01,
            "lr_schedule_epochs": 1.0,
            "weight_decay": 0.10,
        },
        "muon_clip": {
            "family": "muon_clip",
            "learning_rate": 0.0002,
            "min_learning_rate": 0.00002,
            "warmup_fraction": 0.0512,
            "lr_schedule_epochs": 1.0,
            "newton_schulz_steps": 5,
            "update_rms_scale": 0.20,
            "qk_clip_threshold": 100.0,
        },
    }
    profiles = cfg["optimizer_profiles"]
    for arm, expected in expected_profile_fields.items():
        assert arm in profiles
        assert {key: profiles[arm][key] for key in expected} == expected

    assert tokens_per_step(cfg) == 8_192
    assert max_steps(cfg) == 39_063
    permanent_grid = epoch_step_map(cfg)
    assert len(permanent_grid) == 17
    assert len(permanent_grid) >= 10
    assert list(permanent_grid) == sorted(set(permanent_grid))
    assert next(iter(permanent_grid)) == 0
    assert next(reversed(permanent_grid)) == max_steps(cfg)
    assert list(permanent_grid.values()) == pytest.approx(
        [index * 0.25 for index in range(17)]
    )

    assert cfg["weightwatcher"] == {
        "enabled": True,
        "ERG": True,
        "randomize": True,
        "strict": True,
        "min_evals": 20,
        "fix_fingers": "clip_xmax",
        "max_fingers": 10,
        "require_raw_alpha": True,
    }

    statistics = report._stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert statistics["n"] == 5
    assert statistics["mean"] == pytest.approx(3.0)
    assert statistics["ci95_half_width"] == pytest.approx(
        report.T_975_DF4 / np.sqrt(5) * np.std([1, 2, 3, 4, 5], ddof=1)
    )


def test_dated_campaign_contains_reproduction_and_report_entrypoints():
    required = (
        EXPERIMENT_ROOT / "README.md",
        EXPERIMENT_ROOT / "RESULTS.md",
        CAMPAIGN_PATH,
        CONFIG_PATH,
        RUNNER_PATH,
        REPORT_PATH,
        NOTEBOOK_PATH,
        DOCTOR_SMOKE_PATH,
    )
    assert all(path.is_file() for path in required)
    assert "--require-complete" in REPORT_PATH.read_text(encoding="utf-8")
    assert "erg_gap_num_traps" in REPORT_PATH.read_text(encoding="utf-8")
    assert "erg_gap_num_traps" in RUNNER_PATH.read_text(encoding="utf-8")


def test_adam_and_adamw_are_distinct_optimizer_arms():
    cfg = _baseline_config_document()
    model_config = GPTConfig(
        vocab_size=64,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    adam_model = GPT(model_config)
    adamw_model = GPT(model_config)
    adam_handles = make_optimizer_handles(
        adam_model,
        optimizer_profile(cfg, "adam"),
    )
    adamw_handles = make_optimizer_handles(
        adamw_model,
        optimizer_profile(cfg, "adamw"),
    )
    assert len(adam_handles) == len(adamw_handles) == 1
    assert type(adam_handles[0].optimizer) is torch.optim.Adam
    assert type(adamw_handles[0].optimizer) is torch.optim.AdamW
    assert all(
        float(group["weight_decay"]) == 0.0
        for group in adam_handles[0].optimizer.param_groups
    )
    assert any(
        float(group["weight_decay"]) == pytest.approx(0.1)
        for group in adamw_handles[0].optimizer.param_groups
    )


def test_campaign_root_is_explicit_resolved_and_strictly_below_tmp():
    runner = _load_runner()
    resolve_experiment_root = runner.resolve_experiment_root

    with tempfile.TemporaryDirectory(
        prefix="rg-campaign-root-test-",
        dir="/tmp",
    ) as temporary:
        temporary_root = Path(temporary)
        requested = temporary_root / "campaign-output"
        assert resolve_experiment_root(
            {"RG_NANOGPT_EXPERIMENT_ROOT": str(requested)}
        ) == requested.resolve()
        assert not requested.exists(), "path validation must not create output"

        escape = temporary_root / "escape"
        escape.symlink_to(Path("/"), target_is_directory=True)
        with pytest.raises(ValueError):
            resolve_experiment_root(
                {"RG_NANOGPT_EXPERIMENT_ROOT": str(escape / "campaign")}
            )

    invalid_environments = (
        {},
        {"RG_NANOGPT_EXPERIMENT_ROOT": ""},
        {"RG_NANOGPT_EXPERIMENT_ROOT": "relative/output"},
        {"RG_NANOGPT_EXPERIMENT_ROOT": "~/campaign-output"},
        {"RG_NANOGPT_EXPERIMENT_ROOT": "/tmp"},
        {"RG_NANOGPT_EXPERIMENT_ROOT": "/var/tmp/campaign-output"},
        {
            "RG_NANOGPT_EXPERIMENT_ROOT": "/tmp/pretend-home/campaign",
            "HOME": "/tmp/pretend-home",
        },
    )
    for environment in invalid_environments:
        with pytest.raises(ValueError):
            resolve_experiment_root(environment)


def test_weightwatcher_uses_one_clip_xmax_call_for_both_alphas(
    tmp_path,
    monkeypatch,
):
    analyze_calls: list[dict] = []

    class FakeWeightWatcher:
        def __init__(self, *, model):
            self.model = model

        def analyze(self, **kwargs):
            analyze_calls.append(dict(kwargs))
            names = [
                str(item["matrix_name"])
                for item in self.model.matrix_metadata
            ]
            count = len(names)
            return pd.DataFrame(
                {
                    "layer_id": range(1, count + 1),
                    "name": [f"holder.{name}" for name in names],
                    "alpha": [2.0 + index / 10 for index in range(count)],
                    "raw_alpha": [
                        2.5 + index / 10 for index in range(count)
                    ],
                    "num_fingers": [2] * count,
                    "ERG_gap": [0.25] * count,
                    "num_traps": [1] * count,
                    "rand_distance": [0.10] * count,
                }
            )

    monkeypatch.setitem(
        sys.modules,
        "weightwatcher",
        SimpleNamespace(WeightWatcher=FakeWeightWatcher),
    )
    model = GPT(
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
        )
    )
    ww_config = _baseline_config_document()["weightwatcher"]
    summary = run_weightwatcher(
        model,
        tmp_path,
        step=7,
        tokens_seen=4_096,
        train_tokens=8_192,
        config=ww_config,
        seed=1337,
        fingerprint="unit-weightwatcher-fingerprint",
    )

    assert analyze_calls == [
        {
            "ERG": True,
            "randomize": True,
            "plot": False,
            "min_evals": 20,
            "fix_fingers": "clip_xmax",
            "max_fingers": 10,
        }
    ]
    assert summary["n_matrices"] == 6
    assert summary["alpha_raw_n"] == 6
    assert summary["alpha_clip_xmax_n"] == 6

    raw = pd.read_csv(
        tmp_path
        / "spectral"
        / "raw"
        / "weightwatcher_step_0000007.csv"
    )
    layers = pd.read_csv(tmp_path / "spectral" / "layers.csv")
    for frame in (raw, layers):
        assert len(frame) == 6
        assert frame["alpha"].equals(frame["alpha_clip_xmax"])
        assert frame["raw_alpha"].equals(frame["alpha_raw"])
        assert frame["alpha_delta"].tolist() == pytest.approx(
            (
                frame["alpha_raw"] - frame["alpha_clip_xmax"]
            ).tolist()
        )
        assert (frame["weightwatcher_analysis_calls"] == 1).all()
        assert set(frame["primary_alpha_variant"]) == {"clip_xmax"}


def test_source_notebook_is_valid_report_only_papermill_document():
    raw_document = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook = nbformat.from_dict(raw_document)
    nbformat.validate(notebook)

    parameter_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(parameter_cells) == 1
    parameter_source = parameter_cells[0].source
    parameter_tree = ast.parse(
        "".join(parameter_source)
        if isinstance(parameter_source, list)
        else parameter_source
    )
    parameters: dict[str, object] = {}
    for statement in parameter_tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Constant)
        ):
            parameters[statement.targets[0].id] = statement.value.value
    assert parameters == {
        "RESULTS_ROOT": "",
        "OUTPUT_ROOT": "",
        "REQUIRE_COMPLETE": True,
    }

    trees = [
        ast.parse(
            "".join(cell.source)
            if isinstance(cell.source, list)
            else cell.source
        )
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]
    subprocess_calls = [
        node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert len(subprocess_calls) == 1
    keywords = {
        keyword.arg: keyword.value
        for keyword in subprocess_calls[0].keywords
        if keyword.arg is not None
    }
    assert isinstance(keywords.get("check"), ast.Constant)
    assert keywords["check"].value is True
    assert "shell" not in keywords

    source = "\n".join(
        "".join(cell.source)
        if isinstance(cell.source, list)
        else cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    for required_text in (
        "sys.executable",
        "build_report.py",
        "--results-root",
        "--output-root",
        "--require-complete",
        "--allow-incomplete",
        "SUMMARY.md",
        "*.csv",
        "*.png",
    ):
        assert required_text in source
    for forbidden_training_entrypoint in (
        "rg-onehead-train",
        "run_one(",
        "run_optimizer_replicates",
        "rg_nanogpt_one_head.training",
    ):
        assert forbidden_training_entrypoint not in source


def test_dated_launch_materials_do_not_expand_home_or_tilde_paths():
    launch_suffixes = {".md", ".py", ".sh", ".yaml", ".yml", ".ipynb"}
    forbidden_patterns = {
        "shell home expansion": re.compile(r"\$(?:\{HOME\}|HOME\b)"),
        "tilde path": re.compile(r"(?<![A-Za-z0-9_])~[/\\]"),
        "hard-coded home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    }
    violations: list[str] = []
    for path in sorted(EXPERIMENT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in launch_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in forbidden_patterns.items():
            if pattern.search(text):
                violations.append(
                    f"{path.relative_to(EXPERIMENT_ROOT)}: {label}"
                )
    assert not violations, "\n".join(violations)


def test_frozen_config_gate_rejects_any_hyperparameter_change(tmp_path):
    runner = _load_runner()
    cfg = runner._validate_protocol_config(CONFIG_PATH)
    assert runner._canonical_sha256(cfg) == runner.FROZEN_CONFIG_SHA256

    mutated = copy.deepcopy(cfg)
    mutated["optimizer_profiles"]["adamw"]["learning_rate"] = 0.0007
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
    with pytest.raises(runner.CampaignError, match="frozen dated campaign"):
        runner._validate_protocol_config(path)


def test_launcher_redirects_all_xdg_and_live_monitor_state_below_tmp(tmp_path):
    runner = _load_runner()
    root = (tmp_path / "campaign").resolve()
    paths = runner._paths(root)
    runner._create_runtime_directories(paths)
    child = runner._child_environment(root, paths)
    for name in (
        "HOME",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "MPLCONFIGDIR",
        "TMPDIR",
    ):
        assert Path(child[name]).resolve().is_relative_to(root)

    parser = runner._build_parser()
    monitor = parser.parse_args(
        ["monitor", "--optimizer", "muon_clip", "--seed", "4099", "--once"]
    )
    assert monitor.handler is runner._command_monitor

    provisional = parser.parse_args(["analyze", "--allow-incomplete"])
    assert provisional.handler is runner._command_analyze
    assert provisional.allow_incomplete is True


def test_supported_subprocess_lock_rejects_a_duplicate_writer(tmp_path):
    runner = _load_runner()
    lock_path = tmp_path / "adamw_seed_1337.log.lock"
    first = runner._acquire_exclusive_lock(lock_path)
    try:
        with pytest.raises(runner.CampaignError, match="another campaign process"):
            runner._acquire_exclusive_lock(lock_path)
    finally:
        runner._release_exclusive_lock(first)


def test_doctor_invokes_the_full_backend_smoke_gate():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for required in (
        "doctor_smoke.py",
        "doctor_backend_smoke.json",
        "_require_doctor_gate",
        "checkpoint_roundtrip",
        "weightwatcher_analysis_calls",
        "raw_alpha_count",
        "clipped_alpha_count",
    ):
        assert required in source


def test_production_run_requires_a_bound_doctor_gate(tmp_path):
    runner = _load_runner()
    root = (tmp_path / "campaign").resolve()
    paths = runner._paths(root)
    runner._create_runtime_directories(paths)
    runtime = {
        "accelerator": "cuda",
        "device": "cuda",
        "hardware_block_id": "h100-sxm-80gb-cluster-a",
        "hardware_block_id_source": "user",
        "torch_version": "2.8.0",
        "cuda_driver_version": "575.57",
        "cuda_device_name": "NVIDIA H100 80GB HBM3",
        "cuda_device_capability": [9, 0],
        "cuda_device_uuid": "GPU-doctor",
        "python_executable": "/opt/doctor/python",
        "processor": "doctor-host",
    }
    dependencies = {"python": "3.12.0", "torch": "2.8.0"}
    git = {
        "available": True,
        "clean": True,
        "dirty": False,
        "commit": "a" * 40,
        "origin_url": "https://example.invalid/rg_optimizers.git",
    }
    backend_path = paths["provenance"] / "doctor_backend_smoke.json"
    backend = {
        "schema_version": 1,
        "completed": True,
        "runtime": runtime,
        "optimizers": [
            {
                "optimizer": optimizer,
                "checkpoint_roundtrip": True,
                "resumed_optimizer_step": True,
            }
            for optimizer in EXPECTED_ARMS
        ],
        "weightwatcher": {
            "analysis_calls": 1,
            "matrix_count": 6,
            "raw_alpha_count": 6,
            "clipped_alpha_count": 6,
        },
    }
    runner._atomic_json(backend_path, backend)
    doctor = {
        "schema_version": 1,
        "completed": True,
        "experiment_root": str(root),
        "config_sha256": runner._sha256(CONFIG_PATH),
        "resolved_device": "cuda",
        "accelerator": "cuda",
        "runtime": runtime,
        "git": git,
        "dependencies": dependencies,
        "backend_smoke": {
            "summary_path": str(backend_path),
            "summary_sha256": runner._sha256(backend_path),
        },
    }
    runner._atomic_json(paths["provenance"] / "doctor.json", doctor)

    current_runtime = {
        **runtime,
        "cuda_device_uuid": "GPU-production",
        "python_executable": "/opt/production/python",
        "processor": "production-host",
    }
    assert runner._require_doctor_gate(
        paths,
        config=CONFIG_PATH,
        git=git,
        dependencies=dependencies,
        runtime=current_runtime,
    )["completed"] is True

    changed_runtime = {**current_runtime, "cuda_driver_version": "580.00"}
    with pytest.raises(runner.CampaignError, match="runtime/hardware block"):
        runner._require_doctor_gate(
            paths,
            config=CONFIG_PATH,
            git=git,
            dependencies=dependencies,
            runtime=changed_runtime,
        )


def test_explicit_hardware_block_ignores_only_instance_fields():
    runner = _load_runner()
    first = {
        "accelerator": "cuda",
        "hardware_block_id": "h100-sxm-80gb-cluster-a",
        "hardware_block_id_source": "user",
        "torch_version": "2.8.0",
        "cuda_driver_version": "575.57",
        "cuda_device_name": "NVIDIA H100 80GB HBM3",
        "cuda_device_capability": [9, 0],
        "cuda_device_uuid": "GPU-first",
        "cuda_device_count": 8,
        "python_executable": "/opt/host-a/python",
        "processor": "host-a",
    }
    second = {
        **first,
        "cuda_device_uuid": "GPU-second",
        "cuda_device_count": 1,
        "python_executable": "/opt/host-b/python",
        "processor": "host-b",
    }
    assert runner._runtime_block_identity(first) == runner._runtime_block_identity(
        second
    )
    second["cuda_driver_version"] = "580.00"
    assert runner._runtime_block_identity(first) != runner._runtime_block_identity(
        second
    )


def test_report_rejects_wrong_source_dependency_or_optimizer_profile():
    report = _load_report()
    cfg = _baseline_config_document()
    commit = subprocess.run(
        ["git", "-C", str(BASELINE_ROOT.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    packages = {name: "test-version" for name in report.MANIFEST_PACKAGES}
    packages["weightwatcher"] = report.PINNED_WEIGHTWATCHER
    packages["rg-nanogpt-one-head"] = report.PINNED_PACKAGE_VERSION
    profile = dict(cfg["optimizer_profiles"]["adamw"])
    profile["name"] = "adamw"
    manifest = {
        "optimizer": "adamw",
        "seed": 1337,
        "protocol": cfg["protocol"],
        "config_sha256": report.FROZEN_CONFIG_SHA256,
        "initial_model_sha256": "1" * 64,
        "model": cfg["model"],
        "training": cfg["training"],
        "evaluation": cfg["evaluation"],
        "weightwatcher": cfg["weightwatcher"],
        "optimizer_profile": profile,
        "tokens_per_step": 8192,
        "max_steps": report.EXPECTED_TOTAL_STEPS,
        "package_versions": packages,
        "runtime_environment": {
            "accelerator": "cpu",
            "float32_matmul_precision": "highest",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "hardware_block_id": "test-cpu-block",
            "hardware_block_id_source": "test",
        },
        "source_repository": {
            "available": True,
            "dirty": False,
            "commit": commit,
        },
        "data_metadata": {
            "schema_version": 2,
            "dataset_name": cfg["dataset"]["name"],
            "dataset_config": cfg["dataset"]["config"],
            "dataset_split": cfg["dataset"]["split"],
            "dataset_revision": cfg["dataset"]["revision"],
            "tokenizer": cfg["dataset"]["tokenizer"],
            "vocab_size": cfg["model"]["vocab_size"],
            "eot_token": 50_256,
            "dtype": "uint16",
            "document_disjoint_splits": True,
            "splits": {
                "train": cfg["dataset"]["train_tokens"],
                "val": cfg["dataset"]["val_tokens"],
                "test": cfg["dataset"]["test_tokens"],
            },
            "files": {
                split: {
                    "path": f"{split}.bin",
                    "sha256": "a" * 64,
                    "bytes": tokens * 2,
                }
                for split, tokens in {
                    "train": cfg["dataset"]["train_tokens"],
                    "val": cfg["dataset"]["val_tokens"],
                    "test": cfg["dataset"]["test_tokens"],
                }.items()
            },
        },
    }
    report._validate_matched_campaign([manifest], allow_mixed_runtime=False)
    for mutation in ("source", "dependency", "profile"):
        changed = copy.deepcopy(manifest)
        if mutation == "source":
            changed["source_repository"]["commit"] = "0" * 40
        elif mutation == "dependency":
            changed["package_versions"]["weightwatcher"] = "0.0.0"
        else:
            changed["optimizer_profile"]["learning_rate"] = 9.0
        with pytest.raises(report.CampaignValidationError):
            report._validate_matched_campaign([changed], allow_mixed_runtime=True)


def test_archive_inventory_detects_stale_artifact(tmp_path):
    runner = _load_runner()
    root = tmp_path.resolve()
    artifact = root / "report.html"
    artifact.write_text("fresh", encoding="utf-8")
    record = {
        "path": "report.html",
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    assert runner._validated_artifact_inventory(
        [record], root=root, label="artifacts"
    ) == {"report.html": artifact}
    artifact.write_text("stale", encoding="utf-8")
    with pytest.raises(runner.CampaignError, match="stale"):
        runner._validated_artifact_inventory([record], root=root, label="artifacts")


def test_only_dated_archive_records_are_git_trackable():
    repository = BASELINE_ROOT.parent
    readme = (
        "baseline/experiments/nanogpt_one_head_2026_08_21_baseline/"
        "runs/README.md"
    )
    assert subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--quiet", readme]
    ).returncode == 1
    raw_checkpoint = (
        "baseline/experiments/nanogpt_one_head_2026_08_21_baseline/"
        "runs/20260821T000000Z_deadbeef/checkpoint_final.pt"
    )
    assert subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--quiet", raw_checkpoint]
    ).returncode == 0


def test_generated_run_record_reproduction_installs_and_redirects_caches():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for required in (
        "-m pip install --no-deps -e './baseline/nanogpt_one_head'",
        "requirements_replay.txt",
        "dependency_lock.json",
        "verify-lock --lock",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "PIP_CACHE_DIR",
        "run_experiment.py archive",
    ):
        assert required in source


def test_dependency_lock_rejects_nonportable_direct_origin(monkeypatch):
    runner = _load_runner()

    class DirectDistribution:
        metadata = {"Name": "torch"}
        version = "2.8.0"
        requires: list[str] = []

        @staticmethod
        def read_text(filename: str):
            assert filename == "direct_url.json"
            return json.dumps({"url": "file:///tmp/custom-torch.whl"})

    monkeypatch.setattr(
        runner.importlib.metadata,
        "distribution",
        lambda name: DirectDistribution(),
    )
    with pytest.raises(runner.CampaignError, match="direct origin"):
        runner._installed_distribution_lock(
            {"python": sys.version.split()[0], "torch": "2.8.0"}
        )


def test_dependency_lock_includes_installed_transitive_closure(monkeypatch):
    runner = _load_runner()

    class Distribution:
        def __init__(self, name: str, version: str, requires: list[str]):
            self.metadata = {"Name": name}
            self.version = version
            self.requires = requires

        @staticmethod
        def read_text(filename: str):
            assert filename == "direct_url.json"
            return None

    distributions = {
        "torch": Distribution("torch", "2.8.0", ["numpy>=1.24"]),
        "numpy": Distribution("numpy", "2.1.0", []),
    }
    monkeypatch.setattr(
        runner.importlib.metadata,
        "distribution",
        lambda name: distributions[str(name).lower()],
    )
    lock, requirements = runner._installed_distribution_lock(
        {"python": sys.version.split()[0], "torch": "2.8.0"}
    )
    assert lock["schema_version"] == 2
    assert set(lock["packages"]) == {"torch", "numpy"}
    assert requirements.splitlines() == ["numpy==2.1.0", "torch==2.8.0"]


def test_dependency_contract_returns_the_manifest_closure(monkeypatch):
    runner = _load_runner()
    direct = {name: "test-version" for name in runner.DEPENDENCIES}
    direct["rg-nanogpt-one-head"] = runner.PINNED_PACKAGE_VERSION
    direct["weightwatcher"] = runner.PINNED_WEIGHTWATCHER
    monkeypatch.setattr(runner, "_dependency_versions", lambda: direct)
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda name: "not-installed" if name == "torch-xla" else "test-version",
    )
    monkeypatch.setattr(
        runner,
        "_installed_distribution_lock",
        lambda scientific: (
            {
                "schema_version": 2,
                "packages": {
                    "transitive-lib": {
                        "name": "transitive-lib",
                        "version": "9.8.7",
                    }
                },
                "scientific_packages": dict(scientific),
            },
            "transitive-lib==9.8.7\n",
        ),
    )
    contract = runner._require_dependency_contract()
    assert contract["transitive-lib"] == "9.8.7"
