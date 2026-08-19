#!/usr/bin/env python3
"""Build the versioned MNIST tangent-RG experiment notebooks.

The notebooks are generated from auditable Python strings so common protocol,
artifact, uncertainty, and provenance cells cannot silently drift between
methods.  This script needs only the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = EXPERIMENT_ROOT / "notebooks"


def _source(value: str) -> list[str]:
    text = dedent(value).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(value: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def code(value: str, *, tags: tuple[str, ...] = ()) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if tags:
        metadata["tags"] = list(tags)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": _source(value),
    }


def parameters(extra: str = "") -> dict[str, object]:
    base = dedent(
        """
        # Papermill parameters. Override these values in an injected cell.
        RUN_ROOT = ""
        OUTPUT_ROOT = ""
        CHECKPOINT_CACHE_ROOT = ""
        CONFIG_PATH = ""
        PROFILE = "pilot_1000_epochs"
        PROTOCOL_SLUG = ""
        SEEDS = [1337, 2027, 31415]
        CHECKPOINT_PAYLOAD_CACHE_SIZE = 24
        SHOW_PLOTS = True
        REQUIRE_ARTIFACTS = True
        ALLOW_TEMPORARY_LONG_RUN = False
        """
    ).strip()
    extension = dedent(extra).strip()
    source = base if not extension else base + "\n" + extension
    return code(source, tags=("parameters",))


BOOTSTRAP = r"""
from pathlib import Path
from dataclasses import asdict, is_dataclass
from functools import lru_cache
import inspect
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cwd = Path.cwd().resolve()
REPO_ROOT = next(
    (
        candidate
        for candidate in (cwd, *cwd.parents)
        if (candidate / "baseline" / "rg_baselines").is_dir()
    ),
    None,
)
if REPO_ROOT is None:
    raise FileNotFoundError(
        "Could not find baseline/rg_baselines. Launch Jupyter from a clone of "
        "CalculatedContent/rg_optimizers."
    )
BASELINE_ROOT = REPO_ROOT / "baseline"
EXPERIMENT_ROOT = BASELINE_ROOT / "experiments" / "mnist_mlp3_tangent_rg"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

default_root = os.environ.get(
    "RG_MNIST_TANGENT_ROOT", "/tmp/rg-mnist-mlp3-tangent-rg"
)
RUN_ROOT_PATH = Path(RUN_ROOT or default_root).expanduser().resolve()

default_checkpoint_cache_root = os.environ.get(
    "RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT",
    "/tmp/rg-mnist-mlp3-tangent-checkpoints",
)
CHECKPOINT_CACHE_ROOT_PATH = Path(
    CHECKPOINT_CACHE_ROOT or default_checkpoint_cache_root
).expanduser().resolve()

def _suite_name_from_profile():
    if str(PROTOCOL_SLUG).strip():
        return str(PROTOCOL_SLUG).strip()
    candidate = (
        Path(CONFIG_PATH).expanduser()
        if str(CONFIG_PATH).strip()
        else EXPERIMENT_ROOT / "configs" / f"{PROFILE}.yaml"
    )
    if candidate.is_file():
        if candidate.suffix.lower() == ".json":
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            value = payload.get("protocol", {}).get("suite_name")
            if value:
                return str(value)
        else:
            for line in candidate.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("suite_name:"):
                    return stripped.split(":", 1)[1].strip().strip("'\"")
    fallback = {
        "smoke": "mnist_mlp3_tangent_rg_v1_smoke",
        "pilot_1000_epochs": "mnist_mlp3_tangent_rg_v1_pilot1000",
        "long_horizon_10000_epochs": "mnist_mlp3_tangent_rg_v1_reference10000",
    }
    if PROFILE not in fallback:
        raise FileNotFoundError(
            f"Cannot derive suite_name for PROFILE={PROFILE!r}; set CONFIG_PATH "
            "or PROTOCOL_SLUG explicitly."
        )
    return fallback[PROFILE]

PROTOCOL_SLUG = _suite_name_from_profile()
OUTPUT_ROOT_PATH = Path(
    OUTPUT_ROOT or RUN_ROOT_PATH / PROTOCOL_SLUG / "notebook_outputs"
).expanduser().resolve()
OUTPUT_ROOT_PATH.mkdir(parents=True, exist_ok=True)

SEEDS = tuple(int(seed) for seed in SEEDS)
if SEEDS != (1337, 2027, 31415):
    print("WARNING: this is not the preregistered three-seed tuple:", SEEDS)

print("repository:", REPO_ROOT)
print("run root:", RUN_ROOT_PATH)
print("tail checkpoint cache root:", CHECKPOINT_CACHE_ROOT_PATH)
print("effective suite:", PROTOCOL_SLUG)
print("output root:", OUTPUT_ROOT_PATH)
print("seeds:", SEEDS)
"""


COMMON_IMPORTS = r"""
from rg_baselines.statistics import summarize_numeric_metrics
from rg_baselines.tangent_rg import powerlaw_fit, trace_log
"""


COMMON_HELPERS = r"""
def require_path(path, *, description="artifact"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Run the prerequisite numbered notebook or set RUN_ROOT / "
            "OUTPUT_ROOT to the completed protocol directory."
        )
    return path


def first_existing(directory, names, *, description):
    directory = Path(directory)
    candidates = [directory / name for name in names]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing {description} beneath {directory}. Expected one of:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def resolve_protocol_root():
    direct = RUN_ROOT_PATH / PROTOCOL_SLUG
    return direct if direct.is_dir() else RUN_ROOT_PATH


def resolve_arm_dir(optimizer_slug):
    protocol = resolve_protocol_root()
    candidates = [
        protocol / optimizer_slug,
        protocol / "results" / optimizer_slug,
        RUN_ROOT_PATH / optimizer_slug,
        RUN_ROOT_PATH / "results" / optimizer_slug,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    if REQUIRE_ARTIFACTS:
        raise FileNotFoundError(
            f"No completed {optimizer_slug!r} arm was found. Checked:\n"
            + "\n".join(f"  - {path}" for path in candidates)
        )
    return candidates[0]


def resolve_seed_dir(optimizer_slug, seed):
    arm = resolve_arm_dir(optimizer_slug)
    candidates = [
        arm / f"seed_{int(seed)}",
        arm / f"seed_{int(seed):05d}",
        arm / "seeds" / f"seed_{int(seed)}",
        arm / "seeds" / f"seed_{int(seed):05d}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Missing seed directory for optimizer={optimizer_slug}, seed={seed}. "
        f"Checked {candidates}."
    )


def validate_run_identity(seed_dir, *, optimizer_slug, seed):
    seed_dir = Path(seed_dir)
    manifest = json.loads(
        require_path(seed_dir / "manifest.json", description="run manifest")
        .read_text(encoding="utf-8")
    )
    resolved = json.loads(
        require_path(seed_dir / "resolved_config.json", description="resolved config")
        .read_text(encoding="utf-8")
    )
    completion = json.loads(
        require_path(seed_dir / "run_complete.json", description="completion marker")
        .read_text(encoding="utf-8")
    )
    config = dict(resolved.get("config", resolved))
    checks = {
        "manifest suite": (manifest.get("suite_name"), PROTOCOL_SLUG),
        "resolved suite": (config.get("suite_name"), PROTOCOL_SLUG),
        "manifest optimizer": (manifest.get("optimizer"), optimizer_slug),
        "resolved optimizer": (config.get("optimizer"), optimizer_slug),
        "completion optimizer": (completion.get("optimizer"), optimizer_slug),
        "manifest seed": (manifest.get("seed"), int(seed)),
        "resolved seed": (config.get("seed"), int(seed)),
        "completion seed": (completion.get("seed"), int(seed)),
    }
    mismatches = [
        f"{label}: observed={observed!r}, expected={expected!r}"
        for label, (observed, expected) in checks.items()
        if str(observed) != str(expected)
    ]
    fingerprints = {
        str(manifest.get("protocol_fingerprint", "")),
        str(resolved.get("protocol_fingerprint", "")),
        str(completion.get("protocol_fingerprint", "")),
    }
    if "" in fingerprints or len(fingerprints) != 1:
        mismatches.append(
            "manifest/resolved/completion protocol fingerprints are missing or unequal"
        )
    if not bool(completion.get("completed", False)):
        mismatches.append("run_complete.json does not declare completed=true")
    try:
        resolved_epochs = int(config["epochs"])
        completion_epochs = int(completion["epochs"])
        completion_step = int(completion["global_step"])
        best_validation_epoch = int(completion["best_validation_epoch"])
        analysis_plan = dict(resolved["analysis_plan"])
        plan_steps_per_epoch = int(analysis_plan["steps_per_epoch"])
        plan_total_steps = int(analysis_plan["total_steps"])
    except (KeyError, TypeError, ValueError) as error:
        mismatches.append(
            "resolved/completion final-horizon metadata is missing or invalid: "
            f"{type(error).__name__}: {error}"
        )
    else:
        if resolved_epochs < 1 or plan_steps_per_epoch < 1:
            mismatches.append("resolved epochs and steps_per_epoch must be positive")
        if plan_total_steps != resolved_epochs * plan_steps_per_epoch:
            mismatches.append(
                "resolved analysis_plan total_steps does not equal "
                "epochs * steps_per_epoch"
            )
        if completion_epochs != resolved_epochs:
            mismatches.append(
                f"completion epochs={completion_epochs} != resolved epochs={resolved_epochs}"
            )
        if completion_step != plan_total_steps:
            mismatches.append(
                f"completion global_step={completion_step} != resolved "
                f"analysis_plan total_steps={plan_total_steps}"
            )
        if not 0 <= best_validation_epoch <= resolved_epochs:
            mismatches.append(
                f"best_validation_epoch={best_validation_epoch} is outside "
                f"[0, {resolved_epochs}]"
            )
    if mismatches:
        raise RuntimeError(
            f"Run identity/provenance mismatch beneath {seed_dir}:\n  - "
            + "\n  - ".join(mismatches)
        )
    return manifest, resolved, completion


def validate_cross_run_provenance(manifests):
    manifests = list(manifests)
    if not manifests:
        raise RuntimeError("No manifests supplied for cross-run provenance audit")
    invariant_fields = (
        "suite_name", "dataset", "model", "initialization", "normalization",
        "train_indices_sha256", "validation_indices_sha256",
        "test_monitoring_only", "analysis_plan", "device", "software_versions",
        "determinism_settings",
    )
    disagreements = []
    for field in invariant_fields:
        serialized = {
            json.dumps(item.get(field), sort_keys=True, default=str)
            for item in manifests
        }
        if len(serialized) != 1:
            disagreements.append(field)
    if disagreements:
        raise RuntimeError(
            "Matched arms disagree on frozen run provenance fields: "
            + ", ".join(disagreements)
        )
    identities = {
        (str(item.get("optimizer")), int(item.get("seed"))) for item in manifests
    }
    expected = {
        (str(optimizer), int(seed))
        for optimizer in OPTIMIZER_SLUGS
        for seed in SEEDS
    } if "OPTIMIZER_SLUGS" in globals() else identities
    if identities != expected:
        raise RuntimeError(
            f"Manifest optimizer/seed grid is incomplete: observed={sorted(identities)}, "
            f"expected={sorted(expected)}"
        )
    return pd.DataFrame([
        {
            "optimizer": item.get("optimizer"),
            "seed": item.get("seed"),
            "device": item.get("device"),
            "software_versions": json.dumps(
                item.get("software_versions"), sort_keys=True, default=str
            ),
            "determinism_settings": json.dumps(
                item.get("determinism_settings"), sort_keys=True, default=str
            ),
            "pooling_compatibility_policy": (
                "headline pooling requires identical device, software versions, "
                "determinism settings, and scientific invariants across all runs"
            ),
        }
        for item in manifests
    ])


def record_dict(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Cannot convert {type(value).__name__} to an audit row")


def records_from_result(result):
    if result is None:
        return []
    if is_dataclass(result):
        return [record_dict(result)]
    if isinstance(result, dict):
        if "operator_kind" in result:
            return [dict(result)]
        rows = []
        for value in result.values():
            rows.extend(records_from_result(value))
        return rows
    if isinstance(result, (tuple, list)):
        rows = []
        for value in result:
            rows.extend(records_from_result(value))
        return rows
    return [record_dict(result)]


def spectrum_from_record(row):
    for name in (
        "spectrum", "eigenvalues", "singular_values", "rates",
        "positive_spectrum", "gram_spectrum",
    ):
        if name in row:
            values = np.asarray(row[name], dtype=float).reshape(-1)
            return values[np.isfinite(values) & (values > 0.0)]
    raise KeyError(
        "Operator record contains no recognized positive spectrum field. "
        f"Available fields: {sorted(row)}"
    )


def positive_spectrum(values, *, minimum_count=2):
    sample = np.asarray(values, dtype=float).reshape(-1)
    sample = sample[np.isfinite(sample) & (sample > 0.0)]
    sample = np.sort(sample)
    if sample.size < int(minimum_count):
        raise ValueError(
            f"Need at least {minimum_count} finite positive spectral values; "
            f"found {sample.size}."
        )
    return sample


def fit_spectrum_with_trace(
    values,
    *,
    operator_kind,
    map_definition,
    spectrum_kind,
    metadata,
    top_k_values=(0, 1, 2, 3, 4, 5),
    minimum_tail=8,
):
    # Fit amplitudes once, transform that fit to energy, and audit trace-log.
    # The power-law package is never called independently on squared values.
    # Trace-log uses squared values at the amplitude fit's independent rank.

    if str(spectrum_kind) != "amplitude":
        raise ValueError(
            "fit_spectrum_with_trace accepts operator amplitudes only; "
            "energy rows are produced by the exact amplitude-to-energy transform."
        )

    sample = positive_spectrum(values)
    feasible_top_k = tuple(
        int(value) for value in top_k_values if int(value) <= sample.size - 2
    )
    if not feasible_top_k or feasible_top_k[0] != 0:
        feasible_top_k = (0,)
    amplitude_fits = powerlaw_fit.fit_clipping_sensitivity(
        sample,
        top_k_values=feasible_top_k,
        minimum_tail=int(minimum_tail),
        operator_kind=str(operator_kind),
        map_definition=str(map_definition),
        spectrum_kind="amplitude",
        metadata=dict(metadata),
    )
    energy_rows = [
        powerlaw_fit.amplitude_fit_to_energy(row)
        for row in amplitude_fits.to_dict(orient="records")
    ]
    fits = pd.concat(
        [amplitude_fits, pd.DataFrame(energy_rows)],
        ignore_index=True,
        sort=False,
    )
    primary = amplitude_fits.loc[amplitude_fits["clip_top_k"].eq(0)].iloc[0]
    energy = sample ** 2
    trace_row = {
        **dict(metadata),
        "operator_kind": str(operator_kind),
        "map_definition": str(map_definition),
        "spectrum_kind": "energy_derived_from_amplitude",
        "support_rank_source": "powerlaw.Fit package-selected xmin tail count",
        "support_selected_from_same_trace_log": False,
        "support_rank": int(primary.get("n_tail", 0)),
        "trace_log_total": np.nan,
        "trace_log_per_eval": np.nan,
        "lambda_cut_scaled": np.nan,
        "trace_status": "fit_has_no_supported_tail",
    }
    rank = int(primary.get("n_tail", 0))
    if rank > 0:
        evaluated = trace_log.trace_log_at_rank(
            energy,
            rank=min(rank, energy.size),
            normalization_dimension=float(energy.size),
            rank_source="powerlaw.Fit package-selected xmin tail count",
        )
        trace_row.update(evaluated)
        trace_row["trace_status"] = "ok"
    return fits, pd.DataFrame([trace_row])


def save_analysis_frames(method_slug, *, operators, fits, traces):
    destination = OUTPUT_ROOT_PATH / "analyses" / str(method_slug)
    destination.mkdir(parents=True, exist_ok=True)
    required_identity = {
        "optimizer", "seed", "protocol_fingerprint", "source_artifact_kind"
    }
    for label, frame in (("operators", operators), ("fits", fits), ("traces", traces)):
        missing = required_identity - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"{method_slug} {label} lack analysis provenance: {sorted(missing)}"
            )
        if frame[list(required_identity)].isna().any().any():
            raise RuntimeError(f"{method_slug} {label} contain null analysis provenance")
    identity_rows = fits[
        ["optimizer", "seed", "protocol_fingerprint", "source_artifact_kind"]
    ].drop_duplicates()
    duplicate_fingerprints = (
        identity_rows.groupby(["optimizer", "seed"], dropna=False)[
            "protocol_fingerprint"
        ].nunique()
    )
    if (duplicate_fingerprints != 1).any():
        raise RuntimeError(
            f"{method_slug} has multiple protocol fingerprints for one optimizer/seed"
        )
    fingerprint_grid = {
        f"{row.optimizer}:{int(row.seed)}": str(row.protocol_fingerprint)
        for row in identity_rows.itertuples(index=False)
    }
    expected_grid_count = identity_rows[["optimizer", "seed"]].drop_duplicates().shape[0]
    if len(fingerprint_grid) != expected_grid_count:
        raise RuntimeError(f"{method_slug} fingerprint-grid keys are not unique")
    provenance_manifest = {
        "schema_version": 1,
        "suite_name": str(PROTOCOL_SLUG),
        "method_slug": str(method_slug),
        "optimizer_seed_protocol_fingerprints": dict(sorted(fingerprint_grid.items())),
        "source_artifact_kinds": sorted(
            identity_rows["source_artifact_kind"].astype(str).unique().tolist()
        ),
        "operator_row_count": int(len(operators)),
        "fit_row_count": int(len(fits)),
        "trace_row_count": int(len(traces)),
    }
    provenance_manifest["analysis_contract_tokens"] = sorted(
        fits["analysis_contract_token"].dropna().astype(str).unique().tolist()
        if "analysis_contract_token" in fits.columns
        else []
    )
    operators.to_csv(destination / "operator_rows.csv", index=False)
    fits.to_csv(destination / "powerlaw_fits.csv", index=False)
    traces.to_csv(destination / "trace_log_independent_support.csv", index=False)
    (destination / "method_provenance.json").write_text(
        json.dumps(provenance_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def save_spectrum_ccdf_gallery(
    spectral_arrays,
    *,
    method_slug,
    maximum_panels=24,
):
    # Save bounded log-log PDF/CCDF diagnostics for positive amplitudes.
    gallery = OUTPUT_ROOT_PATH / "analyses" / str(method_slug) / "spectrum_pdf_ccdf"
    gallery.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (key, raw) in enumerate(sorted(spectral_arrays.items())):
        if index >= int(maximum_panels):
            break
        sample = positive_spectrum(raw)
        x, ccdf = powerlaw_fit.empirical_ccdf(sample)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        if sample[0] < sample[-1]:
            bins = np.geomspace(sample[0], sample[-1], min(50, max(8, sample.size // 3)))
            axes[0].hist(sample, bins=bins, density=True, histtype="step", linewidth=1.8)
        else:
            axes[0].scatter(sample, np.ones_like(sample), s=15)
        axes[1].step(x, ccdf, where="post", linewidth=1.8)
        for axis in axes:
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(alpha=0.2)
        axes[0].set(xlabel="amplitude b", ylabel="density", title="PDF")
        axes[1].set(xlabel="amplitude b", ylabel="P(B >= b)", title="CCDF")
        fig.suptitle(str(key), fontsize=8)
        fig.tight_layout()
        safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(key))
        path = gallery / f"{index:03d}_{safe[:160]}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if SHOW_PLOTS and index < 3:
            plt.show()
        else:
            plt.close(fig)
        rows.append({"spectrum_key": str(key), "n_positive": int(sample.size), "figure": str(path)})
    index_frame = pd.DataFrame(rows)
    index_frame.to_csv(gallery / "index.csv", index=False)
    return index_frame


def plot_fit_alpha_ci(fits, *, method_slug, title):
    usable = fits.copy()
    if "fit_ok" in usable:
        usable = usable[boolean_series(usable["fit_ok"])]
    if "spectrum_kind" in usable:
        energy = usable[
            usable["spectrum_kind"].astype(str).eq("energy_derived_from_amplitude")
        ]
        if not energy.empty:
            usable = energy
    primary = usable[usable["clip_top_k"].eq(0)] if "clip_top_k" in usable else usable
    if primary.empty:
        raise RuntimeError(
            f"{method_slug}: no successful preregistered raw fits; inspect powerlaw_fits.csv"
        )
    if "state_index" not in primary:
        primary["state_index"] = 0
    groups = tuple(
        name
        for name in (
            "optimizer", "layer", "method", "null_kind", "pair_stride",
            "epsilon", "evidence_role",
        )
        if name in primary
    )
    if not groups:
        primary["method"] = str(method_slug)
        groups = ("method",)
    return plot_seed_ci(
        primary,
        x="state_index",
        metric="alpha",
        groups=groups,
        title=title,
        ylabel="Power-law density exponent alpha",
        reference=2.0,
        allow_incomplete=True,
        incomplete_output_path=(
            OUTPUT_ROOT_PATH / "analyses" / str(method_slug)
            / "incomplete_alpha_ci_groups.csv"
        ),
        output_path=(
            OUTPUT_ROOT_PATH / "analyses" / str(method_slug) / "alpha_95ci.png"
        ),
    )


def call_supported(function, /, *args, **kwargs):
    signature = inspect.signature(function)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **supported)


def boolean_series(values):
    if getattr(values, "dtype", None) == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def ci_summary(
    frame,
    *,
    groups,
    metrics,
    allow_incomplete=False,
    incomplete_output_path=None,
    return_incomplete=False,
):
    missing = set((*groups, *metrics, "seed")) - set(frame.columns)
    if missing:
        raise ValueError(f"CI input is missing columns: {sorted(missing)}")
    # Repeated layers/checkpoints/probes are not independent replicates.  First
    # collapse every declared group to one value per complete training seed.
    replicate = (
        frame.groupby([*groups, "seed"], as_index=False, dropna=False)[list(metrics)]
        .mean(numeric_only=True)
    )
    summary = summarize_numeric_metrics(
        replicate,
        group_columns=tuple(groups),
        metrics=tuple(metrics),
        confidence=0.95,
    )
    if summary.empty:
        raise RuntimeError("Confidence-interval summary is empty after seed aggregation")
    incomplete = summary[pd.to_numeric(summary["n"], errors="coerce") != len(SEEDS)]
    if not incomplete.empty:
        if incomplete_output_path is not None:
            incomplete_output_path = Path(incomplete_output_path)
            incomplete_output_path.parent.mkdir(parents=True, exist_ok=True)
            incomplete.to_csv(incomplete_output_path, index=False)
        if allow_incomplete:
            print(
                "WARNING: dropping incomplete CI identities from the mean/band; "
                "faint individual-seed traces remain visible.\n"
                + incomplete[
                    [name for name in (*groups, "metric", "n") if name in incomplete]
                ].to_string(index=False)
            )
        else:
            identity = [name for name in (*groups, "metric", "n") if name in incomplete]
            raise RuntimeError(
                "Every confidence-interval row requires exactly the preregistered "
                f"{len(SEEDS)} complete seeds. Incomplete identities:\n"
                + incomplete[identity].to_string(index=False)
            )
    complete = summary[pd.to_numeric(summary["n"], errors="coerce") == len(SEEDS)].copy()
    if return_incomplete:
        return complete, incomplete.copy()
    return complete


def plot_seed_ci(
    frame,
    *,
    x,
    metric,
    groups,
    title,
    ylabel,
    reference=None,
    output_path=None,
    allow_incomplete=False,
    incomplete_output_path=None,
):
    groups = tuple(groups)
    if allow_incomplete and incomplete_output_path is None and output_path is not None:
        output_path_for_report = Path(output_path)
        incomplete_output_path = output_path_for_report.with_name(
            output_path_for_report.stem + "_incomplete_ci_groups.csv"
        )
    summary, incomplete = ci_summary(
        frame,
        groups=(*groups, x),
        metrics=(metric,),
        allow_incomplete=allow_incomplete,
        incomplete_output_path=incomplete_output_path,
        return_incomplete=True,
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    if not groups:
        frame = frame.copy()
        frame["series"] = "all"
        groups = ("series",)
    for identity, group in frame.groupby(list(groups), dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        label = ", ".join(f"{key}={value}" for key, value in zip(groups, identity))
        for _, seed_frame in group.groupby("seed"):
            ordered = (
                seed_frame.groupby(x, as_index=False, dropna=False)[metric]
                .mean(numeric_only=True)
                .sort_values(x)
            )
            ax.plot(ordered[x], ordered[metric], alpha=0.16, linewidth=0.9)
        selected = summary.copy()
        for key, value in zip(groups, identity):
            selected = selected[selected[key].astype(str) == str(value)]
        selected = selected[selected["metric"] == metric].sort_values(x)
        if selected.empty:
            pass
        else:
            xv = selected[x].to_numpy(dtype=float)
            mean = selected["mean"].to_numpy(dtype=float)
            low = selected["ci_low"].to_numpy(dtype=float)
            high = selected["ci_high"].to_numpy(dtype=float)
            ax.plot(xv, mean, marker="o", linewidth=2.1, label=label)
            finite = np.isfinite(low) & np.isfinite(high)
            ax.fill_between(xv[finite], low[finite], high[finite], alpha=0.18)
        missing = incomplete.copy()
        for key, value in zip(groups, identity):
            missing = missing[missing[key].astype(str) == str(value)]
        missing = missing[missing["metric"] == metric]
        if not missing.empty:
            ax.scatter(
                missing[x].to_numpy(dtype=float),
                missing["mean"].to_numpy(dtype=float),
                marker="x", color="#555555", alpha=0.65, zorder=4,
            )
    if reference is not None:
        ax.axhline(float(reference), color="#333333", linestyle="--", linewidth=1.4)
    ax.set(xlabel=x, ylabel=ylabel, title=title)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return summary, fig
"""


TAIL_CHECKPOINT_CACHE_HELPERS = r"""
_VERIFIED_TAIL_CACHE_REFS = {}
_VERIFIED_TAIL_CHECKPOINT_IDENTITIES = {}
_VERIFIED_RUN_IDENTITIES = {}


def require_complete_seed(optimizer_slug, seed):
    seed_dir = resolve_seed_dir(optimizer_slug, seed)
    manifest, _, _ = validate_run_identity(
        seed_dir, optimizer_slug=optimizer_slug, seed=seed
    )
    _VERIFIED_RUN_IDENTITIES[(str(optimizer_slug), int(seed))] = {
        "protocol_fingerprint": str(manifest["protocol_fingerprint"]),
        "source_seed_dir": str(Path(seed_dir).resolve()),
    }
    return seed_dir


def verified_run_fingerprint(optimizer_slug, seed):
    identity = _VERIFIED_RUN_IDENTITIES.get((str(optimizer_slug), int(seed)))
    if identity is None:
        raise RuntimeError(
            f"Run identity was not verified for optimizer={optimizer_slug}, seed={seed}"
        )
    return str(identity["protocol_fingerprint"])


def require_tail_checkpoint_cache(optimizer_slug, seed):
    # Establish expected identity from the separately completed run. Never
    # trust identity claimed only by the temporary cache itself.
    source_seed_dir = resolve_seed_dir(optimizer_slug, seed)
    manifest, resolved, completion = validate_run_identity(
        source_seed_dir, optimizer_slug=optimizer_slug, seed=seed
    )
    _VERIFIED_RUN_IDENTITIES[(str(optimizer_slug), int(seed))] = {
        "protocol_fingerprint": str(manifest["protocol_fingerprint"]),
        "source_seed_dir": str(Path(source_seed_dir).resolve()),
    }
    resolved_values = dict(resolved.get("config", resolved))
    if "epochs" not in resolved_values:
        raise KeyError(f"Resolved run config lacks epochs: {source_seed_dir}")
    expected_epochs = tail_checkpoint_epochs(int(resolved_values["epochs"]))
    recorded_cache_root = Path(
        resolved_values.get(
            "tail_checkpoint_cache_root",
            "/tmp/rg-mnist-mlp3-tangent-checkpoints",
        )
    ).expanduser().resolve()
    if recorded_cache_root != CHECKPOINT_CACHE_ROOT_PATH:
        raise RuntimeError(
            "Notebook CHECKPOINT_CACHE_ROOT disagrees with the completed run: "
            f"notebook={CHECKPOINT_CACHE_ROOT_PATH}, recorded={recorded_cache_root}. "
            "Set CHECKPOINT_CACHE_ROOT (or "
            "RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT) to the recorded cache root."
        )
    temporary_root = Path("/tmp").resolve()
    if (
        recorded_cache_root == temporary_root
        or not recorded_cache_root.is_relative_to(temporary_root)
    ):
        raise RuntimeError(
            f"Tail checkpoint cache must be a safe child of /tmp: {recorded_cache_root}"
        )
    cache_seed_dir = (
        recorded_cache_root
        / PROTOCOL_SLUG
        / str(optimizer_slug)
        / f"seed_{int(seed)}"
    ).resolve()
    refs = tuple(
        load_verified_tail_checkpoint_refs(
            cache_seed_dir,
            expected_suite_name=PROTOCOL_SLUG,
            expected_optimizer_name=str(optimizer_slug),
            expected_seed=int(seed),
            expected_fingerprint=str(manifest["protocol_fingerprint"]),
            expected_epochs=expected_epochs,
            validate_payloads=True,
        )
    )
    expected_count = min(100, int(resolved_values["epochs"]))
    if len(refs) != expected_count:
        raise RuntimeError(
            f"Verified cache count {len(refs)} != expected {expected_count}: "
            f"{cache_seed_dir}"
        )
    completed_epochs = int(completion.get("epochs", -1))
    completed_step = int(completion.get("global_step", -1))
    if completed_epochs != int(resolved_values["epochs"]) or completed_step < 1:
        raise RuntimeError(
            f"Completed run horizon is inconsistent beneath {source_seed_dir}"
        )
    steps_per_epoch, remainder = divmod(completed_step, completed_epochs)
    if remainder or steps_per_epoch < 1:
        raise RuntimeError(
            f"Completed step count is not an exact epoch grid beneath {source_seed_dir}"
        )
    expected_pairs = tuple(
        (int(epoch), int(epoch) * int(steps_per_epoch))
        for epoch in expected_epochs
    )
    observed_pairs = tuple((int(ref.epoch), int(ref.global_step)) for ref in refs)
    completion_cache_dir = completion.get("tail_checkpoint_cache_dir")
    if not isinstance(completion_cache_dir, str) or not completion_cache_dir.strip():
        raise RuntimeError(
            f"Completion marker lacks tail_checkpoint_cache_dir: {source_seed_dir}"
        )
    source_cache_checks = {
        "cache_dir": (
            str(Path(completion_cache_dir).expanduser().resolve()),
            str(cache_seed_dir),
        ),
        "checkpoint_count": (completion.get("tail_checkpoint_count"), expected_count),
        "first_epoch": (completion.get("tail_checkpoint_first_epoch"), expected_epochs[0]),
        "last_epoch": (completion.get("tail_checkpoint_last_epoch"), expected_epochs[-1]),
        "epoch_step_grid": (observed_pairs, expected_pairs),
    }
    mismatches = [
        f"{label}: observed={observed!r}, expected={expected!r}"
        for label, (observed, expected) in source_cache_checks.items()
        if observed != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Tail cache disagrees with the persistent run completion marker:\n  - "
            + "\n  - ".join(mismatches)
        )
    resolved_cache_dir = cache_seed_dir.resolve()
    _VERIFIED_TAIL_CACHE_REFS[resolved_cache_dir] = refs
    for ref in refs:
        _VERIFIED_TAIL_CHECKPOINT_IDENTITIES[ref.path.resolve()] = {
            "protocol_fingerprint": str(manifest["protocol_fingerprint"]),
            "optimizer": str(optimizer_slug),
            "seed": int(seed),
            "epoch": int(ref.epoch),
            "global_step": int(ref.global_step),
            "cache_seed_dir": str(resolved_cache_dir),
        }
    return resolved_cache_dir
"""


CHECKPOINT_HELPERS = r"""


def analysis_checkpoint_refs(seed_dir):
    cache_seed_dir = Path(seed_dir).resolve()
    refs = tuple(_VERIFIED_TAIL_CACHE_REFS.get(cache_seed_dir, ()))
    if len(refs) < 2:
        raise RuntimeError(
            "Tail checkpoint cache was not verified in this kernel, or contains "
            f"fewer than two states: {cache_seed_dir}. Call "
            "require_tail_checkpoint_cache(optimizer, seed); analysis notebooks "
            "never fall back to sparse run checkpoints or training."
        )
    return refs


def selected_checkpoint_pairs(seed_dir, *, maximum_pairs=None, stride=1):
    refs = analysis_checkpoint_refs(seed_dir)
    stride = int(stride)
    if stride < 1:
        raise ValueError("PAIR_STRIDE must be positive")
    all_pairs = list(zip(refs[:-stride], refs[stride:]))
    if not all_pairs:
        raise RuntimeError(f"No checkpoint pairs selected from {seed_dir}")
    budget = (
        len(all_pairs)
        if maximum_pairs is None
        else min(int(maximum_pairs), len(all_pairs))
    )
    if budget < 1:
        raise ValueError("maximum_pairs must be positive or None")
    if budget == len(all_pairs):
        indices = list(range(len(all_pairs)))
        complete_role = (
            "complete_verified_tail_adjacent_grid"
            if stride == 1
            else "complete_verified_tail_stride_grid"
        )
        rule = (
            "all chronological pairs from the verified tail checkpoint cache: "
            f"stride={stride}, selected_count={len(indices)}, "
            f"total_available={len(all_pairs)}"
        )
        roles = [complete_role for _ in indices]
    else:
        tail_count = min(max(2, budget // 3), budget)
        broad_budget = max(0, budget - tail_count)
        broad_limit = max(0, len(all_pairs) - tail_count)
        broad_indices = []
        if broad_budget and broad_limit:
            broad_indices = np.unique(
                np.rint(np.geomspace(1, broad_limit, num=broad_budget) - 1).astype(int)
            ).tolist()
            if 0 not in broad_indices:
                broad_indices.insert(0, 0)
        tail_indices = list(range(len(all_pairs) - tail_count, len(all_pairs)))
        indices = sorted(set(broad_indices + tail_indices))
        rule = (
            "deterministic log-index broad pair sample plus consecutive pair tail: "
            f"stride={stride}, requested={maximum_pairs}, selected_indices={indices}, "
            f"tail_count={tail_count}, total_available={len(all_pairs)}"
        )
        tail_start = len(all_pairs) - tail_count
        roles = [
            (
                "consecutive_tail"
                if stride == 1 and index >= tail_start
                else "stride_tail"
                if index >= tail_start
                else "log_index_broad"
            )
            for index in indices
        ]
    return [
        (
            all_pairs[index][0], all_pairs[index][1], rule, role,
        )
        for index, role in zip(indices, roles)
    ]


def selected_checkpoint_pairs_for_strides(seed_dir, *, strides, maximum_pairs=None):
    selected = []
    available_states = len(analysis_checkpoint_refs(seed_dir))
    for stride in tuple(dict.fromkeys(int(value) for value in strides)):
        if stride >= available_states:
            print(
                f"Skipping pair stride={stride}: verified cache has only "
                f"{available_states} states"
            )
            continue
        for previous, current, rule, role in selected_checkpoint_pairs(
            seed_dir, maximum_pairs=maximum_pairs, stride=stride
        ):
            selected.append((stride, previous, current, rule, role))
    if not selected:
        raise RuntimeError(f"No multi-spacing checkpoint pairs selected from {seed_dir}")
    return selected


def selected_trajectory_matrices(seed_dir, *, layers, maximum_checkpoints):
    refs = analysis_checkpoint_refs(seed_dir)
    budget = min(int(maximum_checkpoints), len(refs))
    if budget == len(refs):
        indices = list(range(len(refs)))
        roles = ["complete_verified_tail_state_grid" for _ in indices]
        rule = (
            "all chronological states from the verified tail checkpoint cache: "
            f"selected_count={len(indices)}, total_available={len(refs)}"
        )
    else:
        tail_count = min(max(2, budget // 3), budget)
        broad_budget = max(0, budget - tail_count)
        broad_limit = max(0, len(refs) - tail_count)
        broad_indices = []
        if broad_budget and broad_limit:
            broad_indices = np.unique(
                np.rint(np.geomspace(1, broad_limit, num=broad_budget) - 1)
                .astype(int)
            ).tolist()
            if 0 not in broad_indices:
                broad_indices.insert(0, 0)
        tail_indices = list(range(len(refs) - tail_count, len(refs)))
        indices = sorted(set(broad_indices + tail_indices))
        tail_start = len(refs) - tail_count
        roles = [
            "consecutive_tail" if index >= tail_start else "log_index_broad"
            for index in indices
        ]
        rule = (
            f"deterministic log-index broad sample plus consecutive tail: "
            f"requested={maximum_checkpoints}, selected_indices={indices}, "
            f"tail_count={tail_count}, total_available={len(refs)}"
        )
    selected = [refs[index] for index in indices]
    if len(selected) < 2:
        raise RuntimeError(f"Need at least two trajectory states beneath {seed_dir}")
    for index, ref, role in zip(indices, selected, roles):
        for layer in layers:
            yield ref, str(layer), checkpoint_matrix(ref.path, str(layer)), rule, role


@lru_cache(maxsize=int(CHECKPOINT_PAYLOAD_CACHE_SIZE))
def _load_verified_checkpoint_payload_cached(
    source_text,
    expected_fingerprint,
    expected_optimizer,
    expected_seed,
    expected_epoch,
    expected_global_step,
):
    source = Path(source_text)
    payload = load_analysis_checkpoint(
        source, expected_fingerprint=str(expected_fingerprint)
    )
    checks = {
        "optimizer": (payload.get("optimizer"), str(expected_optimizer)),
        "seed": (payload.get("seed"), int(expected_seed)),
        "epoch": (payload.get("epoch"), int(expected_epoch)),
        "global_step": (payload.get("global_step"), int(expected_global_step)),
    }
    mismatches = [
        f"{field}: observed={observed!r}, expected={expected!r}"
        for field, (observed, expected) in checks.items()
        if str(observed) != str(expected)
    ]
    if mismatches:
        raise RuntimeError(
            f"Checkpoint payload disagrees with verified cache identity: {source}; "
            + "; ".join(mismatches)
        )
    return payload


def load_checkpoint_payload(path):
    source = require_path(path, description="verified tail checkpoint").resolve()
    expected = _VERIFIED_TAIL_CHECKPOINT_IDENTITIES.get(source)
    if expected is None:
        raise RuntimeError(
            f"Checkpoint was not admitted by the strict tail-cache verifier: {source}. "
            "Analysis never falls back to an arbitrary checkpoint path."
        )
    payload = _load_verified_checkpoint_payload_cached(
        str(source),
        str(expected["protocol_fingerprint"]),
        str(expected["optimizer"]),
        int(expected["seed"]),
        int(expected["epoch"]),
        int(expected["global_step"]),
    )
    match = re.fullmatch(
        r"analysis_epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.pt", source.name
    )
    if match is None:
        raise RuntimeError(f"Expected an immutable analysis-checkpoint filename: {source}")
    if (
        int(payload.get("epoch", -1)) != int(match.group("epoch"))
        or int(payload.get("global_step", -1)) != int(match.group("step"))
        or int(payload.get("epoch", -1)) != int(expected["epoch"])
        or int(payload.get("global_step", -1)) != int(expected["global_step"])
    ):
        raise RuntimeError(f"Checkpoint payload epoch/step disagrees with filename: {source}")
    return payload


def checkpoint_state_dict(payload):
    for key in ("model", "model_state_dict", "state_dict"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, dict):
            return value
    raise KeyError(
        "Checkpoint has no model/model_state_dict/state_dict mapping; "
        f"available keys={list(payload) if isinstance(payload, dict) else type(payload)}"
    )


def checkpoint_matrix(path, parameter_name):
    state = checkpoint_state_dict(load_checkpoint_payload(path))
    candidates = (
        parameter_name,
        parameter_name.removeprefix("model."),
        f"model.{parameter_name}",
    )
    for name in candidates:
        if name in state:
            value = state[name]
            if hasattr(value, "detach"):
                return value.detach().cpu().double().numpy()
            return np.asarray(value, dtype=np.float64)
    raise KeyError(
        f"Matrix {parameter_name!r} is absent from {path}; "
        f"available 2-D keys={[key for key, value in state.items() if getattr(value, 'ndim', 0) == 2]}"
    )


def checkpoint_step(payload, fallback):
    for key in ("global_step", "step", "optimizer_step"):
        if isinstance(payload, dict) and key in payload:
            return int(payload[key])
    return int(fallback)


def capture_payloads(seed_dir, *, maximum_captures=None):
    capture_root = Path(seed_dir) / "captures"
    require_path(capture_root, description="dense capture directory")
    paths = tuple(list_capture_files(capture_root))
    if not paths:
        raise FileNotFoundError(
            f"No dense capture files under {capture_root}. Ensure the resolved "
            "config includes burst anchors and rerun training."
        )
    if maximum_captures is not None:
        paths = paths[-int(maximum_captures):]
    manifest = json.loads(
        require_path(Path(seed_dir) / "manifest.json", description="run manifest")
        .read_text(encoding="utf-8")
    )
    loaded = []
    for path in paths:
        payload = load_step_capture(
            path, expected_fingerprint=manifest["protocol_fingerprint"]
        )
        if str(payload.get("optimizer")) != str(manifest.get("optimizer")):
            raise RuntimeError(f"Capture optimizer disagrees with manifest: {path}")
        anchor_match = re.fullmatch(r"burst_epoch_(\d+)", path.parent.name)
        if (
            anchor_match is None
            or int(payload.get("anchor_epoch", -1)) != int(anchor_match.group(1))
        ):
            raise RuntimeError(f"Capture anchor epoch disagrees with directory: {path}")
        loaded.append((path, payload))
    return loaded


def capture_array(parameter_payload, key, *, required=True):
    value = parameter_payload.get(key)
    if value is None:
        if required:
            raise KeyError(
                f"Dense capture parameter is missing required field {key!r}; "
                f"available={sorted(parameter_payload)}"
            )
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def resolved_training_config(seed_dir):
    payload = json.loads(
        require_path(
            Path(seed_dir) / "resolved_config.json",
            description="resolved training config",
        ).read_text(encoding="utf-8")
    )
    values = dict(payload.get("config", payload))
    if not values:
        raise ValueError(f"Resolved config is empty in {seed_dir}")
    values["adamw"] = AdamWProfile(**dict(values.get("adamw", {})))
    muon_values = dict(values.get("muon", {}))
    if "parameter_names" in muon_values:
        muon_values["parameter_names"] = tuple(muon_values["parameter_names"])
    values["muon"] = MuonProfile(**muon_values)
    clip_values = dict(values.get("muonclip_rms", {}))
    if "parameter_names" in clip_values:
        clip_values["parameter_names"] = tuple(clip_values["parameter_names"])
    values["muonclip_rms"] = MuonClipRMSProfile(**clip_values)
    for name in (
        "explicit_analysis_epochs", "dense_burst_anchor_epochs",
        "capture_parameter_names",
    ):
        if name in values and isinstance(values[name], list):
            values[name] = tuple(values[name])
    config = TangentRGConfig(**values)
    config.validate()
    return config
"""


ECS_COVER_METRIC_HELPERS = r"""
ECS_COVER_SOURCE_KIND = (
    "verified_tail_checkpoint_cache_plus_exact_sparse_weightwatcher_trace_metrics"
)
ECS_PRIMARY_TRACE_QUALIFICATION_ROLE = "preregistered_independent_fit_support"


def _strict_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Expected a serialized boolean, found {value!r}")


def _strict_integer_metric(value, *, name):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"{name} is missing or nonnumeric")
    rounded = int(round(float(numeric)))
    if not np.isclose(float(numeric), rounded, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{name} must be integer-valued, found {value!r}")
    return rounded


def load_verified_ecs_metric_tables(optimizer_slug, seed, expected_fingerprint):
    identity = _VERIFIED_RUN_IDENTITIES.get((str(optimizer_slug), int(seed)))
    if identity is None:
        raise RuntimeError(
            f"Run identity was not verified for optimizer={optimizer_slug}, seed={seed}"
        )
    source_seed_dir = Path(identity["source_seed_dir"]).resolve()
    metrics_dir = source_seed_dir / "metrics"
    fit_path = require_path(
        metrics_dir / "weightwatcher_fits.csv",
        description="WeightWatcher fit table used for exact ECS ranks",
    )
    trace_path = require_path(
        metrics_dir / "trace_log.csv",
        description="trace-log table used to audit exact ECS ranks",
    )
    fits = pd.read_csv(fit_path)
    traces = pd.read_csv(trace_path)
    for label, frame, path, diagnostic_columns in (
        (
            "WeightWatcher",
            fits,
            fit_path,
            {"fit_ok", "detX_num"},
        ),
        (
            "trace-log",
            traces,
            trace_path,
            {
                "qualification_role", "sensitivity_only",
                "certification_eligible", "support_rank_source",
                "support_rank", "support_window_start_descending_zero_based",
                "support_window_end_descending_exclusive",
            },
        ),
    ):
        required = {
            "optimizer", "seed", "protocol_fingerprint", "epoch",
            "global_step", "layer", "fit_variant",
        } | diagnostic_columns
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{path} lacks ECS identity columns {sorted(missing)}")
        identity_columns = [
            "optimizer", "seed", "protocol_fingerprint", "epoch",
            "global_step", "layer", "fit_variant",
        ]
        if frame[identity_columns].isna().any().any():
            raise RuntimeError(f"{label} ECS identity contains nulls in {path}")
        identity_checks = {
            "optimizer": (set(frame["optimizer"].dropna().astype(str)), {str(optimizer_slug)}),
            "seed": (set(frame["seed"].dropna().astype(int)), {int(seed)}),
            "protocol_fingerprint": (
                set(frame["protocol_fingerprint"].dropna().astype(str)),
                {str(expected_fingerprint)},
            ),
        }
        mismatches = [
            f"{field}: observed={sorted(observed)}, expected={sorted(expected)}"
            for field, (observed, expected) in identity_checks.items()
            if observed != expected
        ]
        if mismatches:
            raise RuntimeError(
                f"{label} ECS identity mismatch in {path}: " + "; ".join(mismatches)
            )
    return fits, traces, str(fit_path), str(trace_path)


def exact_ecs_cover_rank_record(
    fits,
    traces,
    *,
    optimizer_slug,
    seed,
    epoch,
    global_step,
    layer,
    maximum_rank,
    fit_path,
    trace_path,
):
    identity_mask_fits = (
        fits["optimizer"].astype(str).eq(str(optimizer_slug))
        & pd.to_numeric(fits["seed"], errors="coerce").eq(int(seed))
        & pd.to_numeric(fits["epoch"], errors="coerce").eq(int(epoch))
        & pd.to_numeric(fits["global_step"], errors="coerce").eq(int(global_step))
        & fits["layer"].astype(str).eq(str(layer))
        & fits["fit_variant"].astype(str).eq("clip_xmax")
    )
    fit_rows = fits.loc[identity_mask_fits]
    base = {
        "ecs_rank_status": "unavailable",
        "ecs_rank_metrics_available": False,
        "ecs_full_shell_available": False,
        "ecs_detx_shell_available": False,
        "ecs_rank_fit_variant": "clip_xmax",
        "ecs_rank_fit_path": str(fit_path),
        "ecs_rank_trace_path": str(trace_path),
        "ecs_rank_exact_match_required": True,
        "ecs_rank_exact_epoch_match_found": False,
        "ecs_rank_exact_global_step_match_found": False,
        "ecs_rank_exact_weightwatcher_state_found": False,
        "ecs_rank_exact_trace_state_found": False,
        "ecs_rank_nearest_or_forward_fill_used": False,
    }
    if fit_rows.empty:
        return {
            **base,
            "ecs_rank_unavailable_reason": "no exact sparse WeightWatcher state",
        }
    if len(fit_rows) != 1:
        raise RuntimeError(
            "Expected one exact clip_xmax WeightWatcher row for "
            f"{optimizer_slug}/seed={seed}/epoch={epoch}/{layer}; found {len(fit_rows)}"
        )
    base.update({
        "ecs_rank_exact_epoch_match_found": True,
        "ecs_rank_exact_global_step_match_found": True,
        "ecs_rank_exact_weightwatcher_state_found": True,
    })
    identity_mask_traces = (
        traces["optimizer"].astype(str).eq(str(optimizer_slug))
        & pd.to_numeric(traces["seed"], errors="coerce").eq(int(seed))
        & pd.to_numeric(traces["epoch"], errors="coerce").eq(int(epoch))
        & pd.to_numeric(traces["global_step"], errors="coerce").eq(int(global_step))
        & traces["layer"].astype(str).eq(str(layer))
        & traces["fit_variant"].astype(str).eq("clip_xmax")
    )
    exact_traces = traces.loc[identity_mask_traces]
    if not exact_traces.empty:
        base["ecs_rank_exact_trace_state_found"] = True
    fit = fit_rows.iloc[0]
    if "fit_ok" not in fit_rows.columns or not _strict_bool(fit["fit_ok"]):
        return {
            **base,
            "ecs_rank_unavailable_reason": "exact WeightWatcher fit is not fit_ok",
            "weightwatcher_status": str(fit.get("weightwatcher_status", fit.get("status", "unknown"))),
        }

    primary = exact_traces.loc[
        exact_traces["qualification_role"].astype(str).eq(
            ECS_PRIMARY_TRACE_QUALIFICATION_ROLE
        )
        & ~exact_traces["sensitivity_only"].map(_strict_bool)
    ]
    if len(primary) != 1:
        if len(primary) > 1:
            raise RuntimeError(
                f"Duplicate primary ECS support rows at epoch={epoch}, layer={layer}"
            )
        return {
            **base,
            "ecs_rank_unavailable_reason": "no exact certifying primary trace support",
        }
    primary = primary.iloc[0]
    if not _strict_bool(primary["certification_eligible"]):
        return {
            **base,
            "ecs_rank_unavailable_reason": "primary trace support is not certification eligible",
        }
    detx_rows = exact_traces.loc[
        exact_traces["support_rank_source"].astype(str).eq("weightwatcher_detX")
    ]
    if len(detx_rows) != 1:
        if len(detx_rows) > 1:
            raise RuntimeError(
                f"Duplicate detX ECS support rows at epoch={epoch}, layer={layer}"
            )
        return {
            **base,
            "ecs_rank_unavailable_reason": "no exact detX trace audit row",
        }
    detx_row = detx_rows.iloc[0]
    midpoint_rows = exact_traces.loc[
        exact_traces["support_rank_source"].astype(str).eq(
            "weightwatcher_midpoint"
        )
    ]
    if len(midpoint_rows) != 1:
        if len(midpoint_rows) > 1:
            raise RuntimeError(
                f"Duplicate midpoint ECS audit rows at epoch={epoch}, layer={layer}"
            )
        return {
            **base,
            "ecs_rank_unavailable_reason": "no exact WeightWatcher midpoint audit row",
        }
    midpoint_row = midpoint_rows.iloc[0]

    detx_value = pd.to_numeric(pd.Series([fit.get("detX_num")]), errors="coerce").iloc[0]
    start_value = pd.to_numeric(
        pd.Series([primary.get("support_window_start_descending_zero_based")]),
        errors="coerce",
    ).iloc[0]
    end_value = pd.to_numeric(
        pd.Series([primary.get("support_window_end_descending_exclusive")]),
        errors="coerce",
    ).iloc[0]
    tail_value = pd.to_numeric(pd.Series([primary.get("support_rank")]), errors="coerce").iloc[0]
    if not all(pd.notna(value) for value in (detx_value, start_value, end_value, tail_value)):
        return {
            **base,
            "ecs_rank_unavailable_reason": "nonfinite PL-window or detX rank",
        }
    try:
        k_pl = _strict_integer_metric(end_value, name="PL support-window end")
        k_tl = _strict_integer_metric(detx_value, name="WeightWatcher detX_num")
        window_start = _strict_integer_metric(
            start_value, name="PL support-window start"
        )
        effective_tail = _strict_integer_metric(
            tail_value, name="PL effective-tail rank"
        )
        detx_trace = _strict_integer_metric(
            detx_row["support_rank"], name="trace detX support rank"
        )
        persisted_midpoint = _strict_integer_metric(
            midpoint_row["support_rank"], name="trace WeightWatcher midpoint rank"
        )
    except ValueError as error:
        raise RuntimeError(
            f"Malformed exact ECS rank metric at epoch={epoch}, layer={layer}: {error}"
        ) from error
    if k_pl != window_start + effective_tail:
        raise RuntimeError("ECS PL boundary is not window_start + effective_tail")
    recorded_pl_boundaries = []
    for name, value in (
        ("WeightWatcher pl_support_rank", fit.get("pl_support_rank")),
        (
            "trace pl_support_rank_before_finger_clip",
            primary.get("pl_support_rank_before_finger_clip"),
        ),
    ):
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            try:
                recorded_pl_boundaries.append(
                    (name, _strict_integer_metric(numeric, name=name))
                )
            except ValueError as error:
                raise RuntimeError(
                    f"Malformed exact ECS rank metric at epoch={epoch}, "
                    f"layer={layer}: {error}"
                ) from error
    mismatched_pl_boundaries = [
        f"{name}={value}" for name, value in recorded_pl_boundaries
        if value != k_pl
    ]
    if mismatched_pl_boundaries:
        raise RuntimeError(
            "ECS top-mode PL boundary disagrees with recorded preclip support: "
            + ", ".join(mismatched_pl_boundaries)
        )
    if detx_trace != k_tl:
        raise RuntimeError(
            "Exact detX fit field disagrees with trace audit; fallback is refused"
        )
    expected_persisted_midpoint = int(np.floor((effective_tail + k_tl) / 2.0))
    if persisted_midpoint != expected_persisted_midpoint:
        raise RuntimeError(
            "Persisted WeightWatcher midpoint audit disagrees with its declared "
            "effective-tail/detX convention"
        )
    try:
        selection = single_checkpoint.select_ecs_cover_ranks(
            k_pl,
            k_tl,
            maximum_rank=int(maximum_rank),
        )
    except (TypeError, ValueError) as error:
        return {
            **base,
            "ecs_rank_unavailable_reason": f"invalid exact ECS boundaries: {error}",
            "k_pl": k_pl,
            "k_tl": k_tl,
        }
    result = {
        **base,
        "ecs_rank_status": "ok",
        "ecs_rank_metrics_available": True,
        "ecs_full_shell_available": bool(k_pl < int(maximum_rank)),
        "ecs_detx_shell_available": bool(selection.available),
        "ecs_rank_unavailable_reason": "",
        "ecs_full_shell_unavailable_reason": (
            "" if k_pl < int(maximum_rank)
            else "power-law retained boundary reaches checkpoint numerical rank"
        ),
        "ecs_detx_shell_unavailable_reason": selection.unavailable_reason,
        "k_pl": selection.power_law_rank,
        "k_boundary_mid": selection.boundary_midpoint_rank,
        "weightwatcher_midpoint_rank": persisted_midpoint,
        "k_tl": selection.trace_log_rank,
        "retained_rank": selection.retained_rank,
        "retained_rank_source": selection.retained_rank_source,
        "full_shell_outer_rank": int(maximum_rank),
        "full_shell_outer_rank_source": "checkpoint_numerical_rank",
        "full_shell_rank": max(0, int(maximum_rank) - selection.retained_rank),
        "detx_shell_outer_rank": selection.outer_rank,
        "detx_shell_outer_rank_source": selection.outer_rank_source,
        "detx_shell_rank": selection.shell_rank,
        "rank_selection_rule": selection.selection_rule,
        "pl_boundary_definition": (
            "top-mode boundary index = clipped support window start + effective tail rank; "
            "required by V_k=[v_1,...,v_k]"
        ),
        "pl_effective_tail_rank": effective_tail,
        "pl_support_window_start": window_start,
        "pl_support_window_end": k_pl,
        "pl_support_rank_source": str(primary["support_rank_source"]),
        "pl_support_window_source": str(primary.get("support_window_source", "unknown")),
        "n_fingers_removed": int(round(float(primary.get("n_fingers_removed", window_start)))),
        "detx_rank_source": "finite detX_num in exact WeightWatcher row, cross-audited in trace_log.csv",
    }
    return result
"""


PROVENANCE_ASSERTIONS = r"""
required_provenance = {"operator_kind", "map_definition"}
missing = required_provenance - set(operator_rows.columns)
if missing:
    raise RuntimeError(f"Operator output is missing provenance columns: {sorted(missing)}")
if operator_rows[list(required_provenance)].isna().any().any():
    raise RuntimeError("Operator provenance contains missing values")
display(operator_rows.head(20))
"""


def notebook(name: str, cells: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    for index, cell in enumerate(cells):
        digest = hashlib.sha1(f"{name}:{index}".encode("utf-8")).hexdigest()[:12]
        cell["id"] = digest
    return name, {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def caveat(title: str, operator_kind: str, map_definition: str, text: str) -> dict[str, object]:
    return markdown(
        f"""
        ## {title}

        **`operator_kind`: `{operator_kind}`**

        **`map_definition`: `{map_definition}`**

        **Identifiability caveat.** {text}

        These strings are persisted with every result row. A visually useful
        spectrum does not change the identity of the map that produced it.
        """
    )


TRAINING_IMPORTS = r"""
import subprocess

from rg_baselines.tangent_rg import (
    AdamWProfile,
    MuonClipRMSProfile,
    MuonProfile,
    TangentRGConfig,
    build_analysis_plan,
    list_analysis_checkpoints,
    list_capture_files,
    load_config,
    run_training,
)
from rg_baselines.tangent_rg.checkpoints import load_verified_tail_checkpoint_refs
from rg_baselines.tangent_rg.protocol import tail_checkpoint_epochs
from rg_baselines.tangent_rg import cli as tangent_cli
"""


TRAINING_HELPERS = r"""
def resolve_config_path(profile, explicit=""):
    if explicit:
        return require_path(explicit, description="training configuration")
    names = (f"{profile}.yaml", f"{profile}.yml", f"{profile}.json")
    roots = (
        EXPERIMENT_ROOT / "configs",
        BASELINE_ROOT / "rg_baselines" / "tangent_rg" / "configs",
        BASELINE_ROOT / "configs" / "mnist_mlp3_tangent_rg",
    )
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    epoch_by_profile = {
        "smoke": 2,
        "pilot_1000_epochs": 1_000,
        "long_horizon_10000_epochs": 10_000,
    }
    if profile not in epoch_by_profile:
        raise FileNotFoundError(
            f"No checked-in configuration for profile={profile!r}. Set CONFIG_PATH."
        )
    epochs = epoch_by_profile[profile]
    generated = OUTPUT_ROOT_PATH / "generated_configs" / f"{profile}.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "suite_name": PROTOCOL_SLUG,
            "schema_version": 1,
            "description": f"Notebook-generated preregistered {profile} profile",
        },
        "training": {
            "epochs": epochs,
            "lr_schedule_epochs": min(30, epochs),
            "batch_size": 128,
            "validation_size": 5_000,
            "split_seed": 20_260_807,
            "validation_every_epochs": 1 if epochs == 2 else 5,
            "latest_every_epochs": 1,
            "test_monitoring_only": True,
        },
        "analysis": {
            "log_points": 2 if epochs == 2 else 96,
            "explicit_epochs": [0, 1, 2, 5, 10, 30],
            "dense_burst_anchor_epochs": [0, 1, 10, 100, 1_000],
            "dense_burst_length_steps": 4 if epochs == 2 else 8,
            "capture_parameter_names": ["fc1.weight", "fc2.weight"],
        },
        "runtime": {
            "device": "auto",
            "data_dir": str(RUN_ROOT_PATH / "data"),
            "run_root": str(RUN_ROOT_PATH),
            "tail_checkpoint_cache_root": str(CHECKPOINT_CACHE_ROOT_PATH),
        },
    }
    generated.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    checked = load_config(generated)
    checked.validate()
    print("generated validated config:", generated)
    return generated


def run_cli_training(*, optimizer, profile, seeds, resume):
    config_path = resolve_config_path(profile, CONFIG_PATH)
    launch_config = load_config(config_path)
    if str(launch_config.suite_name) != str(PROTOCOL_SLUG):
        raise RuntimeError(
            f"Resolved config suite_name={launch_config.suite_name!r} does not match "
            f"the notebook output suite {PROTOCOL_SLUG!r}. Clear PROTOCOL_SLUG or "
            "select the matching PROFILE/CONFIG_PATH."
        )
    temporary_root = Path("/tmp").resolve()
    if (
        int(launch_config.epochs) > 2
        and RUN_ROOT_PATH.is_relative_to(temporary_root)
        and not bool(ALLOW_TEMPORARY_LONG_RUN)
    ):
        raise RuntimeError(
            f"Refusing long-horizon training under temporary root {RUN_ROOT_PATH}. "
            "Set RUN_ROOT or RG_MNIST_TANGENT_ROOT to persistent storage. "
            "For a deliberate disposable run only, set "
            "ALLOW_TEMPORARY_LONG_RUN=True."
        )
    for seed in seeds:
        command = [
            sys.executable,
            "-m",
            "rg_baselines.tangent_rg.cli",
            "train",
            "--config",
            str(config_path),
            "--optimizer",
            str(optimizer),
            "--output-root",
            str(RUN_ROOT_PATH),
            "--tail-checkpoint-root",
            str(CHECKPOINT_CACHE_ROOT_PATH),
            "--seed",
            str(int(seed)),
        ]
        if resume:
            command.append("--resume")
        print("running:", " ".join(command))
        subprocess.run(command, cwd=BASELINE_ROOT, check=True)


def load_arm_outputs(optimizer):
    arm = resolve_arm_dir(optimizer)
    manifests = []
    performance_frames = []
    spectral_frames = []
    for seed in SEEDS:
        seed_dir = resolve_seed_dir(optimizer, seed)
        manifest, resolved, completion = validate_run_identity(
            seed_dir, optimizer_slug=optimizer, seed=seed
        )
        manifests.append(manifest)
        performance_path = first_existing(
            seed_dir / "metrics",
            (
                "performance_by_analysis_epoch.csv",
                "performance_by_analysis_state.csv",
            ),
            description="analysis-state performance table",
        )
        performance = pd.read_csv(performance_path)
        for column, expected in (("optimizer", optimizer), ("seed", int(seed))):
            if column not in performance:
                raise KeyError(f"{performance_path} lacks required identity column {column}")
            observed = set(performance[column].dropna().astype(str))
            if observed != {str(expected)}:
                raise RuntimeError(
                    f"{performance_path} identity mismatch for {column}: "
                    f"observed={sorted(observed)}, expected={expected}"
                )
        if "protocol_fingerprint" not in performance:
            raise KeyError(f"{performance_path} lacks protocol_fingerprint")
        performance_fingerprints = set(
            performance["protocol_fingerprint"].dropna().astype(str)
        )
        if performance_fingerprints != {str(manifest["protocol_fingerprint"])}:
            raise RuntimeError(
                f"{performance_path} fingerprint mismatch: "
                f"observed={sorted(performance_fingerprints)}"
            )
        performance["seed"] = int(seed)
        performance["optimizer"] = optimizer
        performance_frames.append(performance)
        spectral_path = first_existing(
            seed_dir / "metrics",
            (
                "spectral_metrics_by_analysis_epoch.csv",
                "weightwatcher_fits.csv",
                "weightwatcher_by_analysis_epoch.csv",
            ),
            description="raw + clip_xmax spectral fit table",
        )
        spectral = pd.read_csv(spectral_path)
        for column, expected in (("optimizer", optimizer), ("seed", int(seed))):
            if column not in spectral:
                raise KeyError(f"{spectral_path} lacks required identity column {column}")
            observed = set(spectral[column].dropna().astype(str))
            if observed != {str(expected)}:
                raise RuntimeError(
                    f"{spectral_path} identity mismatch for {column}: "
                    f"observed={sorted(observed)}, expected={expected}"
                )
        if "protocol_fingerprint" not in spectral:
            raise KeyError(f"{spectral_path} lacks protocol_fingerprint")
        spectral_fingerprints = set(
            spectral["protocol_fingerprint"].dropna().astype(str)
        )
        if spectral_fingerprints != {str(manifest["protocol_fingerprint"])}:
            raise RuntimeError(
                f"{spectral_path} fingerprint mismatch: "
                f"observed={sorted(spectral_fingerprints)}"
            )
        spectral["seed"] = int(seed)
        spectral["optimizer"] = optimizer
        spectral_frames.append(spectral)
    return (
        arm,
        manifests,
        pd.concat(performance_frames, ignore_index=True, sort=False),
        pd.concat(spectral_frames, ignore_index=True, sort=False),
    )


def canonical_columns(performance, spectral):
    performance = performance.copy()
    spectral = spectral.copy()
    for frame in (performance, spectral):
        if "analysis_epoch" in frame and "state_index" not in frame:
            frame["state_index"] = pd.to_numeric(
                frame["analysis_epoch"], errors="coerce"
            )
        if "analysis_state" in frame and "state_index" not in frame:
            frame["state_index"] = pd.to_numeric(
                frame["analysis_state"], errors="coerce"
            )
        if "epoch" in frame and "state_index" not in frame:
            frame["state_index"] = pd.to_numeric(frame["epoch"], errors="coerce")
    if "test_accuracy" not in performance and "test_acc" in performance:
        performance["test_accuracy"] = performance["test_acc"]
    if "layer" not in spectral and "layer_name" in spectral:
        spectral["layer"] = spectral["layer_name"]
    if "fit_variant" not in spectral:
        spectral["fit_variant"] = "unspecified"
    return performance, spectral
"""


def training_notebook(
    *,
    filename: str,
    title: str,
    optimizer_slug: str,
    profile: str,
    description: str,
) -> tuple[str, dict[str, object]]:
    cells = [
        markdown(
            f"""
            # {title}

            {description}

            This notebook delegates training to the tested command-line API;
            it does not define a private optimizer or model. Long runs are
            opt-in so opening the notebook cannot accidentally start a
            multi-day campaign. During training, the runtime also writes the
            final 100 trained epoch-boundary model checkpoints to the dedicated
            temporary cache selected by `CHECKPOINT_CACHE_ROOT`.
            """
        ),
        parameters(
            f"""
            CONFIG_PATH = ""
            PROFILE = "{profile}"
            OPTIMIZER_SLUG = "{optimizer_slug}"
            EXECUTE_TRAINING = False
            RESUME = True
            REQUIRE_ARTIFACTS = False
            """
        ),
        code(BOOTSTRAP + "\nEXPERIMENT_ROOT = BASELINE_ROOT / 'experiments' / 'mnist_mlp3_tangent_rg'"),
        code(COMMON_IMPORTS + "\n" + TRAINING_IMPORTS),
        caveat(
            "Measured object",
            "raw_weight_gram_esd",
            "For each layer W at a scheduled state, eigenvalues of the supported Gram matrix W W^T or W^T W.",
            "The raw weight ESD is the baseline observable. It is not a tangent-space Jacobian. "
            "Optimizer accuracy and late alpha/trace-log behavior validate the training baseline, "
            "not any quotient hypothesis.",
        ),
        code(COMMON_HELPERS + "\n" + TRAINING_HELPERS + "\n" + TAIL_CHECKPOINT_CACHE_HELPERS),
        markdown(
            """
            ## Launch or resume

            Set `CONFIG_PATH` to a resolved `smoke`, `pilot_1000_epochs`, or
            `long_horizon_10000_epochs` configuration and set
            `EXECUTE_TRAINING=True`. The exact command is printed before each
            seeded run. `CHECKPOINT_CACHE_ROOT` defaults to
            `/tmp/rg-mnist-mlp3-tangent-checkpoints` and may also be set with
            `RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT`; it must remain a safe
            child of `/tmp`. The cache is populated online, independently of
            WeightWatcher cadence, and is never reconstructed by an analysis
            notebook. Keep this cache through notebooks `10`, `12`, `13`, and
            `15`, or make an external byte-for-byte backup: losing it after
            completion requires full `--overwrite` retraining because
            `--resume` cannot recreate past epoch boundaries.
            """
        ),
        code(
            """
            if EXECUTE_TRAINING:
                run_cli_training(
                    optimizer=OPTIMIZER_SLUG,
                    profile=PROFILE,
                    seeds=SEEDS,
                    resume=bool(RESUME),
                )
            else:
                print(
                    "Training not launched. Set EXECUTE_TRAINING=True after "
                    "checking CONFIG_PATH, RUN_ROOT, and persistent storage."
                )
            """
        ),
        markdown("## Artifact completeness and protocol audit"),
        code(
            """
            if EXECUTE_TRAINING or REQUIRE_ARTIFACTS:
                arm_dir, manifests, performance, spectral = load_arm_outputs(
                    OPTIMIZER_SLUG
                )
                performance, spectral = canonical_columns(performance, spectral)
                observed_seeds = set(pd.to_numeric(performance["seed"]).astype(int))
                assert observed_seeds == set(SEEDS), (observed_seeds, SEEDS)
                if "test_monitoring_only" in performance:
                    assert performance["test_monitoring_only"].astype(int).eq(1).all()
                tail_cache_rows = []
                for seed in SEEDS:
                    seed_dir = resolve_seed_dir(OPTIMIZER_SLUG, seed)
                    checkpoints_dir = seed_dir / "checkpoints"
                    for name in (
                        "checkpoint_latest.pt",
                        "checkpoint_best.pt",
                        "checkpoint_final.pt",
                    ):
                        require_path(checkpoints_dir / name, description=name)
                    refs = list_analysis_checkpoints(checkpoints_dir)
                    if not refs or refs[0].epoch != 0 or refs[0].global_step != 0:
                        raise RuntimeError(
                            f"{seed_dir} is missing its immutable epoch-zero analysis checkpoint"
                        )
                    cache_seed_dir = require_tail_checkpoint_cache(
                        OPTIMIZER_SLUG, seed
                    )
                    cache_manifest = json.loads(
                        (cache_seed_dir / "manifest.json").read_text(encoding="utf-8")
                    )
                    cache_completion = json.loads(
                        (cache_seed_dir / "cache_complete.json").read_text(encoding="utf-8")
                    )
                    tail_cache_rows.append({
                        "optimizer": OPTIMIZER_SLUG,
                        "seed": int(seed),
                        "cache_seed_dir": str(cache_seed_dir),
                        "checkpoint_count": int(cache_completion["checkpoint_count"]),
                        "first_epoch": int(cache_completion["first_epoch"]),
                        "last_epoch": int(cache_completion["last_epoch"]),
                        "protocol_fingerprint": cache_manifest["protocol_fingerprint"],
                        "completed": bool(cache_completion["completed"]),
                    })
                print("complete arm:", arm_dir)
                display(pd.DataFrame(manifests))
                display(pd.DataFrame(tail_cache_rows))
                display(performance.tail(12))
                display(spectral.tail(18))
            else:
                performance = spectral = pd.DataFrame()
                print("Artifact audit deferred because REQUIRE_ARTIFACTS=False.")
            """
        ),
        markdown("## Performance and fixed-point plots with complete-run uncertainty"),
        code(
            """
            if not performance.empty:
                perf_summary, _ = plot_seed_ci(
                    performance,
                    x="state_index",
                    metric="test_accuracy",
                    groups=("optimizer",) if "optimizer" in performance else (),
                    title=f"{OPTIMIZER_SLUG}: monitoring-only test accuracy",
                    ylabel="Test accuracy",
                    output_path=OUTPUT_ROOT_PATH / OPTIMIZER_SLUG / "test_accuracy_95ci.png",
                )
                valid = spectral.copy()
                if "status" in valid:
                    valid = valid[valid["status"].astype(str).eq("ok")]
                if "selection_role" in valid:
                    preferred = valid[
                        valid["selection_role"].astype(str).isin(
                            ["primary", "preregistered_primary"]
                        )
                    ]
                    if not preferred.empty:
                        valid = preferred
                alpha_summary, _ = plot_seed_ci(
                    valid,
                    x="state_index",
                    metric="alpha",
                    groups=("layer", "fit_variant"),
                    title=f"{OPTIMIZER_SLUG}: layerwise alpha",
                    ylabel="Power-law density exponent alpha",
                    reference=2.0,
                    allow_incomplete=True,
                    incomplete_output_path=(
                        OUTPUT_ROOT_PATH / OPTIMIZER_SLUG
                        / "alpha_incomplete_ci_groups.csv"
                    ),
                    output_path=OUTPUT_ROOT_PATH / OPTIMIZER_SLUG / "alpha_95ci.png",
                )
                perf_summary.to_csv(
                    OUTPUT_ROOT_PATH / OPTIMIZER_SLUG / "performance_notebook_95ci.csv",
                    index=False,
                )
                alpha_summary.to_csv(
                    OUTPUT_ROOT_PATH / OPTIMIZER_SLUG / "alpha_notebook_95ci.csv",
                    index=False,
                )
            """
        ),
        markdown(
            """
            ## Interpretation gate

            High accuracy and a late alpha near two establish only the baseline
            regime. Inspect KS distance, tail count, tail decades, standard
            versus `clip_xmax`, and the independently supported trace-log rows
            before describing a layer as fixed-point-like. FC3 has at most ten
            positive modes and cannot carry the main power-law conclusion.
            """
        ),
    ]
    return notebook(filename, cells)


def smoke_notebook() -> tuple[str, dict[str, object]]:
    cells = [
        markdown(
            """
            # Protocol and bounded smoke validation

            Run this notebook before any pilot or reference campaign. It checks
            importability, declared operator provenance, pure-function algebra,
            CLI discoverability, and (when enabled) the two-epoch artifacts for
            all three optimizer arms.
            """
        ),
        parameters(
            """
            CONFIG_PATH = ""
            PROFILE = "smoke"
            OPTIMIZER_SLUGS = ["adamw", "muon", "muonclip_rms"]
            EXECUTE_TRAINING = False
            RESUME = True
            REQUIRE_ARTIFACTS = False
            """
        ),
        code(BOOTSTRAP + "\nEXPERIMENT_ROOT = BASELINE_ROOT / 'experiments' / 'mnist_mlp3_tangent_rg'"),
        code(
            COMMON_IMPORTS
            + "\n"
            + TRAINING_IMPORTS
            + "\nfrom rg_baselines.tangent_rg import nulls, polar, single_checkpoint, stiefel, two_checkpoint"
        ),
        caveat(
            "Operator inventory",
            "protocol_smoke_inventory",
            "Bounded algebra and artifact checks over explicitly named maps; no learned RG operator is inferred.",
            "Passing smoke tests establishes software consistency only. It does not validate any "
            "heavy-tail, quotient, or fixed-point hypothesis.",
        ),
        code(COMMON_HELPERS + "\n" + TRAINING_HELPERS + "\n" + TAIL_CHECKPOINT_CACHE_HELPERS),
        code(
            """
            import subprocess

            help_command = [
                sys.executable, "-m", "rg_baselines.tangent_rg.cli", "train", "--help"
            ]
            completed = subprocess.run(
                help_command,
                cwd=BASELINE_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            print(completed.stdout)
            assert "--config" in completed.stdout
            assert "--optimizer" in completed.stdout
            assert "--output-root" in completed.stdout
            assert "--tail-checkpoint-root" in completed.stdout
            """
        ),
        markdown("## Deterministic pure-map checks"),
        code(
            """
            rng = np.random.default_rng(20260819)
            smoke_rows = []
            for shape in ((8, 8), (12, 8), (8, 12)):
                weight = rng.normal(size=shape)
                q = polar.polar_factor(weight)
                gram = q.T @ q if shape[0] >= shape[1] else q @ q.T
                error = float(np.linalg.norm(gram - np.eye(min(shape))))
                smoke_rows.append(
                    {
                        "seed": SEEDS[0],
                        "shape": str(shape),
                        "operator_kind": "polar_factor_isometry_smoke",
                        "map_definition": "Pi(W)=U V^T from the thin SVD",
                        "orthogonality_error": error,
                    }
                )
                assert error < 1e-9
            operator_rows = pd.DataFrame(smoke_rows)
            display(operator_rows)
            """
        ),
        code(
            """
            if EXECUTE_TRAINING:
                for optimizer_slug in OPTIMIZER_SLUGS:
                    run_cli_training(
                        optimizer=optimizer_slug,
                        profile=PROFILE,
                        seeds=SEEDS,
                        resume=bool(RESUME),
                    )
            else:
                print("Two-epoch training smoke not launched.")
            """
        ),
        markdown("## Optional artifact and seed-CI smoke"),
        code(
            """
            frames = []
            cache_rows = []
            if EXECUTE_TRAINING or REQUIRE_ARTIFACTS:
                for optimizer_slug in OPTIMIZER_SLUGS:
                    _, _, performance, spectral = load_arm_outputs(optimizer_slug)
                    performance, spectral = canonical_columns(performance, spectral)
                    performance["optimizer"] = optimizer_slug
                    frames.append(performance)
                    for seed in SEEDS:
                        cache_seed_dir = require_tail_checkpoint_cache(
                            optimizer_slug, seed
                        )
                        refs = _VERIFIED_TAIL_CACHE_REFS[cache_seed_dir]
                        cache_rows.append({
                            "optimizer": optimizer_slug,
                            "seed": int(seed),
                            "checkpoint_count": len(refs),
                            "epochs": [int(ref.epoch) for ref in refs],
                            "cache_seed_dir": str(cache_seed_dir),
                        })
                combined = pd.concat(frames, ignore_index=True)
                summary, _ = plot_seed_ci(
                    combined,
                    x="state_index",
                    metric="test_accuracy",
                    groups=("optimizer",),
                    title="Two-epoch smoke: monitoring-only test accuracy",
                    ylabel="Test accuracy",
                    output_path=OUTPUT_ROOT_PATH / "smoke_test_accuracy_95ci.png",
                )
                display(summary)
                display(pd.DataFrame(cache_rows))
            else:
                print("Artifact/CI smoke deferred; set REQUIRE_ARTIFACTS=True after training.")
            """
        ),
        markdown(
            """
            The 1,000-epoch pilot is authorized only after every optimizer arm
            has complete three-seed smoke artifacts, immutable initial
            checkpoints, restartable latest checkpoints, strict fit provenance,
            and monitoring-only test flags.
            """
        ),
    ]
    return notebook("00_Protocol_and_Smoke.ipynb", cells)


def fixed_point_comparison_notebook() -> tuple[str, dict[str, object]]:
    cells = [
        markdown(
            """
            # AdamW / Muon / MuonClip-RMS fixed-point comparison

            This notebook reads complete, matched three-seed arms. It tests the
            baseline premise--high accuracy together with credible late
            `alpha ~= 2` and independently supported trace-log near zero--before
            any transformed tangent operator is interpreted.
            """
        ),
        parameters(
            """
            OPTIMIZER_SLUGS = ["adamw", "muon", "muonclip_rms"]
            ALPHA_TOLERANCE = 0.25
            TRACE_LOG_PER_EVAL_TOLERANCE = 0.10
            MINIMUM_TAIL = 8
            MAXIMUM_KS_D = 0.15
            """
        ),
        code(BOOTSTRAP + "\nEXPERIMENT_ROOT = BASELINE_ROOT / 'experiments' / 'mnist_mlp3_tangent_rg'"),
        code(
            COMMON_IMPORTS
            + "\n"
            + TRAINING_IMPORTS
            + "\nfrom rg_baselines.tangent_rg import plotting, reporting"
        ),
        caveat(
            "Baseline observable",
            "raw_weight_gram_esd_fixed_point_comparison",
            "Matched optimizer comparison of raw layer Gram ESD fits and independently supported trace-log rows.",
            "Agreement with alpha=2 does not identify a quotient or tangent map. This notebook "
            "qualifies the baseline regime only; it cannot select a transformed operator.",
        ),
        code(COMMON_HELPERS + "\n" + TRAINING_HELPERS),
        markdown("## Require all three complete arms"),
        code(
            """
            performance_frames = []
            spectral_frames = []
            legacy_trace_frames = []
            manifests = []
            for optimizer_slug in OPTIMIZER_SLUGS:
                arm, arm_manifests, performance, spectral = load_arm_outputs(
                    optimizer_slug
                )
                performance, spectral = canonical_columns(performance, spectral)
                performance_frames.append(performance)
                spectral_frames.append(spectral)
                manifests.extend(arm_manifests)
                arm_manifest_by_seed = {
                    int(item["seed"]): item for item in arm_manifests
                }
                for seed in SEEDS:
                    seed_dir = resolve_seed_dir(optimizer_slug, seed)
                    trace_path = require_path(
                        seed_dir / "metrics" / "trace_log.csv",
                        description="trace-log audit table",
                    )
                    trace_frame = pd.read_csv(trace_path)
                    for column, expected in (
                        ("optimizer", optimizer_slug), ("seed", int(seed))
                    ):
                        if column not in trace_frame:
                            raise KeyError(f"{trace_path} lacks identity column {column}")
                        observed = set(trace_frame[column].dropna().astype(str))
                        if observed != {str(expected)}:
                            raise RuntimeError(
                                f"{trace_path} identity mismatch for {column}: "
                                f"observed={sorted(observed)}, expected={expected}"
                            )
                    if "protocol_fingerprint" not in trace_frame:
                        raise KeyError(f"{trace_path} lacks protocol_fingerprint")
                    trace_fingerprints = set(
                        trace_frame["protocol_fingerprint"].dropna().astype(str)
                    )
                    expected_fingerprint = str(
                        arm_manifest_by_seed[int(seed)]["protocol_fingerprint"]
                    )
                    if trace_fingerprints != {expected_fingerprint}:
                        raise RuntimeError(
                            f"{trace_path} fingerprint mismatch: "
                            f"observed={sorted(trace_fingerprints)}"
                        )
                    legacy_trace_frames.append(trace_frame)
            performance = pd.concat(performance_frames, ignore_index=True, sort=False)
            spectral = pd.concat(spectral_frames, ignore_index=True, sort=False)
            legacy_trace = pd.concat(
                legacy_trace_frames, ignore_index=True, sort=False
            )
            provenance_audit = validate_cross_run_provenance(manifests)
            assert set(performance["optimizer"]) == set(OPTIMIZER_SLUGS)
            assert set(pd.to_numeric(performance["seed"]).astype(int)) == set(SEEDS)
            if "test_monitoring_only" in performance:
                assert performance["test_monitoring_only"].astype(int).eq(1).all()
            display(pd.DataFrame(manifests))
            display(provenance_audit)
            display(performance.tail(18))
            display(spectral.tail(24))
            """
        ),
        markdown("## Accuracy and alpha trajectories"),
        code(
            """
            performance_summary, _ = plot_seed_ci(
                performance,
                x="state_index",
                metric="test_accuracy",
                groups=("optimizer",),
                title="Monitoring-only test accuracy",
                ylabel="Test accuracy",
                output_path=OUTPUT_ROOT_PATH / "comparison" / "test_accuracy_95ci.png",
            )
            primary_all = spectral.copy()
            required_fit_columns = {"selection_role", "fit_variant"}
            missing_fit_columns = required_fit_columns.difference(primary_all.columns)
            if missing_fit_columns:
                raise KeyError(
                    "Strict fixed-point comparison requires columns "
                    f"{sorted(missing_fit_columns)}"
                )
            primary_all = primary_all[
                primary_all["selection_role"].astype(str).isin(
                    ["primary", "preregistered_primary"]
                )
                & primary_all["fit_variant"].astype(str).eq("clip_xmax")
            ].copy()
            if primary_all.empty:
                raise RuntimeError(
                    "No preregistered primary rows with exact fit_variant=clip_xmax exist"
                )
            alpha_plot = primary_all.copy()
            if "status" in alpha_plot:
                alpha_plot = alpha_plot[alpha_plot["status"].astype(str).eq("ok")]
            alpha_plot = alpha_plot[
                pd.to_numeric(alpha_plot["alpha"], errors="coerce").notna()
            ]
            alpha_summary, _ = plot_seed_ci(
                alpha_plot,
                x="state_index",
                metric="alpha",
                groups=("optimizer", "layer"),
                title="Raw-weight layer alpha: complete-run uncertainty",
                ylabel="Power-law density exponent alpha",
                reference=2.0,
                allow_incomplete=True,
                incomplete_output_path=(
                    OUTPUT_ROOT_PATH / "comparison" / "alpha_incomplete_ci_groups.csv"
                ),
                output_path=OUTPUT_ROOT_PATH / "comparison" / "alpha_95ci.png",
            )
            """
        ),
        markdown("## Independent-support trace-log audit"),
        code(
            """
            trace_column = next(
                (
                    name
                    for name in ("trace_log_per_eval", "trace_log_midpoint_per_eval")
                    if name in primary_all.columns
                ),
                None,
            )
            if trace_column is None:
                raise KeyError(
                    "Spectral table has no trace-log-per-evaluation column. "
                    "Run strict offline spectral analysis first."
                )
            if "support_selected_from_same_trace_log" not in primary_all:
                raise KeyError(
                    "Trace-log rows must declare support_selected_from_same_trace_log."
                )
            same_curve = primary_all["support_selected_from_same_trace_log"]
            if same_curve.dtype != bool:
                same_curve = same_curve.astype(str).str.strip().str.lower().isin(
                    {"1", "true", "yes"}
                )
            independent_all = primary_all[~same_curve].copy()
            independent_all["support_selected_from_same_trace_log"] = False
            if independent_all.empty:
                raise RuntimeError("No independently supported trace-log rows remain")
            trace_plot = independent_all[
                pd.to_numeric(independent_all[trace_column], errors="coerce").notna()
            ].copy()
            if "status" in trace_plot:
                trace_plot = trace_plot[trace_plot["status"].astype(str).eq("ok")]
            trace_summary, _ = plot_seed_ci(
                trace_plot,
                x="state_index",
                metric=trace_column,
                groups=("optimizer", "layer"),
                title="Independent-support trace-log",
                ylabel="Trace-log per retained mode",
                reference=0.0,
                allow_incomplete=True,
                incomplete_output_path=(
                    OUTPUT_ROOT_PATH / "comparison"
                    / "trace_log_incomplete_ci_groups.csv"
                ),
                output_path=OUTPUT_ROOT_PATH / "comparison" / "trace_log_95ci.png",
            )
            """
        ),
        markdown("## Legacy raw-midpoint trace-log compatibility audit (non-certifying)"),
        code(
            """
            compatibility_dir = OUTPUT_ROOT_PATH / "comparison"
            compatibility_dir.mkdir(parents=True, exist_ok=True)
            source_column = next(
                (
                    name for name in ("support_rank_source", "support_source")
                    if name in legacy_trace.columns
                ),
                None,
            )
            legacy_trace_metric = next(
                (
                    name for name in (
                        "trace_log_per_eval", "trace_log_midpoint_per_eval"
                    ) if name in legacy_trace.columns
                ),
                None,
            )
            legacy_raw_midpoint = pd.DataFrame()
            if (
                source_column is not None
                and legacy_trace_metric is not None
                and "fit_variant" in legacy_trace
            ):
                legacy_raw_midpoint = legacy_trace[
                    legacy_trace["fit_variant"].astype(str).eq("raw")
                    & legacy_trace[source_column].astype(str).eq("weightwatcher_midpoint")
                ].copy()
            if legacy_raw_midpoint.empty:
                legacy_compatibility_status = pd.DataFrame([{
                    "operator_kind": "legacy_raw_weightwatcher_midpoint_trace_log",
                    "map_definition": (
                        "historical raw WeightWatcher midpoint support; compatibility "
                        "audit only and unavailable in these trace_log.csv artifacts"
                    ),
                    "available": False,
                    "certification_eligible": False,
                    "reason": "no exact fit_variant=raw, support_source=weightwatcher_midpoint rows",
                }])
                legacy_summary = pd.DataFrame()
            else:
                if legacy_trace_metric != "trace_log_per_eval":
                    legacy_raw_midpoint["trace_log_per_eval"] = pd.to_numeric(
                        legacy_raw_midpoint[legacy_trace_metric], errors="coerce"
                    )
                legacy_raw_midpoint["operator_kind"] = (
                    "legacy_raw_weightwatcher_midpoint_trace_log"
                )
                legacy_raw_midpoint["map_definition"] = (
                    "historical raw WeightWatcher fit with midpoint-defined trace support; "
                    "baseline compatibility audit, not independent-support certification"
                )
                legacy_raw_midpoint["certification_eligible"] = False
                if "state_index" not in legacy_raw_midpoint:
                    legacy_raw_midpoint["state_index"] = pd.to_numeric(
                        legacy_raw_midpoint.get("epoch"), errors="coerce"
                    )
                legacy_summary, _ = plot_seed_ci(
                    legacy_raw_midpoint,
                    x="state_index",
                    metric="trace_log_per_eval",
                    groups=("optimizer", "layer"),
                    title="Legacy raw WeightWatcher midpoint trace-log (audit only)",
                    ylabel="Legacy trace-log per retained mode",
                    reference=0.0,
                    allow_incomplete=True,
                    incomplete_output_path=(
                        compatibility_dir / "legacy_raw_midpoint_incomplete_ci_groups.csv"
                    ),
                    output_path=(
                        compatibility_dir / "legacy_raw_midpoint_trace_log_95ci.png"
                    ),
                )
                legacy_compatibility_status = pd.DataFrame([{
                    "operator_kind": "legacy_raw_weightwatcher_midpoint_trace_log",
                    "map_definition": legacy_raw_midpoint["map_definition"].iloc[0],
                    "available": True,
                    "certification_eligible": False,
                    "reason": "historical compatibility observable only",
                }])
            legacy_raw_midpoint.to_csv(
                compatibility_dir / "legacy_raw_midpoint_trace_log_rows.csv", index=False
            )
            legacy_summary.to_csv(
                compatibility_dir / "legacy_raw_midpoint_trace_log_95ci.csv", index=False
            )
            legacy_compatibility_status.to_csv(
                compatibility_dir / "legacy_raw_midpoint_status.csv", index=False
            )
            display(legacy_compatibility_status)
            if not legacy_summary.empty:
                display(legacy_summary)
            """
        ),
        markdown("## Preregistered late-state qualification table"),
        code(
            """
            # Build the exact preregistered last-five state grid from the
            # analysis-state performance table. A missing spectral row stays in
            # the grid as NaN and fails; it cannot be replaced by an older fit.
            for frame_name, frame in (
                ("performance", performance), ("primary spectral", independent_all)
            ):
                if "step" not in frame:
                    if "global_step" in frame:
                        frame["step"] = pd.to_numeric(
                            frame["global_step"], errors="raise"
                        )
                    else:
                        raise KeyError(f"{frame_name} table has no step/global_step column")
                frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(int)
            primary_identity = ["optimizer", "seed", "layer", "step"]
            duplicated = independent_all.duplicated(primary_identity, keep=False)
            if duplicated.any():
                raise RuntimeError(
                    "Duplicate preregistered primary rows would corrupt the exact late grid:\\n"
                    + independent_all.loc[duplicated, primary_identity]
                    .sort_values(primary_identity)
                    .to_string(index=False)
                )
            expected_layers = ("fc1.weight", "fc2.weight", "fc3.weight")
            late_grid_rows = []
            for optimizer in OPTIMIZER_SLUGS:
                for seed in SEEDS:
                    states = performance[
                        performance["optimizer"].astype(str).eq(optimizer)
                        & pd.to_numeric(performance["seed"], errors="coerce").eq(seed)
                    ].sort_values("step")
                    if states["step"].duplicated().any():
                        raise RuntimeError(
                            f"Duplicate performance analysis step for {optimizer}, seed={seed}"
                        )
                    states = states.tail(5)
                    if len(states) != 5:
                        raise RuntimeError(
                            f"Expected five scheduled late performance states for {optimizer}, "
                            f"seed={seed}; found {len(states)}"
                        )
                    for _, state in states.iterrows():
                        for layer in expected_layers:
                            late_grid_rows.append({
                                "optimizer": optimizer,
                                "seed": int(seed),
                                "layer": layer,
                                "step": int(state["step"]),
                                "expected_epoch": int(state.get("epoch", state["state_index"])),
                                "expected_state_index": int(state["state_index"]),
                            })
            late_grid = pd.DataFrame(late_grid_rows)
            qualification_input = late_grid.merge(
                independent_all,
                on=primary_identity,
                how="left",
                validate="one_to_one",
                indicator="late_grid_merge_status",
            )
            if "ks_D" not in qualification_input and "D" in qualification_input:
                qualification_input["ks_D"] = qualification_input["D"]
            required_status_columns = {"status", "fit_ok", "trace_log_status"}
            missing_status_columns = required_status_columns.difference(
                qualification_input.columns
            )
            if missing_status_columns:
                raise KeyError(
                    "Strict fixed-point qualification requires fit/trace status columns: "
                    f"{sorted(missing_status_columns)}"
                )
            failed_measurement = (
                qualification_input["late_grid_merge_status"].astype(str).ne("both")
                | qualification_input["status"].astype(str).str.lower().ne("ok")
                | ~boolean_series(qualification_input["fit_ok"])
                | qualification_input["trace_log_status"].astype(str).str.lower().ne("ok")
            )
            qualification_input["measurement_valid_for_qualification"] = ~failed_measurement
            for metric in ("alpha", "ks_D", "n_tail", trace_column):
                qualification_input.loc[failed_measurement, metric] = np.nan
            if trace_column != "trace_log_per_eval":
                qualification_input["trace_log_per_eval"] = qualification_input[trace_column]
            qualification = reporting.qualify_fixed_point(
                qualification_input,
                alpha_target=2.0,
                alpha_half_width=ALPHA_TOLERANCE,
                max_ks_D=MAXIMUM_KS_D,
                minimum_tail=MINIMUM_TAIL,
                trace_log_tolerance=TRACE_LOG_PER_EVAL_TOLERANCE,
                persistence_measurements=5,
                required_fraction=0.80,
            ).sort_values(["optimizer", "layer", "seed"])
            expected_layers = set(expected_layers)
            arm_seed_rows = []
            for (optimizer, seed), group in qualification.groupby(
                ["optimizer", "seed"], dropna=False
            ):
                observed_layers = set(group["layer"].astype(str))
                if observed_layers != expected_layers:
                    raise RuntimeError(
                        f"Fixed-point verdict requires all layers for {optimizer}, "
                        f"seed={seed}; observed={sorted(observed_layers)}"
                    )
                arm_seed_rows.append({
                    "optimizer": optimizer,
                    "seed": int(seed),
                    "layers_required": len(expected_layers),
                    "layers_qualified": int(group["fixed_point_qualified"].astype(bool).sum()),
                    "all_layers_qualified": bool(group["fixed_point_qualified"].astype(bool).all()),
                    "fc3_low_rank_warning": True,
                })
            arm_seed_verdict = pd.DataFrame(arm_seed_rows)
            expected_arm_seed_rows = len(OPTIMIZER_SLUGS) * len(SEEDS)
            if len(arm_seed_verdict) != expected_arm_seed_rows:
                raise RuntimeError(
                    f"Expected {expected_arm_seed_rows} optimizer/seed spectral verdicts; "
                    f"found {len(arm_seed_verdict)}"
                )
            for optimizer in OPTIMIZER_SLUGS:
                observed_seeds = set(
                    arm_seed_verdict.loc[
                        arm_seed_verdict["optimizer"].astype(str).eq(optimizer), "seed"
                    ].astype(int)
                )
                if observed_seeds != set(SEEDS):
                    raise RuntimeError(
                        f"Spectral verdict for {optimizer} has seeds "
                        f"{sorted(observed_seeds)}, expected {sorted(SEEDS)}"
                    )
            optimizer_verdict = (
                arm_seed_verdict.groupby("optimizer", as_index=False)
                .agg(
                    seeds_present=("seed", "nunique"),
                    seeds_all_layers_qualified=("all_layers_qualified", "sum"),
                    all_three_seeds_reproduce=("all_layers_qualified", "all"),
                )
            )
            if set(optimizer_verdict["optimizer"].astype(str)) != set(OPTIMIZER_SLUGS):
                raise RuntimeError("Optimizer spectral verdict is missing an arm")
            if not optimizer_verdict["seeds_present"].eq(len(SEEDS)).all():
                raise RuntimeError("Optimizer verdict is missing a preregistered seed")
            baseline_verdict = {
                "require_all_layers": True,
                "require_all_three_seeds": True,
                "all_optimizer_arms_reproduce": bool(
                    optimizer_verdict["all_three_seeds_reproduce"].all()
                ),
                "accuracy_role": "descriptive monitoring-only; no qualification threshold",
            }
            comparison_dir = OUTPUT_ROOT_PATH / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            qualification.to_csv(
                comparison_dir / "late_state_preregistered_qualification.csv",
                index=False,
            )
            qualification_input.to_csv(
                comparison_dir / "late_state_exact_grid_audit.csv", index=False
            )
            arm_seed_verdict.to_csv(
                comparison_dir / "arm_seed_all_layers_verdict.csv", index=False
            )
            optimizer_verdict.to_csv(
                comparison_dir / "optimizer_three_seed_reproducibility.csv", index=False
            )
            provenance_audit.to_csv(
                comparison_dir / "cross_run_provenance_audit.csv", index=False
            )
            (comparison_dir / "baseline_verdict.json").write_text(
                json.dumps(baseline_verdict, indent=2) + "\\n", encoding="utf-8"
            )
            performance_summary.to_csv(
                comparison_dir / "performance_summary_95ci.csv", index=False
            )
            alpha_summary.to_csv(comparison_dir / "alpha_summary_95ci.csv", index=False)
            trace_summary.to_csv(
                comparison_dir / "trace_log_summary_95ci.csv", index=False
            )
            display(qualification)
            display(arm_seed_verdict)
            display(optimizer_verdict)
            display(baseline_verdict)
            """
        ),
        markdown("## Final raw-ESD PDF/CCDF panels with fit-window markers"),
        code(
            """
            panel_dir = comparison_dir / "final_esd_pdf_ccdf"
            panel_dir.mkdir(parents=True, exist_ok=True)
            if "step" not in primary_all:
                if "global_step" not in primary_all:
                    raise KeyError("Primary spectral rows lack step/global_step")
                primary_all["step"] = pd.to_numeric(
                    primary_all["global_step"], errors="raise"
                ).astype(int)
            final_grid_rows = []
            for optimizer in OPTIMIZER_SLUGS:
                for seed in SEEDS:
                    seed_dir = resolve_seed_dir(optimizer, seed)
                    resolved_payload = json.loads(
                        require_path(
                            Path(seed_dir) / "resolved_config.json",
                            description="resolved training config",
                        ).read_text(encoding="utf-8")
                    )
                    resolved_values = dict(
                        resolved_payload.get("config", resolved_payload)
                    )
                    _, _, completion = validate_run_identity(
                        seed_dir, optimizer_slug=optimizer, seed=seed
                    )
                    expected_epoch = int(resolved_values["epochs"])
                    expected_step = int(completion["global_step"])
                    refs = tuple(
                        list_analysis_checkpoints(Path(seed_dir) / "checkpoints")
                    )
                    if not refs:
                        raise RuntimeError(f"No analysis checkpoints beneath {seed_dir}")
                    if (
                        int(refs[-1].epoch) != expected_epoch
                        or int(refs[-1].global_step) != expected_step
                    ):
                        raise RuntimeError(
                            f"Final immutable checkpoint does not match completion for {seed_dir}"
                        )
                    exact_performance = performance[
                        performance["optimizer"].astype(str).eq(optimizer)
                        & pd.to_numeric(performance["seed"], errors="coerce").eq(seed)
                        & pd.to_numeric(performance["epoch"], errors="coerce").eq(expected_epoch)
                        & pd.to_numeric(performance["step"], errors="coerce").eq(expected_step)
                    ]
                    if len(exact_performance) != 1:
                        raise RuntimeError(
                            f"Expected one exact final performance row for {optimizer}, "
                            f"seed={seed}, epoch={expected_epoch}, step={expected_step}; "
                            f"found {len(exact_performance)}"
                        )
                    for layer in ("fc1.weight", "fc2.weight", "fc3.weight"):
                        final_grid_rows.append({
                            "optimizer": optimizer, "seed": int(seed), "layer": layer,
                            "epoch": expected_epoch, "step": expected_step,
                        })
            final_grid = pd.DataFrame(final_grid_rows)
            final_rows = final_grid.merge(
                primary_all,
                on=["optimizer", "seed", "layer", "epoch", "step"],
                how="left",
                validate="one_to_one",
                indicator="final_primary_merge_status",
            )
            missing_final = final_rows[
                final_rows["final_primary_merge_status"].astype(str).ne("both")
            ]
            if not missing_final.empty:
                raise RuntimeError(
                    "Missing exact-final preregistered primary spectral rows; earlier "
                    "states are never backfilled:\\n"
                    + missing_final[["optimizer", "seed", "layer", "epoch", "step"]]
                    .to_string(index=False)
                )
            panel_index = []
            for _, row in final_rows.iterrows():
                optimizer = str(row["optimizer"])
                seed = int(row["seed"])
                layer = str(row["layer"])
                step = int(row.get("step", row.get("global_step", row["state_index"])))
                epoch = int(row.get("epoch", row["state_index"]))
                seed_dir = resolve_seed_dir(optimizer, seed)
                esd_path = require_path(
                    seed_dir / "metrics" / "esd"
                    / f"esd_epoch_{epoch:05d}_step_{step:09d}.npz",
                    description="final raw weight ESD archive",
                )
                with np.load(esd_path) as archive:
                    if layer not in archive:
                        raise KeyError(f"{esd_path} lacks {layer}; keys={archive.files}")
                    values = np.asarray(archive[layer], dtype=float)
                safe_layer = layer.replace(".", "_")
                target = panel_dir / f"{optimizer}_seed_{seed}_{safe_layer}.png"
                finger = row.get("num_fingers", row.get("finger_count", "unknown"))
                variant = row.get("fit_variant", "unknown")
                fig = plotting.plot_pdf_ccdf(
                    values,
                    fit_row=row,
                    title=(
                        f"{optimizer} | seed={seed} | {layer} | "
                        f"variant={variant}, fingers={finger}"
                    ),
                    output=target,
                )
                if SHOW_PLOTS:
                    plt.show()
                else:
                    plt.close(fig)
                panel_index.append({
                    "optimizer": optimizer, "seed": seed, "layer": layer,
                    "epoch": epoch, "step": step, "fit_variant": variant,
                    "finger_policy": "WeightWatcher fix_fingers=clip_xmax",
                    "num_fingers": finger, "xmin": row.get("xmin", np.nan),
                    "xmax": row.get("xmax", np.nan), "figure": str(target),
                    "fit_status": row.get("status", "unknown"),
                    "fit_warning": row.get("warning", ""),
                })
            panel_index = pd.DataFrame(panel_index)
            expected_panels = len(OPTIMIZER_SLUGS) * len(SEEDS) * 3
            if len(panel_index) != expected_panels:
                raise RuntimeError(
                    f"Expected {expected_panels} final layer/arm/seed panels; got {len(panel_index)}"
                )
            panel_index.to_csv(comparison_dir / "final_esd_panel_index.csv", index=False)
            display(panel_index)
            """
        ),
        markdown(
            """
            A failed row falsifies the strict late-state qualification for that
            seed/layer under the declared tolerance; it is not repaired by
            pooling layers or selecting a different checkpoint. A passing FC3
            row remains weak because FC3 has at most ten positive modes.
            """
        ),
    ]
    return notebook("04_Fixed_Point_Comparison.ipynb", cells)


ANALYSIS_IMPORTS = r"""
from rg_baselines.tangent_rg import (
    AdamWProfile,
    MuonClipRMSProfile,
    MuonProfile,
    TangentRGConfig,
    load_analysis_checkpoint,
    list_analysis_checkpoints,
    list_capture_files,
    load_step_capture,
    replay_calibrated_step,
)
from rg_baselines.tangent_rg.checkpoints import load_verified_tail_checkpoint_refs
from rg_baselines.tangent_rg.protocol import tail_checkpoint_epochs
from rg_baselines.tangent_rg import nulls, polar, single_checkpoint, stiefel, two_checkpoint
"""


ANALYSIS_SAVE_AND_PLOT = r"""
operator_rows = pd.DataFrame(operator_records)
fit_rows = pd.concat(fit_frames, ignore_index=True, sort=False)
trace_rows = pd.concat(trace_frames, ignore_index=True, sort=False)
required_provenance = {"operator_kind", "map_definition"}
for name, frame in {
    "operator rows": operator_rows,
    "fit rows": fit_rows,
    "trace rows": trace_rows,
}.items():
    missing = required_provenance - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} missing provenance columns: {sorted(missing)}")
    if frame[list(required_provenance)].isna().any().any():
        raise RuntimeError(f"{name} has null provenance")
analysis_dir = save_analysis_frames(
    METHOD_SLUG,
    operators=operator_rows,
    fits=fit_rows,
    traces=trace_rows,
)
alpha_summary, _ = plot_fit_alpha_ci(
    fit_rows,
    method_slug=METHOD_SLUG,
    title=PLOT_TITLE,
)
alpha_summary.to_csv(analysis_dir / "alpha_summary_95ci.csv", index=False)
display(operator_rows.head(24))
display(fit_rows.head(24))
display(trace_rows.head(24))
print("analysis outputs:", analysis_dir)
"""


def two_checkpoint_notebook() -> tuple[str, dict[str, object]]:
    cells = [
        markdown(
            """
            # Two-checkpoint finite flow and beta surrogates

            Read the independently persisted final-100 checkpoint cache and
            evaluate every selected chronological pair without
            relabelling a secant or finite transfer as a training Jacobian.
            The rectangular radial and Grassmann sectors are retained beside
            the raw displacement so a promising plot cannot hide a failed
            conditioning or support audit.
            """
        ),
        parameters(
            """
            OPTIMIZER_SLUGS = ["adamw", "muon", "muonclip_rms"]
            LAYERS = ["fc1.weight", "fc2.weight", "fc3.weight"]
            MAXIMUM_PAIRS = 100
            PAIR_STRIDES = [1]
            RUN_SPACING_SENSITIVITY = True
            SPACING_SENSITIVITY_STRIDES = [2, 4, 8]
            SPACING_SENSITIVITY_PAIRS_PER_STRIDE = 8
            TOP_K_VALUES = [0, 1, 2, 3, 4, 5]
            MINIMUM_TAIL = 8
            METHOD_SLUG = "two_checkpoint_finite_flow"
            PLOT_TITLE = "Two-checkpoint spectra: alpha with 95% seed CI"
            """
        ),
        code(BOOTSTRAP),
        code(COMMON_IMPORTS + "\n" + ANALYSIS_IMPORTS),
        caveat(
            "Finite-flow object",
            "two_checkpoint_finite_flow_bundle",
            "W0,W1 -> secant, conditioned square transfer, supported Gram rates, and Grassmann principal-angle rates.",
            "Two points identify a displacement and several declared finite maps, not d(beta)/dW. "
            "Checkpoint spacing is therefore a required sensitivity axis.",
        ),
        code(COMMON_HELPERS + "\n" + TAIL_CHECKPOINT_CACHE_HELPERS + "\n" + CHECKPOINT_HELPERS),
        markdown(
            "## Verify each completed tail-checkpoint cache and evaluate declared pairs\n\n"
            "This notebook never launches or resumes training. Missing, partial, stale, "
            "or fingerprint-mismatched caches are fatal prerequisites."
        ),
        code(
            """
            operator_records, fit_frames, trace_frames = [], [], []
            spectral_arrays = {}
            for optimizer in OPTIMIZER_SLUGS:
                for seed in SEEDS:
                    seed_dir = require_tail_checkpoint_cache(optimizer, seed)
                    run_fingerprint = verified_run_fingerprint(optimizer, seed)
                    pairs = selected_checkpoint_pairs_for_strides(
                        seed_dir,
                        strides=PAIR_STRIDES,
                        maximum_pairs=MAXIMUM_PAIRS,
                    )
                    if RUN_SPACING_SENSITIVITY:
                        overlap = set(PAIR_STRIDES) & set(SPACING_SENSITIVITY_STRIDES)
                        if overlap:
                            raise ValueError(f"Primary/sensitivity pair strides overlap: {overlap}")
                        pairs.extend(
                            selected_checkpoint_pairs_for_strides(
                                seed_dir,
                                strides=SPACING_SENSITIVITY_STRIDES,
                                maximum_pairs=SPACING_SENSITIVITY_PAIRS_PER_STRIDE,
                            )
                        )
                    declared_strides = list(PAIR_STRIDES) + (
                        list(SPACING_SENSITIVITY_STRIDES)
                        if RUN_SPACING_SENSITIVITY else []
                    )
                    pair_counts = {
                        int(stride): sum(1 for item in pairs if int(item[0]) == int(stride))
                        for stride in declared_strides
                    }
                    print(
                        f"{optimizer} seed={seed}: "
                        f"{len(analysis_checkpoint_refs(seed_dir))} verified cache states, "
                        f"pair counts by stride={pair_counts}, payload LRU="
                        f"{CHECKPOINT_PAYLOAD_CACHE_SIZE}"
                    )
                    for pair_stride, previous_ref, current_ref, pair_selection_rule, pair_selection_role in pairs:
                        delta_s = int(current_ref.global_step - previous_ref.global_step)
                        if delta_s <= 0:
                            raise RuntimeError("analysis checkpoint steps are not increasing")
                        for layer in LAYERS:
                            W0 = checkpoint_matrix(previous_ref.path, layer)
                            W1 = checkpoint_matrix(current_ref.path, layer)
                            result = two_checkpoint.analyze_two_checkpoints(W0, W1, delta_s)
                            base = {
                                "optimizer": optimizer,
                                "seed": int(seed),
                                "protocol_fingerprint": run_fingerprint,
                                "source_artifact_kind": "verified_tail_checkpoint_cache_model_only",
                                "layer": layer,
                                "state_index": int(current_ref.global_step),
                                "epoch0": int(previous_ref.epoch),
                                "epoch1": int(current_ref.epoch),
                                "step0": int(previous_ref.global_step),
                                "step1": int(current_ref.global_step),
                                "delta_s": delta_s,
                                "pair_stride": int(pair_stride),
                                "pair_spacing_role": (
                                    "primary_adjacent"
                                    if int(pair_stride) in set(PAIR_STRIDES)
                                    else "bounded_spacing_sensitivity"
                                ),
                                "pair_selection_rule": pair_selection_rule,
                                "pair_selection_role": pair_selection_role,
                                "checkpoint_source": "verified_final_100_tail_cache",
                                "checkpoint_cache_seed_dir": str(seed_dir),
                            }
                            operator_records.extend([
                                {**base, "method": "beta_secant", "operator_kind": result.beta.operator_kind,
                                 "map_definition": result.beta.map_definition,
                                 "frobenius_norm": result.beta.beta_norm,
                                 "relative_delta_norm": result.beta.relative_delta_norm},
                                {**base, "method": "square_transfer", "operator_kind": result.square_transfer.operator_kind,
                                 "map_definition": result.square_transfer.map_definition,
                                 "available": result.square_transfer.available,
                                 "condition_number": result.square_transfer.condition_number,
                                 "reconstruction_residual": result.square_transfer.relative_reconstruction_residual},
                                {**base, "method": "aligned_rectangular_transfer", "operator_kind": result.rectangular_transfer.operator_kind,
                                 "map_definition": result.rectangular_transfer.map_definition,
                                 "available": result.rectangular_transfer.available,
                                 "orientation": result.rectangular_transfer.orientation,
                                 "effective_rank": result.rectangular_transfer.effective_rank,
                                 "numerical_rank0": result.rectangular_transfer.numerical_rank0,
                                 "numerical_rank1": result.rectangular_transfer.numerical_rank1,
                                 "rank_rtol": result.rectangular_transfer.rank_rtol,
                                 "rank_threshold0": result.rectangular_transfer.rank_threshold0,
                                 "rank_threshold1": result.rectangular_transfer.rank_threshold1,
                                 "structural_zero_count": result.rectangular_transfer.structural_zero_count,
                                 "condition_number0": result.rectangular_transfer.condition_number0,
                                 "condition_number1": result.rectangular_transfer.condition_number1,
                                 "reconstruction_residual": result.rectangular_transfer.relative_reconstruction_residual,
                                 "unsupported_source_action_residual": result.rectangular_transfer.unsupported_source_action_residual,
                                 "core_reconstruction_residual": result.rectangular_transfer.core_reconstruction_residual,
                                 "subspace_alignment_residual": result.rectangular_transfer.subspace_alignment_residual,
                                 "forced_intersection_zeros": result.rectangular_transfer.forced_intersection_zeros,
                                 "operator_materialized": result.rectangular_transfer.operator_materialized},
                                {**base, "method": "gram_radial", "operator_kind": result.radial.operator_kind,
                                 "map_definition": result.radial.map_definition,
                                 "retained_rank": result.radial.retained_rank,
                                 "dropped_null": result.radial.dropped_initial_null_directions},
                            ])
                            candidates = [
                                ("beta_secant_amplitude", np.linalg.svd(result.beta.beta_surrogate, compute_uv=False),
                                 result.beta.operator_kind, result.beta.map_definition + "; fitted b are singular amplitudes and energy e=b^2 is an exact fit transform"),
                                ("radial_rate_amplitude", result.radial.radial_rate_amplitudes,
                                 result.radial.operator_kind, result.radial.map_definition + "; fitted b are nonnegative rate amplitudes and energy e=b^2 is an exact fit transform"),
                                ("column_grassmann_amplitude", result.grassmann.column.geodesic_rates,
                                 result.grassmann.column.operator_kind, result.grassmann.column.map_definition + "; fitted b are geodesic-rate amplitudes and energy e=b^2 is an exact fit transform"),
                                ("row_grassmann_amplitude", result.grassmann.row.geodesic_rates,
                                 result.grassmann.row.operator_kind, result.grassmann.row.map_definition + "; fitted b are geodesic-rate amplitudes and energy e=b^2 is an exact fit transform"),
                            ]
                            if result.square_transfer.available:
                                candidates.append(
                                    ("square_transfer_amplitude", np.linalg.svd(result.square_transfer.operator, compute_uv=False),
                                     result.square_transfer.operator_kind, result.square_transfer.map_definition + "; fitted b are singular amplitudes and energy e=b^2 is an exact fit transform")
                                )
                            if result.rectangular_transfer.available:
                                candidates.extend([
                                    (
                                        "supported_rectangular_transfer_amplitude",
                                        result.rectangular_transfer.supported_transfer_singular_values,
                                        result.rectangular_transfer.operator_kind,
                                        result.rectangular_transfer.map_definition
                                        + "; fit only nonzero supported finite-transfer singular amplitudes; "
                                        "ambient structural zeros are counted, not fitted; energy e=b^2 is an exact fit transform",
                                    ),
                                    (
                                        "supported_rectangular_transfer_log_rate_amplitude",
                                        result.rectangular_transfer.supported_transfer_rate_amplitudes,
                                        result.rectangular_transfer.operator_kind,
                                        result.rectangular_transfer.map_definition
                                        + "; fitted b are absolute supported log-singular-rate amplitudes; "
                                        "energy e=b^2 is an exact fit transform",
                                    ),
                                    (
                                        "procrustes_aligned_core_transfer_amplitude",
                                        result.rectangular_transfer.core_singular_values,
                                        result.rectangular_transfer.operator_kind,
                                        result.rectangular_transfer.map_definition
                                        + "; separate gauge-aligned rank-r core singular amplitudes; "
                                        "not the ambient transfer and not a Jacobian; energy e=b^2 is an exact fit transform",
                                    ),
                                    (
                                        "procrustes_aligned_core_log_rate_amplitude",
                                        result.rectangular_transfer.core_rate_amplitudes,
                                        result.rectangular_transfer.operator_kind,
                                        result.rectangular_transfer.map_definition
                                        + "; fitted b are absolute log-singular-rate amplitudes of the "
                                        "gauge-aligned rank-r core; energy e=b^2 is an exact fit transform",
                                    ),
                                    (
                                        "aligned_transfer_principal_angle_rate_amplitude",
                                        result.rectangular_transfer.principal_angle_rates,
                                        result.rectangular_transfer.operator_kind,
                                        result.rectangular_transfer.map_definition
                                        + "; fitted b are nonzero Procrustes alignment principal-angle "
                                        "rate amplitudes with forced intersection zeros counted separately; "
                                        "energy e=b^2 is an exact fit transform",
                                    ),
                                ])
                            for method, values, kind, definition in candidates:
                                positive = np.asarray(values, dtype=float)
                                positive = positive[np.isfinite(positive) & (positive > 0)]
                                if positive.size < 2:
                                    continue
                                metadata = {**base, "method": method,
                                            "fc3_rank10_warning": "fc3" in layer.lower()}
                                fits, traces = fit_spectrum_with_trace(
                                    positive,
                                    operator_kind=kind,
                                    map_definition=definition,
                                    spectrum_kind="amplitude",
                                    metadata=metadata,
                                    top_k_values=TOP_K_VALUES,
                                    minimum_tail=MINIMUM_TAIL,
                                )
                                fit_frames.append(fits)
                                trace_frames.append(traces)
                                spectral_arrays[f"{optimizer}_{seed}_{layer}_{current_ref.global_step}_{method}"] = positive
            if not operator_records:
                raise RuntimeError("No two-checkpoint operator rows were produced")
            if not fit_frames:
                raise RuntimeError("No two-checkpoint spectrum had at least two positive modes")
            """
        ),
        code(ANALYSIS_SAVE_AND_PLOT),
        code(
            """
            np.savez_compressed(analysis_dir / "positive_spectra.npz", **spectral_arrays)
            display(save_spectrum_ccdf_gallery(spectral_arrays, method_slug=METHOD_SLUG))
            spacing = fit_rows[
                fit_rows["clip_top_k"].eq(0)
                & fit_rows["spectrum_kind"].astype(str).eq("energy_derived_from_amplitude")
            ].copy()
            spacing_summary = ci_summary(
                spacing,
                groups=("optimizer", "layer", "method", "pair_stride", "delta_s"),
                metrics=("alpha", "ks_D", "n_tail"),
            )
            spacing_summary.to_csv(analysis_dir / "spacing_sensitivity_95ci.csv", index=False)
            display(spacing_summary.tail(30))
            """
        ),
        markdown(
            """
            Retain a finite-flow method only if its conclusion survives the
            checkpoint-spacing table, conditioning audit, explicit top-k
            sensitivity rows, and matched null notebook. A beta secant that
            looks heavy-tailed is still not a beta-function Jacobian.
            """
        ),
    ]
    return notebook("10_Two_Checkpoint_Finite_Flow.ipynb", cells)


def stiefel_tangent_notebook() -> tuple[str, dict[str, object]]:
    cells = [
        markdown(
            """
            # Muon update source on the Stiefel tangent space

            Use persisted one-step captures to distinguish the Muon source,
            ideal polar response, Stiefel projection, implemented
            Newton--Schulz direction, and actual parameter delta. MuonClip-RMS
            is audited for exact nonzero direction RMS `0.20`; QK clipping is
            not applicable to this MLP. These quantities require optimizer
            internals that model-only tail checkpoints intentionally omit, so
            this notebook reads the already saved dense captures and never
            launches or resumes training.
            """
        ),
        parameters(
            """
            OPTIMIZER_SLUGS = ["muon", "muonclip_rms"]
            LAYERS = ["fc1.weight", "fc2.weight"]
            MAXIMUM_CAPTURES = 40
            SMALL_BLOCK_SHAPE = [8, 6]
            SMALL_JACOBIAN_CAPTURE_STRIDE = 8
            NS_STEPS = 5
            NS_EPS = 1e-7
            SOURCE_RANK_RCOND = 1e-4
            SOURCE_RANK_RCOND_SENSITIVITY = [1e-5, 1e-4, 1e-3]
            TOP_K_VALUES = [0, 1, 2, 3, 4, 5]
            MINIMUM_TAIL = 8
            RMS_TOLERANCE = 1e-6
            METHOD_SLUG = "muon_update_stiefel_tangent"
            PLOT_TITLE = "Muon polar/Stiefel spectra: alpha with 95% seed CI"
            """
        ),
        code(BOOTSTRAP),
        code(COMMON_IMPORTS + "\n" + ANALYSIS_IMPORTS),
        caveat(
            "Muon-source geometry",
            "ideal_muon_polar_source_and_stiefel_tangent_bundle",
            "Captured source M -> P(M)=UV^T, its Frechet response, and Pi_P(M)(Z) on the orientation-correct Stiefel tangent space.",
            "D P_M is the derivative of the ideal matrix polar map. It is neither the "
            "full optimizer-step Jacobian nor proof that finite-step NS5 has the same spectrum.",
        ),
        code(COMMON_HELPERS + "\n" + TAIL_CHECKPOINT_CACHE_HELPERS + "\n" + CHECKPOINT_HELPERS),
        markdown("## Require dense captures and measure polar/tangent sectors"),
        code(
            """
            operator_records, fit_frames, trace_frames = [], [], []
            spectral_arrays = {}
            for optimizer in OPTIMIZER_SLUGS:
                for seed in SEEDS:
                    seed_dir = require_complete_seed(optimizer, seed)
                    run_fingerprint = verified_run_fingerprint(optimizer, seed)
                    captures = capture_payloads(
                        seed_dir, maximum_captures=MAXIMUM_CAPTURES
                    )
                    for capture_index, (capture_path, capture) in enumerate(captures):
                        if "parameters" not in capture:
                            raise KeyError(f"{capture_path} lacks parameter captures")
                        step = int(capture["completed_step"])
                        epoch = int(capture["epoch_before_step"])
                        for layer in LAYERS:
                            if layer not in capture["parameters"]:
                                raise KeyError(f"{capture_path} lacks {layer}")
                            values = capture["parameters"][layer]
                            source = capture_array(values, "update_source")
                            perturbation = capture_array(values, "gradient_after_clipping")
                            applied_direction = capture_array(values, "applied_update_direction")
                            captured_polar = capture_array(values, "polar_update")
                            ideal_polar = polar.polar_factor(source)
                            numpy_ns5 = polar.muon_quintic_orthogonalizer(
                                source, steps=NS_STEPS, eps=NS_EPS
                            )
                            ideal_norm = max(np.linalg.norm(ideal_polar), np.finfo(float).tiny)
                            captured_norm = max(np.linalg.norm(captured_polar), np.finfo(float).tiny)
                            captured_vs_ideal = float(
                                np.linalg.norm(captured_polar - ideal_polar) / ideal_norm
                            )
                            captured_vs_numpy_ns5 = float(
                                np.linalg.norm(captured_polar - numpy_ns5)
                                / max(np.linalg.norm(numpy_ns5), np.finfo(float).tiny)
                            )
                            captured_ideal_cosine = float(
                                np.vdot(captured_polar, ideal_polar).real
                                / (captured_norm * ideal_norm)
                            )
                            source_singular = np.linalg.svd(source, compute_uv=False)
                            source_rank_tolerance = (
                                float(SOURCE_RANK_RCOND)
                                * max(float(source_singular[0]), 1.0)
                            )
                            source_rank = int(
                                np.count_nonzero(source_singular > source_rank_tolerance)
                            )
                            source_full_rank = source_rank == min(source.shape)
                            source_rank_sensitivity = {
                                str(float(rcond)): int(
                                    np.count_nonzero(
                                        source_singular
                                        > float(rcond) * max(float(source_singular[0]), 1.0)
                                    )
                                )
                                for rcond in SOURCE_RANK_RCOND_SENSITIVITY
                            }
                            response = (
                                stiefel.muon_polar_source_perturbation(
                                    source, perturbation, rcond=SOURCE_RANK_RCOND
                                )
                                if source_full_rank
                                else None
                            )
                            projection = stiefel.project_stiefel_tangent(
                                ideal_polar, applied_direction
                            )
                            pullback = (
                                polar.polar_pullback_spectrum(
                                    source, rcond=SOURCE_RANK_RCOND
                                )
                                if source_full_rank
                                else None
                            )
                            effective_rms = float(np.sqrt(np.mean(applied_direction ** 2)))
                            declared_rms = values.get("declared_rms_scale", np.nan)
                            base = {
                                "optimizer": optimizer,
                                "seed": int(seed),
                                "protocol_fingerprint": run_fingerprint,
                                "source_artifact_kind": "verified_dense_update_capture",
                                "layer": layer,
                                "state_index": step,
                                "epoch": epoch,
                                "capture": str(capture_path),
                                "fc3_rank10_warning": False,
                                "source_numerical_rank": source_rank,
                                "source_full_rank": bool(source_full_rank),
                                "polar_geometry_unique": bool(source_full_rank),
                                "source_rank_rcond": float(SOURCE_RANK_RCOND),
                                "source_rank_tolerance": source_rank_tolerance,
                                "source_rank_sensitivity": json.dumps(
                                    source_rank_sensitivity, sort_keys=True
                                ),
                            }
                            response_row = (
                                {**base, "method": "ideal_polar_directional_response",
                                 "operator_kind": response.operator_kind,
                                 "map_definition": response.map_definition,
                                 "available": True,
                                 "source_numerical_rank": source_rank,
                                 "source_rank_tolerance": source_rank_tolerance,
                                 "source_rank_rcond": float(SOURCE_RANK_RCOND),
                                 "source_rank_sensitivity": json.dumps(source_rank_sensitivity, sort_keys=True),
                                 "central_vs_analytic_relative_error": response.central_vs_analytic_relative_error,
                                 "tangent_residual": response.polar_response_tangent_residual,
                                 "ambient_projection_difference": response.ambient_projection_relative_difference}
                                if response is not None
                                else {**base, "method": "ideal_polar_directional_response",
                                 "operator_kind": "unavailable_rank_deficient_ideal_polar_derivative",
                                 "map_definition": "D P_M requires full rectangular rank; rank-deficient captured source retained while finite NS5 remains measurable",
                                 "available": False,
                                 "source_numerical_rank": source_rank,
                                 "source_rank_tolerance": source_rank_tolerance,
                                 "source_rank_rcond": float(SOURCE_RANK_RCOND),
                                 "source_rank_sensitivity": json.dumps(source_rank_sensitivity, sort_keys=True),
                                 "unavailable_reason": "captured source is not full rank on the smaller matrix side"}
                            )
                            operator_records.extend([
                                response_row,
                                {**base, "method": "captured_applied_direction_projection",
                                 "operator_kind": (
                                     projection.operator_kind if source_full_rank
                                     else "svd_completion_dependent_stiefel_projection_sensitivity"
                                 ),
                                 "map_definition": projection.map_definition + (
                                     "; rank-deficient source makes the ideal polar factor and "
                                     "associated tangent gauge SVD-completion-dependent"
                                     if not source_full_rank else ""
                                 ),
                                 "evidence_role": (
                                     "primary_unique_polar_geometry" if source_full_rank
                                     else "sensitivity_only_svd_completion"
                                 ),
                                 "tangent_residual": projection.tangent_constraint_residual,
                                 "normal_fraction": float(np.linalg.norm(projection.normal) / max(np.linalg.norm(applied_direction), np.finfo(float).tiny)),
                                 "effective_direction_rms": effective_rms,
                                 "declared_rms_scale": declared_rms},
                                {**base, "method": "captured_ns5_vs_ideal_polar",
                                 "operator_kind": "captured_finite_ns5_output_vs_ideal_polar_factor",
                                 "map_definition": (
                                     "same captured Muon source M: compare implemented NS5(M) with "
                                     "ideal P(M)=UV^T; output discrepancy, not a Jacobian; when M "
                                     "is rank deficient, the ideal factor uses an arbitrary SVD completion"
                                 ),
                                 "evidence_role": (
                                     "well_defined_finite_ns5_vs_svd_completion_sensitivity"
                                     if not source_full_rank else "finite_ns5_vs_unique_ideal_polar"
                                 ),
                                 "captured_vs_ideal_relative_error": captured_vs_ideal,
                                 "captured_vs_numpy_ns5_relative_error": captured_vs_numpy_ns5,
                                 "captured_ideal_cosine": captured_ideal_cosine,
                                 "ns_steps": int(NS_STEPS), "ns_eps": float(NS_EPS)},
                            ])
                            if optimizer == "muonclip_rms":
                                if not np.isfinite(declared_rms):
                                    raise RuntimeError("MuonClip-RMS capture lacks declared_rms_scale")
                                if abs(effective_rms - float(declared_rms)) > RMS_TOLERANCE:
                                    raise RuntimeError(
                                        f"MuonClip-RMS direction RMS mismatch at {capture_path}, {layer}: "
                                        f"observed={effective_rms}, declared={declared_rms}"
                                    )
                            projection_method = (
                                "captured_tangent_projection_amplitude"
                                if source_full_rank
                                else "svd_completion_tangent_projection_sensitivity_amplitude"
                            )
                            projection_kind = (
                                projection.operator_kind
                                if source_full_rank
                                else "svd_completion_dependent_stiefel_projection_sensitivity"
                            )
                            projection_definition = (
                                projection.map_definition
                                + "; fitted b are one-probe singular amplitudes and energy e=b^2 "
                                "is an exact fit transform"
                                + (
                                    "; rank-deficient source makes polar factor/tangent gauge "
                                    "SVD-completion-dependent; sensitivity only"
                                    if not source_full_rank else ""
                                )
                            )
                            candidates = [
                                (
                                    "captured_finite_ns5_output_amplitude",
                                    np.linalg.svd(captured_polar, compute_uv=False),
                                    "captured_finite_muon_ns5_output_singular_amplitudes",
                                    "captured implemented NS5(source) output; finite algorithmic map "
                                    "is well-defined even when the ideal polar derivative is non-unique",
                                    "well_defined_finite_map_control",
                                ),
                                (
                                    "captured_applied_direction_amplitude",
                                    np.linalg.svd(applied_direction, compute_uv=False),
                                    "captured_applied_muon_direction_singular_amplitudes",
                                    "captured post-NS5 and post-RMS-scaling direction actually supplied "
                                    "to the parameter update; not a training-map Jacobian",
                                    "well_defined_applied_direction_control",
                                ),
                                (
                                    projection_method,
                                    np.linalg.svd(projection.tangent, compute_uv=False),
                                    projection_kind,
                                    projection_definition,
                                    (
                                        "primary_unique_polar_geometry"
                                        if source_full_rank else "sensitivity_only_svd_completion"
                                    ),
                                ),
                            ]
                            if pullback is not None:
                                candidates.insert(
                                    0,
                                    ("polar_pullback_amplitude", pullback.singular_amplitudes,
                                     pullback.operator_kind, pullback.map_definition,
                                     "primary_unique_polar_geometry"),
                                )
                            if capture_index % int(SMALL_JACOBIAN_CAPTURE_STRIDE) == 0:
                                block_rows = min(int(SMALL_BLOCK_SHAPE[0]), source.shape[0])
                                block_cols = min(int(SMALL_BLOCK_SHAPE[1]), source.shape[1])
                                small_source = source[:block_rows, :block_cols].copy()
                                maximum_dimension = int(small_source.size)
                                ns_jacobian = polar.explicit_muon_newton_schulz_jacobian(
                                    small_source,
                                    steps=NS_STEPS,
                                    eps=NS_EPS,
                                    max_input_dimension=maximum_dimension,
                                    rank_rtol=SOURCE_RANK_RCOND,
                                )
                                small_singular = np.linalg.svd(small_source, compute_uv=False)
                                small_tolerance = (
                                    float(SOURCE_RANK_RCOND)
                                    * max(float(small_singular[0]), 1.0)
                                )
                                small_full_rank = int(
                                    np.count_nonzero(small_singular > small_tolerance)
                                ) == min(small_source.shape)
                                try:
                                    if not small_full_rank:
                                        raise np.linalg.LinAlgError(
                                            "small source block is rank deficient; ideal polar derivative is non-unique"
                                        )
                                    ideal_jacobian = polar.explicit_polar_jacobian(
                                        small_source,
                                        max_input_dimension=maximum_dimension,
                                        rank_rtol=SOURCE_RANK_RCOND,
                                    )
                                    compared = min(
                                        ns_jacobian.singular_values.size,
                                        ideal_jacobian.singular_values.size,
                                    )
                                    spectral_discrepancy = float(
                                        np.linalg.norm(
                                            ns_jacobian.singular_values[:compared]
                                            - ideal_jacobian.singular_values[:compared]
                                        )
                                        / max(
                                            np.linalg.norm(ideal_jacobian.singular_values[:compared]),
                                            np.finfo(float).tiny,
                                        )
                                    )
                                    ideal_rank = ideal_jacobian.numerical_rank
                                    ideal_unavailable = ""
                                except np.linalg.LinAlgError as error:
                                    ideal_jacobian = None
                                    spectral_discrepancy = np.nan
                                    ideal_rank = np.nan
                                    ideal_unavailable = f"{type(error).__name__}: {error}"
                                operator_records.append({
                                    **base,
                                    "method": "small_block_ns5_vs_ideal_jacobian",
                                    "operator_kind": "small_coordinate_block_ns5_vs_ideal_polar_jacobian_probe",
                                    "map_definition": (
                                        f"central-difference Jacobians on the fixed top-left {small_source.shape} "
                                        "source block, comparing configured finite NS5 with ideal polar; "
                                        "a local coordinate probe, not the full-layer Jacobian"
                                    ),
                                    "ns5_numerical_rank": ns_jacobian.numerical_rank,
                                    "ideal_numerical_rank": ideal_rank,
                                    "relative_singular_spectrum_error": spectral_discrepancy,
                                    "ideal_comparison_available": ideal_jacobian is not None,
                                    "ideal_unavailable_reason": ideal_unavailable,
                                    "source_rank_rcond": float(SOURCE_RANK_RCOND),
                                    "small_rank_tolerance": small_tolerance,
                                    "ns_steps": int(NS_STEPS), "ns_eps": float(NS_EPS),
                                })
                                candidates.append(
                                    ("small_block_ns5_jacobian_amplitude", ns_jacobian.singular_values,
                                     ns_jacobian.operator_kind,
                                     ns_jacobian.map_definition + f"; fixed block={small_source.shape}; local coordinate probe only",
                                     "well_defined_finite_map_local_coordinate_probe")
                                )
                                if ideal_jacobian is not None:
                                    candidates.append(
                                        ("small_block_ideal_polar_jacobian_amplitude", ideal_jacobian.singular_values,
                                         ideal_jacobian.operator_kind,
                                         ideal_jacobian.map_definition + f"; fixed block={small_source.shape}; local coordinate probe only",
                                         "unique_ideal_polar_local_coordinate_probe")
                                    )
                            for method, spectrum, kind, definition, evidence_role in candidates:
                                metadata = {
                                    **base, "method": method,
                                    "evidence_role": evidence_role,
                                }
                                fits, traces = fit_spectrum_with_trace(
                                    spectrum,
                                    operator_kind=kind,
                                    map_definition=definition,
                                    spectrum_kind="amplitude",
                                    metadata=metadata,
                                    top_k_values=TOP_K_VALUES,
                                    minimum_tail=MINIMUM_TAIL,
                                )
                                fit_frames.append(fits)
                                trace_frames.append(traces)
                                spectral_arrays[f"{optimizer}_{seed}_{layer}_{step}_{method}"] = np.asarray(spectrum)
            if not operator_records or not fit_frames:
                raise RuntimeError("No Stiefel/polar capture results were produced")
            """
        ),
        code(ANALYSIS_SAVE_AND_PLOT),
        code(
            """
            np.savez_compressed(analysis_dir / "positive_spectra.npz", **spectral_arrays)
            display(save_spectrum_ccdf_gallery(spectral_arrays, method_slug=METHOD_SLUG))
            numeric = operator_rows.select_dtypes(include=[np.number]).columns
            audit_metrics = [name for name in (
                "central_vs_analytic_relative_error", "tangent_residual",
                "ambient_projection_difference", "normal_fraction",
                "effective_direction_rms",
            ) if name in numeric]
            stability_parts = []
            for metric in audit_metrics:
                available = operator_rows[operator_rows[metric].notna()].copy()
                if available.empty:
                    continue
                stability_parts.append(
                    ci_summary(
                        available,
                        groups=tuple(
                            name for name in (
                                "optimizer", "layer", "method", "evidence_role"
                            ) if name in available
                        ),
                        metrics=(metric,),
                    )
                )
            stability = pd.concat(stability_parts, ignore_index=True, sort=False)
            stability.to_csv(analysis_dir / "geometry_audit_95ci.csv", index=False)
            display(stability)
            """
        ),
        markdown(
            """
            A heavy tail in the analytic polar pullback is a property of
            `D P_M`; a tail in a single projected captured update is a property
            of that probe. Neither is promoted to a training-map Jacobian.
            Check analytic/central-difference agreement and NS5-versus-ideal
            discrepancies before interpreting optimizer-specific separation.
            """
        ),
    ]
    return notebook("11_Muon_Update_Stiefel_Tangent.ipynb", cells)


def radial_angular_notebook() -> tuple[str, dict[str, object]]:
    cells = [
        markdown(
            """
            # Radial and angular quotient sectors

            Read the independently persisted final-100 checkpoint cache and
            split its finite checkpoint motion into supported generalized-Gram
            radial rates and orientation-correct row/column Grassmann motion.
            Relative-polar tilt/twist rates are reported separately so zeros
            forced by rectangular dimension are never treated as fitted data.
            """
        ),
        parameters(
            """
            OPTIMIZER_SLUGS = ["adamw", "muon", "muonclip_rms"]
            LAYERS = ["fc1.weight", "fc2.weight", "fc3.weight"]
            MAXIMUM_PAIRS = 100
            PAIR_STRIDES = [1]
            RUN_SPACING_SENSITIVITY = True
            SPACING_SENSITIVITY_STRIDES = [2, 4, 8]
            SPACING_SENSITIVITY_PAIRS_PER_STRIDE = 8
            TOP_K_VALUES = [0, 1, 2, 3, 4, 5]
            MINIMUM_TAIL = 8
            METHOD_SLUG = "radial_angular_quotients"
            PLOT_TITLE = "Radial/angular quotient spectra: alpha with 95% seed CI"
            """
        ),
        code(BOOTSTRAP),
        code(COMMON_IMPORTS + "\n" + ANALYSIS_IMPORTS),
        caveat(
            "Quotient-sector bundle",
            "finite_radial_angular_quotient_rates",
            "Generalized Gram radial log-rates plus row/column Grassmann and relative-polar angular rates between W0 and W1.",
            "There is no uniquely privileged quotient from two checkpoints. Rectangular gauges, "
            "forced intersections, endpoint atoms, and checkpoint spacing remain explicit.",
        ),
        code(COMMON_HELPERS + "\n" + TAIL_CHECKPOINT_CACHE_HELPERS + "\n" + CHECKPOINT_HELPERS),
        code(
            """
            operator_records, fit_frames, trace_frames = [], [], []
            spectral_arrays = {}
            for optimizer in OPTIMIZER_SLUGS:
                for seed in SEEDS:
                    seed_dir = require_tail_checkpoint_cache(optimizer, seed)
                    run_fingerprint = verified_run_fingerprint(optimizer, seed)
                    pairs = selected_checkpoint_pairs_for_strides(
                        seed_dir, strides=PAIR_STRIDES, maximum_pairs=MAXIMUM_PAIRS
                    )
                    if RUN_SPACING_SENSITIVITY:
                        overlap = set(PAIR_STRIDES) & set(SPACING_SENSITIVITY_STRIDES)
                        if overlap:
                            raise ValueError(f"Primary/sensitivity pair strides overlap: {overlap}")
                        pairs.extend(
                            selected_checkpoint_pairs_for_strides(
                                seed_dir,
                                strides=SPACING_SENSITIVITY_STRIDES,
                                maximum_pairs=SPACING_SENSITIVITY_PAIRS_PER_STRIDE,
                            )
                        )
                    declared_strides = list(PAIR_STRIDES) + (
                        list(SPACING_SENSITIVITY_STRIDES)
                        if RUN_SPACING_SENSITIVITY else []
                    )
                    pair_counts = {
                        int(stride): sum(1 for item in pairs if int(item[0]) == int(stride))
                        for stride in declared_strides
                    }
                    print(
                        f"{optimizer} seed={seed}: "
                        f"{len(analysis_checkpoint_refs(seed_dir))} verified cache states, "
                        f"pair counts by stride={pair_counts}, payload LRU="
                        f"{CHECKPOINT_PAYLOAD_CACHE_SIZE}"
                    )
                    for pair_stride, previous_ref, current_ref, pair_selection_rule, pair_selection_role in pairs:
                        delta_s = int(current_ref.global_step - previous_ref.global_step)
                        for layer in LAYERS:
                            W0 = checkpoint_matrix(previous_ref.path, layer)
                            W1 = checkpoint_matrix(current_ref.path, layer)
                            result = two_checkpoint.analyze_two_checkpoints(W0, W1, delta_s)
                            base = {
                                "optimizer": optimizer, "seed": int(seed), "layer": layer,
                                "protocol_fingerprint": run_fingerprint,
                                "source_artifact_kind": "verified_tail_checkpoint_cache_model_only",
                                "state_index": int(current_ref.global_step),
                                "epoch0": int(previous_ref.epoch), "epoch1": int(current_ref.epoch),
                                "delta_s": delta_s,
                                "pair_stride": int(pair_stride),
                                "pair_spacing_role": (
                                    "primary_adjacent"
                                    if int(pair_stride) in set(PAIR_STRIDES)
                                    else "bounded_spacing_sensitivity"
                                ),
                                "pair_selection_rule": pair_selection_rule,
                                "pair_selection_role": pair_selection_role,
                                "checkpoint_source": "verified_final_100_tail_cache",
                                "checkpoint_cache_seed_dir": str(seed_dir),
                                "fc3_rank10_warning": "fc3" in layer.lower(),
                            }
                            sectors = [
                                ("radial_log_rate_amplitude", result.radial.radial_rate_amplitudes, result.radial),
                                ("column_grassmann_geodesic_amplitude", result.grassmann.column.geodesic_rates, result.grassmann.column),
                                ("row_grassmann_geodesic_amplitude", result.grassmann.row.geodesic_rates, result.grassmann.row),
                                ("relative_polar_tilt_amplitude", result.relative_polar.tilt_geodesic_rates, result.relative_polar),
                                ("relative_polar_twist_amplitude", result.relative_polar.twist_geodesic_rates, result.relative_polar),
                            ]
                            if result.rectangular_transfer.available:
                                sectors.extend([
                                    (
                                        "supported_rectangular_transfer_log_rate_amplitude",
                                        result.rectangular_transfer.supported_transfer_rate_amplitudes,
                                        result.rectangular_transfer,
                                    ),
                                    (
                                        "procrustes_aligned_core_log_rate_amplitude",
                                        result.rectangular_transfer.core_rate_amplitudes,
                                        result.rectangular_transfer,
                                    ),
                                    (
                                        "aligned_transfer_principal_angle_rate_amplitude",
                                        result.rectangular_transfer.principal_angle_rates,
                                        result.rectangular_transfer,
                                    ),
                                ])
                            operator_records.extend([
                                {**base, "method": "radial", "operator_kind": result.radial.operator_kind,
                                 "map_definition": result.radial.map_definition,
                                 "retained_rank": result.radial.retained_rank,
                                 "dropped_null": result.radial.dropped_initial_null_directions},
                                {**base, "method": "column_grassmann", "operator_kind": result.grassmann.column.operator_kind,
                                 "map_definition": result.grassmann.column.map_definition,
                                 "forced_zero_atoms": result.grassmann.column.forced_intersection_zeros,
                                 "endpoint_atoms": result.grassmann.column.unmatched_dimensions},
                                {**base, "method": "row_grassmann", "operator_kind": result.grassmann.row.operator_kind,
                                 "map_definition": result.grassmann.row.map_definition,
                                 "forced_zero_atoms": result.grassmann.row.forced_intersection_zeros,
                                 "endpoint_atoms": result.grassmann.row.unmatched_dimensions},
                                {**base, "method": "relative_polar", "operator_kind": result.relative_polar.operator_kind,
                                 "map_definition": result.relative_polar.map_definition,
                                 "tilt_zero_atoms": result.relative_polar.tilt_zero_atoms,
                                 "twist_zero_atoms": result.relative_polar.twist_zero_atoms,
                                 "twist_unique": result.relative_polar.twist_unique},
                                {**base, "method": "aligned_rectangular_transfer", "operator_kind": result.rectangular_transfer.operator_kind,
                                 "map_definition": result.rectangular_transfer.map_definition,
                                 "available": result.rectangular_transfer.available,
                                 "orientation": result.rectangular_transfer.orientation,
                                 "effective_rank": result.rectangular_transfer.effective_rank,
                                 "numerical_rank0": result.rectangular_transfer.numerical_rank0,
                                 "numerical_rank1": result.rectangular_transfer.numerical_rank1,
                                 "rank_rtol": result.rectangular_transfer.rank_rtol,
                                 "rank_threshold0": result.rectangular_transfer.rank_threshold0,
                                 "rank_threshold1": result.rectangular_transfer.rank_threshold1,
                                 "condition_number0": result.rectangular_transfer.condition_number0,
                                 "condition_number1": result.rectangular_transfer.condition_number1,
                                 "structural_zero_count": result.rectangular_transfer.structural_zero_count,
                                 "reconstruction_residual": result.rectangular_transfer.relative_reconstruction_residual,
                                 "unsupported_source_action_residual": result.rectangular_transfer.unsupported_source_action_residual,
                                 "core_reconstruction_residual": result.rectangular_transfer.core_reconstruction_residual,
                                 "subspace_alignment_residual": result.rectangular_transfer.subspace_alignment_residual,
                                 "forced_intersection_zeros": result.rectangular_transfer.forced_intersection_zeros},
                            ])
                            for method, spectrum, record in sectors:
                                positive = np.asarray(spectrum, dtype=float)
                                positive = positive[np.isfinite(positive) & (positive > 0)]
                                if positive.size < 2:
                                    continue
                                definition = record.map_definition + "; fitted b are nonzero rate amplitudes and energy e=b^2 is an exact fit transform"
                                metadata = {**base, "method": method}
                                fits, traces = fit_spectrum_with_trace(
                                    positive,
                                    operator_kind=record.operator_kind,
                                    map_definition=definition,
                                    spectrum_kind="amplitude",
                                    metadata=metadata,
                                    top_k_values=TOP_K_VALUES,
                                    minimum_tail=MINIMUM_TAIL,
                                )
                                fit_frames.append(fits); trace_frames.append(traces)
                                spectral_arrays[f"{optimizer}_{seed}_{layer}_{current_ref.global_step}_{method}"] = positive
            if not operator_records or not fit_frames:
                raise RuntimeError("No supported radial/angular spectra were produced")
            """
        ),
        code(ANALYSIS_SAVE_AND_PLOT),
        code(
            """
            np.savez_compressed(analysis_dir / "positive_spectra.npz", **spectral_arrays)
            display(save_spectrum_ccdf_gallery(spectral_arrays, method_slug=METHOD_SLUG))
            primary = fit_rows[
                fit_rows["clip_top_k"].eq(0)
                & fit_rows["spectrum_kind"].astype(str).eq("energy_derived_from_amplitude")
            ]
            spacing_summary = ci_summary(
                primary,
                groups=("optimizer", "layer", "method", "pair_stride", "delta_s"),
                metrics=("alpha", "ks_D", "n_tail"),
            )
            spacing_summary.to_csv(analysis_dir / "spacing_sensitivity_95ci.csv", index=False)
            display(spacing_summary.tail(36))
            """
        ),
        markdown(
            """
            Interpret radial, column-space, row-space, tilt, and twist results
            as different declared observables. Agreement is corroboration;
            disagreement is not grounds to select the prettiest quotient after
            inspection. FC3 remains a rank-10 diagnostic only.
            """
        ),
    ]
    return notebook("12_Radial_Angular_Quotients.ipynb", cells)


def single_checkpoint_notebook() -> tuple[str, dict[str, object]]:
    cells = [
        markdown(
            """
            # Single-checkpoint analytic map Jacobians

            Starting from one saved weight matrix `W`, define six explicit
            candidate RG map families and form the actual Jacobian of every map:
            angular polar, normalized Gram, centered trace-log Gram, centered
            log-singular radial, the exact configured finite Muon NS5 map, and
            the restricted retracted-core ECS Grassmann Cartan cover for wide
            `fc1.weight`. The ECS family is evaluated with the requested full
            numerical row shell and with a detX-bounded shell sensitivity.
            The notebook fits only spectra of these derivatives--never the ESD
            of an undifferentiated quotient or a checkpoint displacement.
            Every trained matrix is read from the verified final-100 cache;
            this notebook cannot launch or resume training.
            """
        ),
        parameters(
            """
            OPTIMIZER_SLUGS = ["adamw", "muon", "muonclip_rms"]
            LAYERS = ["fc1.weight", "fc2.weight", "fc3.weight"]
            TOP_K_VALUES = [0, 1, 2, 3, 4, 5]
            MINIMUM_TAIL = 8
            MAXIMUM_CHECKPOINTS = 100
            NUMERICAL_SHAPE = [6, 8]
            ECS_COVER_LAYER = "fc1.weight"
            ECS_VALIDATION_RETAINED_RANK = 3
            ECS_VALIDATION_OUTER_RANK = 5
            ECS_RANK_RCOND = 1e-9
            ECS_FIT_CONTRACT_TOKEN = (
                "ecs_grassmann_cartan_cover_group_qualified_v1"
            )
            NS_STEPS = 5
            NS_EPS = 1e-7
            METHOD_SLUG = "single_checkpoint_map_jacobians"
            PLOT_TITLE = "Single-checkpoint declared-map spectra: alpha with 95% seed CI"
            """
        ),
        code(BOOTSTRAP),
        code(COMMON_IMPORTS + "\n" + ANALYSIS_IMPORTS),
        caveat(
            "Declared single-point maps",
            "single_checkpoint_algebraic_map_derivative_bundle",
            "J_i(W)=D F_i(W) for the first five maps; "
            "J_ECS(W)=D_E Phi_W(0) for the checkpoint-anchored retracted "
            "ECS Grassmann Cartan cover.",
            "Every fitted object is the singular spectrum of an actual derivative of a "
            "declared candidate RG map computable from W alone. Numerical materialization "
            "is restricted to fixed small formula-validation matrices.",
        ),
        markdown(
            r"""
            ## Wide-FC1 ECS Grassmann Cartan cover

            For the checkpoint SVD $W=U\Sigma V^T$, take
            $V_k=[v_1,\ldots,v_k]$ and
            $V_c=[v_{k+1},\ldots,v_q]$. Radial changes of
            $\Sigma_k$, left-angular motion, and rotations internal to
            $V_k$ are null directions. An ambient perturbation $E$ is
            reduced to the quotient coordinate

            \[
            K_W(E)=V_c^T E^T U_k\Sigma_k^{-1}.
            \]

            The notebook differentiates the nonlinear, checkpoint-anchored
            retraction

            \[
            V(K)=(V_k+V_cK)(I+K^TK)^{-1/2},\quad
            R_W(K)=U_k\Sigma_kV(K)^T,
            \]
            \[
            \Phi_W(E)=V_c^T\{2P_{\rm row}[R_W(K_W(E))]-I\}V_k.
            \]

            Thus $D_E\Phi_W(0)[E]=2K_W(E)$, a true Jacobian with
            amplitudes $2/\sigma_i$, energies $4/\sigma_i^2$, and rank
            $k(q-k)$. The factor two is the one-cross-block Cartan
            convention; canonical $K$ would give $1/\sigma_i$, while the
            full projector tangent in Frobenius norm gives
            $\sqrt{2}/\sigma_i$.

            Rank diagnostics are admitted only from exact checkpoint matches.
            The finger-aware power-law top-mode boundary defines $k$. The
            requested primary cover takes $q$ to be the checkpoint numerical
            row rank; a separately labelled sensitivity takes $q$ from finite
            `detX_num`. Both use the same exact trace-audited states, and roles
            are never swapped. Missing, coincident, or reversed detX
            boundaries are audit rows, never nearest-state fills.
            At fixed $k$, changing $q$ only changes the uniform multiplicity
            $q-k$ of every $2/\sigma_i$ amplitude. The detX-$q$ construction is
            therefore a shell-dimension/multiplicity sensitivity, not an
            independent spectral-shape or alpha corroboration. Finger
            sensitivities remove whole $(q-k)$-mode core groups, never partial
            degenerate blocks. A fit is retained only when the
            package-selected tail contains at least `MINIMUM_TAIL` retained-core
            groups; the expanded mode count alone cannot qualify it.
            Repeated shell modes and checkpoints are not independent samples;
            uncertainty is computed across complete seeded runs.
            """
        ),
        code(
            COMMON_HELPERS
            + "\n"
            + TAIL_CHECKPOINT_CACHE_HELPERS
            + "\n"
            + CHECKPOINT_HELPERS
            + "\n"
            + ECS_COVER_METRIC_HELPERS
        ),
        markdown("## Exact large-layer spectra and small numerical validation"),
        code(
            """
            operator_records, fit_frames, trace_frames = [], [], []
            spectral_arrays = {}
            for optimizer in OPTIMIZER_SLUGS:
                for seed in SEEDS:
                    seed_dir = require_tail_checkpoint_cache(optimizer, seed)
                    run_fingerprint = verified_run_fingerprint(optimizer, seed)
                    (
                        ecs_metric_fits,
                        ecs_metric_traces,
                        ecs_fit_path,
                        ecs_trace_path,
                    ) = load_verified_ecs_metric_tables(
                        optimizer, seed, run_fingerprint
                    )
                    cache_refs = analysis_checkpoint_refs(seed_dir)
                    final_cache_epoch = int(cache_refs[-1].epoch)
                    print(
                        f"{optimizer} seed={seed}: analyzing "
                        f"{len(cache_refs)} verified cache states "
                        f"with payload LRU={CHECKPOINT_PAYLOAD_CACHE_SIZE}"
                    )
                    for selected, layer, W, selection_rule, selection_role in selected_trajectory_matrices(
                        seed_dir,
                        layers=LAYERS,
                        maximum_checkpoints=MAXIMUM_CHECKPOINTS,
                    ):
                        checkpoint_singular_values = np.linalg.svd(W, compute_uv=False)
                        polar_record = polar.polar_pullback_spectrum(
                            W,
                            precomputed_singular_values=checkpoint_singular_values,
                            include_mode_labels=False,
                        )
                        gram_record = single_checkpoint.normalized_gram_analytic_spectrum(
                            W,
                            precomputed_singular_values=checkpoint_singular_values,
                        )
                        log_gram_record = single_checkpoint.centered_log_gram_analytic_spectrum(
                            W,
                            precomputed_singular_values=checkpoint_singular_values,
                        )
                        radial_record = single_checkpoint.centered_log_singular_analytic_spectrum(
                            W,
                            precomputed_singular_values=checkpoint_singular_values,
                        )
                        ns5_record = polar.muon_newton_schulz_analytic_spectrum(
                            W,
                            steps=NS_STEPS,
                            eps=NS_EPS,
                            precomputed_singular_values=checkpoint_singular_values,
                        )
                        base = {
                            "optimizer": optimizer, "seed": int(seed), "layer": layer,
                            "protocol_fingerprint": run_fingerprint,
                            "source_artifact_kind": "verified_tail_checkpoint_cache_model_only",
                            "state_index": int(selected.global_step), "epoch": int(selected.epoch),
                            "trajectory_selection_rule": selection_rule,
                            "trajectory_selection_role": selection_role,
                            "checkpoint_source": "verified_final_100_tail_cache",
                            "checkpoint_cache_seed_dir": str(seed_dir),
                            "fc3_rank10_warning": "fc3" in layer.lower(),
                        }
                        operator_records.extend([
                            {**base, "method": "polar_pullback", "operator_kind": polar_record.operator_kind,
                             "map_definition": polar_record.map_definition,
                             "derivative_rank": polar_record.derivative_rank,
                             "zero_count": polar_record.zero_count},
                            {**base, "method": "normalized_gram_pullback", "operator_kind": gram_record.operator_kind,
                             "map_definition": gram_record.map_definition,
                             "derivative_rank": gram_record.derivative_rank,
                             "zero_count": gram_record.zero_count,
                             "scale_null_residual": 0.0,
                             "scale_null_audit": "analytic identity D F_W[W]=0"},
                            {**base, "method": "centered_log_gram_pullback", "operator_kind": log_gram_record.operator_kind,
                             "map_definition": log_gram_record.map_definition,
                             "derivative_rank": log_gram_record.derivative_rank,
                             "zero_count": log_gram_record.zero_count,
                             "scale_null_residual": 0.0,
                             "scale_null_audit": "analytic identity D L_W[W]=0"},
                            {**base, "method": "centered_log_singular_radial_pullback", "operator_kind": radial_record.operator_kind,
                             "map_definition": radial_record.map_definition,
                             "derivative_rank": radial_record.derivative_rank,
                             "zero_count": radial_record.zero_count,
                             "scale_null_residual": 0.0,
                             "scale_null_audit": "analytic identity D R_W[W]=0"},
                            {**base, "method": "finite_muon_ns5_pullback", "operator_kind": ns5_record.operator_kind,
                             "map_definition": ns5_record.map_definition,
                             "derivative_rank": ns5_record.derivative_rank,
                             "zero_count": ns5_record.zero_count,
                             "ns_steps": int(NS_STEPS), "ns_eps": float(NS_EPS)},
                        ])
                        candidates = [
                            ("polar_pullback", polar_record.singular_amplitudes, polar_record, base),
                            ("normalized_gram_pullback", gram_record.singular_amplitudes, gram_record, base),
                            ("centered_log_gram_pullback", log_gram_record.singular_amplitudes, log_gram_record, base),
                            ("centered_log_singular_radial_pullback", radial_record.singular_amplitudes, radial_record, base),
                            ("finite_muon_ns5_pullback", ns5_record.singular_amplitudes, ns5_record, base),
                        ]
                        if layer == ECS_COVER_LAYER:
                            rank_tolerance = (
                                float(ECS_RANK_RCOND)
                                * float(checkpoint_singular_values[0])
                            )
                            checkpoint_rank = int(
                                np.count_nonzero(
                                    checkpoint_singular_values > rank_tolerance
                                )
                            )
                            ecs_ranks = exact_ecs_cover_rank_record(
                                ecs_metric_fits,
                                ecs_metric_traces,
                                optimizer_slug=optimizer,
                                seed=seed,
                                epoch=int(selected.epoch),
                                global_step=int(selected.global_step),
                                layer=layer,
                                maximum_rank=checkpoint_rank,
                                fit_path=ecs_fit_path,
                                trace_path=ecs_trace_path,
                            )
                            cover_common = {
                                **base,
                                **ecs_ranks,
                                "source_artifact_kind": ECS_COVER_SOURCE_KIND,
                                "checkpoint_numerical_rank": checkpoint_rank,
                                "checkpoint_rank_rcond": float(ECS_RANK_RCOND),
                                "checkpoint_rank_tolerance": rank_tolerance,
                            }
                            cover_variants = (
                                (
                                    "ecs_grassmann_cartan_cover_full_row_shell_pullback",
                                    "full_checkpoint_numerical_row_shell",
                                    bool(ecs_ranks["ecs_full_shell_available"]),
                                    ecs_ranks.get("full_shell_outer_rank"),
                                    ecs_ranks.get("full_shell_outer_rank_source"),
                                    ecs_ranks.get("full_shell_unavailable_reason", ""),
                                ),
                                (
                                    "ecs_grassmann_cartan_cover_detx_shell_pullback",
                                    "detx_bounded_outer_shell_sensitivity",
                                    bool(ecs_ranks["ecs_detx_shell_available"]),
                                    ecs_ranks.get("detx_shell_outer_rank"),
                                    ecs_ranks.get("detx_shell_outer_rank_source"),
                                    ecs_ranks.get("detx_shell_unavailable_reason", ""),
                                ),
                            )
                            for (
                                cover_method,
                                shell_variant,
                                cover_available,
                                cover_outer_rank,
                                cover_outer_source,
                                variant_unavailable_reason,
                            ) in cover_variants:
                                cover_base = {
                                    **cover_common,
                                    "method": cover_method,
                                    "ecs_shell_variant": shell_variant,
                                    "ecs_shell_comparison_role": (
                                        "primary_full_numerical_row_shell"
                                        if shell_variant
                                        == "full_checkpoint_numerical_row_shell"
                                        else "shell_dimension_multiplicity_sensitivity_only"
                                    ),
                                    "ecs_normalized_esd_shape_invariant_to_q_at_fixed_k": True,
                                    "ecs_alpha_across_q_not_independent_corroboration": True,
                                    "ecs_finger_clipping_unit": (
                                        "whole_retained_core_groups_of_q_minus_k_modes"
                                    ),
                                    "ecs_minimum_tail_unit": "retained_core_amplitude_groups",
                                    "analysis_contract_token": ECS_FIT_CONTRACT_TOKEN,
                                    "outer_rank": cover_outer_rank,
                                    "outer_rank_source": cover_outer_source,
                                    "shell_rank": (
                                        int(cover_outer_rank)
                                        - int(ecs_ranks["retained_rank"])
                                        if cover_available else 0
                                    ),
                                }
                                if cover_available:
                                    cover_record = (
                                        single_checkpoint.ecs_grassmann_cover_analytic_spectrum(
                                            W,
                                            retained_rank=int(ecs_ranks["retained_rank"]),
                                            outer_rank=int(cover_outer_rank),
                                            rcond=float(ECS_RANK_RCOND),
                                            precomputed_singular_values=checkpoint_singular_values,
                                        )
                                    )
                                    cover_base.update({
                                        "ambient_row_dimension": cover_record.ambient_row_dimension,
                                        "ambient_cover_dimension": cover_record.ambient_cover_dimension,
                                        "restricted_cover_dimension": cover_record.restricted_cover_dimension,
                                        "excluded_ambient_dimension": int(
                                            cover_record.retained_rank
                                            * (cover_record.ambient_row_dimension - cover_record.outer_rank)
                                        ),
                                        "core_amplitude_group_count": cover_record.core_amplitude_group_count,
                                        "numerically_distinct_core_amplitude_count": (
                                            cover_record.numerically_distinct_core_amplitude_count
                                        ),
                                        "deterministic_shell_multiplicity": cover_record.deterministic_shell_multiplicity,
                                        "boundary_singular_gap": cover_record.boundary_singular_gap,
                                        "outer_boundary_singular_gap": cover_record.outer_boundary_singular_gap,
                                        "cover_coordinate_scale": cover_record.coordinate_scale,
                                        "cover_metric_convention": cover_record.metric_convention,
                                        "deterministic_replication_warning": (
                                            "k retained-core amplitude groups, each repeated q-k times; "
                                            "only complete seeded training runs are independent--"
                                            "checkpoints and (i,a) copies are repeated observations"
                                        ),
                                    })
                                    operator_records.append({
                                        **cover_base,
                                        "operator_kind": cover_record.operator_kind,
                                        "map_definition": cover_record.map_definition,
                                        "derivative_rank": cover_record.derivative_rank,
                                        "zero_count": cover_record.zero_count,
                                        "available": True,
                                    })
                                    candidates.append((
                                        cover_method,
                                        cover_record.singular_amplitudes,
                                        cover_record,
                                        cover_base,
                                    ))
                                else:
                                    operator_records.append({
                                        **cover_base,
                                        "operator_kind": (
                                            "unavailable_exact_sparse_ecs_grassmann_cartan_cover"
                                        ),
                                        "map_definition": (
                                            "requested anchored retracted-core cover "
                                            "J[E]=2V_c^T E^T U_k Sigma_k^-1"
                                        ),
                                        "available": False,
                                        "variant_unavailable_reason": (
                                            variant_unavailable_reason
                                            or ecs_ranks.get(
                                                "ecs_rank_unavailable_reason",
                                                "exact ECS rank metrics unavailable",
                                            )
                                        ),
                                    })
                        for method, spectrum, record, candidate_base in candidates:
                            positive = np.asarray(spectrum, dtype=float)
                            positive = positive[np.isfinite(positive) & (positive > 0)]
                            if positive.size < 2:
                                continue
                            metadata = {**candidate_base, "method": method}
                            fit_top_k_values = TOP_K_VALUES
                            is_ecs_cover = method.startswith(
                                "ecs_grassmann_cartan_cover_"
                            )
                            if is_ecs_cover:
                                shell_multiplicity = int(
                                    record.deterministic_shell_multiplicity
                                )
                                core_group_count = int(
                                    record.core_amplitude_group_count
                                )
                                feasible_group_clips = tuple(
                                    int(group_count)
                                    for group_count in TOP_K_VALUES
                                    if int(group_count) <= core_group_count - 2
                                )
                                if not feasible_group_clips or feasible_group_clips[0] != 0:
                                    feasible_group_clips = (0,)
                                fit_top_k_values = tuple(
                                    group_count * shell_multiplicity
                                    for group_count in feasible_group_clips
                                )
                            fits, traces = fit_spectrum_with_trace(
                                positive,
                                operator_kind=record.operator_kind,
                                map_definition=record.map_definition,
                                spectrum_kind="amplitude",
                                metadata=metadata,
                                top_k_values=fit_top_k_values,
                                minimum_tail=MINIMUM_TAIL,
                            )
                            if is_ecs_cover:
                                core_amplitudes = (
                                    2.0
                                    / np.asarray(
                                        record.retained_singular_values,
                                        dtype=float,
                                    )
                                )
                                fits = powerlaw_fit.qualify_replicated_group_fits(
                                    fits,
                                    core_amplitudes,
                                    group_multiplicity=shell_multiplicity,
                                    minimum_tail_groups=MINIMUM_TAIL,
                                )
                                fits["ecs_clip_core_groups"] = fits[
                                    "clip_group_count"
                                ]
                                fits["ecs_tail_core_group_count"] = fits[
                                    "tail_group_count"
                                ]
                                fits["ecs_minimum_tail_core_groups"] = fits[
                                    "minimum_tail_groups"
                                ]
                                fits["ecs_group_tail_qualified"] = fits[
                                    "group_tail_qualified"
                                ]
                                fits["ecs_mode_level_fit_ok_before_group_gate"] = fits[
                                    "mode_level_fit_ok_before_group_gate"
                                ]
                            fit_frames.append(fits); trace_frames.append(traces)
                            # Fits cover every selected state.  Persist only final-state
                            # arrays so the six exact Jacobian families do not retain
                            # the full trajectory (many GiB for the 3x3 long campaign).
                            if int(selected.epoch) == final_cache_epoch:
                                spectral_arrays[
                                    f"{optimizer}_{seed}_{layer}_{selected.global_step}_{method}"
                                ] = positive

            rng = np.random.default_rng(20260819)
            small = rng.normal(size=tuple(NUMERICAL_SHAPE))
            small[:min(small.shape), :min(small.shape)] += 2.0 * np.eye(min(small.shape))
            maximum_dimension = int(np.prod(NUMERICAL_SHAPE))
            cover_coordinate_direction = rng.normal(
                size=(
                    int(ECS_VALIDATION_OUTER_RANK - ECS_VALIDATION_RETAINED_RANK),
                    int(ECS_VALIDATION_RETAINED_RANK),
                )
            )
            cover_validation_epsilon = 1e-6
            cover_plus = single_checkpoint.ecs_grassmann_retracted_core(
                small,
                cover_validation_epsilon * cover_coordinate_direction,
                retained_rank=ECS_VALIDATION_RETAINED_RANK,
                outer_rank=ECS_VALIDATION_OUTER_RANK,
                rcond=ECS_RANK_RCOND,
            )
            cover_minus = single_checkpoint.ecs_grassmann_retracted_core(
                small,
                -cover_validation_epsilon * cover_coordinate_direction,
                retained_rank=ECS_VALIDATION_RETAINED_RANK,
                outer_rank=ECS_VALIDATION_OUTER_RANK,
                rcond=ECS_RANK_RCOND,
            )
            cover_retraction_derivative = (
                cover_plus.cartan_cross_block - cover_minus.cartan_cross_block
            ) / (2.0 * cover_validation_epsilon)
            cover_retraction_relative_error = float(
                np.linalg.norm(
                    cover_retraction_derivative - 2.0 * cover_coordinate_direction
                )
                / max(
                    np.linalg.norm(2.0 * cover_coordinate_direction),
                    np.finfo(float).tiny,
                )
            )
            validations = [
                (
                    "polar",
                    polar.explicit_polar_jacobian(
                        small, max_input_dimension=maximum_dimension, rank_rtol=1e-9
                    ),
                    polar.polar_pullback_spectrum(small),
                ),
                (
                    "normalized_gram",
                    polar.central_difference_jacobian(
                        lambda value: single_checkpoint.normalized_gram_map(value).value,
                        small,
                        max_input_dimension=maximum_dimension,
                        rank_rtol=1e-9,
                        operator_kind="explicit_numerical_normalized_gram_jacobian",
                        map_definition="central-difference validation of D[dG/||W||_F^2]",
                    ),
                    single_checkpoint.normalized_gram_analytic_spectrum(small),
                ),
                (
                    "centered_log_gram",
                    polar.central_difference_jacobian(
                        lambda value: single_checkpoint.centered_log_gram_map(value).value,
                        small,
                        max_input_dimension=maximum_dimension,
                        rank_rtol=1e-9,
                        operator_kind="explicit_numerical_centered_log_gram_jacobian",
                        map_definition="central-difference validation of the centered log-Gram candidate RG Jacobian",
                    ),
                    single_checkpoint.centered_log_gram_analytic_spectrum(small),
                ),
                (
                    "centered_log_singular_radial",
                    polar.central_difference_jacobian(
                        lambda value: single_checkpoint.centered_log_singular_map(value).value,
                        small,
                        max_input_dimension=maximum_dimension,
                        rank_rtol=1e-9,
                        operator_kind="explicit_numerical_centered_log_singular_jacobian",
                        map_definition="central-difference validation of the centered log-singular candidate RG Jacobian",
                    ),
                    single_checkpoint.centered_log_singular_analytic_spectrum(small),
                ),
                (
                    "finite_muon_ns5",
                    polar.explicit_muon_newton_schulz_jacobian(
                        small,
                        steps=NS_STEPS,
                        eps=NS_EPS,
                        max_input_dimension=maximum_dimension,
                        rank_rtol=1e-9,
                    ),
                    polar.muon_newton_schulz_analytic_spectrum(
                        small, steps=NS_STEPS, eps=NS_EPS
                    ),
                ),
                (
                    "ecs_grassmann_cartan_cover",
                    polar.central_difference_jacobian(
                        lambda value: single_checkpoint.ecs_grassmann_cover_map(
                            small,
                            value,
                            retained_rank=ECS_VALIDATION_RETAINED_RANK,
                            outer_rank=ECS_VALIDATION_OUTER_RANK,
                            rcond=ECS_RANK_RCOND,
                        ).value,
                        np.zeros_like(small),
                        max_input_dimension=maximum_dimension,
                        rank_rtol=1e-9,
                        operator_kind=(
                            "explicit_numerical_retracted_core_ecs_grassmann_"
                            "cartan_cover_jacobian"
                        ),
                        map_definition=(
                            "central-difference materialization of the nonlinear "
                            "checkpoint-anchored retracted ECS Cartan cover map"
                        ),
                    ),
                    single_checkpoint.ecs_grassmann_cover_analytic_spectrum(
                        small,
                        retained_rank=ECS_VALIDATION_RETAINED_RANK,
                        outer_rank=ECS_VALIDATION_OUTER_RANK,
                        rcond=ECS_RANK_RCOND,
                    ),
                ),
            ]
            for validation_name, numerical, analytic in validations:
                analytic_amplitudes = np.asarray(analytic.singular_amplitudes, dtype=float)
                numerical_amplitudes = np.asarray(
                    numerical.singular_values[: analytic_amplitudes.size], dtype=float
                )
                if numerical.numerical_rank != analytic.derivative_rank:
                    raise RuntimeError(
                        f"small {validation_name} Jacobian rank mismatch: "
                        f"numeric={numerical.numerical_rank}, analytic={analytic.derivative_rank}"
                    )
                agreement = float(
                    np.linalg.norm(numerical_amplitudes - analytic_amplitudes)
                    / max(np.linalg.norm(analytic_amplitudes), np.finfo(float).tiny)
                )
                operator_records.append({
                    "optimizer": "synthetic", "seed": int(SEEDS[0]),
                    "layer": str(tuple(NUMERICAL_SHAPE)),
                    "protocol_fingerprint": "not_applicable_synthetic_formula_validation",
                    "source_artifact_kind": "fixed_seed_synthetic_formula_validation",
                    "state_index": 0,
                    "method": f"small_explicit_{validation_name}_validation",
                    "operator_kind": numerical.operator_kind,
                    "map_definition": numerical.map_definition,
                    "analytic_operator_kind": analytic.operator_kind,
                    "relative_spectral_error": agreement,
                    "numeric_rank": int(numerical.numerical_rank),
                    "analytic_rank": int(analytic.derivative_rank),
                    "retracted_core_directional_error": (
                        cover_retraction_relative_error
                        if validation_name == "ecs_grassmann_cartan_cover"
                        else np.nan
                    ),
                    "fc3_rank10_warning": False,
                })
                if agreement > 1e-5:
                    raise RuntimeError(
                        f"analytic/numerical {validation_name} spectrum mismatch: {agreement:.3e}"
                    )
            if cover_retraction_relative_error > 1e-5:
                raise RuntimeError(
                    "ECS Grassmann retraction/Cartan derivative mismatch: "
                    f"{cover_retraction_relative_error:.3e}"
                )
            if not fit_frames:
                raise RuntimeError("No single-checkpoint spectrum was fit")
            """
        ),
        code(ANALYSIS_SAVE_AND_PLOT),
        code(
            """
            np.savez_compressed(analysis_dir / "positive_spectra.npz", **spectral_arrays)
            display(save_spectrum_ccdf_gallery(spectral_arrays, method_slug=METHOD_SLUG))
            cover_rank_audit = operator_rows[
                operator_rows["method"].astype(str).isin({
                    "ecs_grassmann_cartan_cover_full_row_shell_pullback",
                    "ecs_grassmann_cartan_cover_detx_shell_pullback",
                })
            ].copy()
            if cover_rank_audit.empty:
                raise RuntimeError("No FC1 ECS cover rank attempts were audited")
            cover_rank_audit["available"] = cover_rank_audit["available"].map(
                _strict_bool
            )
            for column in (
                "ecs_rank_exact_weightwatcher_state_found",
                "ecs_rank_exact_trace_state_found",
            ):
                cover_rank_audit[column] = cover_rank_audit[column].map(_strict_bool)
            cover_coverage = (
                cover_rank_audit.groupby(
                    ["method", "optimizer", "seed"], as_index=False
                )
                .agg(
                    cache_states_audited=("state_index", "count"),
                    exact_weightwatcher_states=(
                        "ecs_rank_exact_weightwatcher_state_found", "sum"
                    ),
                    exact_trace_states=("ecs_rank_exact_trace_state_found", "sum"),
                    rank_eligible_cover_states=("available", "sum"),
                )
            )
            cover_method_mask = (
                fit_rows["method"].astype(str).eq(
                    "ecs_grassmann_cartan_cover_full_row_shell_pullback"
                )
                | fit_rows["method"].astype(str).eq(
                    "ecs_grassmann_cartan_cover_detx_shell_pullback"
                )
            )
            cover_primary_fits = fit_rows[
                cover_method_mask
                & fit_rows["spectrum_kind"].astype(str).eq("amplitude")
                & fit_rows["clip_top_k"].eq(0)
            ].copy()
            if "fit_ok" not in cover_primary_fits.columns:
                raise RuntimeError("ECS cover primary fits lack fit_ok status")
            cover_fit_attempt_counts = (
                cover_primary_fits
                .groupby(["method", "optimizer", "seed"])["state_index"]
                .nunique()
                .rename("primary_fit_attempt_states")
                .reset_index()
            )
            cover_success_counts = (
                cover_primary_fits[boolean_series(cover_primary_fits["fit_ok"])]
                .groupby(["method", "optimizer", "seed"])["state_index"]
                .nunique()
                .rename("successful_primary_fit_states")
                .reset_index()
            )
            cover_coverage = cover_coverage.merge(
                cover_fit_attempt_counts,
                on=["method", "optimizer", "seed"],
                how="left",
                validate="one_to_one",
            ).merge(
                cover_success_counts,
                on=["method", "optimizer", "seed"],
                how="left",
                validate="one_to_one",
            )
            for column in (
                "primary_fit_attempt_states", "successful_primary_fit_states"
            ):
                cover_coverage[column] = cover_coverage[column].fillna(0).astype(int)
            cover_rank_audit.to_csv(
                analysis_dir / "ecs_cover_exact_rank_audit.csv", index=False
            )
            cover_coverage.to_csv(
                analysis_dir / "ecs_cover_coverage_by_run.csv", index=False
            )
            fig, ax = plt.subplots(figsize=(10.0, 4.5))
            labels = [
                (
                    f"{'full-q' if 'full_row' in row.method else 'detX-q'}:"
                    f"{row.optimizer}/s{int(row.seed)}"
                )
                for row in cover_coverage.itertuples(index=False)
            ]
            positions = np.arange(len(labels))
            width = 0.24
            ax.bar(
                positions - width,
                cover_coverage["exact_weightwatcher_states"].to_numpy(dtype=float),
                width=width,
                label="exact WW states",
            )
            ax.bar(
                positions,
                cover_coverage["exact_trace_states"].to_numpy(dtype=float),
                width=width,
                label="exact trace states",
            )
            ax.bar(
                positions + width,
                cover_coverage["rank_eligible_cover_states"].to_numpy(dtype=float),
                width=width,
                label="rank-eligible cover states",
            )
            ax.scatter(
                positions + width,
                cover_coverage["successful_primary_fit_states"].to_numpy(dtype=float),
                color="black",
                marker="x",
                label="successful primary fits",
            )
            ax.set_xticks(positions, labels, rotation=45, ha="right")
            ax.set_ylabel("checkpoint states")
            ax.set_title("ECS cover: exact sparse-rank coverage in final-100 cache")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(analysis_dir / "ecs_cover_exact_rank_coverage.png", dpi=180)
            if SHOW_PLOTS:
                plt.show()
            else:
                plt.close(fig)
            display(cover_coverage)
            numerical_audit = operator_rows[
                operator_rows["method"].astype(str).str.startswith("small_explicit_")
            ]
            numerical_audit.to_csv(analysis_dir / "small_numerical_validation.csv", index=False)
            display(numerical_audit)
            """
        ),
        markdown(
            """
            The six explicit Jacobian families are formula validations on
            `6x8`, not MLP estimates. The large-layer spectra are exact analytic
            pullback spectra of the same six maps, avoiding full Jacobian
            materialization. Any apparent alpha near two belongs to the named
            candidate RG map only. The ECS cover uses only exact intersections
            between the final-100 cache and sparse WeightWatcher/trace states;
            missing rank states are audited and never filled from neighbors.
            Its full-$q$ and detX-$q$ rows differ only in deterministic shell
            multiplicity at fixed $k$, so detX-$q$ is not an independent alpha
            check. ECS clipping removes complete core-amplitude groups and
            `fit_ok` additionally requires at least `MINIMUM_TAIL` physical
            retained-core groups above the package-selected `xmin`.
            """
        ),
    ]
    return notebook("13_Single_Checkpoint_Map_Jacobians.ipynb", cells)


def calibrated_training_map_notebook() -> tuple[str, dict[str, object]]:
    cells = [
        markdown(
            """
            # Calibrated local training-map response

            Replay the exact captured minibatch, loss, gradient clipping,
            optimizer state, scheduler rates, model state, and pre-forward RNG
            state. Central differences perturb one named weight matrix while
            every other calibration variable stays fixed. This is the first
            notebook in the suite that estimates a derivative of a specified
            *training step* rather than an algebraic weight map. It requires
            saved batches, RNG state, and optimizer state that model-only tail
            checkpoints intentionally omit, so it reads the existing dense
            captures and never launches or resumes training.
            """
        ),
        parameters(
            """
            OPTIMIZER_SLUGS = ["adamw", "muon", "muonclip_rms"]
            LAYERS = ["fc1.weight", "fc2.weight"]
            MAXIMUM_CALIBRATIONS = 1
            PROBE_COUNT = 16
            FINITE_DIFFERENCE_EPSILONS = [1e-3, 3e-3, 1e-2]
            REPLAY_DEVICE = "manifest"
            STRICT_REPLAY = True
            ALLOW_UNQUALIFIED_REPLAY = False
            REPLAY_MAX_ABS_TOLERANCE = 1e-4
            TOP_K_VALUES = [0, 1, 2, 3, 4, 5]
            MINIMUM_TAIL = 8
            METHOD_SLUG = "calibrated_local_training_map"
            PLOT_TITLE = "Calibrated training-map JVP spectra: alpha with 95% seed CI"
            """
        ),
        code(BOOTSTRAP),
        code(COMMON_IMPORTS + "\n" + ANALYSIS_IMPORTS),
        caveat(
            "Calibrated map",
            "calibrated_training_map_finite_difference_jvp_not_w_only",
            "T_{B,S}(W): one captured MLP3 loss/clip/optimizer step on fixed batch B and complete pre-step state S, with one matrix coordinate perturbed.",
            "The derivative is local to the saved batch, optimizer state, scheduler rates, RNG "
            "state, device, direction, and epsilon. It is not identifiable from W alone and is not "
            "automatically a continuum RG beta-function Jacobian.",
        ),
        code(COMMON_HELPERS + "\n" + TAIL_CHECKPOINT_CACHE_HELPERS + "\n" + CHECKPOINT_HELPERS),
        markdown("## Calibration artifact audit and unperturbed replay"),
        code(
            """
            import torch

            calibrated_captures = {}
            replay_rows = []
            replay_qualification_by_capture = {}
            for optimizer in OPTIMIZER_SLUGS:
                for seed in SEEDS:
                    seed_dir = require_complete_seed(optimizer, seed)
                    run_fingerprint = verified_run_fingerprint(optimizer, seed)
                    config = resolved_training_config(seed_dir)
                    manifest = json.loads(
                        (Path(seed_dir) / "manifest.json").read_text(encoding="utf-8")
                    )
                    original_device = str(manifest.get("device", "unknown"))
                    replay_device = (
                        original_device
                        if str(REPLAY_DEVICE).strip().lower() in {"", "manifest"}
                        else str(REPLAY_DEVICE)
                    )
                    available = [
                        (path, payload)
                        for path, payload in capture_payloads(seed_dir)
                        if isinstance(payload.get("calibration_state"), dict)
                    ]
                    if not available:
                        raise FileNotFoundError(
                            f"No first-in-burst calibration_state for {optimizer}, seed={seed}. "
                            "The capture must include inputs, targets, model/optimizer state, "
                            "scheduler rates, loss/clip definition, and pre-forward RNG."
                        )
                    available = available[-int(MAXIMUM_CALIBRATIONS):]
                    calibrated_captures[(optimizer, int(seed))] = (
                        config, available, original_device, replay_device,
                        run_fingerprint,
                    )
                    for path, payload in available:
                        replay = replay_calibrated_step(
                            payload,
                            config,
                            parameter_perturbations=None,
                            device=replay_device,
                            expected_fingerprint=str(payload["protocol_fingerprint"]),
                        )
                        errors = replay.get("reference_max_abs_error", {})
                        maximum_error = max(errors.values(), default=np.nan)
                        replay_qualified = bool(
                            np.isfinite(maximum_error)
                            and maximum_error <= REPLAY_MAX_ABS_TOLERANCE
                        )
                        replay_qualification_by_capture[str(path)] = replay_qualified
                        replay_rows.append({
                            "optimizer": optimizer, "seed": int(seed),
                            "protocol_fingerprint": run_fingerprint,
                            "source_artifact_kind": "verified_calibrated_dense_capture",
                            "state_index": int(payload["completed_step"]),
                            "capture": str(path),
                            "original_training_device": original_device,
                            "replay_device": replay_device,
                            "maximum_reference_abs_error": maximum_error,
                            "replay_max_abs_tolerance": float(REPLAY_MAX_ABS_TOLERANCE),
                            "replay_qualified": replay_qualified,
                            "evidence_role": (
                                "qualified_calibrated_replay"
                                if replay_qualified else "exploratory_unqualified_replay"
                            ),
                            "operator_kind": "unperturbed_calibrated_step_replay_audit",
                            "map_definition": "exact stored batch/state/RNG replay; original-device replay is the bitwise target",
                        })
                        if not replay_qualified and (
                            bool(STRICT_REPLAY) or not bool(ALLOW_UNQUALIFIED_REPLAY)
                        ):
                            raise RuntimeError(
                                f"Replay error {maximum_error} exceeds tolerance at {path}. "
                                f"Original device={original_device!r}, replay device={replay_device!r}. "
                                "Use the manifest device before rejecting the artifact. For a "
                                "non-qualifying exploratory run only, set STRICT_REPLAY=False and "
                                "ALLOW_UNQUALIFIED_REPLAY=True."
                            )
            replay_audit = pd.DataFrame(replay_rows)
            display(replay_audit)
            """
        ),
        markdown("## Random-subspace Jacobian sketch with epsilon sensitivity"),
        code(
            """
            operator_records = replay_rows.copy()
            fit_frames, trace_frames = [], []
            spectral_arrays = {}
            for (optimizer, seed), (
                config, captures, original_device, replay_device, run_fingerprint
            ) in calibrated_captures.items():
                for capture_path, capture in captures:
                    calibration = capture["calibration_state"]
                    step = int(capture["completed_step"])
                    replay_qualified = bool(
                        replay_qualification_by_capture[str(capture_path)]
                    )
                    for layer_index, layer in enumerate(LAYERS):
                        if layer not in calibration["model_state_before_step"]:
                            raise KeyError(f"Calibration state lacks {layer}")
                        base_tensor = calibration["model_state_before_step"][layer].detach().cpu()
                        base_weight = base_tensor.double().numpy()
                        map_definition = (
                            f"T_B,S({layer}): replay one {optimizer} step at capture {capture_path.name}; "
                            f"batch_shape={tuple(calibration['inputs'].shape)}, loss={calibration['loss_definition']}, "
                            f"clip={calibration['gradient_clipping']}, device={replay_device}; all non-{layer} state fixed"
                        )

                        def calibrated_map(candidate, batch, optimizer_state):
                            if batch is None or optimizer_state is None:
                                raise ValueError("batch and optimizer state are calibration requirements")
                            delta = np.asarray(candidate, dtype=np.float64) - base_weight
                            perturbation = torch.as_tensor(delta, dtype=base_tensor.dtype)
                            replay = replay_calibrated_step(
                                capture,
                                config,
                                parameter_perturbations={layer: perturbation},
                                device=replay_device,
                                expected_fingerprint=str(
                                    capture["protocol_fingerprint"]
                                ),
                            )
                            return replay["model_state_after_step"][layer].double().numpy()

                        if int(PROBE_COUNT) < int(MINIMUM_TAIL):
                            raise ValueError("PROBE_COUNT must be at least MINIMUM_TAIL")
                        probe_seed = 10_000_019 * int(seed) + 1009 * layer_index + step
                        rng = np.random.default_rng(probe_seed)
                        raw_directions = rng.normal(
                            size=(base_weight.size, int(PROBE_COUNT))
                        )
                        subspace_basis, _ = np.linalg.qr(raw_directions, mode="reduced")
                        del raw_directions
                        for epsilon in FINITE_DIFFERENCE_EPSILONS:
                            response_columns = []
                            directional_records = []
                            for probe in range(int(PROBE_COUNT)):
                                direction = subspace_basis[:, probe].reshape(base_weight.shape)
                                derivative = single_checkpoint.calibrated_training_map_finite_difference(
                                    calibrated_map,
                                    base_weight,
                                    direction,
                                    batch=calibration["inputs"],
                                    optimizer_state=calibration["optimizer_state_before_step"],
                                    map_definition=map_definition,
                                    epsilon=float(epsilon),
                                )
                                response = np.asarray(
                                    derivative.directional_derivative, dtype=np.float64
                                ).reshape(-1)
                                response_columns.append(response)
                                directional_base = {
                                    "optimizer": optimizer, "seed": int(seed), "layer": layer,
                                    "protocol_fingerprint": run_fingerprint,
                                    "source_artifact_kind": "verified_calibrated_dense_capture",
                                    "state_index": step, "capture": str(capture_path),
                                    "probe": int(probe), "epsilon": float(epsilon),
                                    "probe_seed": int(probe_seed),
                                    "probe_count": int(PROBE_COUNT),
                                    "original_training_device": original_device,
                                    "replay_device": replay_device,
                                    "replay_qualified": replay_qualified,
                                    "evidence_role": (
                                        "qualified_calibrated_replay"
                                        if replay_qualified
                                        else "exploratory_unqualified_replay"
                                    ),
                                    "method": "calibrated_training_step_jvp_norm_diagnostic",
                                    "fc3_rank10_warning": False,
                                }
                                directional_records.append({
                                    **directional_base,
                                    "operator_kind": derivative.operator_kind,
                                    "map_definition": derivative.map_definition,
                                    "backend": derivative.backend,
                                    "response_frobenius_norm": float(np.linalg.norm(response)),
                                    "batch_supplied": derivative.batch_supplied,
                                    "optimizer_state_supplied": derivative.optimizer_state_supplied,
                                    "weight_only": derivative.weight_only,
                                })
                            operator_records.extend(directional_records)
                            response_matrix = np.column_stack(response_columns)
                            update_response_matrix = response_matrix - subspace_basis
                            training_definition = (
                                map_definition
                                + f"; Y=J V for a fixed {PROBE_COUNT}-dimensional orthonormal "
                                + f"random domain V (probe_seed={probe_seed}); fitted amplitudes are "
                                + "svd(Y), a restricted-domain Jacobian sketch, not the full-layer ESD"
                            )
                            update_definition = (
                                map_definition
                                + f"; B(W)=T_B,S(W)-W with delta_s=1, so Y_B=(J_T-I)V "
                                + f"on the same fixed {PROBE_COUNT}-dimensional orthonormal random "
                                + f"domain (probe_seed={probe_seed}); restricted sketch, not full ESD"
                            )
                            sketch_candidates = (
                                ("calibrated_training_map_random_subspace_jacobian_sketch",
                                 "calibrated_training_map_random_subspace_jacobian_sketch",
                                 training_definition, response_matrix),
                                ("calibrated_update_beta_random_subspace_jacobian_sketch",
                                 "calibrated_update_beta_random_subspace_jacobian_sketch",
                                 update_definition, update_response_matrix),
                            )
                            for method, operator_kind, sketch_definition, matrix in sketch_candidates:
                                restricted_gram = matrix.T @ matrix
                                amplitudes = np.sqrt(
                                    np.maximum(np.linalg.eigvalsh(restricted_gram), 0.0)
                                )[::-1]
                                galerkin = subspace_basis.T @ matrix
                                galerkin_amplitudes = np.linalg.svd(galerkin, compute_uv=False)
                                base = {
                                    "optimizer": optimizer, "seed": int(seed), "layer": layer,
                                    "protocol_fingerprint": run_fingerprint,
                                    "source_artifact_kind": "verified_calibrated_dense_capture",
                                    "state_index": step, "capture": str(capture_path),
                                    "epsilon": float(epsilon), "probe_seed": int(probe_seed),
                                    "probe_count": int(PROBE_COUNT), "method": method,
                                    "original_training_device": original_device,
                                    "replay_device": replay_device,
                                    "replay_qualified": replay_qualified,
                                    "evidence_role": (
                                        "qualified_calibrated_replay"
                                        if replay_qualified
                                        else "exploratory_unqualified_replay"
                                    ),
                                    "fc3_rank10_warning": False,
                                }
                                operator_records.append({
                                    **base,
                                    "operator_kind": operator_kind,
                                    "map_definition": sketch_definition,
                                    "restricted_rank": int(np.count_nonzero(amplitudes > 0.0)),
                                    "largest_restricted_singular_amplitude": float(amplitudes[0]),
                                    "smallest_restricted_singular_amplitude": float(amplitudes[-1]),
                                    "galerkin_rank": int(np.count_nonzero(galerkin_amplitudes > 0.0)),
                                })
                                fits, traces = fit_spectrum_with_trace(
                                    amplitudes,
                                    operator_kind=operator_kind,
                                    map_definition=sketch_definition + "; energy e=b^2 is an exact fit transform",
                                    spectrum_kind="amplitude",
                                    metadata=base,
                                    top_k_values=TOP_K_VALUES,
                                    minimum_tail=MINIMUM_TAIL,
                                )
                                fit_frames.append(fits); trace_frames.append(traces)
                                spectral_arrays[f"{optimizer}_{seed}_{layer}_{step}_{method}_eps{epsilon}"] = amplitudes
                                spectral_arrays[f"{optimizer}_{seed}_{layer}_{step}_{method}_galerkin_eps{epsilon}"] = galerkin_amplitudes
            if not fit_frames:
                raise RuntimeError("No calibrated training-map responses were fit")
            """
        ),
        code(ANALYSIS_SAVE_AND_PLOT),
        code(
            """
            np.savez_compressed(analysis_dir / "positive_spectra.npz", **spectral_arrays)
            display(save_spectrum_ccdf_gallery(spectral_arrays, method_slug=METHOD_SLUG))
            replay_audit.to_csv(analysis_dir / "unperturbed_replay_audit.csv", index=False)
            primary_energy = fit_rows[
                fit_rows["clip_top_k"].eq(0)
                & fit_rows["spectrum_kind"].astype(str).eq("energy_derived_from_amplitude")
            ]
            epsilon_summary = ci_summary(
                primary_energy,
                groups=("optimizer", "layer", "method", "epsilon"),
                metrics=("alpha", "ks_D", "n_tail"),
            )
            epsilon_summary.to_csv(analysis_dir / "epsilon_sensitivity_95ci.csv", index=False)
            display(epsilon_summary)
            """
        ),
        markdown(
            """
            The unperturbed replay is a mandatory calibration audit and defaults
            to the manifest's original device. Cross-device replay is
            exploratory: unless the recorded tolerance passes, its Jacobian
            sketches are labelled non-qualifying and excluded by notebook `15`.
            The fitted spectrum is `svd(JV)` for the declared orthonormal
            random domain, not the SVD of a reshaped single JVP and not the full
            Jacobian ESD. Probe and epsilon rows are repeated measurements
            within each seed, not extra independent replicates.
            """
        ),
    ]
    return notebook("14_Calibrated_Local_Training_Map.ipynb", cells)


def nulls_stability_notebook() -> tuple[str, dict[str, object]]:
    cells = [
        markdown(
            """
            # Method nulls and stability comparison

            Require the prior method outputs, compare only their preregistered
            energy-derived primary fits, and recompute matched scale, rotation,
            Gaussian, and Haar/Stiefel controls on the same final pair from the
            verified tail-checkpoint cache. Invariance controls and distributional nulls answer
            different questions and are never pooled.
            """
        ),
        parameters(
            """
            OPTIMIZER_SLUGS = ["adamw", "muon", "muonclip_rms"]
            LAYERS = ["fc1.weight", "fc2.weight", "fc3.weight"]
            REQUIRED_METHODS = [
                "two_checkpoint_finite_flow",
                "muon_update_stiefel_tangent",
                "radial_angular_quotients",
                "single_checkpoint_map_jacobians",
                "calibrated_local_training_map",
            ]
            REQUIRED_METHOD_SOURCES = {
                "two_checkpoint_finite_flow": ["verified_tail_checkpoint_cache_model_only"],
                "muon_update_stiefel_tangent": ["verified_dense_update_capture"],
                "radial_angular_quotients": ["verified_tail_checkpoint_cache_model_only"],
                "single_checkpoint_map_jacobians": [
                    "verified_tail_checkpoint_cache_model_only",
                    "verified_tail_checkpoint_cache_plus_exact_sparse_weightwatcher_trace_metrics",
                ],
                "calibrated_local_training_map": ["verified_calibrated_dense_capture"],
            }
            REQUIRED_ECS_PRIMARY_METHOD = (
                "ecs_grassmann_cartan_cover_full_row_shell_pullback"
            )
            REQUIRED_ECS_FIT_CONTRACT_TOKEN = (
                "ecs_grassmann_cartan_cover_group_qualified_v1"
            )
            TOP_K_VALUES = [0, 1, 2, 3, 4, 5]
            MINIMUM_TAIL = 8
            METHOD_SLUG = "method_nulls_stability"
            PLOT_TITLE = "Matched single-checkpoint nulls: alpha with 95% seed CI"
            """
        ),
        code(BOOTSTRAP),
        code(COMMON_IMPORTS + "\n" + ANALYSIS_IMPORTS),
        caveat(
            "Null and stability bundle",
            "matched_null_and_preregistered_method_stability_comparison",
            "Frozen prior-method fit tables plus scale/rotation invariance audits and matched Gaussian/Haar controls on the same matrices.",
            "A method is not validated by alpha proximity alone. It must separate from relevant "
            "distributional nulls while passing the invariances its map claims and remaining stable "
            "over seed, spacing, epsilon, probes, and explicit finger sensitivity.",
        ),
        code(COMMON_HELPERS + "\n" + TAIL_CHECKPOINT_CACHE_HELPERS + "\n" + CHECKPOINT_HELPERS),
        markdown("## Require every prior analysis artifact"),
        code(
            """
            current_run_fingerprints = {}
            for optimizer in OPTIMIZER_SLUGS:
                for seed in SEEDS:
                    require_complete_seed(optimizer, seed)
                    current_run_fingerprints[f"{optimizer}:{int(seed)}"] = (
                        verified_run_fingerprint(optimizer, seed)
                    )
            prior_frames = []
            for method_slug in REQUIRED_METHODS:
                method_dir = require_path(
                    OUTPUT_ROOT_PATH / "analyses" / method_slug,
                    description=f"{method_slug} analysis output",
                )
                fit_path = require_path(
                    method_dir / "powerlaw_fits.csv",
                    description=f"{method_slug} power-law fit table",
                )
                provenance_path = require_path(
                    method_dir / "method_provenance.json",
                    description=f"{method_slug} method provenance manifest",
                )
                method_provenance = json.loads(
                    provenance_path.read_text(encoding="utf-8")
                )
                method_optimizers = (
                    ["muon", "muonclip_rms"]
                    if method_slug == "muon_update_stiefel_tangent"
                    else list(OPTIMIZER_SLUGS)
                )
                expected_method_grid = {
                    f"{optimizer}:{int(seed)}": current_run_fingerprints[
                        f"{optimizer}:{int(seed)}"
                    ]
                    for optimizer in method_optimizers
                    for seed in SEEDS
                }
                provenance_checks = {
                    "schema_version": (method_provenance.get("schema_version"), 1),
                    "suite_name": (method_provenance.get("suite_name"), PROTOCOL_SLUG),
                    "method_slug": (method_provenance.get("method_slug"), method_slug),
                    "optimizer_seed_protocol_fingerprints": (
                        method_provenance.get("optimizer_seed_protocol_fingerprints"),
                        dict(sorted(expected_method_grid.items())),
                    ),
                    "source_artifact_kinds": (
                        method_provenance.get("source_artifact_kinds"),
                        sorted(REQUIRED_METHOD_SOURCES[method_slug]),
                    ),
                }
                if method_slug == "single_checkpoint_map_jacobians":
                    provenance_checks["analysis_contract_tokens"] = (
                        method_provenance.get("analysis_contract_tokens"),
                        [REQUIRED_ECS_FIT_CONTRACT_TOKEN],
                    )
                provenance_mismatches = [
                    f"{name}: observed={observed!r}, expected={expected!r}"
                    for name, (observed, expected) in provenance_checks.items()
                    if observed != expected
                ]
                if provenance_mismatches:
                    raise RuntimeError(
                        f"Stale or incompatible method provenance for {method_slug}:\\n  - "
                        + "\\n  - ".join(provenance_mismatches)
                    )
                frame = pd.read_csv(fit_path)
                if method_provenance.get("fit_row_count") != int(len(frame)):
                    raise RuntimeError(
                        f"{fit_path} row count disagrees with method_provenance.json"
                    )
                required = {
                    "optimizer", "seed", "protocol_fingerprint",
                    "source_artifact_kind", "operator_kind", "map_definition",
                    "method", "alpha", "clip_top_k", "spectrum_kind", "fit_ok",
                }
                missing = required - set(frame.columns)
                if missing:
                    raise ValueError(f"{fit_path} missing {sorted(missing)}")
                if method_slug == "single_checkpoint_map_jacobians":
                    ecs_contract_columns = {
                        "analysis_contract_token",
                        "ecs_clip_core_groups",
                        "ecs_tail_core_group_count",
                        "ecs_minimum_tail_core_groups",
                        "ecs_group_tail_qualified",
                        "ecs_mode_level_fit_ok_before_group_gate",
                        "mode_group_count_consistency_verified",
                    }
                    missing_ecs_contract = ecs_contract_columns - set(frame.columns)
                    if missing_ecs_contract:
                        raise RuntimeError(
                            "Stale single-checkpoint fit artifact lacks ECS group "
                            f"qualification fields: {sorted(missing_ecs_contract)}"
                        )
                    ecs_contract_rows = frame[
                        frame["method"].astype(str).str.startswith(
                            "ecs_grassmann_cartan_cover_"
                        )
                    ].copy()
                    if ecs_contract_rows.empty:
                        raise RuntimeError(
                            "Single-checkpoint artifact contains no ECS cover fit rows"
                        )
                    observed_contract_tokens = set(
                        ecs_contract_rows["analysis_contract_token"]
                        .dropna().astype(str)
                    )
                    if observed_contract_tokens != {
                        REQUIRED_ECS_FIT_CONTRACT_TOKEN
                    }:
                        raise RuntimeError(
                            "Single-checkpoint ECS fit contract token is stale: "
                            f"{sorted(observed_contract_tokens)}"
                        )
                    if ecs_contract_rows[
                        [
                            "ecs_clip_core_groups",
                            "ecs_tail_core_group_count",
                            "ecs_minimum_tail_core_groups",
                            "ecs_group_tail_qualified",
                            "ecs_mode_level_fit_ok_before_group_gate",
                            "mode_group_count_consistency_verified",
                        ]
                    ].isna().any().any():
                        raise RuntimeError(
                            "Single-checkpoint ECS group qualification contains nulls"
                        )
                    if not boolean_series(
                        ecs_contract_rows[
                            "mode_group_count_consistency_verified"
                        ]
                    ).all():
                        raise RuntimeError(
                            "Single-checkpoint ECS mode/group count invariant failed"
                        )
                identity_columns = [
                    "optimizer", "seed", "protocol_fingerprint",
                    "source_artifact_kind",
                ]
                if frame[identity_columns].isna().any().any():
                    raise RuntimeError(
                        f"{fit_path} contains null optimizer/seed/fingerprint/source identity"
                    )
                observed_fit_grid = {}
                for identity, identity_rows in frame.groupby(
                    ["optimizer", "seed"], dropna=False
                ):
                    optimizer, seed = str(identity[0]), int(identity[1])
                    fingerprints = set(
                        identity_rows["protocol_fingerprint"].dropna().astype(str)
                    )
                    sources = set(
                        identity_rows["source_artifact_kind"].dropna().astype(str)
                    )
                    if len(fingerprints) != 1 or sources != set(
                        REQUIRED_METHOD_SOURCES[method_slug]
                    ):
                        raise RuntimeError(
                            f"{fit_path} has mixed/missing provenance for "
                            f"optimizer={optimizer}, seed={seed}"
                        )
                    observed_fit_grid[f"{optimizer}:{seed}"] = next(iter(fingerprints))
                if observed_fit_grid != expected_method_grid:
                    raise RuntimeError(
                        f"{fit_path} optimizer/seed fingerprint grid is stale or incomplete"
                    )
                frame["analysis_method"] = method_slug
                prior_frames.append(frame)
            prior_fits = pd.concat(prior_frames, ignore_index=True, sort=False)
            prior_primary = prior_fits[
                prior_fits["clip_top_k"].eq(0)
                & prior_fits["spectrum_kind"].astype(str).eq("energy_derived_from_amplitude")
            ].copy()
            prior_primary = prior_primary[
                boolean_series(prior_primary["fit_ok"])
            ]
            expected_ecs_primary_grid = {
                (str(optimizer), int(seed))
                for optimizer in OPTIMIZER_SLUGS
                for seed in SEEDS
            }
            successful_ecs_primary = prior_primary[
                prior_primary["analysis_method"].astype(str).eq(
                    "single_checkpoint_map_jacobians"
                )
                & prior_primary["method"].astype(str).eq(
                    REQUIRED_ECS_PRIMARY_METHOD
                )
                & boolean_series(prior_primary["ecs_group_tail_qualified"])
            ].copy()
            observed_ecs_primary_grid = {
                (str(row.optimizer), int(row.seed))
                for row in successful_ecs_primary[
                    ["optimizer", "seed"]
                ].drop_duplicates().itertuples(index=False)
            }
            if observed_ecs_primary_grid != expected_ecs_primary_grid:
                missing_ecs_primary = sorted(
                    expected_ecs_primary_grid - observed_ecs_primary_grid
                )
                unexpected_ecs_primary = sorted(
                    observed_ecs_primary_grid - expected_ecs_primary_grid
                )
                raise RuntimeError(
                    "Method comparison is incomplete: the primary full-q ECS "
                    "Grassmann cover lacks a successful group-qualified "
                    "energy fit for the exact optimizer/seed grid. "
                    f"missing={missing_ecs_primary}, "
                    f"unexpected={unexpected_ecs_primary}"
                )
            calibrated_mask = prior_primary["analysis_method"].astype(str).eq(
                "calibrated_local_training_map"
            )
            if calibrated_mask.any() and "replay_qualified" not in prior_primary:
                raise RuntimeError(
                    "Calibrated-map fit rows lack mandatory replay_qualified provenance"
                )
            replay_ok = (
                boolean_series(prior_primary["replay_qualified"])
                if "replay_qualified" in prior_primary
                else pd.Series(False, index=prior_primary.index)
            )
            unqualified_calibrated_rows = prior_primary[
                calibrated_mask & ~replay_ok
            ].copy()
            prior_primary = prior_primary[~calibrated_mask | replay_ok].copy()
            if prior_primary.empty:
                raise RuntimeError("No successful prior energy-derived primary fits")
            if not unqualified_calibrated_rows.empty:
                print(
                    "Excluded calibrated-map rows whose unperturbed replay did not "
                    "qualify; they cannot enter method retention."
                )
                display(unqualified_calibrated_rows.head(20))
            display(prior_primary.head(30))
            """
        ),
        markdown("## Matched final-checkpoint nulls and invariance checks"),
        code(
            """
            operator_records, fit_frames, trace_frames = [], [], []
            spectral_arrays = {}
            for optimizer in OPTIMIZER_SLUGS:
                for seed in SEEDS:
                    seed_dir = require_tail_checkpoint_cache(optimizer, seed)
                    run_fingerprint = verified_run_fingerprint(optimizer, seed)
                    checkpoint_refs = analysis_checkpoint_refs(seed_dir)
                    previous_ref, final_ref = checkpoint_refs[-2], checkpoint_refs[-1]
                    for layer_index, layer in enumerate(LAYERS):
                        W = checkpoint_matrix(final_ref.path, layer)
                        rng = np.random.default_rng(31_337 * int(seed) + layer_index)
                        invariant = nulls.check_invariances(
                            lambda matrix: single_checkpoint.normalized_gram_map(matrix),
                            W,
                            rng=rng,
                            value_fn=lambda record: record.eigenvalues,
                        )
                        base = {
                            "optimizer": optimizer, "seed": int(seed), "layer": layer,
                            "protocol_fingerprint": run_fingerprint,
                            "source_artifact_kind": "verified_tail_checkpoint_cache_model_only",
                            "state_index": int(final_ref.global_step),
                            "checkpoint_source": "verified_final_100_tail_cache",
                            "checkpoint_cache_seed_dir": str(seed_dir),
                            "fc3_rank10_warning": "fc3" in layer.lower(),
                        }
                        for case in invariant.cases:
                            operator_records.append({
                                **base, "method": "normalized_gram_invariance",
                                "null_kind": case.name,
                                "operator_kind": case.operator_kind,
                                "map_definition": case.map_definition,
                                "passed": case.passed,
                                "absolute_error": case.absolute_error,
                                "relative_error": case.relative_error,
                            })
                        identical = two_checkpoint.finite_difference_beta(W, W, 1.0)
                        operator_records.append({
                            **base, "method": "identical_checkpoint_zero",
                            "null_kind": "identical_checkpoint",
                            "operator_kind": identical.operator_kind,
                            "map_definition": identical.map_definition,
                            "beta_norm": identical.beta_norm,
                            "expected_zero": bool(identical.beta_norm == 0.0),
                        })

                        W_previous = checkpoint_matrix(previous_ref.path, layer)
                        delta_s = int(final_ref.global_step - previous_ref.global_step)
                        scaled_flow = nulls.scale_null(W_previous, rng=rng, scale=1.7)
                        rotated_flow = nulls.rotation_null(W_previous, rng=rng, side="both")
                        finite_flow_cases = (
                            ("trained", W_previous, W,
                             "chronological adjacent final checkpoint pair"),
                            ("temporally_reversed", W, W_previous,
                             "same adjacent pair with temporal order reversed"),
                            ("known_global_scale", W_previous, scaled_flow.sample,
                             scaled_flow.map_definition),
                            ("pure_left_right_rotation", W_previous, rotated_flow.sample,
                             rotated_flow.map_definition),
                        )
                        for null_kind, flow_start, flow_end, null_definition in finite_flow_cases:
                            flow = two_checkpoint.analyze_two_checkpoints(
                                flow_start, flow_end, delta_s
                            )
                            operator_records.append({
                                **base,
                                "method": "finite_flow_radial_angular_null",
                                "null_kind": null_kind,
                                "operator_kind": flow.operator_kind,
                                "map_definition": flow.map_definition + "; control=" + null_definition,
                                "delta_s": delta_s,
                                "radial_retained_rank": flow.radial.retained_rank,
                                "column_geodesic_rate": flow.grassmann.column.geodesic_rate,
                                "row_geodesic_rate": flow.grassmann.row.geodesic_rate,
                                "rectangular_transfer_available": flow.rectangular_transfer.available,
                                "rectangular_transfer_orientation": flow.rectangular_transfer.orientation,
                                "rectangular_transfer_rank0": flow.rectangular_transfer.numerical_rank0,
                                "rectangular_transfer_rank1": flow.rectangular_transfer.numerical_rank1,
                                "rectangular_transfer_rank_rtol": flow.rectangular_transfer.rank_rtol,
                                "rectangular_transfer_threshold0": flow.rectangular_transfer.rank_threshold0,
                                "rectangular_transfer_threshold1": flow.rectangular_transfer.rank_threshold1,
                                "rectangular_transfer_condition0": flow.rectangular_transfer.condition_number0,
                                "rectangular_transfer_condition1": flow.rectangular_transfer.condition_number1,
                                "rectangular_transfer_structural_zeros": flow.rectangular_transfer.structural_zero_count,
                                "rectangular_transfer_reconstruction_residual": flow.rectangular_transfer.relative_reconstruction_residual,
                                "rectangular_transfer_unsupported_residual": flow.rectangular_transfer.unsupported_source_action_residual,
                                "rectangular_transfer_core_residual": flow.rectangular_transfer.core_reconstruction_residual,
                                "rectangular_transfer_alignment_residual": flow.rectangular_transfer.subspace_alignment_residual,
                            })
                            flow_spectra = [
                                ("finite_flow_radial_null", flow.radial.radial_rate_amplitudes,
                                 flow.radial.operator_kind, flow.radial.map_definition),
                                ("finite_flow_column_grassmann_null", flow.grassmann.column.geodesic_rates,
                                 flow.grassmann.column.operator_kind, flow.grassmann.column.map_definition),
                            ]
                            if flow.rectangular_transfer.available:
                                flow_spectra.extend([
                                    (
                                        "supported_rectangular_transfer_log_rate_null",
                                        flow.rectangular_transfer.supported_transfer_rate_amplitudes,
                                        flow.rectangular_transfer.operator_kind,
                                        flow.rectangular_transfer.map_definition,
                                    ),
                                    (
                                        "procrustes_aligned_core_log_rate_null",
                                        flow.rectangular_transfer.core_rate_amplitudes,
                                        flow.rectangular_transfer.operator_kind,
                                        flow.rectangular_transfer.map_definition,
                                    ),
                                ])
                            for method, amplitudes, kind, definition in flow_spectra:
                                positive = np.asarray(amplitudes, dtype=float)
                                positive = positive[np.isfinite(positive) & (positive > 0.0)]
                                if positive.size < 2:
                                    operator_records.append({
                                        **base, "method": method, "null_kind": null_kind,
                                        "operator_kind": kind,
                                        "map_definition": definition + "; control=" + null_definition,
                                        "fit_available": False,
                                        "unavailable_reason": "fewer than two positive null amplitudes",
                                    })
                                    continue
                                metadata = {
                                    **base, "method": method, "null_kind": null_kind,
                                    "delta_s": delta_s,
                                }
                                fits, traces = fit_spectrum_with_trace(
                                    positive,
                                    operator_kind=kind,
                                    map_definition=definition + "; control=" + null_definition + "; energy e=b^2 is an exact fit transform",
                                    spectrum_kind="amplitude",
                                    metadata=metadata,
                                    top_k_values=TOP_K_VALUES,
                                    minimum_tail=MINIMUM_TAIL,
                                )
                                fit_frames.append(fits); trace_frames.append(traces)
                                spectral_arrays[f"{optimizer}_{seed}_{layer}_{method}_{null_kind}"] = positive

                        samples = [("trained", W, None)]
                        for generated in (
                            nulls.scale_null(W, rng=rng, scale=1.7),
                            nulls.rotation_null(W, rng=rng, side="both"),
                            nulls.gaussian_null(W, rng=rng, match="frobenius"),
                            nulls.haar_polar_null(W.shape, rng=rng),
                        ):
                            samples.append((generated.null_kind, generated.sample, generated))
                        for null_kind, sample, null_record in samples:
                            observables = (
                                polar.polar_pullback_spectrum(sample),
                                single_checkpoint.normalized_gram_analytic_spectrum(sample),
                            )
                            for observable in observables:
                                kind = observable.operator_kind
                                definition = observable.map_definition
                                if null_record is not None:
                                    kind = f"{kind}|under:{null_record.operator_kind}"
                                    definition = f"{definition}; null transform: {null_record.map_definition}"
                                method = (
                                    "polar_pullback_null"
                                    if "polar" in observable.operator_kind
                                    else "normalized_gram_pullback_null"
                                )
                                metadata = {**base, "method": method, "null_kind": null_kind}
                                amplitudes = observable.singular_amplitudes
                                operator_records.append({
                                    **metadata,
                                    "operator_kind": kind,
                                    "map_definition": definition,
                                    "derivative_rank": observable.derivative_rank,
                                    "zero_count": observable.zero_count,
                                })
                                fits, traces = fit_spectrum_with_trace(
                                    amplitudes,
                                    operator_kind=kind,
                                    map_definition=definition + "; fitted b are derivative singular amplitudes and energy e=b^2 is an exact fit transform",
                                    spectrum_kind="amplitude",
                                    metadata=metadata,
                                    top_k_values=TOP_K_VALUES,
                                    minimum_tail=MINIMUM_TAIL,
                                )
                                fit_frames.append(fits); trace_frames.append(traces)
                                spectral_arrays[f"{optimizer}_{seed}_{layer}_{method}_{null_kind}"] = amplitudes
            if not fit_frames:
                raise RuntimeError("No matched-null spectrum was fit")
            """
        ),
        code(ANALYSIS_SAVE_AND_PLOT),
        markdown("## Complete-run method stability and null-separation tables"),
        code(
            """
            np.savez_compressed(analysis_dir / "positive_spectra.npz", **spectral_arrays)
            display(save_spectrum_ccdf_gallery(spectral_arrays, method_slug=METHOD_SLUG))
            unqualified_calibrated_rows.to_csv(
                analysis_dir / "excluded_unqualified_calibrated_replay_rows.csv",
                index=False,
            )
            prior_groups = ["analysis_method"]
            for candidate in (
                "optimizer", "layer", "method", "pair_stride", "delta_s", "epsilon",
                "evidence_role", "probe_count", "pair_selection_role",
            ):
                if candidate in prior_primary:
                    prior_groups.append(candidate)
            method_summary = ci_summary(
                prior_primary,
                groups=tuple(prior_groups),
                metrics=tuple(name for name in ("alpha", "ks_D", "n_tail", "tail_decades") if name in prior_primary),
            )
            method_summary.to_csv(analysis_dir / "prior_method_stability_95ci.csv", index=False)
            null_primary = fit_rows[
                fit_rows["clip_top_k"].eq(0)
                & fit_rows["spectrum_kind"].astype(str).eq("energy_derived_from_amplitude")
            ].copy()
            if "fit_ok" in null_primary:
                null_primary = null_primary[boolean_series(null_primary["fit_ok"])]
            if null_primary.empty:
                raise RuntimeError("No successful matched-null primary energy fits")
            null_summary = ci_summary(
                null_primary,
                groups=("optimizer", "layer", "method", "null_kind"),
                metrics=("alpha", "ks_D", "n_tail", "tail_decades"),
            )
            null_summary.to_csv(analysis_dir / "matched_null_summary_95ci.csv", index=False)
            pairing_keys = ["optimizer", "seed", "layer", "method"]
            trained = null_primary[null_primary["null_kind"].eq("trained")][
                pairing_keys + ["alpha", "ks_D", "n_tail"]
            ].rename(columns={
                "alpha": "trained_alpha", "ks_D": "trained_ks_D",
                "n_tail": "trained_n_tail",
            })
            controls = null_primary[~null_primary["null_kind"].eq("trained")].copy()
            paired = controls.merge(
                trained,
                on=pairing_keys,
                how="inner",
                validate="many_to_one",
            )
            if paired.empty:
                raise RuntimeError("No trained/null fit pairs were available for separation")
            paired["trained_minus_null_alpha"] = (
                pd.to_numeric(paired["trained_alpha"], errors="coerce")
                - pd.to_numeric(paired["alpha"], errors="coerce")
            )
            paired["trained_minus_null_ks_D"] = (
                pd.to_numeric(paired["trained_ks_D"], errors="coerce")
                - pd.to_numeric(paired["ks_D"], errors="coerce")
            )
            paired["trained_minus_null_n_tail"] = (
                pd.to_numeric(paired["trained_n_tail"], errors="coerce")
                - pd.to_numeric(paired["n_tail"], errors="coerce")
            )
            paired.to_csv(analysis_dir / "paired_trained_null_rows.csv", index=False)
            separation_summary = ci_summary(
                paired,
                groups=("optimizer", "layer", "method", "null_kind"),
                metrics=(
                    "trained_minus_null_alpha",
                    "trained_minus_null_ks_D",
                    "trained_minus_null_n_tail",
                ),
            )
            alpha_decisions = separation_summary[
                separation_summary["metric"].eq("trained_minus_null_alpha")
                & separation_summary["null_kind"].astype(str).isin(
                    {"iid_gaussian", "haar_rectangular_polar", "temporally_reversed"}
                )
            ].copy()
            alpha_decisions["paired_alpha_ci_excludes_zero"] = (
                (pd.to_numeric(alpha_decisions["ci_low"]) > 0.0)
                | (pd.to_numeric(alpha_decisions["ci_high"]) < 0.0)
            )
            alpha_decisions["retention_decision"] = np.where(
                alpha_decisions["paired_alpha_ci_excludes_zero"],
                "alpha_separates_only; require all other retention gates",
                "not_retained: paired alpha does not separate",
            )
            invariance_alpha = separation_summary[
                separation_summary["metric"].eq("trained_minus_null_alpha")
                & (
                    separation_summary["null_kind"].astype(str).str.contains(
                        "global_scale", case=False, regex=False
                    )
                    | separation_summary["null_kind"].astype(str).str.contains(
                        "rotation", case=False, regex=False
                    )
                )
            ].copy()
            invariance_alpha["alpha_equality_ci_contains_zero"] = (
                (pd.to_numeric(invariance_alpha["ci_low"]) <= 0.0)
                & (pd.to_numeric(invariance_alpha["ci_high"]) >= 0.0)
            )
            invariance_alpha["invariance_decision"] = np.where(
                invariance_alpha["alpha_equality_ci_contains_zero"],
                "consistent with expected alpha invariance",
                "invariance audit failed: paired alpha shift excludes zero",
            )
            separation_summary.to_csv(
                analysis_dir / "paired_null_separation_95ci.csv", index=False
            )
            alpha_decisions.to_csv(
                analysis_dir / "paired_null_retention_decisions.csv", index=False
            )
            invariance_alpha.to_csv(
                analysis_dir / "paired_invariance_alpha_decisions.csv", index=False
            )
            invariance_rows = operator_rows[
                operator_rows["method"].eq("normalized_gram_invariance")
            ]
            if not invariance_rows["passed"].fillna(False).all():
                raise RuntimeError("A claimed normalized-Gram invariance failed")
            display(method_summary.tail(40))
            display(null_summary.tail(40))
            display(alpha_decisions.tail(40))
            display(invariance_alpha.tail(40))
            """
        ),
        markdown(
            """
            Scale/rotation checks test claimed invariance; Gaussian and Haar
            controls test distributional specificity. A method can pass one
            and fail the other. Validation requires credible tail support,
            null separation, parameter stability, late stationarity, and
            reproduction across the three complete seeds; FC3 cannot carry the
            conclusion.
            """
        ),
    ]
    return notebook("15_Method_Nulls_Stability_Comparison.ipynb", cells)


def build_all_notebooks() -> tuple[tuple[str, dict[str, object]], ...]:
    return (
        smoke_notebook(),
        training_notebook(
            filename="01_Long_Horizon_AdamW.ipynb",
            title="Long-horizon AdamW baseline",
            optimizer_slug="adamw",
            profile="pilot_1000_epochs",
            description="Train or resume the preregistered AdamW control at the selected 2/1,000/10,000-epoch stage.",
        ),
        training_notebook(
            filename="02_Long_Horizon_Muon.ipynb",
            title="Long-horizon Muon baseline",
            optimizer_slug="muon",
            profile="pilot_1000_epochs",
            description="Train or resume canonical Muon on FC1/FC2 with auxiliary AdamW on FC3 and biases.",
        ),
        training_notebook(
            filename="03_Long_Horizon_MuonClip_RMS.ipynb",
            title="Long-horizon MuonClip-RMS baseline",
            optimizer_slug="muonclip_rms",
            profile="pilot_1000_epochs",
            description="Train or resume canonical Muon EMA + NS5 + exact RMS=0.20 matrix directions; QK clipping is N/A.",
        ),
        fixed_point_comparison_notebook(),
        two_checkpoint_notebook(),
        stiefel_tangent_notebook(),
        radial_angular_notebook(),
        single_checkpoint_notebook(),
        calibrated_training_map_notebook(),
        nulls_stability_notebook(),
    )


def main() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    built = build_all_notebooks()
    expected = {
        "00_Protocol_and_Smoke.ipynb",
        "01_Long_Horizon_AdamW.ipynb",
        "02_Long_Horizon_Muon.ipynb",
        "03_Long_Horizon_MuonClip_RMS.ipynb",
        "04_Fixed_Point_Comparison.ipynb",
        "10_Two_Checkpoint_Finite_Flow.ipynb",
        "11_Muon_Update_Stiefel_Tangent.ipynb",
        "12_Radial_Angular_Quotients.ipynb",
        "13_Single_Checkpoint_Map_Jacobians.ipynb",
        "14_Calibrated_Local_Training_Map.ipynb",
        "15_Method_Nulls_Stability_Comparison.ipynb",
    }
    observed = {name for name, _ in built}
    if observed != expected:
        raise RuntimeError(
            f"Notebook inventory drift: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    for name, payload in built:
        path = NOTEBOOK_ROOT / name
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(EXPERIMENT_ROOT))


if __name__ == "__main__":
    main()
