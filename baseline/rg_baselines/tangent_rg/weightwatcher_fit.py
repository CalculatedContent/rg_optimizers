"""Dual WeightWatcher fits and raw MLP weight ESD extraction.

WeightWatcher is authoritative for weight-matrix fits.  Every measurement is
run twice: once unchanged and once with ``fix_fingers='clip_xmax'``.  The two
rows remain separate throughout the pipeline so clipping can never be hidden.
Derived quotient/Jacobian spectra use :mod:`.powerlaw_fit` instead.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


WEIGHTWATCHER_EFFECTIVE_TAIL_SUPPORT_SOURCE = (
    "weightwatcher_backend_xmax_exact_fit_tail"
)
WEIGHTWATCHER_INFERRED_INTERNAL_SLICE_FALLBACK_SUPPORT_SOURCE = (
    "weightwatcher_inferred_internal_slice_fallback_tail"
)
WEIGHTWATCHER_RAW_SUPPORT_SOURCE = "weightwatcher_raw_num_pl_spikes"
WEIGHTWATCHER_REPORTED_FINGER_SENSITIVITY_SOURCE = (
    "weightwatcher_reported_finger_count_sensitivity_only"
)
WEIGHTWATCHER_INTERNAL_SLICE_SENSITIVITY_SOURCE = (
    "weightwatcher_inferred_internal_slice_count_sensitivity_only"
)
WEIGHTWATCHER_PRIMARY_TAIL_SUPPORT_SOURCES = (
    WEIGHTWATCHER_EFFECTIVE_TAIL_SUPPORT_SOURCE,
    WEIGHTWATCHER_INFERRED_INTERNAL_SLICE_FALLBACK_SUPPORT_SOURCE,
    WEIGHTWATCHER_RAW_SUPPORT_SOURCE,
)
EXPECTED_MLP3_LAYERS = ("fc1.weight", "fc2.weight", "fc3.weight")


@dataclass(frozen=True)
class WeightWatcherMeasurement:
    """Long-form details and raw, unrescaled eigenvalues for one checkpoint."""

    details: pd.DataFrame
    esds: dict[str, np.ndarray]


@dataclass(frozen=True)
class WeightWatcherValidation:
    """Structural acquisition errors and nonfatal statistical failures."""

    structural_errors: tuple[str, ...]
    primary_fit_failures: tuple[str, ...]
    raw_audit_warnings: tuple[str, ...]

    @property
    def primary_ok(self) -> bool:
        return not self.structural_errors and not self.primary_fit_failures

    @property
    def acquisition_usable(self) -> bool:
        return not self.structural_errors

    @property
    def primary_errors(self) -> tuple[str, ...]:
        """Compatibility view containing structural and primary-fit issues."""

        return self.structural_errors + self.primary_fit_failures


def _canonical_layer(value: Any) -> str:
    text = str(value).replace("module.", "").strip()
    for layer in ("fc1", "fc2", "fc3"):
        if text == layer or text.endswith(f".{layer}") or f"{layer}.weight" in text:
            return f"{layer}.weight"
    return text if text.endswith(".weight") else f"{text}.weight"


def _layer_from_row(row: pd.Series) -> str:
    candidates = [row.get(name, "") for name in ("longname", "name")]
    canonical = {"fc1.weight", "fc2.weight", "fc3.weight"}
    converted = [_canonical_layer(value) for value in candidates if str(value).strip()]
    return next((value for value in converted if value in canonical), converted[0] if converted else "")


def extract_weight_esds(model: Any) -> dict[str, np.ndarray]:
    """Compute nonzero ``W W^T``/``W^T W`` eigenvalues via singular values."""

    esds: dict[str, np.ndarray] = {}
    for name, parameter in model.named_parameters():
        if name not in {"fc1.weight", "fc2.weight", "fc3.weight"}:
            continue
        singular = parameter.detach().float().cpu().numpy()
        values = np.linalg.svd(singular, compute_uv=False) ** 2
        values = np.sort(values[np.isfinite(values) & (values > 0.0)])
        esds[name] = values
    missing = {"fc1.weight", "fc2.weight", "fc3.weight"} - set(esds)
    if missing:
        raise RuntimeError(f"MLP3 spectral layers missing: {sorted(missing)}")
    return esds


@contextlib.contextmanager
def _isolated_rng(seed: int):
    """Make randomized WW diagnostics reproducible without advancing training RNG."""

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch = None
    torch_state = cuda_state = None
    try:
        try:
            import torch as torch_module

            torch = torch_module
            torch_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                cuda_state = torch.cuda.get_rng_state_all()
        except ImportError:
            pass
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
        if torch is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def _analyze(
    model: Any,
    *,
    fix_fingers: str | bool,
    max_fingers: int,
    min_evals: int,
    max_evals: int | None,
    svd_method: str,
    randomize: bool,
    analysis_seed: int,
) -> pd.DataFrame:
    try:
        import weightwatcher as ww
    except ImportError as error:
        raise RuntimeError(
            "WeightWatcher is required for weight ESD fits; install the "
            "baseline experiment dependencies"
        ) from error

    watcher = ww.WeightWatcher(model=model)
    kwargs: dict[str, Any] = {
        "plot": False,
        "randomize": bool(randomize),
        "min_evals": int(min_evals),
        "savefig": False,
        "vectors": False,
        "start_ids": 0,
        "ERG": True,
        "fix_fingers": fix_fingers,
        "max_fingers": int(max_fingers),
        "svd_method": str(svd_method),
    }
    if max_evals is not None:
        kwargs["max_evals"] = int(max_evals)
    try:
        parameters = inspect.signature(watcher.analyze).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    except (TypeError, ValueError):
        pass
    if "ERG" not in kwargs:
        raise RuntimeError("WeightWatcher analyze() does not expose the required ERG=True")
    with _isolated_rng(analysis_seed):
        details = watcher.analyze(**kwargs)
    if not isinstance(details, pd.DataFrame) or details.empty:
        raise RuntimeError("WeightWatcher returned no layer details")
    return details.copy()


def _standardize_details(
    details: pd.DataFrame,
    *,
    fit_variant: str,
    primary_variant: str,
    max_fingers: int,
    esds: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = details.copy()
    if not any(column in frame.columns for column in ("longname", "name")):
        raise RuntimeError("WeightWatcher details contain neither longname nor name")
    frame["layer"] = frame.apply(_layer_from_row, axis=1)
    frame = frame[frame["layer"].isin({"fc1.weight", "fc2.weight", "fc3.weight"})]
    frame["operator_kind"] = "weight_esd"
    frame["map_definition"] = "W -> nonzero_eigenvalues(W W^T)"
    frame["fit_backend"] = "WeightWatcher.analyze"
    frame["fit_variant"] = str(fit_variant)
    frame["finger_policy"] = (
        "none" if fit_variant == "raw" else "fix_fingers=clip_xmax"
    )
    frame["max_fingers"] = int(max_fingers) if fit_variant != "raw" else 0
    frame["selection_role"] = (
        "preregistered_primary" if fit_variant == primary_variant else "required_audit"
    )
    frame["qk_clip_applicable"] = False
    frame["qk_clip_reason"] = "MLP3 has no attention Q/K pair"

    aliases = {
        "D": "ks_D",
        "raw_alpha": "alpha_before_finger_clip",
    }
    for old, new in aliases.items():
        if old in frame.columns and new not in frame.columns:
            frame[new] = frame[old]
    for required in ("alpha", "sigma", "ks_D", "xmin", "xmax"):
        if required not in frame.columns:
            frame[required] = np.nan
    frame["backend_xmax"] = pd.to_numeric(frame["xmax"], errors="coerce")
    original_n_tail = (
        pd.to_numeric(frame["n_tail"], errors="coerce")
        if "n_tail" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    support = (
        pd.to_numeric(frame["num_pl_spikes"], errors="coerce")
        if "num_pl_spikes" in frame.columns
        else original_n_tail
    )
    fingers = (
        pd.to_numeric(frame["num_fingers"], errors="coerce").fillna(0.0)
        if "num_fingers" in frame.columns
        else pd.Series(0.0, index=frame.index, dtype=float)
    ).clip(lower=0.0)
    frame["num_fingers"] = fingers
    frame["num_fingers_reported"] = fingers
    frame["pl_support_rank"] = support
    frame["n_tail_before_finger_clip"] = support
    inferred_internal = fingers.where(fingers <= 0.0, fingers + 1.0)
    if fit_variant == "raw":
        inferred_internal = pd.Series(0.0, index=frame.index, dtype=float)
    frame["num_fingers_inferred_internal_slice"] = inferred_internal

    # WeightWatcher 0.7.7 reports ``num_fingers=idx-1`` although the selected
    # clip candidate is ``evals[:-idx]``.  A finite backend xmax is therefore
    # the authoritative fit endpoint: compute membership directly from the
    # saved ESD rather than guessing which count convention the API intended.
    effective_tail_values: list[float] = []
    primary_removed_values: list[float] = []
    effective_xmax: list[float] = []
    effective_xmax_sources: list[str] = []
    membership_sources: list[str] = []
    support_sources: list[str] = []
    endpoint_tail_counts: list[float] = []
    removed_above_endpoint: list[float] = []
    for index, row in frame.iterrows():
        values = np.asarray(esds[str(row["layer"])], dtype=float)
        values = np.sort(values[np.isfinite(values) & (values > 0.0)])
        reported = int(round(float(fingers.loc[index])))
        support_value = float(support.loc[index])
        xmin_value = pd.to_numeric(
            pd.Series([row.get("xmin", np.nan)]), errors="coerce"
        ).iloc[0]
        backend_xmax = float(frame.loc[index, "backend_xmax"])
        if fit_variant == "raw":
            tail = max(0.0, support_value) if np.isfinite(support_value) else float("nan")
            effective_tail_values.append(tail)
            primary_removed_values.append(0.0)
            effective_xmax.append(float(values[-1]) if values.size else float("nan"))
            effective_xmax_sources.append("observed_esd_max")
            membership_sources.append("raw_num_pl_spikes_support")
            support_sources.append(WEIGHTWATCHER_RAW_SUPPORT_SOURCE)
            endpoint_tail_counts.append(float("nan"))
            removed_above_endpoint.append(0.0)
        elif np.isfinite(backend_xmax):
            if pd.notna(xmin_value) and np.isfinite(float(xmin_value)):
                in_fit = (values >= float(xmin_value)) & (values <= backend_xmax)
                tail = float(np.count_nonzero(in_fit))
            else:
                tail = float("nan")
            removed = float(np.count_nonzero(values > backend_xmax))
            effective_tail_values.append(tail)
            primary_removed_values.append(removed)
            effective_xmax.append(backend_xmax)
            effective_xmax_sources.append("weightwatcher_backend_xmax")
            membership_sources.append("exact_saved_esd_membership_in_[xmin,backend_xmax]")
            support_sources.append(WEIGHTWATCHER_EFFECTIVE_TAIL_SUPPORT_SOURCE)
            endpoint_tail_counts.append(tail)
            removed_above_endpoint.append(removed)
        else:
            inferred_removed = int(round(float(inferred_internal.loc[index])))
            tail = (
                max(0.0, support_value - inferred_removed)
                if np.isfinite(support_value)
                else float("nan")
            )
            retained_index = int(values.size) - inferred_removed - 1
            effective_tail_values.append(tail)
            primary_removed_values.append(float(inferred_removed))
            if values.size == 0 or retained_index < 0:
                effective_xmax.append(float("nan"))
                effective_xmax_sources.append(
                    "invalid_inferred_internal_slice_count_fallback"
                )
            else:
                effective_xmax.append(float(values[retained_index]))
                effective_xmax_sources.append(
                    "observed_esd_inferred_internal_slice_fallback"
                )
            membership_sources.append(
                "inferred_idx=num_fingers+1_fallback_backend_xmax_missing"
            )
            support_sources.append(
                WEIGHTWATCHER_INFERRED_INTERNAL_SLICE_FALLBACK_SUPPORT_SOURCE
            )
            endpoint_tail_counts.append(float("nan"))
            removed_above_endpoint.append(float("nan"))

    effective_tail = pd.Series(effective_tail_values, index=frame.index, dtype=float)
    primary_removed = pd.Series(primary_removed_values, index=frame.index, dtype=float)
    frame["n_fingers_removed"] = primary_removed
    frame["n_tail"] = effective_tail
    frame["n_tail_fit"] = effective_tail
    frame["endpoint_fit_tail_count"] = endpoint_tail_counts
    frame["removed_above_backend_xmax"] = removed_above_endpoint
    frame["primary_tail_membership_source"] = membership_sources
    frame["effective_tail_support_source"] = support_sources
    frame["xmax"] = effective_xmax
    frame["effective_xmax_source"] = effective_xmax_sources
    frame["reported_count_tail_sensitivity"] = (support - fingers).clip(lower=0.0)
    frame["inferred_internal_slice_tail_sensitivity"] = (
        support - inferred_internal
    ).clip(lower=0.0)
    endpoint_removed = pd.to_numeric(
        frame["removed_above_backend_xmax"], errors="coerce"
    )
    frame["reported_minus_endpoint_removed"] = fingers - endpoint_removed
    frame["inferred_slice_minus_endpoint_removed"] = (
        inferred_internal - endpoint_removed
    )
    frame["clip_count_ambiguity_sensitivity_only"] = (
        fit_variant == "clip_xmax"
    ) & fingers.gt(0.0)
    count_column = next(
        (name for name in ("num_evals", "M", "N") if name in frame.columns),
        None,
    )
    total = (
        pd.to_numeric(frame[count_column], errors="coerce")
        if count_column is not None
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    frame["tail_fraction"] = effective_tail / total.where(total > 0.0)
    frame["finger_count_valid"] = (
        support.notna()
        & fingers.notna()
        & primary_removed.notna()
        & (fingers <= support)
        & (primary_removed <= support)
    )
    alpha = pd.to_numeric(frame["alpha"], errors="coerce")
    sigma = pd.to_numeric(frame["sigma"], errors="coerce")
    distance = pd.to_numeric(frame["ks_D"], errors="coerce")
    xmin = pd.to_numeric(frame["xmin"], errors="coerce")
    xmax = pd.to_numeric(frame["xmax"], errors="coerce")
    frame["tail_window_valid"] = xmin.gt(0.0) & xmax.ge(xmin)
    if "status" in frame.columns:
        frame["weightwatcher_status"] = frame["status"].astype(str)
        normalized_status = frame["status"].astype(str).str.strip().str.lower()
        backend_ok = normalized_status.str.contains(
            r"success|passed|(?:^|[^a-z])ok(?:[^a-z]|$)", regex=True
        )
    else:
        frame["weightwatcher_status"] = "not_reported"
        backend_ok = pd.Series(False, index=frame.index)
    frame["backend_status_ok"] = backend_ok
    frame["fit_ok"] = (
        np.isfinite(alpha)
        & np.isfinite(sigma)
        & np.isfinite(distance)
        & np.isfinite(xmin)
        & np.isfinite(xmax)
        & frame["tail_window_valid"]
        & frame["finger_count_valid"]
        & (effective_tail >= 2)
        & backend_ok
    )
    frame["status"] = np.where(frame["fit_ok"], "ok", "failed")
    if "warning" not in frame.columns:
        frame["warning"] = ""
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["tail_decades"] = np.log10(xmax / xmin)
    frame["low_rank_warning"] = frame["layer"].eq("fc3.weight")
    return frame.reset_index(drop=True)


def _row_validation_errors(row: pd.Series) -> list[str]:
    """Return strict, human-readable errors for one standardized fit row."""

    errors: list[str] = []
    layer = str(row.get("layer", "unknown"))
    numeric: dict[str, float] = {}
    for column in ("alpha", "sigma", "ks_D", "xmin", "xmax", "n_tail"):
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        numeric[column] = float(value) if pd.notna(value) else float("nan")
        if not np.isfinite(numeric[column]):
            errors.append(f"{layer}: {column} is missing/nonfinite")
    if not (
        np.isfinite(numeric["xmin"])
        and np.isfinite(numeric["xmax"])
        and numeric["xmax"] >= numeric["xmin"] > 0.0
    ):
        errors.append(f"{layer}: invalid fit window xmax>=xmin>0")
    support = float(row.get("pl_support_rank", np.nan))
    fingers = float(row.get("n_fingers_removed", 0.0))
    if not (
        np.isfinite(support)
        and np.isfinite(fingers)
        and 0.0 <= fingers <= support
    ):
        errors.append(f"{layer}: num_fingers exceeds or invalidates num_pl_spikes")
    if not bool(row.get("backend_status_ok", False)):
        errors.append(f"{layer}: WeightWatcher backend status is not success")
    if not bool(row.get("fit_ok", False)):
        errors.append(f"{layer}: fit_ok is false")
    return errors


def validate_weightwatcher_measurement(
    measurement: WeightWatcherMeasurement,
    *,
    primary_variant: str = "clip_xmax",
    expected_layers: Iterable[str] = EXPECTED_MLP3_LAYERS,
) -> WeightWatcherValidation:
    """Validate primary fits strictly and retain raw failures as warnings.

    Missing or duplicate expected rows are structural and make the acquisition
    unusable. Nonfinite metrics, failed backend fit status, invalid fit windows,
    invalid finger counts, and ``fit_ok=False`` are persisted statistical
    failures but do not abort a long training run: early checkpoints and FC3's
    rank-ten spectrum are expected to fail fit qualification occasionally.
    """

    if primary_variant not in {"raw", "clip_xmax"}:
        raise ValueError("primary_variant must be 'raw' or 'clip_xmax'")
    details = measurement.details
    layers = tuple(str(layer) for layer in expected_layers)

    def validate_variant(variant: str) -> tuple[list[str], list[str]]:
        structural: list[str] = []
        fit_issues: list[str] = []
        selected = details[details["fit_variant"].eq(variant)]
        for layer in layers:
            rows = selected[selected["layer"].eq(layer)]
            if len(rows) != 1:
                structural.append(
                    f"{variant}/{layer}: expected exactly one row, found {len(rows)}"
                )
                continue
            fit_issues.extend(
                f"{variant}/{message}"
                for message in _row_validation_errors(rows.iloc[0])
            )
        extras = sorted(set(selected["layer"].astype(str)) - set(layers))
        if extras:
            structural.append(f"{variant}: unexpected layers {extras}")
        return structural, fit_issues

    primary_structural, primary_failures = validate_variant(primary_variant)
    if primary_variant == "raw":
        raw_structural: list[str] = []
        raw_warnings: list[str] = []
    else:
        raw_structural, raw_warnings = validate_variant("raw")
    return WeightWatcherValidation(
        structural_errors=tuple(primary_structural + raw_structural),
        primary_fit_failures=tuple(primary_failures),
        raw_audit_warnings=tuple(raw_warnings),
    )


def analyze_weightwatcher_dual(
    model: Any,
    *,
    min_evals: int = 8,
    max_evals: int | None = None,
    max_fingers: int = 10,
    svd_method: str = "accurate",
    randomize: bool = True,
    analysis_seed: int = 904_271,
    primary_variant: str = "clip_xmax",
    metadata: dict[str, Any] | None = None,
) -> WeightWatcherMeasurement:
    """Run standard and ``clip_xmax`` WeightWatcher fits side by side."""

    if primary_variant not in {"raw", "clip_xmax"}:
        raise ValueError("primary_variant must be 'raw' or 'clip_xmax'")
    # Match the audited baseline path: WW analyzes an isolated CPU copy so it
    # cannot mutate a live CUDA/MPS model or depend on device-specific tensor
    # conversion behavior.
    model_for_analysis = copy.deepcopy(model).to("cpu")
    model_for_analysis.eval()
    raw = _analyze(
        model_for_analysis,
        fix_fingers=False,
        max_fingers=max_fingers,
        min_evals=min_evals,
        max_evals=max_evals,
        svd_method=svd_method,
        randomize=randomize,
        analysis_seed=analysis_seed,
    )
    clipped = _analyze(
        model_for_analysis,
        fix_fingers="clip_xmax",
        max_fingers=max_fingers,
        min_evals=min_evals,
        max_evals=max_evals,
        svd_method=svd_method,
        randomize=randomize,
        analysis_seed=analysis_seed,
    )
    esds = extract_weight_esds(model_for_analysis)
    combined = pd.concat(
        [
            _standardize_details(
                raw,
                fit_variant="raw",
                primary_variant=primary_variant,
                max_fingers=max_fingers,
                esds=esds,
            ),
            _standardize_details(
                clipped,
                fit_variant="clip_xmax",
                primary_variant=primary_variant,
                max_fingers=max_fingers,
                esds=esds,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    for key, value in dict(metadata or {}).items():
        combined[key] = value
    return WeightWatcherMeasurement(
        details=combined,
        esds=esds,
    )


def weightwatcher_trace_log_rows(
    measurement: WeightWatcherMeasurement,
    *,
    fit_variant: str = "clip_xmax",
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate trace-log at PL, detX, and midpoint supports.

    A finite WeightWatcher backend ``xmax`` defines the preregistered support
    exactly as the saved-ESD members in ``[xmin, xmax]``.  WeightWatcher's
    reported finger count and its inferred internal ``idx`` count are emitted
    as sensitivity-only rows because 0.7.7 exposes an ``idx``/``idx-1``
    ambiguity.  If xmax is absent, the source-backed internal ``idx`` count is
    the primary fallback and only the reported count remains a sensitivity.
    Same-curve detX and midpoint rows are audits only.
    """

    from .trace_log import normalize_esd, trace_log_at_rank

    details = measurement.details[measurement.details["fit_variant"] == fit_variant]
    rows: list[dict[str, Any]] = []
    for _, detail in details.iterrows():
        layer = str(detail["layer"])
        values = measurement.esds[layer]
        count = int(values.size)
        normalized_descending = normalize_esd(
            values,
            dimension=float(count),
        )[::-1]

        def numeric(name: str, default: float = np.nan) -> float:
            value = pd.to_numeric(
                pd.Series([detail.get(name, default)]), errors="coerce"
            ).iloc[0]
            return float(value) if pd.notna(value) else float("nan")

        tail_value = numeric("n_tail")
        finger_value = numeric("n_fingers_removed", 0.0)
        support_value = numeric("pl_support_rank")
        support_before_clip = (
            max(0, int(round(float(support_value))))
            if np.isfinite(support_value)
            else 0
        )
        pl_rank = int(round(float(tail_value))) if np.isfinite(tail_value) else 0
        window_start = (
            max(0, int(round(float(finger_value))))
            if fit_variant == "clip_xmax" and np.isfinite(finger_value)
            else 0
        )
        support_source = str(
            detail.get(
                "effective_tail_support_source",
                WEIGHTWATCHER_INFERRED_INTERNAL_SLICE_FALLBACK_SUPPORT_SOURCE
                if fit_variant == "clip_xmax"
                else WEIGHTWATCHER_RAW_SUPPORT_SOURCE,
            )
        )
        reported = max(0, int(round(numeric("num_fingers_reported", 0.0))))
        inferred = max(
            0,
            int(round(numeric("num_fingers_inferred_internal_slice", 0.0))),
        )
        backend_xmax = numeric("backend_xmax")
        endpoint_removed = numeric("removed_above_backend_xmax")

        provenance = {
            "layer": layer,
            "fit_variant": fit_variant,
            "operator_kind": "weight_esd_trace_log",
            "pl_support_rank_before_finger_clip": support_before_clip,
            "num_fingers_reported": reported,
            "num_fingers_inferred_internal_slice": inferred,
            "removed_above_backend_xmax": endpoint_removed,
            "reported_minus_endpoint_removed": numeric(
                "reported_minus_endpoint_removed"
            ),
            "inferred_slice_minus_endpoint_removed": numeric(
                "inferred_slice_minus_endpoint_removed"
            ),
            "backend_xmax": backend_xmax,
            "backend_xmax_finite": bool(np.isfinite(backend_xmax)),
            "primary_tail_membership_source": str(
                detail.get("primary_tail_membership_source", "not_reported")
            ),
            **dict(metadata or {}),
        }

        def window_row(
            *,
            rank: int,
            start: int,
            source: str,
            window_source: str,
            sensitivity_only: bool,
        ) -> dict[str, Any]:
            end = start + rank
            valid = rank >= 1 and start >= 0 and end <= normalized_descending.size
            if valid:
                selected = normalized_descending[start:end]
                trace_total = float(np.sum(np.log(selected)))
                trace_per_eval = trace_total / rank
                lambda_cut = float(selected[-1])
                status = "ok_sensitivity_only" if sensitivity_only else "ok"
            else:
                trace_total = float("nan")
                trace_per_eval = float("nan")
                lambda_cut = float("nan")
                status = (
                    "invalid_sensitivity_window"
                    if sensitivity_only
                    else "invalid_effective_tail_window"
                )
            return {
                **provenance,
                "support_rank": max(0, rank),
                "support_rank_source": source,
                "support_selected_from_same_trace_log": False,
                "normalization_dimension": float(count),
                "trace_log_total": trace_total,
                "trace_log_per_eval": trace_per_eval,
                "lambda_cut_scaled": lambda_cut,
                "trace_log_status": status,
                "map_definition": (
                    "mean-one full ESD, descending independently supplied fit window"
                ),
                "support_window_start_descending_zero_based": start,
                "support_window_end_descending_exclusive": end,
                "support_window_source": window_source,
                "n_fingers_removed": start,
                "effective_fit_tail_rank": max(0, rank),
                "primary_effective_fit_tail_rank": max(0, pl_rank),
                "sensitivity_only": bool(sensitivity_only),
                "certification_eligible": bool(valid and not sensitivity_only),
                "qualification_role": (
                    "sensitivity_only_cannot_certify"
                    if sensitivity_only
                    else "preregistered_independent_fit_support"
                ),
            }

        rows.append(
            window_row(
                rank=pl_rank,
                start=window_start,
                source=support_source,
                window_source=str(
                    detail.get(
                        "primary_tail_membership_source",
                        "standardized WeightWatcher fit membership",
                    )
                ),
                sensitivity_only=False,
            )
        )

        if fit_variant == "clip_xmax" and reported > 0:
            reported_rank_value = numeric("reported_count_tail_sensitivity")
            inferred_rank_value = numeric("inferred_internal_slice_tail_sensitivity")
            sensitivity_windows = [
                (
                    WEIGHTWATCHER_REPORTED_FINGER_SENSITIVITY_SOURCE,
                    reported,
                    reported_rank_value,
                    "API num_fingers count convention; sensitivity only",
                ),
            ]
            # When xmax is finite, the inferred internal slice is an alternate
            # convention worth retaining.  When xmax is absent it is already
            # the primary source-backed fallback and must not be duplicated as
            # a sensitivity-only row.
            if (
                support_source
                != WEIGHTWATCHER_INFERRED_INTERNAL_SLICE_FALLBACK_SUPPORT_SOURCE
            ):
                sensitivity_windows.append(
                    (
                        WEIGHTWATCHER_INTERNAL_SLICE_SENSITIVITY_SOURCE,
                        inferred,
                        inferred_rank_value,
                        "inferred WeightWatcher 0.7.7 internal "
                        "idx=num_fingers+1; sensitivity only",
                    )
                )
            for source, removed, rank_value, source_note in sensitivity_windows:
                rank = int(round(rank_value)) if np.isfinite(rank_value) else 0
                rows.append(
                    window_row(
                        rank=rank,
                        start=removed,
                        source=source,
                        window_source=source_note,
                        sensitivity_only=True,
                    )
                )

        detx_value = pd.to_numeric(
            pd.Series(
                [detail.get("detX_num", detail.get("detx_num", np.nan))]
            ),
            errors="coerce",
        ).iloc[0]
        detx_rank = (
            int(np.clip(round(float(detx_value)), 1, count))
            if pd.notna(detx_value)
            else max(1, min(count, pl_rank if pl_rank else 1))
        )
        midpoint = int(
            np.clip(
                np.floor((max(1, pl_rank) + detx_rank) / 2.0),
                1,
                count,
            )
        )
        for source, rank, same_curve in (
            ("weightwatcher_detX", detx_rank, True),
            ("weightwatcher_midpoint", midpoint, True),
        ):
            row = trace_log_at_rank(
                values,
                rank=rank,
                normalization_dimension=float(count),
                rank_source=source,
            )
            row["support_selected_from_same_trace_log"] = same_curve
            row.update(
                {
                    **provenance,
                    "layer": layer,
                    "fit_variant": fit_variant,
                    "operator_kind": "weight_esd_trace_log",
                    "map_definition": "top-m logdet of mean-one weight ESD",
                    "support_window_start_descending_zero_based": 0,
                    "support_window_end_descending_exclusive": rank,
                    "support_window_source": "same-curve top-m audit",
                    "pl_support_rank_before_finger_clip": support_before_clip,
                    "n_fingers_removed": window_start,
                    "effective_fit_tail_rank": max(0, pl_rank),
                    "primary_effective_fit_tail_rank": max(0, pl_rank),
                    "trace_log_status": "same_curve_audit",
                    "sensitivity_only": False,
                    "certification_eligible": False,
                    "qualification_role": "same_curve_audit_cannot_certify",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
