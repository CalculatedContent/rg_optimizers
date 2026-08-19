"""Power-law fits for spectra that are not analyzed by WeightWatcher.

The scientific contract in this experiment is deliberately narrow:

* ``powerlaw.Fit`` performs its own continuous-MLE/KS search for ``xmin``;
* no home-grown regression or custom ``xmin`` grid is used;
* top-eigenvalue removal is reported as an explicit sensitivity curve rather
  than silently choosing the removal count with the most attractive fit; and
* amplitude-to-energy results are an exact change of variables on the same
  fitted sample, never a second fit that could choose a different tail.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


FIT_BACKEND = "powerlaw.Fit"


def positive_values(values: Any, *, minimum_count: int = 2) -> np.ndarray:
    """Return sorted finite positive continuous observations."""

    sample = np.asarray(values, dtype=float).reshape(-1)
    sample = np.sort(sample[np.isfinite(sample) & (sample > 0.0)])
    if sample.size < int(minimum_count):
        raise ValueError(
            f"need at least {int(minimum_count)} finite positive observations; "
            f"found {sample.size}"
        )
    return sample


def _fit_attributes(package_fit: Any) -> tuple[float, float, float]:
    distribution = getattr(package_fit, "power_law", package_fit)
    alpha_value = getattr(distribution, "alpha", None)
    sigma_value = getattr(distribution, "sigma", None)
    distance_value = getattr(distribution, "D", None)
    if alpha_value is None:
        alpha_value = getattr(package_fit, "alpha")
    if sigma_value is None:
        sigma_value = getattr(package_fit, "sigma")
    if distance_value is None:
        distance_value = getattr(package_fit, "D", float("nan"))
    alpha = float(alpha_value)
    sigma = float(sigma_value)
    distance = float(distance_value)
    return alpha, sigma, distance


def fit_powerlaw(
    values: Any,
    *,
    clip_top_k: int = 0,
    minimum_tail: int = 8,
    operator_kind: str,
    map_definition: str,
    spectrum_kind: str = "amplitude",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one continuous sample with package-selected ``xmin``.

    ``clip_top_k`` is an explicitly labelled sensitivity setting.  Callers
    must not select a preferred value by minimizing ``D`` after looking at the
    data.  The preregistered primary derived-spectrum fit is ``clip_top_k=0``.
    """

    if int(clip_top_k) < 0:
        raise ValueError("clip_top_k must be non-negative")
    if int(minimum_tail) < 2:
        raise ValueError("minimum_tail must be at least two")

    # Two observations are enough to retain an auditable failed-fit row; the
    # independent ``minimum_tail`` gate below prevents a package call or a
    # scientific qualification until the declared support is large enough.
    sample = positive_values(values, minimum_count=int(clip_top_k) + 2)
    removed = int(clip_top_k)
    used = sample[:-removed] if removed else sample.copy()
    row: dict[str, Any] = {
        **dict(metadata or {}),
        "operator_kind": str(operator_kind),
        "map_definition": str(map_definition),
        "spectrum_kind": str(spectrum_kind),
        "fit_backend": FIT_BACKEND,
        "fit_variant": "raw" if removed == 0 else f"clip_top_{removed}",
        "clip_top_k": removed,
        "selection_role": "primary" if removed == 0 else "sensitivity_only",
        "n_total": int(sample.size),
        "n_used": int(used.size),
        "observed_min": float(used[0]),
        "observed_max": float(used[-1]),
        "alpha": float("nan"),
        "sigma": float("nan"),
        "ks_D": float("nan"),
        "xmin": float("nan"),
        "xmax": float(used[-1]),
        "n_tail": 0,
        "tail_fraction": 0.0,
        "tail_decades": float("nan"),
        "fit_ok": False,
        "warning": "",
    }

    if used.size < int(minimum_tail):
        row["warning"] = (
            f"only {used.size} observations remain; minimum_tail={minimum_tail}"
        )
        return row

    try:
        import powerlaw

        # This exact call is part of the protocol.  In particular, xmin is not
        # supplied: the package scans candidate thresholds using its MLE/KS
        # procedure.
        package_fit = powerlaw.Fit(used, discrete=False, verbose=False)
        alpha, sigma, distance = _fit_attributes(package_fit)
        xmin = float(package_fit.xmin)
        tail = used[used >= xmin]
        n_tail = int(tail.size)
        xmax = float(tail[-1]) if n_tail else float(used[-1])
        warning = ""
        if n_tail < int(minimum_tail):
            warning = (
                f"package-selected tail has {n_tail} observations; "
                f"minimum_tail={minimum_tail}"
            )
        finite = np.isfinite([alpha, sigma, distance, xmin]).all()
        row.update(
            {
                "alpha": alpha,
                "sigma": sigma,
                "ks_D": distance,
                "xmin": xmin,
                "xmax": xmax,
                "n_tail": n_tail,
                "tail_fraction": float(n_tail / used.size),
                "tail_decades": (
                    float(np.log10(xmax / xmin))
                    if n_tail and xmax >= xmin > 0.0
                    else float("nan")
                ),
                "fit_ok": bool(finite and n_tail >= int(minimum_tail)),
                "warning": warning,
            }
        )
    except Exception as error:  # retain failed rows for an auditable run table
        row["warning"] = f"{type(error).__name__}: {error}"
    return row


def fit_clipping_sensitivity(
    values: Any,
    *,
    top_k_values: Iterable[int] = range(0, 6),
    minimum_tail: int = 8,
    operator_kind: str,
    map_definition: str,
    spectrum_kind: str = "amplitude",
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return an explicit, unranked top-``k`` finger sensitivity table."""

    candidates = tuple(dict.fromkeys(int(value) for value in top_k_values))
    if not candidates or candidates[0] != 0 or any(value < 0 for value in candidates):
        raise ValueError("top_k_values must start with zero and be non-negative")
    rows = [
        fit_powerlaw(
            values,
            clip_top_k=value,
            minimum_tail=minimum_tail,
            operator_kind=operator_kind,
            map_definition=map_definition,
            spectrum_kind=spectrum_kind,
            metadata=metadata,
        )
        for value in candidates
    ]
    frame = pd.DataFrame(rows)
    frame["sensitivity_selected"] = False
    return frame


def qualify_replicated_group_fits(
    fits: pd.DataFrame,
    group_amplitudes: Any,
    *,
    group_multiplicity: int,
    minimum_tail_groups: int,
) -> pd.DataFrame:
    """Apply a group-level qualification to a deterministically repeated ESD.

    Some analytic Jacobians contain one amplitude per physical coordinate
    group, repeated a fixed number of times by symmetry.  ``powerlaw.Fit`` is
    still run on the declared expanded ESD, but neither those copies nor a
    partial removal of a degenerate block are independent evidence.  This
    helper therefore requires clipping in whole groups and gates ``fit_ok`` on
    the number of retained physical groups above the package-selected
    ``xmin``.  It supports the amplitude rows and their exact squared-energy
    transforms produced by :func:`amplitude_fit_to_energy`.
    """

    multiplicity = int(group_multiplicity)
    minimum_groups = int(minimum_tail_groups)
    if multiplicity < 1:
        raise ValueError("group_multiplicity must be positive")
    if minimum_groups < 2:
        raise ValueError("minimum_tail_groups must be at least two")
    required = {
        "clip_top_k", "spectrum_kind", "xmin", "fit_ok", "n_used", "n_tail"
    }
    missing = required - set(fits.columns)
    if missing:
        raise ValueError(f"fit table lacks required columns: {sorted(missing)}")

    # Keep a one-group spectrum as an auditable failed qualification rather
    # than aborting the notebook.  ``minimum_tail_groups >= 2`` still makes it
    # scientifically ineligible.
    groups = positive_values(group_amplitudes, minimum_count=1)
    result = fits.copy()
    clip_group_counts: list[int] = []
    used_group_counts: list[int] = []
    tail_group_counts: list[int] = []
    package_fit_statuses: list[bool] = []
    qualified_statuses: list[bool] = []
    warnings: list[str] = []
    for row in result.to_dict(orient="records"):
        clipped_modes = int(row["clip_top_k"])
        if clipped_modes % multiplicity:
            raise ValueError(
                "clip_top_k must remove complete replicated groups: "
                f"clip_top_k={clipped_modes}, multiplicity={multiplicity}"
            )
        clipped_groups = clipped_modes // multiplicity
        if clipped_groups > groups.size - 1:
            raise ValueError(
                "group clipping must leave at least one physical group: "
                f"groups={groups.size}, clipped_groups={clipped_groups}"
            )
        used_groups = groups[:-clipped_groups] if clipped_groups else groups.copy()
        spectrum_kind = str(row["spectrum_kind"])
        if spectrum_kind == "energy_derived_from_amplitude":
            used_groups = used_groups**2
        elif spectrum_kind != "amplitude":
            raise ValueError(
                "replicated-group qualification supports amplitude or exact "
                f"derived-energy rows, not {spectrum_kind!r}"
            )
        xmin = float(row["xmin"])
        tail_groups = (
            int(np.count_nonzero(used_groups >= xmin))
            if np.isfinite(xmin) and xmin > 0.0
            else 0
        )
        expected_used_modes = int(used_groups.size * multiplicity)
        expected_tail_modes = int(tail_groups * multiplicity)
        if int(row["n_used"]) != expected_used_modes:
            raise ValueError(
                "expanded ESD used-mode count is inconsistent with complete "
                f"groups: observed={row['n_used']}, expected={expected_used_modes}"
            )
        if int(row["n_tail"]) != expected_tail_modes:
            raise ValueError(
                "expanded ESD tail count is inconsistent with complete groups: "
                f"observed={row['n_tail']}, expected={expected_tail_modes}"
            )
        package_ok = bool(row["fit_ok"])
        group_ok = tail_groups >= minimum_groups
        warning = str(row.get("warning", "") or "")
        if not group_ok:
            group_warning = (
                f"package-selected tail has {tail_groups} physical groups; "
                f"minimum_tail_groups={minimum_groups}"
            )
            warning = f"{warning}; {group_warning}" if warning else group_warning
        clip_group_counts.append(clipped_groups)
        used_group_counts.append(int(used_groups.size))
        tail_group_counts.append(tail_groups)
        package_fit_statuses.append(package_ok)
        qualified_statuses.append(bool(package_ok and group_ok))
        warnings.append(warning)

    result["clip_group_count"] = clip_group_counts
    result["group_multiplicity"] = multiplicity
    result["used_group_count"] = used_group_counts
    result["tail_group_count"] = tail_group_counts
    result["minimum_tail_groups"] = minimum_groups
    result["mode_level_fit_ok_before_group_gate"] = package_fit_statuses
    result["group_tail_qualified"] = [
        count >= minimum_groups for count in tail_group_counts
    ]
    result["fit_ok"] = qualified_statuses
    result["clipping_unit"] = "whole_replicated_physical_groups"
    result["mode_group_count_consistency_verified"] = True
    result["warning"] = warnings
    return result


def amplitude_fit_to_energy(row: Mapping[str, Any]) -> dict[str, Any]:
    r"""Apply the exact change of variables ``e=b^2`` to a fit row.

    If ``p(b) ~ b^{-alpha_b}``, then

    ``alpha_e=(alpha_b+1)/2``, ``sigma_e=sigma_b/2``, and both fit-window
    endpoints are squared.  KS distance and tail membership are unchanged
    because this is a monotone transform of the identical observations.
    """

    result = dict(row)
    if str(result.get("spectrum_kind", "amplitude")) != "amplitude":
        raise ValueError("source row must describe an amplitude spectrum")
    result["spectrum_kind"] = "energy_derived_from_amplitude"
    result["fit_backend"] = f"exact_transform_of:{row.get('fit_backend', FIT_BACKEND)}"
    result["alpha"] = (float(row["alpha"]) + 1.0) / 2.0
    result["sigma"] = float(row["sigma"]) / 2.0
    for key in ("observed_min", "observed_max", "xmin", "xmax"):
        result[key] = float(row[key]) ** 2
    result["tail_decades"] = 2.0 * float(row["tail_decades"])
    result["derived_not_refit"] = True
    return result


def empirical_ccdf(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates for a right-continuous empirical CCDF plot."""

    x = positive_values(values)
    count = x.size
    return x, (count - np.arange(count, dtype=float)) / float(count)
