"""Frozen per-matrix WeightWatcher fits and confidence/tail aggregation.

The fitting call is exactly WeightWatcher(model).analyze(), with no fit-range
override. Only fc2/fc3/fc4 (128 by 128) meet the declared minimum dimension 64.
Saved eigenvalues are ascending squared singular values of an unnormalised
float64 CPU copy of W; they are diagnostic data, not a replacement WW fit.
"""

import contextlib
import importlib.metadata
import io
import json
import math
import random
import statistics

import numpy as np
import torch

ELIGIBLE_MATRICES = ("fc2.weight", "fc3.weight", "fc4.weight")
MIN_DIMENSION = 64
METHOD = ("Unmodified WeightWatcher 0.7.7 analyze per matrix; powerlaw 2.0.0; "
          "ordinary fitted alpha; weights sqrt(num_pl_spikes)/(D+0.02); "
          "all three declared eligible matrices, no matrix-size weight. "
          "Diagnostic eigenvalues are sorted SVD(W.float64)^2 without normalization.")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def confidence_weight(fit, dimension=None):
    """Return the prespecified fit confidence weight, or None for invalid fits."""
    alpha, distance, tail = fit.get("alpha"), fit.get("D"), fit.get("num_pl_spikes")
    size = fit.get("num_evals", dimension)
    if (fit.get("status") == "success" and finite(alpha) and alpha > 0
            and finite(distance) and 0 <= distance <= 1
            and isinstance(tail, int) and not isinstance(tail, bool) and tail > 0
            and isinstance(size, int) and not isinstance(size, bool) and tail <= size
            and (dimension is None or size <= dimension)):
        return math.sqrt(tail) / (distance + 0.02)
    return None


def aggregate_matrices(matrices):
    """Require one valid fit for each declared matrix; never average survivors."""
    names = [row.get("matrix") for row in matrices]
    values = []
    for row in matrices:
        fit = row.get("fit") or {}
        shape = row.get("shape") or []
        weight = confidence_weight(fit, min(shape)) if shape == [128, 128] else None
        row["weight"] = weight
        if weight is not None:
            values.append((fit["alpha"], weight))
    complete = (len(names) == len(set(names)) == len(ELIGIBLE_MATRICES)
                and set(names) == set(ELIGIBLE_MATRICES) and len(values) == len(ELIGIBLE_MATRICES))
    return {
        "n_valid": len(values), "complete": complete,
        "confidence_tail_alpha": (math.fsum(a * w for a, w in values)
                                  / math.fsum(w for _, w in values)) if complete else None,
        # This dispersion is across layers, not an uncertainty interval on seeds.
        "layer_alpha_sd": statistics.stdev(a for a, _ in values) if complete else None,
    }


def analyze_model(model):
    """Analyze without changing model weights, mode, or any training RNG state."""
    saved_torch = torch.get_rng_state().clone()
    saved_numpy, saved_python = np.random.get_state(), random.getstate()
    state = model.state_dict()
    rows = []
    excluded = []
    versions = {}
    dependency_error = None
    try:
        for package, expected in (("weightwatcher", "0.7.7"), ("powerlaw", "2.0.0")):
            versions[package] = importlib.metadata.version(package)
            if versions[package] != expected:
                raise RuntimeError(f"{package} {versions[package]} != required {expected}")
        import weightwatcher as ww
    except Exception as exc:
        dependency_error = f"{type(exc).__name__}: {exc}"
    try:
        for name, tensor in state.items():
            if tensor.ndim == 2 and name not in ELIGIBLE_MATRICES:
                excluded.append({"matrix": name, "shape": list(tensor.shape),
                                 "reason": ("minimum matrix dimension below 64"
                                            if min(tensor.shape) < MIN_DIMENSION
                                            else "not in the prespecified eligible matrix list")})
        for name in ELIGIBLE_MATRICES:
            tensor = state.get(name)
            row = {"matrix": name, "shape": list(tensor.shape) if tensor is not None else None,
                   "alpha": None, "D": None, "num_pl_spikes": None, "xmin": None,
                   "status": "error", "weight": None, "fit": {}, "eigenvalues": []}
            rows.append(row)
            captured = io.StringIO()
            try:
                if tensor is None or tensor.ndim != 2 or min(tensor.shape) < MIN_DIMENSION:
                    raise ValueError("declared matrix missing or below minimum dimension 64")
                weight_array = tensor.detach().cpu().double().numpy().copy()
                if not np.isfinite(weight_array).all():
                    raise ValueError("matrix contains nonfinite values")
                singular_values = np.linalg.svd(weight_array, compute_uv=False)
                row["eigenvalues"] = np.sort(singular_values ** 2).tolist()
                if dependency_error:
                    raise RuntimeError(dependency_error)
                copy = torch.nn.Sequential(torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False))
                with torch.no_grad():
                    copy[0].weight.copy_(tensor.detach().cpu())
                with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                    frame = ww.WeightWatcher(copy).analyze()
                fits = json.loads(frame.to_json(orient="records"))
                if len(fits) != 1:
                    row["returned_fits"] = fits
                    raise ValueError(f"expected one WW fit, received {len(fits)}")
                row["fit"] = fits[0]
                for field in ("alpha", "D", "num_pl_spikes", "xmin", "status"):
                    row[field] = fits[0].get(field)
                row["weight"] = confidence_weight(fits[0], min(tensor.shape))
                if row["weight"] is None:
                    row["diagnostic"] = "WW result failed the prespecified validity conditions"
            except Exception as exc:
                row["diagnostic"] = f"{type(exc).__name__}: {exc}"
            if captured.getvalue():
                row["ww_output"] = captured.getvalue()[-4000:]
    finally:
        torch.set_rng_state(saved_torch)
        np.random.set_state(saved_numpy)
        random.setstate(saved_python)
    return {"method": METHOD, "versions": versions,
            "eligible_matrices": list(ELIGIBLE_MATRICES), "excluded_matrices": excluded,
            "n_eligible": len(ELIGIBLE_MATRICES), **aggregate_matrices(rows), "matrices": rows}
