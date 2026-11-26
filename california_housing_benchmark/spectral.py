"""Prespecified WeightWatcher fits for the published California Housing MLP.

The frozen architecture is 8 -> 784 -> 784 -> 1. The minimum matrix
dimension of 64 admits exactly one matrix: its 784 x 784 hidden weight.
Consequently its confidence/tail average equals that single fitted alpha;
there is no estimate of dispersion across layers. Uncertainty across seeds
is computed separately in analysis.py.

The fit is the unchanged WeightWatcher(model).analyze() call. Diagnostic
eigenvalues are sorted squared singular values of an unnormalised float64
CPU copy and do not replace WeightWatcher's fitting procedure.
"""

import contextlib
import importlib.metadata
import io
import json
import math
import random

import numpy as np
import torch

MIN_DIMENSION = 64
EXPECTED_ELIGIBLE_SHAPES = ((784, 784),)
METHOD = (
    "Unmodified WeightWatcher 0.7.7 analyze per unique eligible matrix; "
    "powerlaw 2.0.0; ordinary fitted alpha; weights "
    "sqrt(num_pl_spikes)/(D+0.02); no matrix-size factor. Minimum matrix "
    "dimension 64 admits one 784x784 hidden matrix, so the model average "
    "equals that fit and layer dispersion is unavailable. Diagnostic "
    "eigenvalues are sorted SVD(W.float64)^2 without normalization."
)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def confidence_weight(fit, dimension=None):
    """Return the frozen confidence/tail weight, or None for an invalid fit."""
    alpha, distance, tail = fit.get("alpha"), fit.get("D"), fit.get("num_pl_spikes")
    size = fit.get("num_evals", dimension)
    if (fit.get("status") == "success" and finite(alpha) and alpha > 0
            and finite(distance) and 0 <= distance <= 1
            and isinstance(tail, int) and not isinstance(tail, bool) and tail > 0
            and isinstance(size, int) and not isinstance(size, bool) and tail <= size
            and (dimension is None or size <= dimension)):
        return math.sqrt(tail) / (distance + 0.02)
    return None


def aggregate_matrices(matrices, eligible_matrices=None):
    """Recalculate the average from raw fits, requiring the one declared matrix.

    Never replace missing or invalid fits by a survivor-only average. The
    optional name list checks that the stored raw fits match the declaration.
    This function does not mutate the supplied records.
    """
    names = [row.get("matrix") for row in matrices]
    declared = names if eligible_matrices is None else list(eligible_matrices)
    shapes = [tuple(row.get("shape") or ()) for row in matrices]
    values = []
    for row, shape in zip(matrices, shapes):
        if len(shape) != 2 or min(shape) < MIN_DIMENSION:
            continue
        weight = confidence_weight(row.get("fit") or {}, min(shape))
        if weight is not None:
            values.append((row["fit"]["alpha"], weight))
    complete = (len(names) == len(set(names)) == len(declared) == 1
                and all(isinstance(name, str) and name for name in names)
                and set(names) == set(declared)
                and tuple(shapes) == EXPECTED_ELIGIBLE_SHAPES
                and len(values) == 1)
    return {
        "n_valid": len(values),
        "complete": complete,
        "confidence_tail_alpha": values[0][0] if complete else None,
        "layer_alpha_sd": None,
        "layer_dispersion_defined": False,
        "model_average_equals_single_matrix_fit": complete,
    }


def measure(model):
    """Measure unique matrices without changing model, mode, or training RNG.

    The benchmark runs on CPU. CUDA RNG is also preserved if CUDA has
    already been initialized, without initializing it solely for diagnosis.
    """
    saved_torch = torch.get_rng_state().clone()
    saved_numpy, saved_python = np.random.get_state(), random.getstate()
    saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    rows, excluded, eligible = [], [], []
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
        # named_parameters removes tied Parameter objects. The additional
        # storage check also avoids fitting a shared weight alias twice.
        seen = set()
        for name, tensor in model.named_parameters():
            if tensor.ndim != 2:
                continue
            identity = (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
            if identity in seen:
                excluded.append({"matrix": name, "shape": list(tensor.shape),
                                 "reason": "alias of an already counted unique weight matrix"})
                continue
            seen.add(identity)
            if min(tensor.shape) < MIN_DIMENSION:
                excluded.append({"matrix": name, "shape": list(tensor.shape),
                                 "reason": "minimum matrix dimension below 64"})
                continue
            eligible.append(name)
            row = {"matrix": name, "shape": list(tensor.shape),
                   "alpha": None, "D": None, "num_pl_spikes": None, "xmin": None,
                   "status": "error", "weight": None, "fit": {}, "eigenvalues": []}
            rows.append(row)
            captured = io.StringIO()
            try:
                weight_array = tensor.detach().cpu().double().numpy().copy()
                if not np.isfinite(weight_array).all():
                    raise ValueError("matrix contains nonfinite values")
                singular_values = np.linalg.svd(weight_array, compute_uv=False)
                row["eigenvalues"] = np.sort(singular_values ** 2).tolist()
                if dependency_error:
                    raise RuntimeError(dependency_error)
                copied = torch.nn.Sequential(torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False))
                with torch.no_grad():
                    copied[0].weight.copy_(tensor.detach().cpu())
                with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                    frame = ww.WeightWatcher(copied).analyze()
                fits = json.loads(frame.to_json(orient="records"))
                if len(fits) != 1:
                    row["returned_fits"] = fits
                    raise ValueError(f"expected one WW fit, received {len(fits)}")
                row["fit"] = fits[0]  # retain every field, not only the average inputs
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
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)
    aggregation = aggregate_matrices(rows, eligible)
    output = {"method": METHOD, "versions": versions, "minimum_matrix_dimension": MIN_DIMENSION,
              "eligible_matrices": eligible, "excluded_matrices": excluded,
              "n_eligible": len(eligible), **aggregation, "matrices": rows}
    if tuple(tuple(row["shape"]) for row in rows) != EXPECTED_ELIGIBLE_SHAPES:
        output["diagnostic"] = "eligible matrices differ from frozen 8->784->784->1 architecture"
    return output


# Preserve the old descriptive name for callers importing that API.
analyze_model = measure
