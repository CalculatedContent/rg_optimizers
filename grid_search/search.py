"""Small SGD-momentum grid search: train, measure all three alphas, save a CSV."""
import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import os
import platform
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data import SPLIT_SEED, load_data

PEAK_LRS = {"MNIST": [0.056864187, 0.379835549, 1.724122529],
            "FashionMNIST": [0.018476187, 0.179621415, 0.547559449]}
DEFAULTS = dict(bias=True, dropout=0.0, fc1_weight_decay=0.0,
                lr1_factor=1.0, lr2_factor=1.0, lr3_factor=1.0,
                momentum=0.9, batch_size=256, warmup_epochs=5, lr_floor=0.1,
                nesterov=False, init_scale=1.0)


class MLP(nn.Module):
    def __init__(self, bias=True, dropout=0.0, init_scale=1.0):
        super().__init__()
        # Draw the biases even when removing them, matching paired initialization.
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = dropout
        with torch.no_grad():
            for layer in (self.fc1, self.fc2, self.fc3):
                layer.weight.mul_(init_scale)
                if not bias:
                    layer.register_parameter("bias", None)

    def forward(self, x):
        h = F.dropout(F.relu(self.fc1(x)), self.dropout, self.training)
        return self.fc3(F.relu(self.fc2(h)))


def parameter_grid(spec):
    unknown = set(spec) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown grid parameters: {sorted(unknown)}")
    grid = {k: spec.get(k, [v]) for k, v in DEFAULTS.items()}
    if any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("Every grid value must be a nonempty JSON list")
    for values in itertools.product(*grid.values()):
        c = dict(zip(grid, values))
        if not 0 <= c["dropout"] < 1 or not 0 <= c["momentum"] < 1:
            raise ValueError("Dropout and momentum must be in [0, 1)")
        if not 0 <= c["lr_floor"] <= 1 or c["fc1_weight_decay"] < 0:
            raise ValueError("LR floor must be in [0, 1]; decay must be nonnegative")
        if any(c[k] <= 0 for k in ("lr1_factor", "lr2_factor", "lr3_factor", "init_scale")):
            raise ValueError("LR factors and initialization scale must be positive")
        if type(c["batch_size"]) is not int or c["batch_size"] <= 0:
            raise ValueError("Batch size must be a positive integer")
        if type(c["warmup_epochs"]) is not int or c["warmup_epochs"] < 0:
            raise ValueError("Warm-up epochs must be a nonnegative integer")
        if type(c["bias"]) is not bool or type(c["nesterov"]) is not bool:
            raise ValueError("bias and nesterov must be JSON booleans")
        if c["nesterov"] and c["momentum"] == 0:
            raise ValueError("Nesterov requires positive momentum")
        yield c


def schedule(step, steps_per_epoch, epochs, warmup_epochs, floor):
    warm, total = warmup_epochs * steps_per_epoch, epochs * steps_per_epoch
    if step < warm:
        return 0.01 + 0.99 * step / max(1, warm - 1)
    q = min(1.0, max(0.0, (step - warm) / max(1, total - warm - 1)))
    return floor + (1 - floor) * (1 + math.cos(math.pi * q)) / 2


def make_optimizer(model, c, peaks):
    groups = []
    for i, layer in enumerate((model.fc1, model.fc2, model.fc3)):
        groups.append(dict(params=[layer.weight], lr=peaks[i], peak_lr=peaks[i],
                           weight_decay=c["fc1_weight_decay"] if i == 0 else 0.0))
        if layer.bias is not None:
            groups.append(dict(params=[layer.bias], lr=peaks[i], peak_lr=peaks[i], weight_decay=0.0))
    return torch.optim.SGD(groups, lr=peaks[1], momentum=c["momentum"],
                           nesterov=c["nesterov"], foreach=False)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "xla":
        import torch_xla
        return torch_xla.device()
    return torch.device(name)


def sync_xla(device):
    if device.type == "xla":
        import torch_xla
        torch_xla.sync()


@torch.no_grad()
def evaluate(model, x, y, ids, device):
    model.eval()  # Dropout is disabled for accuracy and interpolation checks.
    loss, errors = 0.0, 0
    for start in range(0, len(ids), 2048):
        ix = ids[start:start + 2048]
        target = y[ix].to(device)
        logits = model(x[ix].to(device))
        loss += F.cross_entropy(logits, target, reduction="sum").item()
        errors += (logits.argmax(1) != target).sum().item()
    return dict(loss=loss / len(ids), errors=int(errors), accuracy=1 - errors / len(ids))


def alphas(model):
    import weightwatcher as ww
    # Raw matrices, including the 10-row FC3; no overrides to analyze().
    cpu_model = copy.deepcopy(model).cpu().eval()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        frame = ww.WeightWatcher(cpu_model).analyze()
    if len(frame) != 3:
        raise ValueError(f"Expected three WeightWatcher layers, got {len(frame)}")
    return {f"alpha_fc{i}": float(a) for i, a in enumerate(frame["alpha"], 1)}


def train_one(dataset, c, seed, data, device, epochs=120, max_epochs=None, test_data=None):
    """Return one flat row. No early stopping in fixed-budget searches."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed, device)
    model = MLP(c["bias"], c["dropout"], c["init_scale"]).to(device)
    peaks = [v * c[f"lr{i}_factor"] for i, v in enumerate(PEAK_LRS[dataset], 1)]
    opt = make_optimizer(model, c, peaks)
    x, y, train, validation = data
    rng = np.random.default_rng(seed + 10000)
    steps = math.ceil(len(train) / c["batch_size"])
    streak, first = 0, None
    started = time.monotonic()
    row = dict(status="ok", error="")
    for epoch in range(1, (max_epochs or epochs) + 1):
        model.train()
        ids = rng.permutation(train)
        for start in range(0, len(ids), c["batch_size"]):
            factor = schedule((epoch - 1) * steps + start // c["batch_size"], steps,
                              epochs, c["warmup_epochs"], c["lr_floor"])
            for group in opt.param_groups:
                group["lr"] = group["peak_lr"] * factor
            ix = ids[start:start + c["batch_size"]]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[ix].to(device)), y[ix].to(device))
            loss.backward()
            opt.step()
            sync_xla(device)
        training = evaluate(model, x, y, train, device)
        valid = evaluate(model, x, y, validation, device)
        if not math.isfinite(training["loss"]) or not math.isfinite(valid["loss"]):
            row.update(status="nonfinite", error="Nonfinite training or validation loss")
            streak = 0
            break
        streak = streak + 1 if training["errors"] == 0 else 0
        if first is None and training["errors"] == 0:
            first = epoch
        if epoch >= epochs and streak >= 10:
            break
    row.update(epoch=epoch, zero_error_streak=streak, sustained_interpolation=streak >= 10,
               first_zero_error_epoch=first, train_updates=epoch * steps)
    for prefix, metrics in (("train", training), ("validation", valid)):
        row.update({f"{prefix}_{k}": v for k, v in metrics.items()})
    row.update({f"alpha_fc{i}": float("nan") for i in range(1, 4)})
    row["alpha_error"] = ""
    if row["status"] == "ok":
        try:
            row.update(alphas(model))
        except Exception as exc:
            row["alpha_error"] = f"{type(exc).__name__}: {exc}"
    a = np.array([row[f"alpha_fc{i}"] for i in range(1, 4)])
    row.update(alpha_mean=float(a.mean()), alpha_max_distance=float(np.abs(a - 2).max()),
               all_alphas_within_0_1=bool(np.isfinite(a).all() and (np.abs(a - 2) <= 0.1).all()))
    if test_data is not None and row["status"] == "ok":
        tx, ty = test_data
        row.update({f"test_{k}": v for k, v in evaluate(model, tx, ty, np.arange(len(ty)), device).items()})
    row.update(seconds=time.monotonic() - started,
               **{f"peak_lr_fc{i}": v for i, v in enumerate(peaks, 1)})
    return row


def main(dataset, default_grid):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=default_grid)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, cuda:1, or xla")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-epochs", type=int, help="Confirmation only: train beyond --epochs until 10 zero-error epochs")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Run only the first N config/seed pairs (also limits dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Print run count and parameter grid; do not load data")
    parser.add_argument("--test", action="store_true", help="Evaluate a frozen SINGLE configuration on test data")
    args = parser.parse_args()
    configs = list(parameter_grid(json.loads(args.grid.read_text())))
    if args.epochs <= 0 or (args.max_epochs is not None and args.max_epochs < args.epochs):
        parser.error("Require 0 < epochs <= max-epochs")
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("threads and limit must be positive")
    if any(c["warmup_epochs"] >= args.epochs for c in configs):
        parser.error("Warm-up must be shorter than epochs")
    if args.test and len(configs) != 1:
        parser.error("--test requires exactly one frozen configuration, not a sweep")
    seeds = args.seeds or ([501, 502, 503, 504, 505] if args.test else [401, 402, 403])
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        parser.error("Seeds must be distinct integers in [0, 2**32)")
    jobs = list(itertools.product(configs, seeds))[:args.limit]
    print(f"{dataset}: {len(configs)} configurations, {len(jobs)} runs; {args.epochs} epochs minimum", flush=True)
    if args.dry_run:
        print(args.grid.read_text())
        return pd.DataFrame()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    torch.use_deterministic_algorithms(device.type == "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "pandas", "scipy", "scikit-learn", "weightwatcher", "powerlaw")}
    if device.type == "xla":
        versions["torch_xla"] = importlib.metadata.version("torch_xla")
    output = args.output or Path(__file__).parent / "results" / f"{dataset}_{args.grid.stem}{'_test' if args.test else ''}.csv"
    rows = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {r["run_id"] for r in rows}
    data = load_data(args.data_dir, dataset)
    test_data = load_data(args.data_dir, dataset, test=True) if args.test else None
    source_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("data.py").read_bytes()).hexdigest()
    for number, (config, seed) in enumerate(jobs, 1):
        identity = dict(dataset=dataset, **config, seed=seed, epochs=args.epochs,
                        max_epochs=args.max_epochs or args.epochs, split_seed=SPLIT_SEED,
                        device=str(device), threads=args.threads, test_evaluated=args.test,
                        source_sha256=source_hash, versions=json.dumps(versions, sort_keys=True),
                        python=platform.python_version())
        run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        if run_id in done:
            continue
        result = train_one(dataset, config, seed, data, device, args.epochs, args.max_epochs, test_data)
        rows.append(dict(run_id=run_id, **identity, **result))
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temp, index=False)
        temp.replace(output)
        done.add(run_id)
        print(f"[{number}/{len(jobs)}] seed={seed} val={result['validation_accuracy']:.4%} "
              f"errors={result['train_errors']} alpha="
              f"{[round(result[f'alpha_fc{i}'], 3) for i in range(1, 4)]} -> {output}", flush=True)
    return pd.DataFrame(rows)
