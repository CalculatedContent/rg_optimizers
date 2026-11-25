"""Readiness checks for a later explicit launch; never allocate a TPU or install packages.

CPU spectral checks and TPU kernel checks run in separate processes so JAX and
WeightWatcher cannot take ownership of the trainer's device or random state.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import hashlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REFERENCE = REPO / "baseline/gpt2_small"
SOURCE = REFERENCE / "speedrun30/vendor/llmc_train_gpt2.py"
MANIFEST = REFERENCE / "speedrun30/data_manifest.json"
SOURCE_SHA256 = "757d0cea0d48cbc4c7d7d70371f955d49cdf3a7cfb4c87701a720c2fe0905c34"
MANIFEST_SHA256 = "028141b267e9a6c99af8beea42bed58fe1d7418bdf034d630f1792aeb6a20e19"
DATA_REVISION = "889765ea1f903759787add96995d81171b632d0c"


def load_config(path):
    from config import TrainingConfig
    config = TrainingConfig.from_json(Path(path))
    config.validate()
    return config


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_checks():
    observed = {"model": sha256(SOURCE), "data_manifest": sha256(MANIFEST)}
    if observed != {"model": SOURCE_SHA256, "data_manifest": MANIFEST_SHA256}:
        raise RuntimeError("Pinned model source or data manifest changed; refusing to launch")
    return observed


def required_data_files(config, manifest):
    """Continuous stream: preserve boundaries independently of microbatch size."""
    if config.mode == "smoke":
        return []
    names = ["fineweb_val_000000.bin"]
    remaining = int(config.budget_tokens) + 1
    for name, entry in sorted(manifest["files"].items()):
        if name.startswith("fineweb_train_"):
            names.append(name)
            remaining -= (int(entry["size"]) - 1024) // 2
            if remaining <= 0:
                break
    if len(names) == 1:
        raise RuntimeError("Pinned manifest contains no training shards")
    return names


def check_data(config, data_dir):
    manifest = json.loads(MANIFEST.read_text())
    if manifest["repo"] != "kjj0/fineweb10B-gpt2" or manifest["revision"] != DATA_REVISION:
        raise RuntimeError("Unexpected benchmark dataset identity")
    checks = []
    for name in required_data_files(config, manifest):
        path = Path(data_dir) / name
        entry = manifest["files"][name]
        if not path.is_file():
            raise RuntimeError(f"Missing {path}. Run prepare_data.py for this budget first.")
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Dataset size/SHA256 mismatch: {path}; no file was replaced")
        with path.open("rb") as handle:
            header = struct.unpack("<256i", handle.read(1024))
        if header[0] != 20240520 or header[1] != 1:
            raise RuntimeError(f"Unexpected GPT-2 token shard header: {path}")
        if header[2] * 2 + 1024 != entry["size"]:
            raise RuntimeError(f"Token count and size disagree: {path}")
        if name.startswith("fineweb_val_"):
            end = int(config.validation_offset) + int(config.validation_tokens) + 1
            if end > header[2]:
                raise RuntimeError("Requested validation slice exceeds the pinned validation shard")
        checks.append({"file": name, "bytes": entry["size"], "sha256": entry["sha256"]})
    return {"repo": manifest["repo"], "revision": DATA_REVISION,
            "synthetic_smoke": config.mode == "smoke", "files": checks}


def existing_parent(path):
    path = Path(path)
    while not path.exists():
        if path.parent == path:
            raise RuntimeError("No existing parent for output path")
        path = path.parent
    if not path.is_dir():
        raise RuntimeError(f"Output parent is not a directory: {path}")
    return path


def check_output(run_dir, *, device, min_free_gib):
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        raise ValueError("--run-dir must be an explicit absolute path")
    parent = existing_parent(run_dir).resolve()
    if device == "tpu" and any(parent == base or base in parent.parents
                                for base in (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))):
        raise RuntimeError("TPU outputs require a persistent mounted volume, not a temporary directory")
    mount = None
    if shutil.which("findmnt"):
        result = subprocess.run(["findmnt", "--json", "--target", str(parent),
                                 "--output", "TARGET,SOURCE,FSTYPE"],
                                capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0:
            mount = json.loads(result.stdout)["filesystems"][0]
    if device == "tpu":
        if not mount or mount.get("fstype") in ("tmpfs", "ramfs", "overlay") or mount.get("target") == "/":
            raise RuntimeError("Mount a persistent output volume and pass its absolute --run-dir; "
                               "the observed path is temporary, the root filesystem, or unverifiable")
    free = shutil.disk_usage(parent).free
    if free < min_free_gib * 1024 ** 3:
        raise RuntimeError(f"Only {free / 1024**3:.1f} GiB free at {parent}; "
                           f"require at least {min_free_gib:g} GiB")
    # A reversible local write test, with no run directory or checkpoint creation.
    with tempfile.TemporaryFile(dir=parent) as probe:
        probe.write(b"sgdm-output-preflight\n")
        probe.flush()
        os.fsync(probe.fileno())
    return {"requested_path": str(run_dir), "existing_parent": str(parent),
            "free_bytes": free, "minimum_free_gib": min_free_gib,
            "writable": True, "mount": mount,
            "persistence_note": "Verify this volume survives your planned TPU lifecycle."}


def check_versions(device, skip_spectral):
    expected = {"torch": "2.6.0", "numpy": "1.26.4"}
    if not skip_spectral:
        expected["weightwatcher"] = "0.7.7"
    if device == "tpu":
        expected.update({"torch-xla": "2.6.0", "libtpu": "0.0.7.1",
                         "jax": "0.4.38", "jaxlib": "0.4.38",
                         "ml-dtypes": "0.5.1", "opt-einsum": "3.4.0"})
    actual = {}
    for name, version in expected.items():
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing {name}; install the provided requirements in your venv") from exc
        actual[name] = observed
        if observed.split("+", 1)[0] != version:
            raise RuntimeError(f"Expected {name}=={version}; observed {observed}")
    if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
        raise RuntimeError("Use Python 3.10–3.12 for this pinned Torch/XLA 2.6 environment")
    return {"python": sys.version.split()[0], "packages": actual}


def checked_child(kind, config_path, *, timeout):
    command = [sys.executable, str(HERE / "preflight.py"), "--internal-check", kind]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    if kind == "spectral":
        env.update(PJRT_DEVICE="CPU", JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    else:
        env["PJRT_DEVICE"] = "TPU"
        env.pop("JAX_PLATFORMS", None)
    result = subprocess.run(command, env=env, text=True, capture_output=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{kind} preflight failed: " + (result.stdout + result.stderr)[-4000:])
    for line in reversed(result.stdout.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("preflight_check") == kind:
            return row["result"]
    raise RuntimeError(f"{kind} preflight did not produce a capability record")


def run_preflight(*, config_path, data_dir, run_dir, device="tpu", skip_spectral=False,
                  min_free_gib=30.0, probe_timeout=1200):
    config = load_config(config_path)
    if skip_spectral and config.mode != "smoke":
        raise ValueError("Production runs cannot skip all-layer spectral tracking")
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("Minimum free space must be finite and nonnegative")
    report = {"status": "checking", "config_sha256": config.config_sha256,
              "data_dir": str(Path(data_dir).resolve()), "run_dir": str(Path(run_dir).resolve()),
              "mode": config.mode, "device": device, "checked_unix": time.time()}
    report["sources"] = source_checks()
    report["environment"] = check_versions(device, skip_spectral)
    report["output"] = check_output(run_dir, device=device, min_free_gib=min_free_gib)
    report["data"] = check_data(config, data_dir)
    if skip_spectral:
        report["spectral"] = {"status": "explicitly_skipped_for_smoke_only"}
    else:
        report["spectral"] = checked_child("spectral", None, timeout=180)
        if not report["spectral"].get("ok"):
            raise RuntimeError("WeightWatcher native-field/ERG capability check failed: "
                               + json.dumps(report["spectral"]))
    if device == "tpu":
        report["tpu"] = checked_child("tpu", config_path, timeout=probe_timeout)
    report["status"] = "passed"
    report["finished_unix"] = time.time()
    return report


def tpu_probe(config_path):
    """One full-model update at the real microbatch shape, isolated and disposable."""
    config = load_config(config_path)
    import torch
    import torch_xla.runtime as xr
    xr.use_spmd()
    observed = {"global_devices": xr.global_runtime_device_count(),
                "addressable_devices": xr.addressable_runtime_device_count()}
    if observed != {"global_devices": 8, "addressable_devices": 8}:
        raise RuntimeError(f"Require one host with eight TPU chips; observed {observed}")
    sys.path.insert(0, str(REFERENCE / "muon_speedrun"))
    try:
        from runtime import Runtime, attention_check
        import stock_model
    finally:
        sys.path.remove(str(REFERENCE / "muon_speedrun"))
    if config.microbatch_sequences % 8:
        raise RuntimeError("The TPU microbatch must divide across eight chips; use CPU for configs/smoke.json")
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="sgdm-tpu-preflight-") as temporary:
        # This checks the exact configured attention shape and its derivatives.
        attention_check(Path(temporary), config.microbatch_sequences,
                        n_head=config.n_head, head_dim=config.n_embd // config.n_head)
        attention = json.loads((Path(temporary) / "ATTENTION_CHECK.json").read_text())
        # attention_check initializes XLA's process-wide persistent cache here;
        # keep its directory alive through the subsequent full-model compile.
        return _tpu_model_probe(config, observed, Runtime, stock_model, attention)


def _tpu_model_probe(config, observed, Runtime, stock_model, attention):
    import torch
    rt = Runtime("tpu")
    stock_model.configure_attention(rt.attention("flash"))
    architecture = stock_model.GPTConfig(block_size=config.context_length, vocab_size=config.vocab_size,
                                         n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd)
    model = stock_model.make_model(architecture, seed=config.seed, device=rt.device)
    for value in (*model.parameters(), *model.buffers()):
        rt.replicate(value)
    groups = [{"params": [p for p in model.parameters() if p.ndim >= 2],
               "weight_decay": config.weight_decay},
              {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}]
    optimizer = torch.optim.SGD(groups, lr=rt.scalar(config.peak_lr * config.schedule_factor(0)),
                                 momentum=config.momentum, nesterov=False, dampening=0.0,
                                 foreach=False)
    generator = torch.Generator().manual_seed(config.seed)
    tokens = torch.randint(config.vocab_size,
                            (config.microbatch_sequences, config.context_length + 1), generator=generator)
    model.zero_grad(set_to_none=False)
    accumulation = config.global_batch_sequences // config.microbatch_sequences
    with rt.autocast():
        _, loss = model(rt.put(tokens[:, :-1].contiguous()), rt.put(tokens[:, 1:].contiguous()),
                         return_logits=False)
    (loss / accumulation).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    for grad in gradients:
        rt.replicate(grad)
    norm = sum(grad.detach().float().square().sum() for grad in gradients).sqrt()
    scale = torch.clamp(config.grad_clip / (norm + 1e-6), max=1.0)
    with torch.no_grad():
        for grad in gradients:
            grad.mul_(scale)
    optimizer.step()
    finite = torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
    rt.step(wait=True)
    loss_value, norm_value = float(loss.cpu()), float(norm.cpu())
    if not bool(finite.cpu()) or not math.isfinite(loss_value) or not math.isfinite(norm_value):
        raise RuntimeError("Nonfinite full-model TPU optimizer probe")
    if model.lm_head.weight is not model.transformer.wte.weight:
        raise RuntimeError("Tied embedding/output alias was lost")
    return {**observed, "attention": attention, "parameters": sum(p.numel() for p in model.parameters()),
            "microbatch_sequences": config.microbatch_sequences, "loss": loss_value,
            "gradient_norm": norm_value, "status": "passed",
            "note": "One disposable microbatch update; no claim of convergence or accumulation parity."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "tpu"), default="tpu")
    parser.add_argument("--skip-spectral", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=30.0)
    parser.add_argument("--probe-timeout", type=int, default=1200)
    parser.add_argument("--internal-check", choices=("spectral", "tpu"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.internal_check:
        if args.internal_check == "spectral":
            from metrics import capability_check
            result = capability_check()
        else:
            result = tpu_probe(args.config)
        print(json.dumps({"preflight_check": args.internal_check, "result": result}, allow_nan=False))
        return 0
    if not all((args.config, args.data_dir, args.run_dir)):
        parser.error("--config, --data-dir, and --run-dir are required")
    report = run_preflight(config_path=args.config.resolve(), data_dir=args.data_dir.resolve(),
                           run_dir=args.run_dir, device=args.device, skip_spectral=args.skip_spectral,
                           min_free_gib=args.min_free_gib, probe_timeout=args.probe_timeout)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
