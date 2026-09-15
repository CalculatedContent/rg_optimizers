#!/usr/bin/env python3
"""Controlled, suffix-supervised nanoGPT memorization experiments.

Uses the repository GPT, AdamW/Muon implementations and frozen baseline profile.
No corpus downloads; all data are synthetic. WeightWatcher never edits training
weights. Importing this module needs only NumPy; training imports are lazy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CONFIG = HERE / "configs/suite.json"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


@dataclass(frozen=True)
class Record:
    eid: str
    group: str
    prompt: tuple[int, ...]
    target: tuple[int, ...]


def pack(records: list[Record], block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad to the original context length; score only the answer suffix."""
    if not records:
        raise ValueError("empty batch")
    x = np.zeros((len(records), block_size), dtype=np.int64)
    y = np.full_like(x, -100)
    for i, record in enumerate(records):
        if not record.prompt or not record.target:
            raise ValueError("empty prompt/target")
        seq = record.prompt + record.target
        if len(seq) > block_size + 1:
            raise ValueError("record exceeds the model context")
        padded = np.array((0,) * (block_size + 1 - len(seq)) + seq)
        x[i] = padded[:-1]
        start = block_size - len(record.target)
        y[i, start:] = padded[start + 1:]
    return x, y


class Study:
    """Stateless step-indexed sampling, paired across optimizer arms."""
    def __init__(self, cfg: dict, condition: str, stage: str, seed: int, batch: int):
        if condition not in cfg["conditions"]:
            raise ValueError(condition)
        self.cfg, self.condition, self.stage = cfg, condition, cfg["stages"][stage]
        self.seed, self.batch = seed, batch
        self.steps = int(self.stage["steps"])
        self.boundary = max(1, self.steps // 2)
        self.rng = np.random.default_rng(cfg["data_seed"])
        self.audit: list[Record] = []
        self.train: list[Record] = []
        self.second: list[Record] = []
        self.injections: dict[int, Record] = {}
        self._build()

    def tokens(self, n: int) -> tuple[int, ...]:
        return tuple(int(x) for x in self.rng.integers(16, 272, n))

    def _build(self) -> None:
        c, n = self.condition, int(self.cfg["associations"])
        if c.startswith("verbatim"):
            doses = []
            for dose in self.stage["doses"]:
                for i in range(self.stage["canaries_per_dose"]):
                    r = Record(f"canary_d{dose}_{i}", f"dose_{dose}",
                               self.tokens(self.cfg["canary_prefix_tokens"]),
                               self.tokens(self.cfg["canary_suffix_tokens"]))
                    self.audit.append(r)
                    doses.extend([r] * dose)
            # Exact lifetime presentations, not corpus-duplication multipliers.
            slots = self.boundary * self.batch
            if len(doses) > slots:
                raise ValueError("injection window too small for requested doses")
            schedule_rng = np.random.default_rng(np.random.SeedSequence([self.cfg["data_seed"], self.seed, 99]))
            where = schedule_rng.choice(slots, len(doses), replace=False)
            self.injections = dict(zip(map(int, where), doses))
        elif c.startswith("rule_"):
            p = int(self.cfg["modulus"])
            pairs = self.rng.permutation(p * p)
            train_end, val_end = int(0.5 * len(pairs)), int(0.75 * len(pairs))
            fraction = {"rule_clean": 0.0, "rule_half_noise": 0.5, "rule_random": 1.0}[c]
            corrupt = set(self.rng.permutation(train_end)[:round(fraction * train_end)])
            # Sample replacements even in clean condition so all corpus identities align.
            replacements = self.rng.integers(p, size=train_end)
            for i, pair in enumerate(pairs):
                a, b = divmod(int(pair), p)
                truth = (16 + (a + b) % p,)
                prompt = (1, 16 + a, 2, 16 + b, 3)
                if i < train_end:
                    target = (16 + int(replacements[i]),) if i in corrupt else truth
                    kind = "randomized" if i in corrupt else "clean"
                    r = Record(f"pair_{a}_{b}", f"train_observed_{kind}", prompt, target)
                    self.train.append(r)
                    self.audit.append(r)
                    if i in corrupt:
                        self.audit.append(replace(r, group="train_true_randomized", target=truth))
                else:
                    group = "validation_rule" if i < val_end else "test_rule"
                    self.audit.append(Record(f"pair_{a}_{b}", group, prompt, truth))
        else:
            # Unique, disjoint keys. B values are explicitly different from A values.
            keys: list[tuple[int, ...]] = []
            while len(keys) < 3 * n:
                key = self.tokens(8)
                if key not in keys:
                    keys.append(key)
            values = [self.tokens(self.cfg["value_tokens"]) for _ in keys]
            for i in range(n):
                if values[n + i] == values[i]:
                    raise RuntimeError("A/B target collision; change the preregistered data seed")
                a = Record(f"A_{i}", "memory_A", (1, 4) + keys[i] + (3,), values[i])
                bkey = keys[i] if c == "forgetting_conflict" else keys[n + i]
                b = Record(f"B_{i}", "memory_B", (1, 4) + bkey + (3,), values[n + i])
                self.train.append(a)
                self.second.append(b)
                self.audit.append(a)
                if c.startswith("forgetting"):
                    self.audit.append(b)
                else:
                    self.audit.extend([
                        replace(a, group="seen_key_new_template", prompt=(1, 6) + keys[i] + (3,)),
                        replace(a, group="wrong_key_control", prompt=(1, 4) + keys[(i + 1) % n] + (3,)),
                        Record(f"unseen_{i}", "unseen_key_control", (1, 4) + keys[2*n+i] + (3,), values[2*n+i]),
                    ])

    def sample(self, step: int) -> list[Record]:
        if not 0 <= step < self.steps:
            raise ValueError("step outside the frozen budget")
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg["data_seed"], self.seed, step, 7]))
        out = []
        for i in range(self.batch):
            if self.condition.startswith("verbatim"):
                # Same background draws in present/absent conditions, even at injections.
                prefix = tuple(map(int, rng.integers(16, 272, self.cfg["canary_prefix_tokens"])))
                suffix = tuple(16 + (v - 16 + 1) % 256 for v in prefix[-self.cfg["canary_suffix_tokens"]:])
                r = Record("background", "background", prefix, suffix)
                if self.condition == "verbatim":
                    r = self.injections.get(step * self.batch + i, r)
            else:
                pool = self.second if self.condition.startswith("forgetting") and step >= self.boundary else self.train
                r = pool[int(rng.integers(len(pool)))]
                if self.condition == "associations":
                    # Two seen templates; the third remains audit-only.
                    template = int(rng.choice([4, 5]))
                    r = replace(r, prompt=(1, template) + r.prompt[2:])
            out.append(r)
        return out

    def probes(self, final: bool = False) -> list[Record]:
        out, counts = [], Counter()
        for r in self.audit:
            if r.group == "test_rule" and not final:
                continue
            if counts[r.group] >= self.stage["probe_limit"]:
                continue
            counts[r.group] += 1
            lengths = self.cfg["prefix_lengths_final"] if final and self.condition.startswith("verbatim") else [len(r.prompt)]
            for length in lengths:
                group = f"{r.group}/prefix_{length}" if self.condition.startswith("verbatim") else r.group
                out.append(replace(r, group=group, prompt=r.prompt[-length:]))
        return out

    def identity(self) -> str:
        return digest({"audit": [asdict(r) for r in self.audit],
                       "injections": {str(k): r.eid for k, r in self.injections.items()},
                       "train": [asdict(r) for r in self.train],
                       "second": [asdict(r) for r in self.second]})


def resolve_profile(source: dict, optimizer: str, recipe: str) -> dict:
    """Optional control: same auxiliary AdamW and per-step matrix decay factors."""
    import copy
    profile = copy.deepcopy(source["optimizer_profiles"][optimizer])
    if recipe == "shared_aux_decay" and optimizer == "muon":
        adam = source["optimizer_profiles"]["adamw"]
        ratio = adam["learning_rate"] / profile["matrix_learning_rate"]
        if not math.isclose(ratio, adam["min_learning_rate"] / profile["matrix_min_learning_rate"]):
            raise ValueError("cannot match decay factors with unequal LR floor ratios")
        profile.update(aux_learning_rate=adam["learning_rate"],
                       aux_min_learning_rate=adam["min_learning_rate"],
                       aux_weight_decay=adam["weight_decay"],
                       matrix_weight_decay=adam["weight_decay"] * ratio,
                       warmup_fraction=adam["warmup_fraction"],
                       lr_schedule_epochs=adam["lr_schedule_epochs"],
                       beta1=adam["beta1"], beta2=adam["beta2"], epsilon=adam["epsilon"])
    return profile


def read_rows(path: Path) -> list[dict]:
    """Only an interrupted final JSONL row may be ignored."""
    lines = path.read_text().splitlines()
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return out


def load_source(cfg: dict) -> dict:
    """Fail closed instead of silently switching baseline/profile versions."""
    import yaml
    for rel, expected in cfg["source_blobs"].items():
        content = (REPO / rel).read_bytes()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != expected:
            raise RuntimeError(f"upstream source changed: {rel}; review and version suite.json")
    return yaml.safe_load((REPO / cfg["source_config"]).read_text())


def set_environment(root: Path) -> Path:
    root = root.expanduser().resolve()
    valid = any(root.is_relative_to(Path(base)) and root != Path(base) for base in ("/tmp", "/private/tmp"))
    if not valid:
        raise ValueError("output root must be a dedicated directory beneath /tmp or /private/tmp")
    root.mkdir(parents=True, exist_ok=True)
    for key, sub in {"HOME": "home", "XDG_CACHE_HOME": "xdg/cache", "XDG_CONFIG_HOME": "xdg/config",
                     "XDG_DATA_HOME": "xdg/data", "XDG_STATE_HOME": "xdg/state", "MPLCONFIGDIR": "matplotlib",
                     "HF_HOME": "huggingface", "TORCH_HOME": "torch", "PIP_CACHE_DIR": "pip",
                     "TIKTOKEN_CACHE_DIR": "tiktoken", "TMPDIR": "tmp"}.items():
        path = root / "cache" / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return root


def weightwatch(model, dest: Path, cfg: dict, seed: int, step: int) -> None:
    import torch
    import weightwatcher as ww
    from rg_nanogpt_one_head.model import transformer_matrix_items
    if importlib.metadata.version("weightwatcher") != cfg["version"]:
        raise RuntimeError("WeightWatcher version must match the suite pin")
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        # Creating nn.Linear modules consumes the CPU RNG: isolate that too.
        with torch.random.fork_rng(devices=[]):
            np.random.seed((seed + step + 104729) % (2**32))
            random.seed(seed + step + 104729)
            torch.random.default_generator.manual_seed(seed + step + 104729)
            holder = torch.nn.ModuleDict()
            for name, _, _, w in transformer_matrix_items(model):
                layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="cpu")
                with torch.no_grad():
                    layer.weight.copy_(w.detach().float().cpu())
                holder[name] = layer
            args = {k: v for k, v in cfg.items() if k != "version"}
            table = ww.WeightWatcher(model=holder).analyze(**args)
            required = {"alpha", "raw_alpha", "D", "rand_distance", "ERG_gap", "num_traps"}
            if not required.issubset(table.columns) or len(table) != len(holder):
                raise RuntimeError(f"unexpected WW schema/coverage: {list(table.columns)}, rows={len(table)}")
            # Retain all WW columns; never substitute a proxy for an unavailable field.
            table["alpha_clip_xmax"] = table["alpha"]
            table["alpha_raw"] = table["raw_alpha"]
            table["step"], table["diagnostic_seed"] = step, seed + step + 104729
            names = []
            for _, row in table.iterrows():
                text = " ".join(str(row.get(k, "")) for k in ("longname", "name"))
                matches = [name for name in holder if name in text]
                if len(matches) != 1:
                    raise RuntimeError(f"cannot bind WW row to a matrix: {text}")
                names.append(matches[0])
            if len(set(names)) != len(holder):
                raise RuntimeError("duplicate/missing matrix in WeightWatcher output")
            table["matrix_name"] = names
            # This flag concerns fit support ONLY, not evidence of learned correlations.
            tail = table["num_pl_spikes"] if "num_pl_spikes" in table else np.full(len(table), np.nan)
            table["tail_support_at_least_20"] = np.asarray(tail, dtype=float) >= 20
            table["finite_alpha_and_D"] = np.isfinite(table["alpha"]) & np.isfinite(table["D"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(".tmp")
            table.to_csv(temp, index=False)
            os.replace(temp, dest)
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def evaluate(model, records: list[Record], batch_size: int, device: str) -> dict:
    import torch
    import torch.nn.functional as F
    was_training = model.training
    model.eval()
    grouped: dict[str, list[dict]] = {}
    try:
        with torch.inference_mode():
            # Separate target lengths so autoregressive continuations have equal context.
            for length in sorted({len(r.target) for r in records}):
                rows = [r for r in records if len(r.target) == length]
                for start in range(0, len(rows), batch_size):
                    batch = rows[start:start + batch_size]
                    x, y = pack(batch, model.cfg.block_size)
                    x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
                    logits, _ = model(x)
                    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                             reduction="none", ignore_index=-100).reshape(y.shape)
                    valid = y.ne(-100)
                    nll = (losses.sum(1) / valid.sum(1)).cpu().tolist()
                    accuracy = ((logits.argmax(-1).eq(y) & valid).sum(1) / valid.sum(1)).cpu().tolist()
                    prompt = x[:, :model.cfg.block_size + 1 - length]
                    generated = model.generate_greedy(prompt, length)[:, -length:].cpu().numpy()
                    for j, r in enumerate(batch):
                        target = np.array(r.target)
                        entry = {"eid": r.eid, "nll": nll[j], "teacher_forced_accuracy": accuracy[j],
                                 "exact_match": float(np.array_equal(generated[j], target)),
                                 "continuation_token_accuracy": float((generated[j] == target).mean())}
                        grouped.setdefault(r.group, []).append(entry)
        return {group: {"n": len(rows), "mean": {key: float(np.mean([r[key] for r in rows]))
                      for key in ("nll", "teacher_forced_accuracy", "exact_match", "continuation_token_accuracy")},
                      "examples": rows} for group, rows in grouped.items()}
    finally:
        model.train(was_training)


def state_digest(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def run(args, cfg: dict) -> None:
    root = set_environment(Path(args.root or cfg["output_root"]))
    source = load_source(cfg)
    sys.path.insert(0, str(REPO / "baseline/nanogpt_one_head/src"))
    import torch
    from rg_nanogpt_one_head.model import GPT, GPTConfig
    from rg_nanogpt_one_head.optimizers import make_optimizer_handles, set_learning_rates, zero_grad, optimizer_step
    import fcntl
    if importlib.metadata.version("weightwatcher") != cfg["weightwatcher"]["version"]:
        raise RuntimeError("install the repository's pinned WeightWatcher before running")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    model = GPT(GPTConfig(**source["model"])).to(args.device)
    initial_hash = state_digest(model)
    profile = resolve_profile(source, args.optimizer, args.recipe)
    handles = make_optimizer_handles(model, profile)
    train = source["training"]
    effective_batch = train["batch_size"] * train["grad_accum_steps"]
    study = Study(cfg, args.condition, args.stage, args.seed, effective_batch)
    tokens_per_step = effective_batch * source["model"]["block_size"]
    schedule_steps = math.ceil(source["dataset"]["train_tokens"] * profile["lr_schedule_epochs"] / tokens_per_step)
    warmup = min(schedule_steps - 1, math.ceil(schedule_steps * profile["warmup_fraction"]))
    run_dir = root / args.stage / args.recipe / args.condition / args.optimizer / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    device_info = {"device": args.device, "platform": platform.platform(), "machine": platform.machine(),
                   "processor": platform.processor(), "python": platform.python_version(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "weightwatcher": importlib.metadata.version("weightwatcher"),
                   "packages": sorted((d.metadata.get("Name", "unknown"), d.version)
                                      for d in importlib.metadata.distributions())}
    if args.device == "cuda":
        device_info["accelerator"] = str(torch.cuda.get_device_properties(0))
        device_info["cuda"] = torch.version.cuda
    if args.device == "mps":
        import subprocess
        device_info["accelerator"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    manifest = {"suite": cfg, "source_model": source["model"], "profile": profile,
                "condition": args.condition, "stage": args.stage, "recipe": args.recipe, "seed": args.seed,
                "initial_model_sha256": initial_hash, "data_sha256": study.identity(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_tokens_per_step": tokens_per_step, "schedule_steps": schedule_steps,
                "warmup_steps": warmup, "device": device_info,
                "objective": "answer_suffix_cross_entropy", "protected_test_used_for_selection": False}
    fingerprint = digest(manifest)
    manifest["fingerprint"] = fingerprint
    manifest = json.loads(json.dumps(manifest))
    for other in cfg["optimizers"]:
        if other == args.optimizer:
            continue
        peer = run_dir.parent.parent / other / run_dir.name / "manifest.json"
        if peer.exists():
            paired = json.loads(peer.read_text())
            for key in ("initial_model_sha256", "data_sha256", "source_model", "suite",
                        "stage", "recipe", "seed", "runner_sha256", "device"):
                if paired[key] != manifest[key]:
                    raise RuntimeError(f"paired optimizer arm mismatch: {key}; use a new root")
    latest = run_dir / "checkpoint_latest.pt"
    metrics_path = run_dir / "metrics.jsonl"
    first, exposures = 0, Counter()
    if (run_dir / "manifest.json").exists():
        old = json.loads((run_dir / "manifest.json").read_text())
        if old != manifest:
            raise RuntimeError("run fingerprint changed; use a new output root")
        if not args.resume:
            raise RuntimeError("run exists; pass --resume or use a new root")
        if (run_dir / "complete.json").exists():
            print(f"Already complete: {run_dir}"); return
        if not latest.exists():
            raise RuntimeError("no restart checkpoint; inspect the failed preflight and use a new root")
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint["fingerprint"] != fingerprint:
            raise RuntimeError("checkpoint fingerprint mismatch")
        model.load_state_dict(checkpoint["model"])
        for handle, state in zip(handles, checkpoint["optimizers"], strict=True):
            handle.optimizer.load_state_dict(state)
        torch.set_rng_state(checkpoint["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["device_rng"])
        elif args.device == "mps":
            torch.mps.set_rng_state(checkpoint["device_rng"])
        first, exposures = checkpoint["step"], Counter(checkpoint["exposures"])
        # Discard incomplete rows/files after the last atomic restart point.
        if metrics_path.exists():
            rows = [row for row in read_rows(metrics_path) if row["step"] <= first]
            metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        for path in (run_dir / "spectral").glob("step_*.csv"):
            if int(path.stem.split("_")[-1]) > first:
                path.unlink()
    else:
        atomic_json(run_dir / "manifest.json", manifest)
        atomic_json(run_dir / "probe_inventory.json", [asdict(r) for r in study.audit])
        atomic_json(run_dir / "injection_schedule.json", {str(k): r.eid for k, r in study.injections.items()})
    permanent = {round(study.steps * i / 16) for i in range(17)}

    def checkpoint(step: int, loss: float | None) -> None:
        before = state_digest(model)
        audit = evaluate(model, study.probes(final=(step == study.steps)), train["batch_size"], args.device)
        weightwatch(model, run_dir / "spectral" / f"step_{step:08d}.csv", cfg["weightwatcher"], args.seed, step)
        if state_digest(model) != before:
            raise RuntimeError("evaluation/diagnostics changed training weights")
        row = {"step": step, "phase": "A" if step <= study.boundary else "B", "last_train_loss": loss,
               "input_tokens": step * tokens_per_step, "model_sha256": before, "audit": audit}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())
        state = {"step": step, "fingerprint": fingerprint, "model": model.state_dict(),
                 "optimizers": [h.optimizer.state_dict() for h in handles], "exposures": dict(exposures),
                 "torch_rng": torch.get_rng_state(), "device_rng": None}
        if args.device == "cuda":
            state["device_rng"] = torch.cuda.get_rng_state_all()
        elif args.device == "mps":
            state["device_rng"] = torch.mps.get_rng_state()
        temp = latest.with_suffix(".tmp")
        torch.save(state, temp); os.replace(temp, latest)
        if step in permanent or step == study.boundary:
            dest = run_dir / f"model_step_{step:08d}.pt"
            torch.save({"model": state["model"], "step": step, "fingerprint": fingerprint}, dest.with_suffix(".tmp"))
            os.replace(dest.with_suffix(".tmp"), dest)
        counts = {r.eid: exposures.get(r.eid, 0) for r in study.audit}
        counts.update(exposures)
        atomic_json(run_dir / "exposures.json", counts)
        print(f"{args.condition} {args.optimizer} seed={args.seed} step={step}/{study.steps} loss={loss} WW=clip_xmax", flush=True)

    if first == 0 and not latest.exists():
        checkpoint(0, None)
    model.train()
    for step in range(first, study.steps):
        zero_grad(handles)
        set_learning_rates(handles, update_index=step, total_steps=schedule_steps, warmup_steps=warmup)
        records = study.sample(step)
        total_loss = 0.0
        for i in range(0, effective_batch, train["batch_size"]):
            batch = records[i:i + train["batch_size"]]
            x, y = pack(batch, source["model"]["block_size"])
            _, loss = model(torch.as_tensor(x, device=args.device), torch.as_tensor(y, device=args.device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            (loss / train["grad_accum_steps"]).backward()
            total_loss += float(loss.detach()) / train["grad_accum_steps"]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
        if not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite gradient at step {step}")
        optimizer_step(handles)
        exposures.update(r.eid for r in records)
        done = step + 1
        if done % 25 == 0:
            print(f"step={done} train_loss={total_loss:.6f}", flush=True)
        if done in permanent or done in cfg["early_eval_steps"] or done == study.boundary or done % study.stage["eval_every"] == 0 or done == study.steps:
            checkpoint(done, total_loss)
    atomic_json(run_dir / "complete.json", {"fingerprint": fingerprint, "steps": study.steps, "status": "complete"})
    print(f"Complete: {run_dir}")


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--stage", choices=cfg["stages"], default="pilot")
    train = sub.add_parser("run")
    train.add_argument("--stage", choices=cfg["stages"], default="pilot")
    train.add_argument("--condition", choices=cfg["conditions"], required=True)
    train.add_argument("--optimizer", choices=cfg["optimizers"], required=True)
    train.add_argument("--seed", type=int, choices=cfg["seeds"], default=cfg["seeds"][0])
    train.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    train.add_argument("--root")
    train.add_argument("--recipe", choices=["repository", "shared_aux_decay"], default="repository")
    train.add_argument("--resume", action="store_true")
    monitor = sub.add_parser("monitor")
    monitor.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args, cfg)
    elif args.command == "plan":
        seeds = cfg["seeds"] if args.stage == "full" else cfg["seeds"][:1]
        print(f"# {len(seeds)*len(cfg['optimizers'])*len(cfg['conditions'])} runs; {cfg['stages'][args.stage]['steps']} steps/run; synthetic suffix loss, NOT FineWeb NLL.")
        for condition in cfg["conditions"]:
            for seed in seeds:
                for optimizer in cfg["optimizers"]:
                    print(f"python run.py run --stage {args.stage} --condition {condition} --optimizer {optimizer} --seed {seed} --device mps --resume")
    else:
        rows = read_rows(args.run_dir / "metrics.jsonl")
        if not rows:
            raise SystemExit("No completed audit yet")
        latest = rows[-1]
        print(json.dumps({"step": latest["step"], "audit": {k: v["mean"] for k, v in latest["audit"].items()}}, indent=2))
        print((args.run_dir / "spectral" / f"step_{latest['step']:08d}.csv").read_text())


if __name__ == "__main__":
    main()
