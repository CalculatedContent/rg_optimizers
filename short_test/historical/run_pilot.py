"""One reproducible, single-thread CPU nanoGPT timing pilot; no model search."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import time
PROCESS_START = time.perf_counter()
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import statistics
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run"
OUT.mkdir(exist_ok=True)
LOCK = (OUT / "run.lock").open("a")
fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def cgroup():
    base = Path("/sys/fs/cgroup")
    result = {}
    for name in ("memory.current", "memory.max", "memory.events", "cpu.max"):
        p = base / name
        if p.exists():
            result[name] = p.read_text().strip()
    return result

if (OUT / "summary.json").exists():
    print("Completed pilot exists; skipping training.", flush=True)
    print((OUT / "summary.json").read_text(), flush=True)
    sys.exit(0)

before = cgroup()
stats = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
headroom = int(before["memory.max"]) - int(stats["anon"]) - int(stats["kernel"])
if headroom < 950 * 1024**2:
    raise RuntimeError("Insufficient non-file memory headroom; leave existing sweeps untouched and retry later")

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
sys.path.insert(0, str(ROOT / "nanoGPT"))
from model import GPT, GPTConfig

source = ROOT / "nanoGPT" / "model.py"
input_path = ROOT / "nanoGPT" / "data" / "shakespeare_char" / "input.txt"
text = input_path.read_text()
chars = sorted(set(text))
assert len(text) == 1115394 and len(chars) == 65, "Unexpected Shakespeare dataset"
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)
n = len(ids)
a, b = int(.90 * n), int(.95 * n)
splits = {"train": ids[:a], "validation": ids[a:b], "test": ids[b:]}
config = {
    "purpose": "Single fixed CPU timing pilot, not a hyperparameter or optimizer comparison",
    "upstream_url": "https://github.com/karpathy/nanoGPT",
    "upstream_commit": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
    "model_source_sha256": sha(source), "runner_sha256": sha(Path(__file__)),
    "data_url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "input_sha256": sha(input_path), "seed": 1337,
    "architecture": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 64,
                     "vocab_size": 65, "dropout": 0.0, "bias": False},
    "optimizer": "AdamW", "learning_rate": .001, "min_lr": .0001,
    "warmup_steps": 100, "steps": 2000, "batch_size": 12,
    "weight_decay": .1, "betas": [.9, .99], "grad_clip": 1.,
    "device": "cpu", "dtype": "float32", "threads": 1, "compile": False,
    "split_policy": "Contiguous first 90% train, next 5% validation, last 5% test; unlike upstream 90/10",
    "split_lengths": {k: len(v) for k, v in splits.items()},
    "evaluation": "Fixed sampled train/validation batches every 250 steps; full validation and test only at fixed endpoint",
    "selection": "Fixed endpoint; no early stopping, tuning, or test-guided selection",
    "vocabulary": "All 65 characters from the corpus; held-out text never enters gradient updates",
}
config_path = OUT / "config.json"
if config_path.exists():
    assert json.loads(config_path.read_text()) == config, "Frozen pilot config mismatch"
else:
    write_json(config_path, config)

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
train_rng = torch.Generator().manual_seed(config["seed"] + 1)
model = GPT(GPTConfig(**config["architecture"]))
optimizer = model.configure_optimizers(.1, .001, (.9, .99), "cpu")
step = 0
history, timings, events = [], [], []
checkpoint = OUT / "last.pt"
if checkpoint.exists():
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["config"] == config
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    train_rng.set_state(saved["train_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    step, history, timings, events = saved["steps"], saved["history"], saved["timings"], saved["events"]
    del saved

def batch(split, rng):
    data = splits[split]
    ix = torch.randint(len(data) - 64, (12,), generator=rng)
    x = torch.stack([torch.from_numpy(data[i:i+64].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+65].astype(np.int64)) for i in ix])
    return x, y

@torch.no_grad()
def sampled_eval():
    model.eval()
    result = {}
    for index, split in enumerate(("train", "validation")):
        rng = torch.Generator().manual_seed(10001 + index)
        losses = [model(*batch(split, rng))[1].item() for _ in range(20)]
        result[split] = statistics.mean(losses)
    model.train()
    return result

@torch.no_grad()
def full_eval(split):
    # All possible next-character targets exactly once, with context reset every 64 characters.
    model.eval()
    data = splits[split]
    total_loss, correct, count = 0., 0, 0
    for start in range(0, len(data) - 1, 12 * 64):
        length = min(12 * 64, len(data) - 1 - start)
        whole = length // 64
        if whole:
            size = whole * 64
            x = torch.from_numpy(data[start:start+size].astype(np.int64)).view(whole, 64)
            y = torch.from_numpy(data[start+1:start+size+1].astype(np.int64)).view(whole, 64)
            logits, loss = model(x, y)
            total_loss += loss.item() * size
            correct += (logits.argmax(-1) == y).sum().item()
            count += size
        if length % 64:
            offset, tail = start + whole * 64, length % 64
            x = torch.from_numpy(data[offset:offset+tail].astype(np.int64))[None, :]
            y = torch.from_numpy(data[offset+1:offset+tail+1].astype(np.int64))[None, :]
            logits, loss = model(x, y)
            total_loss += loss.item() * tail
            correct += (logits.argmax(-1) == y).sum().item()
            count += tail
    assert count == len(data) - 1
    loss = total_loss / count
    model.train()
    return {"loss_nats_per_character": loss, "perplexity": math.exp(loss),
            "bits_per_character": loss / math.log(2), "next_character_accuracy": correct / count,
            "evaluated_targets": count}

def save_checkpoint():
    tmp = checkpoint.with_suffix(".pt.tmp")
    torch.save({"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "steps": step, "torch_rng": torch.get_rng_state(), "train_rng": train_rng.get_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "history": history, "timings": timings, "events": events}, tmp)
    tmp.replace(checkpoint)
    write_json(OUT / "progress.json", {"steps": step, "total": 2000, "last_eval": history[-1],
                                       "training_seconds": sum(timings)})

started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
train_start, cpu_start, start_step = time.perf_counter(), time.process_time(), step
print(json.dumps({"event": "start", "utc": started_utc, "start_step": step,
                  "parameters": model.get_num_params(non_embedding=False), "config": config}), flush=True)
if step == 0:
    history.append({"step": 0, **sampled_eval()})
    print(json.dumps(history[-1]), flush=True)
while step < 2000:
    t0 = time.perf_counter()
    if step < 100:
        lr = .001 * (step + 1) / 101
    else:
        lr = .0001 + .5 * (1 + math.cos(math.pi * (step - 100) / 1900)) * .0009
    for group in optimizer.param_groups:
        group["lr"] = lr
    x, y = batch("train", train_rng)
    _, loss = model(x, y)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    step += 1
    timings.append(time.perf_counter() - t0)
    if step % 50 == 0:
        print(json.dumps({"step": step, "loss": loss.item(), "seconds_per_step_last50": statistics.mean(timings[-50:]),
                          "elapsed_seconds": time.perf_counter() - train_start}), flush=True)
    if step % 250 == 0:
        history.append({"step": step, **sampled_eval()})
        save_checkpoint()
        print(json.dumps({"evaluation": history[-1], "checkpoint_saved": True}), flush=True)

training_loop_seconds = time.perf_counter() - train_start
evaluation_start = time.perf_counter()
validation, test = full_eval("validation"), full_eval("test")
full_evaluation_seconds = time.perf_counter() - evaluation_start
model.eval()
with torch.random.fork_rng():
    torch.manual_seed(20261021)
    sample = model.generate(torch.tensor([[stoi['\n']]]), 500, temperature=.8, top_k=40)
(OUT / "sample.txt").write_text(''.join(chars[i] for i in sample[0].tolist()))
summary = {
    "completed": True, "completed_steps": step, "updates_this_session": step - start_step,
    "started_utc": started_utc, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "trainable_parameters": model.get_num_params(non_embedding=False),
    "training_tokens_processed": step * 12 * 64,
    "timing_seconds": {"training_steps_only": sum(timings), "training_loop_with_evals_and_checkpoints": training_loop_seconds,
                       "final_full_validation_and_test": full_evaluation_seconds,
                       "process_including_imports_setup_evaluation_and_generation": time.perf_counter() - PROCESS_START,
                       "cpu_this_session_after_setup": time.process_time() - cpu_start,
                       "median_step": statistics.median(timings), "mean_last_100_steps": statistics.mean(timings[-100:])},
    "training_characters_per_second": step * 12 * 64 / sum(timings),
    "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "uniform_baseline_loss_nats": math.log(65), "uniform_baseline_perplexity": 65,
    "sampled_losses": history, "validation": validation, "test": test,
    "torch_version": torch.__version__, "numpy_version": np.__version__,
    "cgroup_before": before, "cgroup_after": cgroup(),
    "checkpoint_sha256": sha(checkpoint), "config_sha256": sha(config_path),
    "caveat": "One fixed-budget seed on one CPU thread alongside six existing workers; not time to convergence or a general CPU benchmark",
}
write_json(OUT / "summary.json", summary)
print(json.dumps({"event": "finished", "summary": summary}), flush=True)
