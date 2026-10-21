"""Validation-only convergence continuation of the frozen 2,000-step pilot."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
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
OUT = ROOT / "convergence"
OUT.mkdir(exist_ok=True)
lock = (OUT / "run.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def atomic_json(p, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)

def memory_events():
    return dict((k, int(v)) for k, v in (line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()))

if (OUT / "summary.json").exists():
    print("Completed continuation exists; no further training.", flush=True)
    sys.exit(0)

original = json.loads((ROOT / "run/config.json").read_text())
pilot = json.loads((ROOT / "run/summary.json").read_text())
assert sha(ROOT / "run/last.pt") == pilot["checkpoint_sha256"]
assert sha(ROOT / "nanoGPT/model.py") == original["model_source_sha256"]
assert sha(ROOT / "nanoGPT/data/shakespeare_char/input.txt") == original["input_sha256"]
plan = {
    "purpose": "Measure time to a declared practical validation plateau, not global optimization convergence",
    "runner_sha256": sha(Path(__file__)), "initial_checkpoint_sha256": pilot["checkpoint_sha256"],
    "initial_step": 2000, "max_total_steps": 20000, "eval_every_steps": 500,
    "initial_learning_rate": .0001, "confirmation_learning_rate": .00005,
    "meaningful_improvement_nats_per_character": .005, "patience_evaluations": 3,
    "stop_rule": "After 3 consecutive full-validation checks without cumulative improvement of 0.005 from the significant-improvement anchor, reduce LR once from 1e-4 to 5e-5; after another such 3-check plateau stop. Any significant improvement resets patience. A hard-cap stop is NOT convergence.",
    "selection": "Lowest full validation loss, including the original step-2000 checkpoint; never test or alpha",
    "unchanged": ["architecture", "vocabulary", "train/validation/test partitions", "batch size 12", "optimizer moments", "training minibatch and random states", "weight decay", "gradient clipping"],
    "test_policy": "The pilot's test score was already exposed. No intermediate test checks. Re-evaluate that same held-out partition only after validation selection and stopping; no retuning afterward.",
    "original_config": original,
}
if (OUT / "plan.json").exists():
    assert json.loads((OUT / "plan.json").read_text()) == plan, "Frozen continuation plan mismatch"
else:
    atomic_json(OUT / "plan.json", plan)
mem = dict((k, int(v)) for k, v in (line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()))
headroom = int(Path("/sys/fs/cgroup/memory.max").read_text()) - mem["anon"] - mem["kernel"]
assert headroom >= 950 * 1024**2, "Insufficient memory headroom; retry without disturbing existing sweeps"
oom_before = memory_events()

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
sys.path.insert(0, str(ROOT / "nanoGPT"))
from model import GPT, GPTConfig

text = (ROOT / "nanoGPT/data/shakespeare_char/input.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)
a, b = int(.90 * len(ids)), int(.95 * len(ids))
splits = {"train": ids[:a], "validation": ids[a:b], "test": ids[b:]}
model = GPT(GPTConfig(**original["architecture"]))
optimizer = model.configure_optimizers(.1, .0001, (.9, .99), "cpu")
train_rng = torch.Generator()
resume = (OUT / "last.pt").exists()
state = torch.load(OUT / "last.pt" if resume else ROOT / "run/last.pt", map_location="cpu", weights_only=False)
if resume:
    assert state["plan"] == plan
else:
    assert state["config"] == original
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
torch.set_rng_state(state["torch_rng"])
train_rng.set_state(state["train_rng"])
np.random.set_state(state["numpy_rng"])
random.setstate(state["python_rng"])
step = state["steps"]
history = state.get("continuation_history", [])
timings = state.get("continuation_timings", [])
lr = state.get("continuation_lr", .0001)
anchor = state.get("anchor", float("inf"))
bad_checks = state.get("bad_checks", 0)
best_loss = state.get("best_loss", float("inf"))
best_step = state.get("best_step", 2000)
reductions = state.get("reductions", [])
sessions = state.get("sessions", [])
del state

def batch(split, rng):
    data = splits[split]
    ix = torch.randint(len(data) - 64, (12,), generator=rng)
    return (torch.stack([torch.from_numpy(data[i:i+64].astype(np.int64)) for i in ix]),
            torch.stack([torch.from_numpy(data[i+1:i+65].astype(np.int64)) for i in ix]))

@torch.no_grad()
def sampled_train_loss():
    model.eval()
    rng = torch.Generator().manual_seed(10001)
    result = statistics.mean(model(*batch("train", rng))[1].item() for _ in range(20))
    model.train()
    return result

@torch.no_grad()
def full_eval(split):
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
    model.train()
    loss = total_loss / count
    return {"loss_nats_per_character": loss, "perplexity": math.exp(loss),
            "next_character_accuracy": correct / count, "evaluated_targets": count}

started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
session_start, session_step = time.perf_counter(), step

def save(name):
    current_session = {"started_utc": started_utc, "first_step": session_step, "last_step": step,
                       "elapsed_seconds": time.perf_counter() - session_start}
    obj = {"plan": plan, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
           "steps": step, "torch_rng": torch.get_rng_state(), "train_rng": train_rng.get_state(),
           "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
           "continuation_history": history, "continuation_timings": timings, "continuation_lr": lr,
           "anchor": anchor, "bad_checks": bad_checks, "best_loss": best_loss, "best_step": best_step,
           "reductions": reductions, "sessions": sessions + [current_session]}
    path = OUT / name
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
    atomic_json(OUT / "progress.json", {"step": step, "best_step": best_step, "best_loss": best_loss,
                                       "lr": lr, "bad_checks": bad_checks, "history": history,
                                       "training_seconds": sum(timings), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})

print(json.dumps({"event": "start", "utc": started_utc, "step": step, "plan": plan}), flush=True)
if not resume:
    validation = full_eval("validation")
    assert abs(validation["loss_nats_per_character"] - pilot["validation"]["loss_nats_per_character"]) < 1e-7
    best_loss = anchor = validation["loss_nats_per_character"]
    history.append({"step": step, "train_sample_loss": sampled_train_loss(), "validation": validation,
                    "lr": lr, "anchor": anchor, "bad_checks": 0})
    save("best.pt")
    save("last.pt")

stop_reason = None
if lr == .00005 and bad_checks >= 3:
    stop_reason = "validation_plateau_confirmed_at_lower_learning_rate"
while step < 20000 and stop_reason is None:
    t0 = time.perf_counter()
    for group in optimizer.param_groups:
        group["lr"] = lr
    x, y = batch("train", train_rng)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    assert torch.isfinite(loss), "Nonfinite training loss"
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    step += 1
    timings.append(time.perf_counter() - t0)
    if step % 100 == 0:
        print(json.dumps({"step": step, "loss": loss.item(), "lr": lr,
                          "seconds_per_step_last100": statistics.mean(timings[-100:])}), flush=True)
    if step % 500 == 0:
        validation = full_eval("validation")
        value = validation["loss_nats_per_character"]
        if anchor - value >= .005:
            anchor, bad_checks = value, 0
        else:
            bad_checks += 1
        entry = {"step": step, "train_sample_loss": sampled_train_loss(), "validation": validation,
                 "lr": lr, "anchor": anchor, "bad_checks": bad_checks,
                 "elapsed_this_session": time.perf_counter() - session_start}
        history.append(entry)
        is_best = value < best_loss
        if is_best:
            best_loss, best_step = value, step
        if bad_checks >= 3:
            if lr > .00005:
                reductions.append({"step": step, "from": lr, "to": .00005, "validation_loss": value})
                lr, anchor, bad_checks = .00005, value, 0
            else:
                stop_reason = "validation_plateau_confirmed_at_lower_learning_rate"
        if is_best:
            save("best.pt")
        save("last.pt")
        print(json.dumps({"evaluation": entry, "best_step": best_step, "best_loss": best_loss,
                          "next_lr": lr, "stop_reason": stop_reason}), flush=True)
    elif step % 250 == 0:
        save("last.pt")
if stop_reason is None:
    stop_reason = "hard_cap_reached_without_confirmed_plateau"

training_loop_seconds = sum(s["elapsed_seconds"] for s in sessions) + time.perf_counter() - session_start
endpoint = history[-1]
assert endpoint["step"] == step
# Freeze selection before opening the test evaluator.
selection = {"training_stop_step": step, "selected_step": best_step, "validation_loss": best_loss,
             "stop_reason": stop_reason, "selected_checkpoint_sha256": sha(OUT / "best.pt"),
             "selection_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
atomic_json(OUT / "selection.json", selection)
best = torch.load(OUT / "best.pt", map_location="cpu", weights_only=False)
assert best["steps"] == best_step
assert all(torch.isfinite(t).all() for t in best["model"].values())
assert all(int(s["step"].item()) == best_step for s in best["optimizer"]["state"].values())
model.load_state_dict(best["model"])
del best
evaluation_start = time.perf_counter()
full_train = full_eval("train")
selected_validation = full_eval("validation")
assert abs(selected_validation["loss_nats_per_character"] - best_loss) < 1e-7
selected_test = full_eval("test")
evaluation_seconds = time.perf_counter() - evaluation_start
model.eval()
with torch.random.fork_rng():
    torch.manual_seed(20261021)
    sample = model.generate(torch.tensor([[stoi['\n']]]), 500, temperature=.8, top_k=40)
(OUT / "sample.txt").write_text(''.join(chars[i] for i in sample[0].tolist()))
summary = {"completed": True, "practical_validation_plateau_confirmed": stop_reason.startswith("validation_plateau"),
           "stop_reason": stop_reason, "training_stop_step": step, "selected_step": best_step,
           "selected_full_train": full_train, "selected_validation": selected_validation, "selected_test": selected_test,
           "endpoint": endpoint, "history": history, "lr_reductions": reductions,
           "timing_seconds": {"continuation_training_updates": sum(timings), "continuation_training_with_monitoring": training_loop_seconds,
                              "final_full_train_validation_test": evaluation_seconds,
                              "this_process_including_setup_evaluation_generation": time.perf_counter() - PROCESS_START,
                              "pilot_plus_continuation_training_updates": pilot["timing_seconds"]["training_steps_only"] + sum(timings)},
           "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           "memory_events_before": oom_before, "memory_events_after": memory_events(),
           "selected_checkpoint_sha256": sha(OUT / "best.pt"), "last_checkpoint_sha256": sha(OUT / "last.pt"),
           "plan_sha256": sha(OUT / "plan.json"), "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "test_disclosure": plan["test_policy"],
           "limits": "Operational validation plateau for one resumed seed and this training schedule; not a global optimum, a universal target loss, or proof that training loss has converged."}
atomic_json(OUT / "summary.json", summary)
print(json.dumps({"event": "complete", "summary": summary}), flush=True)
