"""Real CPU training and fresh-process continuation exercise, invoked by pytest."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from rg_nanogpt_one_head.muonclip import install_muonclip_extension
install_muonclip_extension()
from rg_nanogpt_one_head.config import load_config
from rg_nanogpt_one_head.continuation import make_continuation_config, TrainingPaused, file_sha256
from rg_nanogpt_one_head.checkpoints import optimizer_state_sha256
from rg_nanogpt_one_head.completion import validate_completed_run
from rg_nanogpt_one_head.monitor import load_series_frames
from rg_nanogpt_one_head.training import run_one

root = Path(sys.argv[1])
cfg = deepcopy(load_config("configs/muonclip_reference.yaml"))
cfg["dataset"].update(name="unit/fineweb", config="unit", revision="unit-revision", train_tokens=2048, val_tokens=512, test_tokens=512)
cfg["model"].update(vocab_size=64, block_size=8, n_layer=1, n_head=1, n_embd=32, dropout=0.0)
cfg["training"].update(seeds=[13], batch_size=2, grad_accum_steps=2, target_epochs=0.125, max_steps=8, epoch_interval=1.0, eval_interval_steps=2, eval_batches=1, checkpoint_interval_steps=2)
cfg["evaluation"].update(bleu_examples=2, bleu_prompt_tokens=3, bleu_continuation_tokens=2, bleu_batch_size=2, test_interval_steps=2)
cfg["weightwatcher"].update(min_evals=5, fix_fingers=False)
for p in cfg["optimizer_profiles"].values():
    p.pop("lr_schedule_epochs", None)
cfg["optimizer_profiles"]["muon_clip"].update(learning_rate=2e-5, min_learning_rate=2e-5, warmup_fraction=0.0, qk_diagnostics_interval=2)
data = root / "data"
data.mkdir(parents=True)
rng = np.random.default_rng(7)
splits = {k: cfg["dataset"][k + "_tokens"] for k in ("train", "val", "test")}
files = {}
for split, size in splits.items():
    path = data / f"{split}.bin"
    rng.integers(0, 64, size=size, dtype=np.uint16).tofile(path)
    files[split] = {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
(data / "meta.json").write_text(json.dumps({
    "schema_version": 2, "tokenizer": "gpt2", "vocab_size": 64, "dtype": "uint16", "splits": splits,
    "document_disjoint_splits": True, "dataset_name": "unit/fineweb", "dataset_config": "unit",
    "dataset_split": "train", "dataset_revision": "unit-revision", "eot_token": 0, "files": files,
}))

def run(configuration, name):
    return run_one(cfg=configuration, data_root=data, results_root=root / name, optimizer_name="muon_clip", seed=13, device="cpu", progress=False)

def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def same_training_state(a, b):
    assert a["model_state_sha256"] == b["model_state_sha256"]
    assert torch.equal(a["train_generator_state"], b["train_generator_state"])
    assert torch.equal(a["torch_random_state"], b["torch_random_state"])
    aa, bb = deepcopy(a["optimizers"]), deepcopy(b["optimizers"])
    # Phase-local QK statistics have no effect on the update rule.
    aa[0].pop("muonclip_global_state")
    bb[0].pop("muonclip_global_state")
    assert optimizer_state_sha256(aa) == optimizer_state_sha256(bb)

whole = run(cfg, "whole")
parent_cfg = deepcopy(cfg)
parent_cfg["training"].update(max_steps=4, target_epochs=0.0625)
parent = run(parent_cfg, "parent")
parent_path = parent / "checkpoint_final.pt"
parent_hash = file_sha256(parent_path)
extension = make_continuation_config(parent_path, steps=4, learning_rate=None, test_interval=2, stop_file=root / "STOP", min_free_disk_gb=0)
child = run(extension, "child")
validate_completed_run(child)
same_training_state(load(whole / "checkpoint_final.pt"), load(child / "checkpoint_final.pt"))
same_training_state(load(parent_path), load(child / "checkpoint_initial.pt"))
assert load(child / "checkpoint_final.pt")["global_step"] == 8
frame = pd.read_csv(child / "metrics.csv")
assert frame.global_step.tolist() == [4, 6, 8]
assert frame.test_accuracy.notna().all()
assert frame.loc[frame.step > 0, "primary_lr"].eq(2e-5).all()

# Pause after a saved update; resume must reproduce exactly the uninterrupted state.
import rg_nanogpt_one_head.train_loop as loop
original_save = loop.save_training_checkpoint
def pause_after_checkpoint(path, **kwargs):
    result = original_save(path, **kwargs)
    if Path(path).name == "checkpoint_latest.pt" and kwargs["step"] == 2:
        (root / "STOP").touch()
    return result
loop.save_training_checkpoint = pause_after_checkpoint
try:
    run(extension, "paused")
    raise AssertionError("pause was not honored")
except TrainingPaused as exc:
    assert exc.code == 75
finally:
    loop.save_training_checkpoint = original_save
(root / "STOP").unlink()
resumed = run(extension, "paused")
validate_completed_run(resumed)
same_training_state(load(child / "checkpoint_final.pt"), load(resumed / "checkpoint_final.pt"))

# Mutating a protected parent-derived field must be rejected before child training.
bad = deepcopy(extension)
bad["optimizer_profiles"]["muon_clip"]["momentum"] = 0.5
try:
    run(bad, "bad")
    raise AssertionError("optimizer mutation accepted")
except RuntimeError as exc:
    assert "cannot change optimizer" in str(exc)
assert file_sha256(parent_path) == parent_hash

# Exercise actual CLI -> CPU supervisor -> fresh worker across THREE segments,
# strict completion audit, bounded checkpoint retention, and joined monitoring.
series = root / "series"
command = [sys.executable, "-m", "rg_nanogpt_one_head.muonclip_continue", "start",
    "--series-root", str(series), "--from-checkpoint", str(parent_path), "--data-root", str(data),
    "--device", "cpu", "--additional-steps", "6", "--segment-steps", "2", "--test-interval-steps", "2",
    "--keep-segments", "2", "--min-free-disk-gb", "0"]
subprocess.run(command, check=True)
state = json.loads((series / "series.json").read_text())
assert state["status"] == "completed" and state["completed_steps"] == 6
assert len(state["completed_segments"]) == 3
first = series / "segments/segment_000001/muon_clip/seed_13"
assert not (first / "checkpoint_final.pt").exists()
assert (first / "checkpoints_pruned.json").is_file()
assert (first / "metrics.csv").is_file()
assert Path(state["latest_checkpoint"]).is_file()
assert load(Path(state["latest_checkpoint"]))["global_step"] == 10
metrics, layers = load_series_frames(series)
assert metrics.step.tolist() == [0, 2, 4, 6, 8, 10]
assert layers.step.max() == 10
assert file_sha256(parent_path) == parent_hash
# A completed finite series resumes idempotently even with pruned older phases.
subprocess.run([sys.executable, "-m", "rg_nanogpt_one_head.muonclip_continue", "resume", "--series-root", str(series)], check=True)
print("CONTINUATION_SCENARIO_PASSED")
