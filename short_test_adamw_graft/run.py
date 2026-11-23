"""Five-seed TinyShakespeare comparison using the original CPU initialization and data stream."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
PORTABLE = HERE.parent / "short_test_portable"
sys.path.insert(0, str(PORTABLE))
import comparison as previous
import runtime
import torch
from graft import MagnitudeGraft

base = previous.base
SEEDS = (1340, 1341, 1342, 1343, 1344)
ARMS = ("adamw", "sgdm", "graft")
SCHEDULE = dict(max_steps=20000, warmup_steps=100, cosine_end=17000, floor_fraction=0.1,
                eval_every=500, checkpoint_every=250, grad_clip=1.0)
ADAM = dict(optimizer="AdamW", peak_lr=0.001, betas=[0.9, 0.99], weight_decay=0.0)
SGDM = dict(optimizer="SGDM", peak_lr=0.1, momentum=0.9, weight_decay=0.0,
            dampening=0.0, nesterov=False)
RATIO = SGDM["peak_lr"] / ADAM["peak_lr"]


def setup(arm, seed):
    recipe = SGDM if arm == "sgdm" else ADAM
    model, optimizer, rng = previous.setup(seed, recipe)
    if arm == "graft":
        optimizer = MagnitudeGraft(model, optimizer, reference_lr_ratio=RATIO)
    return model, optimizer, rng, recipe


def design():
    sources = [HERE / "run.py", HERE / "graft.py", PORTABLE / "comparison.py",
               PORTABLE / "runtime.py", PORTABLE / "portable_common.py",
               base.SOURCE / "common.py", base.SOURCE / "vendor/model.py"]
    return dict(schema=1, seeds=list(SEEDS), arms=list(ARMS), schedule=SCHEDULE,
                recipes=dict(adamw=ADAM, sgdm=SGDM,
                             graft=dict(native=ADAM, reference=SGDM, reference_lr_ratio=RATIO)),
                architecture=base.ARCHITECTURE, batch_size=12, split="contiguous 90/5/5",
                input_sha256=base.INPUT_SHA, device=runtime.environment(),
                packages={name: importlib.metadata.version(name)
                          for name in ("torch", "numpy", "scipy", "weightwatcher", "powerlaw")},
                sources={str(p.relative_to(HERE.parent)): base.sha(p) for p in sources},
                primary="Paired test cross-entropy in nats per character after 20000 updates",
                secondary="Test cross-entropy at lowest full validation loss, assessed every 500 updates")


def save_checkpoint(path, model, opt, rng, progress, identity):
    snapshot = dict(identity=identity, model=model.state_dict(), optimizer=opt.state_dict(),
                    rng=base.rng_state(rng), progress=progress)
    temp = path.with_suffix(".pt.tmp")
    torch.save(runtime.cpu_tree(snapshot), temp)
    temp.replace(path)


def restore_checkpoint(path, model, opt, rng, identity):
    # Only load checkpoints created locally by this experiment.
    record = torch.load(path, map_location="cpu", weights_only=False)
    if record["identity"] != identity:
        raise ValueError(f"Checkpoint does not belong to this arm, seed, and design: {path}")
    model.load_state_dict(record["model"])
    opt.load_state_dict(record["optimizer"])
    base.restore_rng(record["rng"], rng)
    return record["progress"]


def examine(model, data, step):
    validation = base.full_eval(model, data["validation"])
    return dict(step=step, validation_loss=validation["loss_nats_per_character"],
                validation=validation, train_sample_loss=base.sampled_loss(model, data["train"]),
                alpha=base.spectrum(model.state_dict()))


def measure(model, data, record):
    metrics = {key: (record["validation"] if key == "validation" else base.full_eval(model, ids))
               for key, ids in data.items()}
    return dict(step=record["step"], metrics=metrics, alpha=record["alpha"])


def trial(output, data, arm, seed, schedule, until, design_hash):
    folder = output / arm / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(arm=arm, seed=seed, design_sha256=design_hash)
    with previous.lock(folder / "run.lock"):
        result = folder / "result.json"
        if result.exists():
            if previous.read(result)["identity"] != identity:
                raise ValueError(f"Existing result belongs to another design: {result}")
            print(f"{arm} seed {seed} already complete", flush=True)
            return
        model, opt, rng, recipe = setup(arm, seed)
        last = folder / "last.pt"
        if last.exists():
            p = restore_checkpoint(last, model, opt, rng, identity)
        else:
            p = dict(step=0, history=[examine(model, data, 0)], best_state=None,
                     best_step=None, best_loss=None, training_seconds=0.0)
        print(f"{arm} seed {seed} starting at step {p['step']}", flush=True)
        while p["step"] < until:
            step = p["step"]
            start = time.monotonic()
            try:
                loss, norm = previous.update(model, opt, rng, data["train"], step, recipe, schedule)
            except FloatingPointError as exc:
                base.write_json(folder / "numerical_failure.json",
                                dict(**identity, step=step, error=str(exc)))
                raise
            runtime.synchronize()
            p["training_seconds"] += time.monotonic() - start
            p["step"] = step + 1
            if p["step"] % schedule["eval_every"] == 0:
                entry = examine(model, data, p["step"])
                entry.update(lr=opt.param_groups[0]["lr"], last_minibatch_loss=loss,
                             gradient_norm_before_clip=norm)
                p["history"].append(entry)
                val = entry["validation_loss"]
                if not math.isfinite(val):
                    base.write_json(folder / "numerical_failure.json",
                                    dict(**identity, step=p["step"], error="Nonfinite validation loss"))
                    raise FloatingPointError("Nonfinite validation loss")
                if p["best_loss"] is None or val < p["best_loss"]:
                    p.update(best_loss=val, best_step=p["step"],
                             best_state=runtime.cpu_tree(model.state_dict()))
                print(f"{arm} seed {seed} step {p['step']} validation {val:.6f}", flush=True)
            if p["step"] % schedule["checkpoint_every"] == 0:
                save_checkpoint(last, model, opt, rng, p, identity)
                base.write_json(folder / "progress.json",
                                dict(**identity, step=p["step"], best_step=p["best_step"],
                                     best_validation_loss=p["best_loss"]))
        if p["step"] != schedule["max_steps"]:
            return
        endpoint = measure(model, data, p["history"][-1])
        best = p["best_step"]
        if best == p["step"]:
            selected = endpoint
        else:
            model.load_state_dict(p["best_state"])
            selected = measure(model, data, next(row for row in p["history"] if row["step"] == best))
        base.write_json(result, dict(**identity, identity=identity, completed=True,
                                     endpoint=endpoint, validation_selected=selected,
                                     history=p["history"], training_seconds=p["training_seconds"]))


def weights_digest(model):
    digest = hashlib.sha256()
    for p in model.parameters():
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def preflight(data):
    initial = None
    reports = {}
    for arm in ARMS:
        model, opt, rng, recipe = setup(arm, SEEDS[0])
        pair = (weights_digest(model), rng.get_state())
        if initial is None:
            initial = pair
        elif initial[0] != pair[0] or not torch.equal(initial[1], pair[1]):
            raise AssertionError("Arms did not start with identical weights and minibatch RNG")
        for step in range(2):
            previous.update(model, opt, rng, data["train"], step, recipe, SCHEDULE)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "last.pt"
            identity = dict(arm=arm, seed=SEEDS[0])
            save_checkpoint(checkpoint, model, opt, rng, {"step": 2}, identity)
            expected_loss, _ = previous.update(model, opt, rng, data["train"], 2, recipe, SCHEDULE)
            expected = runtime.cpu_tree(model.state_dict())
            model, opt, rng, _ = setup(arm, SEEDS[0])
            restore_checkpoint(checkpoint, model, opt, rng, identity)
            replay_loss, _ = previous.update(model, opt, rng, data["train"], 2, recipe, SCHEDULE)
            atol = 0.0 if runtime.DEVICE == "cpu" else 1e-5
            if not math.isclose(expected_loss, replay_loss, rel_tol=0, abs_tol=atol):
                raise AssertionError("Minibatch loss changed after replay")
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.detach().cpu(), expected[name], atol=atol, rtol=0)
        fits = base.spectrum(model.state_dict())
        if fits["unique_matrices"] != 18 or fits["successful_fits"] != 18:
            raise AssertionError("Expected 18 successful raw WeightWatcher fits during preflight")
        reports[arm] = dict(replayed_step=3, loss=replay_loss, fitted_matrices=fits["successful_fits"])
    print(json.dumps(dict(device=runtime.DEVICE, shared_initialization=True, checks=reports), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--device", choices=("cpu", "mps", "tpu"), default="cpu")
    parser.add_argument("--data", type=Path, default=HERE / "data" / "input.txt")
    parser.add_argument("--output", type=Path, help="Defaults to runs/<device> beside the notebook")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--until-step", type=int, default=SCHEDULE["max_steps"],
                        help="Stage training at a multiple of 250. Results are emitted at 20000 only.")
    args = parser.parse_args()
    if not 0 < args.until_step <= SCHEDULE["max_steps"] or args.until_step % SCHEDULE["checkpoint_every"]:
        parser.error("--until-step must be a multiple of 250 between 250 and 20000")
    runtime.DEVICE = args.device
    os.environ["SHORT_GPT_DEVICE"] = args.device
    runtime.initialize()
    data = base.load_data(args.data)
    if args.command == "preflight":
        preflight(data)
        return
    output = (args.output or HERE / "runs" / args.device).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "design.json"
    # Independent seed workers may create the same manifest concurrently.
    with (output / "design.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous.freeze(manifest, design())
    design_hash = base.sha(manifest)
    for i, seed in enumerate(SEEDS):
        if args.seed is not None and args.seed != seed:
            continue
        # Rotate ordering across seeds. Every arm gets the same training stream.
        for arm in ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]:
            if args.arm is None or args.arm == arm:
                trial(output, data, arm, seed, SCHEDULE, args.until_step, design_hash)


if __name__ == "__main__":
    main()
