"""Resolve a source-pinned study plan. This does not launch training."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve(repo: Path, study: dict, *, verify_modules: bool = True) -> dict:
    source = study["source"]
    raw = (repo / source["recipe"]).read_bytes()
    if git_blob_sha(raw) != source["recipe_git_blob_sha"]:
        raise RuntimeError("Baseline recipe changed: review and repin the study; do not silently reuse it.")
    if verify_modules:
        root = repo / "baseline/nanogpt_one_head/src/rg_nanogpt_one_head"
        for name, expected in source["modules"].items():
            if git_blob_sha((root / name).read_bytes()) != expected:
                raise RuntimeError(f"Baseline {name} changed: review and repin the study.")
    base = yaml.safe_load(raw)
    if study["optimizers"] != ["adamw", "muon"]:
        raise ValueError("Primary comparison is AdamW versus Muon, not silently substituted MuonClip.")
    for key in study["optimizers"]:
        if base["optimizer_profiles"][key]["family"] != key:
            raise ValueError(f"Optimizer family mismatch for {key}.")
    if study["seeds"] != base["training"]["seeds"]:
        raise ValueError("Paired seeds differ from the selected repository campaign.")
    if base["dataset"]["train_tokens"] != study["training"]["reference_tokens"]:
        raise ValueError("Reference token horizon does not match the inherited corpus.")
    for key in ("fix_fingers", "ERG", "randomize", "min_evals", "max_fingers", "require_raw_alpha"):
        if base["weightwatcher"][key] != study["monitoring"][key]:
            raise ValueError(f"WeightWatcher setting mismatch: {key}.")
    t = base["training"]
    tokens_per_step = t["batch_size"] * t["grad_accum_steps"] * base["model"]["block_size"]
    ref = study["training"]["reference_tokens"]
    epochs = study["training"]["target_reference_epochs"]
    steps_at = lambda epoch: math.ceil(epoch * ref / tokens_per_step)
    spectral = set(study["monitoring"]["spectral_early_steps"])
    interval = study["monitoring"]["spectral_every_reference_epochs"]
    spectral.update(steps_at(i * interval) for i in range(math.floor(epochs / interval) + 1))
    # Warmup end is resolved by the baseline trainer, not guessed by this planner.
    runs = []
    for condition in study["conditions"]:
        for seed in study["seeds"]:
            for optimizer in study["optimizers"]:
                cfg = copy.deepcopy(base)
                cfg["protocol"]["name"] = study["name"]
                cfg["protocol"]["description"] = "Memorization study derived from source-backed settings; not a qualification lock."
                cfg["training"]["seeds"] = [seed]
                cfg["training"]["target_epochs"] = epochs
                cfg["optimizer_profiles"] = {optimizer: cfg["optimizer_profiles"][optimizer]}
                # Historical display text says Muon was not an arm of the older campaign.
                cfg["optimizer_profiles"][optimizer]["display_name"] = {
                    "adamw": "AdamW", "muon": "Muon + auxiliary AdamW"
                }[optimizer]
                runs.append({"run_id": f"{condition}/{optimizer}/seed_{seed}",
                             "condition": condition, "optimizer": optimizer, "seed": seed,
                             "baseline_config": cfg})
    result = {
        "study": study, "tokens_per_update": tokens_per_step,
        "schedule_steps": steps_at(1), "total_steps": steps_at(epochs),
        "actual_target_tokens_per_run": steps_at(epochs) * tokens_per_step,
        "spectral_steps_before_warmup_union": sorted(s for s in spectral | {steps_at(epochs)} if s <= steps_at(epochs)),
        "full_exposure_steps": [steps_at(e) for e in study["monitoring"]["full_exposure_reference_epochs"]],
        "prefix_sweep_steps": [steps_at(e) for e in study["monitoring"]["prefix_sweep_reference_epochs"]],
        "runs": runs,
        "execution_status": "plan_only_training_adapter_not_implemented",
    }
    result["fingerprint"] = fingerprint(result)
    return result


def write_plan(plan: dict, out: Path) -> None:
    # Refuse reuse instead of silently overwriting another experiment.
    out.mkdir(parents=True, exist_ok=False)
    (out / "resolved_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for run in plan["runs"]:
        path = out / "configs" / (run["run_id"].replace("/", "__") + ".yaml")
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(run["baseline_config"], sort_keys=False))
    print(f"Wrote {len(plan['runs'])} planned runs to {out}. No training was started.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_plan(resolve(args.repo, json.loads((HERE / "study.json").read_text())), args.out)
