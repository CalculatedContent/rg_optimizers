"""Run one unchanged parent continuation or one frozen fresh confirmation seed."""
import argparse
import json
from pathlib import Path

from .long_train import train_long
from .search import (augment_result, check_identity, digest, execution_record, exclusive,
                     load_context, read_json, retain_weights, validate_frozen_selection,
                     validate_result, write_json)


def run_one(root, phase, seed):
    root, plan, identity = load_context(root)
    if phase not in ("continuation", "confirmation"):
        raise ValueError("Unknown run phase")
    parent = None
    frozen_hash = None
    if phase == "continuation":
        if seed not in plan["continuation_seeds"]:
            raise ValueError("Unexpected continuation seed")
        config = plan["continuation_config"]
        parent = root / "parent" / f"seed_{seed}" / "checkpoint20k.pt"
        if not parent.is_file():
            raise FileNotFoundError(f"Pinned parent checkpoint is unavailable: {parent}")
        expected = plan["continuation"]["checkpoints"][str(seed)]["sha256"]
        if digest(parent) != expected:
            raise ValueError("Parent checkpoint bytes differ from the frozen plan")
        directory = root / "continuations" / f"seed_{seed}"
        start = int(plan["continuation"]["checkpoint_step"])
    else:
        if seed not in plan["confirmation_seeds"]:
            raise ValueError("Unexpected confirmation seed")
        frozen_path = root / "frozen_selection.json"
        frozen = read_json(frozen_path)
        validate_frozen_selection(frozen, plan, identity)
        config = frozen["selected"]
        frozen_hash = digest(frozen_path)
        directory = root / "runs" / f"seed_{seed}"
        start = 0
    with exclusive(directory, ".wrapper.lock"):
        result_path = directory / "result.json"
        record = read_json(result_path) if result_path.exists() else None
        if record is None or record.get("status") != "complete":
            execution_record(directory, identity, seed, phase)
            record = train_long(plan, config, root / "data", directory, seed,
                                phase=phase, parent_checkpoint=parent, resume=True)
        validate_result(record, plan, identity, config, seed, phase)
        record = augment_result(directory, record, identity,
                                frozen_selection_sha256=frozen_hash)
        required = {int(step) for step in plan["checkpoint_updates"]
                    if start <= int(step) <= int(plan["total_updates"])}
        required.add(int(plan["total_updates"]))
        retained = retain_weights(directory, record, require_full_steps=required)
        completion = {**identity, "seed": seed, "phase": phase, "status": "complete",
                      "completed_updates": record["completed_updates"],
                      "result_sha256": digest(result_path), "retained_files": retained,
                      "frozen_selection_sha256": frozen_hash,
                      "parent_checkpoint_sha256": record.get("parent_checkpoint_sha256")}
        existing = directory / "completion.json"
        if not existing.exists() or read_json(existing) != completion:
            write_json(existing, completion)
        print(f"Completed or reused {phase} seed={seed}, updates={record['completed_updates']}", flush=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", choices=("continuation", "confirmation"), required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    result = run_one(args.root, args.phase, args.seed)
    print(json.dumps({key: result[key] for key in ("seed", "phase", "status", "completed_updates")}))


if __name__ == "__main__":
    main()
