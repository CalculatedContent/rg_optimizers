"""Plan and audit SGDM searches; never allocate hardware or launch training.

All promotions are fresh runs with a newly hashed training horizon. The selected
recipe is the best completed member of this search, not an established optimum.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from config import (BENCHMARK_VALIDATION_TOKENS, FULL_BUDGET_TOKENS,
                    TrainingConfig, canonical_json, scientific_hash, smoke_config)


HERE = Path(__file__).resolve().parent
COMMON_TOKEN_QUANTUM = 512 * 1_024
DEFAULT_STAGES = (
    {"stage_id": "screen", "budget_tokens": 134_217_728, "promote": 4,
     "eval_interval_tokens": 33_554_432, "spectral_interval_tokens": 67_108_864,
     "checkpoint_interval_tokens": 33_554_432, "validation_kind": "tuning"},
    {"stage_id": "refine", "budget_tokens": 1_073_741_824, "promote": 1,
     "eval_interval_tokens": 131_072_000, "spectral_interval_tokens": 131_072_000,
     "checkpoint_interval_tokens": 131_072_000, "validation_kind": "tuning"},
    {"stage_id": "full", "budget_tokens": FULL_BUDGET_TOKENS, "promote": 0,
     "eval_interval_tokens": 131_072_000, "spectral_interval_tokens": 131_072_000,
     "checkpoint_interval_tokens": 131_072_000, "validation_kind": "benchmark"},
)


def _slug(value: float) -> str:
    return format(value, ".8g").replace(".", "p")


def candidate_configs() -> list[TrainingConfig]:
    """Ten proposed small-batch configurations, plus a historical HP control."""
    result = []
    for batch in (16, 64):
        for lr in (0.01, 0.03, 0.1, 0.3, 1.0):
            candidate_id = f"sgdm-b{batch:04d}-lr{_slug(lr)}-m0p98"
            result.append(TrainingConfig(candidate_id=candidate_id,
                                         run_name=f"{candidate_id}-candidate-seed42",
                                         global_batch_sequences=batch,
                                         microbatch_sequences=16, peak_lr=lr))
    control_id = "control-b0512-lr0p3-m0p9-wd3em5"
    result.append(TrainingConfig(candidate_id=control_id,
                                 run_name=f"{control_id}-candidate-seed42",
                                 global_batch_sequences=512, microbatch_sequences=16,
                                 peak_lr=0.3, momentum=0.9, weight_decay=3e-5,
                                 warmup_fraction=700 / 19_560,
                                 proposal_status="historical_hyperparameter_control"))
    return result


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    body = dict(value)
    body.pop("manifest_sha256", None)
    document = {**body, "manifest_sha256": scientific_hash(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != document:
            raise FileExistsError(f"Refusing to replace immutable manifest: {path}")
        return path
    with path.open("x") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def read_immutable_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    body = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if document.get("manifest_sha256") != scientific_hash(body):
        raise ValueError(f"Manifest hash mismatch: {path}")
    return document


def expected_snapshot_steps(config: TrainingConfig) -> list[int]:
    interval = config.spectral_interval_tokens // config.tokens_per_step
    return sorted({0, config.max_steps, *range(interval, config.max_steps, interval)})


def comparison_contract(config: TrainingConfig) -> dict[str, Any]:
    """Fields that must be shared when ranking different optimizer candidates."""
    names = ("mode", "architecture", "model_source_sha256", "dataset", "tokenizer",
             "n_layer", "n_head", "n_embd", "vocab_size", "context_length",
             "seed", "data_seed", "budget_tokens", "validation_kind",
             "validation_offset", "validation_tokens")
    return {name: getattr(config, name) for name in names}


def stage_config(config: TrainingConfig, spec: Mapping[str, Any]) -> TrainingConfig:
    return config.at_budget(spec["budget_tokens"], stage_id=spec["stage_id"],
                            validation_kind=spec["validation_kind"],
                            eval_interval_tokens=spec["eval_interval_tokens"],
                            spectral_interval_tokens=spec["spectral_interval_tokens"],
                            checkpoint_interval_tokens=spec["checkpoint_interval_tokens"],
                            stop_at_target=False)


def write_stage(output: Path, candidates: Iterable[TrainingConfig], stages: list[dict[str, Any]],
                *, parent_selection_sha256: str | None = None) -> Path:
    if not stages:
        raise ValueError("No stage remains to plan")
    spec = stages[0]
    entries = []
    contracts = []
    for base in candidates:
        config = stage_config(base, spec)
        path = Path("configs") / f"{config.candidate_id}.json"
        config.write_json(output / path)
        entries.append({"candidate_id": config.candidate_id, "config": str(path),
                        "config_sha256": config.config_sha256})
        contracts.append(comparison_contract(config))
    if not entries or any(value != contracts[0] for value in contracts[1:]):
        raise ValueError("A stage must have candidates with one shared comparison contract")
    manifest = {
        "schema_version": 1, "stage": spec, "remaining_stages": stages[1:],
        "candidates": entries, "comparison_contract": contracts[0],
        "parent_selection_sha256": parent_selection_sha256,
        "fresh_initialization_required": True,
        "resume_rule": "Identical configuration hash only; never resume a promotion.",
        "selection_metric": "final tuning-window validation NLL at equal processed-token budget",
        "claim": "Proposed search; no optimum or target-reaching recipe has been established.",
    }
    return write_immutable_json(output / "stage.json", manifest)


def make_plan(output: Path, *, seed: int = 42) -> Path:
    candidates = [replace(c, seed=seed, run_name=f"{c.candidate_id}-candidate-seed{seed}")
                  for c in candidate_configs()]
    manifest = write_immutable_json(output / "plan.json", {
        "schema_version": 1, "seed": seed, "data_seed": 42,
        "stages": list(DEFAULT_STAGES),
        "candidate_templates": [{"candidate_id": c.candidate_id, "config_sha256": c.config_sha256,
                                 "config": c.as_dict()} for c in candidates],
        "full_budget_tokens": FULL_BUDGET_TOKENS,
        "single_seed_pilot": True,
        "confirmation_seeds_proposed_not_run": [42, 43, 44],
        "comparison": "Equal training tokens and identical held-out tuning window, fresh initialization at each stage.",
        "limitation": "Budget-dependent rankings can change; promotion may discard the eventual best full-budget candidate.",
    })
    write_stage(output / "screen", candidates, [dict(s) for s in DEFAULT_STAGES])
    return manifest


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def check_result(config: TrainingConfig, result: Mapping[str, Any],
                 tracking: Mapping[str, Any] | None) -> list[str]:
    """Reject wrong-budget/holdout endpoints and incomplete scientific artifacts."""
    errors = []
    successful_statuses = {"budget_complete_target_met", "budget_complete_target_not_met",
                           "schedule_complete_target_met", "schedule_complete_target_not_met", "completed"}
    if result.get("status") not in successful_statuses or result.get("terminal") is not True:
        errors.append("training status is not a successfully completed horizon")
    for name, expected in (("config_sha256", config.config_sha256), ("candidate_id", config.candidate_id),
                           ("stage_id", config.stage_id), ("seed", config.seed)):
        if result.get(name) != expected:
            errors.append(f"{name} mismatch")
    if result.get("full_training_recipe_completed") is not True:
        errors.append("training horizon is incomplete")
    if result.get("tokens_seen") != config.budget_tokens or result.get("step") != config.max_steps:
        errors.append("training endpoint does not equal the planned token budget")
    validation = result.get("validation")
    if not isinstance(validation, Mapping):
        errors.append("missing final validation")
        validation = {}
    if validation.get("validation_kind") != config.validation_kind:
        errors.append("validation split kind mismatch")
    if validation.get("validation_offset") != config.validation_offset:
        errors.append("validation offset mismatch")
    if validation.get("evaluation_tokens") != config.validation_tokens:
        errors.append("partial or mismatched validation token count")
    if validation.get("tokens_seen") != config.budget_tokens or validation.get("step") != config.max_steps:
        errors.append("validation is not from the final planned checkpoint")
    nll = validation.get("val_nll")
    if not _number(nll) or nll < 0:
        errors.append("validation NLL is nonfinite, missing, or negative")
    if tracking is None:
        errors.append("missing TRACKING_STATUS.json")
    else:
        if tracking.get("complete") is not True or tracking.get("training_done") is not True:
            errors.append("spectral worker has not confirmed training and measurement completion")
        if tracking.get("failed") != 0 or tracking.get("pending") != 0:
            errors.append("spectral measurements have failed or remain pending")
        expected = expected_snapshot_steps(config)
        completed_steps = tracking.get("completed_steps")
        valid_steps = (isinstance(completed_steps, list)
                       and all(not isinstance(step, bool) and isinstance(step, int)
                               and 0 <= step <= config.max_steps for step in completed_steps))
        if not valid_steps or not set(expected).issubset(set(completed_steps)):
            errors.append("completed spectral snapshots are missing scheduled steps or contain invalid steps")
        elif tracking.get("completed") != len(set(completed_steps)):
            errors.append("completed spectral count differs from its distinct committed steps")
        declared = result.get("expected_snapshot_steps")
        if declared is not None:
            valid_declared = (isinstance(declared, list)
                              and all(not isinstance(step, bool) and isinstance(step, int)
                                      and 0 <= step <= config.max_steps for step in declared))
            if not valid_declared or not valid_steps or set(declared) != set(completed_steps):
                errors.append("not every trainer-declared snapshot has been committed")
        if tracking.get("missing_expected_steps"):
            errors.append("spectral worker reports missing expected snapshots")
        if tracking.get("config_sha256") not in (None, config.config_sha256):
            errors.append("tracking configuration hash mismatch")
    return errors


def rank_results(configs: Iterable[TrainingConfig], records: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any] | None]],
                 *, require_all_terminal: bool = True) -> dict[str, Any]:
    configs = list(configs)
    if not configs:
        raise ValueError("Cannot rank an empty candidate set")
    contract = comparison_contract(configs[0])
    if contract["mode"] != "production":
        raise ValueError("Synthetic smoke results cannot select a production candidate")
    if contract["validation_kind"] != "tuning":
        raise ValueError("Hyperparameters must be selected using the tuning window, not the official benchmark")
    if any(comparison_contract(config) != contract for config in configs):
        raise ValueError("Cannot rank different token budgets, seeds, datasets, models, or validation windows")
    eligible, excluded, blocked = [], [], []
    for config in configs:
        record = records.get(config.candidate_id)
        if record is None:
            blocked.append({"candidate_id": config.candidate_id, "reasons": ["missing training result"]})
            continue
        result, tracking = record
        if (all(result.get(name) == getattr(config, name) for name in
                ("config_sha256", "candidate_id", "stage_id", "seed"))
                and result.get("status") in {"diverged", "failed_nonfinite"}
                and result.get("terminal") is True):
            excluded.append({"candidate_id": config.candidate_id, "reasons": ["explicit terminal numerical divergence"],
                             "failure_reason": result.get("failure_reason") or result.get("error")})
            continue
        errors = check_result(config, result, tracking)
        if errors:
            blocked.append({"candidate_id": config.candidate_id, "reasons": errors})
        else:
            eligible.append({"candidate_id": config.candidate_id, "config_sha256": config.config_sha256,
                             "val_nll": float(result["validation"]["val_nll"]),
                             "tokens_seen": config.budget_tokens, "seed": config.seed})
    if blocked and require_all_terminal:
        raise ValueError("Selection blocked by incomplete or mismatched evidence: " + canonical_json(blocked))
    if not eligible:
        raise ValueError("No completed, fully measured candidate is eligible")
    eligible.sort(key=lambda row: (row["val_nll"], row["candidate_id"]))
    return {"ranking": eligible, "excluded_divergence": excluded, "incomplete_or_invalid": blocked,
            "comparison_contract": contract, "selection_scope": "best completed candidates in the declared searched set",
            "single_seed_pilot": True, "optimum_claim": False}


def _load_stage(stage_dir: Path) -> tuple[dict[str, Any], list[TrainingConfig]]:
    manifest = read_immutable_json(stage_dir / "stage.json")
    configs = []
    for entry in manifest["candidates"]:
        config = TrainingConfig.from_json(stage_dir / entry["config"])
        if config.config_sha256 != entry["config_sha256"] or config.candidate_id != entry["candidate_id"]:
            raise ValueError("Stage manifest and configuration identity disagree")
        configs.append(config)
    return manifest, configs


def _read_records(results_dir: Path, configs: Iterable[TrainingConfig]) -> dict[str, tuple[dict[str, Any], dict[str, Any] | None]]:
    expected = {config.config_sha256: config.candidate_id for config in configs}
    result_paths = sorted(results_dir.rglob("TRAINING_STATUS.json"))
    # A run may publish both aliases. Read TRAINING_STATUS preferentially.
    result_paths += [p for p in sorted(results_dir.rglob("status.json"))
                     if not (p.parent / "TRAINING_STATUS.json").exists()]
    records = {}
    for path in result_paths:
        result = json.loads(path.read_text())
        candidate = expected.get(result.get("config_sha256"))
        if candidate is None:
            continue
        if candidate in records:
            raise ValueError(f"Multiple results for one immutable candidate; specify an unambiguous results directory: {candidate}")
        tracking_path = path.parent / "TRACKING_STATUS.json"
        tracking = json.loads(tracking_path.read_text()) if tracking_path.exists() else None
        records[candidate] = (result, tracking)
    return records


def select_stage(stage_dir: Path, results_dir: Path, output: Path) -> Path:
    manifest, configs = _load_stage(stage_dir)
    if not manifest["remaining_stages"]:
        raise ValueError("The full stage is confirmation, not another hyperparameter selection")
    ranking = rank_results(configs, _read_records(results_dir, configs))
    count = min(manifest["stage"]["promote"], len(ranking["ranking"]))
    chosen = {row["candidate_id"] for row in ranking["ranking"][:count]}
    selected = [config for config in configs if config.candidate_id in chosen]
    selection = {**ranking, "schema_version": 1,
                 "source_stage_sha256": manifest["manifest_sha256"],
                 "selected_candidate_ids": [row["candidate_id"] for row in ranking["ranking"][:count]],
                 "promotion": "Fresh initialization; no training state or learning-rate history is continued."}
    selection_path = write_immutable_json(output / "selection.json", selection)
    selection_sha = read_immutable_json(selection_path)["manifest_sha256"]
    write_stage(output, selected, manifest["remaining_stages"], parent_selection_sha256=selection_sha)
    if manifest["remaining_stages"][0]["stage_id"] == "full":
        if len(selected) != 1:
            raise ValueError("Full confirmation must receive one tuning-selected candidate")
        winner = stage_config(selected[0], manifest["remaining_stages"][0])
        winner.write_json(output / "winner.json")
    return selection_path


def confirmation_plan(config_path: Path, output: Path, seeds: list[int]) -> Path:
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Confirmation requires at least three distinct, explicitly recorded seeds")
    base = TrainingConfig.from_json(config_path)
    if base.mode != "production" or base.validation_kind != "benchmark":
        raise ValueError("Confirmation requires an explicit full-production candidate")
    entries = []
    for seed in seeds:
        config = replace(base, seed=seed, stage_id="confirmation",
                         run_name=f"{base.candidate_id}-confirmation-seed{seed}")
        path = Path("configs") / f"seed{seed}.json"
        config.write_json(output / path)
        entries.append({"seed": seed, "config": str(path), "config_sha256": config.config_sha256})
    return write_immutable_json(output / "confirmation.json", {
        "schema_version": 1, "source_config_sha256": base.config_sha256,
        "fresh_initialization_required": True, "seeds": seeds,
        "configurations": entries, "status": "planned_not_run",
        "reporting_rule": "Report every seed and the mean/spread; do not select the best seed.",
    })


def replan_longer(config_path: Path, output: Path, budget_tokens: int, schedule: str) -> Path:
    base = TrainingConfig.from_json(config_path)
    if budget_tokens <= base.budget_tokens or budget_tokens % COMMON_TOKEN_QUANTUM:
        raise ValueError("A longer plan requires a larger budget divisible by 524288 tokens")
    config = base.at_budget(budget_tokens, stage_id="longer-fresh", validation_kind="benchmark",
                            schedule=schedule, warmup_fraction=0.1, stable_fraction=0.8,
                            stop_at_target=False, proposal_status="proposed_unverified")
    config.write_json(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Create a single-seed staged search; launches no jobs")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seed", type=int, default=42)
    select = commands.add_parser("select", help="Audit completed stage results and export fresh promotions")
    select.add_argument("--stage-dir", type=Path, required=True)
    select.add_argument("--results-dir", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    confirm = commands.add_parser("confirm", help="Plan independent seed confirmation; launches no jobs")
    confirm.add_argument("--config", type=Path, required=True)
    confirm.add_argument("--output", type=Path, required=True)
    confirm.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    longer = commands.add_parser("replan-longer", help="Create a larger fresh-initialization horizon, never a resume")
    longer.add_argument("--config", type=Path, required=True)
    longer.add_argument("--output", type=Path, required=True)
    longer.add_argument("--budget-tokens", type=int, required=True)
    longer.add_argument("--schedule", choices=("cosine", "wsd"), default="wsd")
    args = parser.parse_args()
    if args.command == "plan":
        path = make_plan(args.output, seed=args.seed)
    elif args.command == "select":
        path = select_stage(args.stage_dir, args.results_dir, args.output)
    elif args.command == "confirm":
        path = confirmation_plan(args.config, args.output, args.seeds)
    else:
        path = replan_longer(args.config, args.output, args.budget_tokens, args.schedule)
    print(json.dumps({"created": str(path), "launched_training": False}, allow_nan=False))


if __name__ == "__main__":
    main()
