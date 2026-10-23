"""Opt-in fixed-shape canary evaluation; historical evaluator is untouched.

The training batch, injection schedule, model and optimizer are unchanged.
Only the full-sweep worker installs this evaluator, and its batch size/version
are included in the run fingerprint. All output metrics retain their original
per-canary definitions and CSV names.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch
import torch.nn.functional as F

EVALUATOR_VERSION = "fixed-shape-batched-canaries-v1"
FIELDS = ("step", "epoch", "id", "dose", "nll", "teacher_accuracy",
          "token_accuracy", "exact_match")


def _mark(device: torch.device) -> None:
    if device.type == "xla":
        from .runtime import mark_step
        mark_step(device)


@torch.no_grad()
def score_batch(model, tokens: torch.Tensor, *, prefix: int, suffix: int,
                device: torch.device) -> torch.Tensor:
    """Return [canaries, 4] CPU metrics, using one fixed decode shape.

    Metrics: mean suffix NLL, teacher-forced token accuracy, greedy token
    accuracy, and exact greedy continuation recall. The padded future cannot
    affect a selected earlier position under the model's causal attention.
    """
    if tokens.ndim != 2 or prefix < 1 or suffix < 1:
        raise ValueError("tokens must be [batch, sequence]; prefix/suffix positive")
    if prefix + suffix > tokens.shape[1] or prefix + suffix > model.cfg.block_size:
        raise ValueError("canary continuation exceeds available tokens/context")
    device = torch.device(device)
    sequence = tokens[:, :prefix + suffix].to(device=device, dtype=torch.long)
    targets = sequence[:, prefix:prefix + suffix]
    hidden = model.hidden_states(sequence[:, :-1])
    logits = model.lm_head(hidden[:, prefix - 1:prefix - 1 + suffix])
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                          targets.reshape(-1), reduction="none").view(-1, suffix).mean(1)
    teacher = logits.argmax(-1).eq(targets).float().mean(1)
    # Execute the fixed teacher graph before constructing the decode graph.
    _mark(device)
    del hidden, logits
    buffer = torch.cat((sequence[:, :prefix], torch.zeros_like(targets)), dim=1)
    position = torch.full((sequence.shape[0], 1), prefix - 1,
                          dtype=torch.long, device=device)
    for _ in range(suffix):
        states = model.hidden_states(buffer)
        selected = states.gather(1, position.unsqueeze(-1).expand(-1, 1, states.shape[-1]))
        next_token = model.lm_head(selected).argmax(-1)
        position = position + 1
        buffer = buffer.scatter(1, position, next_token)
        # Tensor position, fixed buffer shape: no growing-prefix graph variants.
        _mark(device)
    hit = buffer[:, prefix:prefix + suffix].eq(targets)
    metrics = torch.stack((nll, teacher, hit.float().mean(1),
                           hit.all(1).float()), dim=1).detach().cpu()
    if not bool(torch.isfinite(metrics).all()):
        raise FloatingPointError("non-finite batched canary metric")
    return metrics


def write_rows(path: Path, rows: list[dict]) -> None:
    """Atomically replace a replayed suffix; the worker archives pre-resume data.

    Never combine old post-checkpoint canary observations with replayed states.
    Historical runs do not call this helper.
    """
    if not rows:
        raise ValueError("empty canary evaluation")
    step = int(rows[0]["step"])
    previous = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"unexpected canary CSV schema: {path}")
            for row in reader:
                if None in row or any(row.get(k) is None for k in FIELDS):
                    raise ValueError(f"incomplete canary CSV row: {path}")
                if int(row["step"]) < step:
                    previous.append(row)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(previous)
    writer.writerows(rows)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate_batched(experiment, model, *, device, step: int, epoch: float,
                     batch_size: int = 40) -> dict[str, float]:
    if batch_size < 1 or not experiment.canaries:
        raise ValueError("batch_size and canary inventory must be positive")
    was_training = model.training
    rows = []
    started = time.perf_counter()
    model.eval()
    try:
        for start in range(0, len(experiment.canaries), batch_size):
            group = experiment.canaries[start:start + batch_size]
            sequences = torch.stack([c["tokens"] for c in group])
            if len(group) < batch_size:
                # Fixed evaluation shape even for a final short batch.
                sequences = torch.cat((sequences, sequences[-1:].expand(
                    batch_size - len(group), -1)), dim=0)
            scores = score_batch(model, sequences, prefix=experiment.prefix,
                                 suffix=experiment.suffix, device=device)[:len(group)]
            for canary, values in zip(group, scores.tolist()):
                rows.append(dict(zip(FIELDS, (int(step), float(epoch), canary["id"],
                    int(canary["dose"]), *values))))
    finally:
        model.train(was_training)
    write_rows(Path(experiment.csv_path), rows)
    timing = {"step": int(step), "epoch": float(epoch), "evaluator": EVALUATOR_VERSION,
              "batch_size": batch_size, "canaries": len(rows),
              "seconds": time.perf_counter() - started,
              "forward_calls": math.ceil(len(rows) / batch_size) * (experiment.suffix + 1)}
    with (Path(experiment.run_dir) / "canary_evaluation_timing.jsonl").open("a") as handle:
        handle.write(json.dumps(timing) + "\n")
    exposed = [r for r in rows if r["dose"] > 0]
    zero = [r for r in rows if r["dose"] == 0]
    return {"exact_match": sum(r["exact_match"] for r in exposed) / max(1, len(exposed)),
            "token_accuracy": sum(r["token_accuracy"] for r in exposed) / max(1, len(exposed)),
            "zero_exact_match": sum(r["exact_match"] for r in zero) / max(1, len(zero))}


def install_batched_canary_evaluation(batch_size: int = 40) -> None:
    """Opt-in, process-local install. Does not alter canary construction/injection."""
    from .random_canaries import RandomCanaryExperiment
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    def evaluate(self, model, *, device, step, epoch):
        return evaluate_batched(self, model, device=device, step=step,
                                epoch=epoch, batch_size=batch_size)
    RandomCanaryExperiment.evaluate = evaluate
