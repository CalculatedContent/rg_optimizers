from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .runtime import is_xla_device, mark_step, synchronize, tree_to_cpu


@dataclass(frozen=True)
class BleuProbe:
    prompts: torch.Tensor
    references: torch.Tensor
    prompt_tokens: int
    continuation_tokens: int


def random_batch(
    data: np.memmap,
    *,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("data split is too short for the configured block size")
    starts = torch.randint(
        len(data) - block_size - 1,
        (int(batch_size),),
        generator=generator,
    ).tolist()
    x = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[start : start + block_size],
                    dtype=np.int64,
                )
            )
            for start in starts
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[start + 1 : start + 1 + block_size],
                    dtype=np.int64,
                )
            )
            for start in starts
        ]
    )
    return x, y


def fixed_probe(
    data: np.memmap,
    *,
    batch_size: int,
    block_size: int,
    n_batches: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return [
        random_batch(
            data,
            batch_size=int(batch_size),
            block_size=int(block_size),
            generator=generator,
        )
        for _ in range(int(n_batches))
    ]


def fixed_bleu_probe(
    data: np.memmap,
    *,
    examples: int,
    prompt_tokens: int,
    continuation_tokens: int,
    seed: int,
) -> BleuProbe:
    total = int(prompt_tokens) + int(continuation_tokens)
    if len(data) <= total + 1:
        raise ValueError("test split is too short for BLEU continuation probes")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    starts = torch.randint(
        len(data) - total - 1,
        (int(examples),),
        generator=generator,
    ).tolist()
    prompts = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[start : start + prompt_tokens],
                    dtype=np.int64,
                )
            )
            for start in starts
        ]
    )
    references = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[
                        start + prompt_tokens :
                        start + prompt_tokens + continuation_tokens
                    ],
                    dtype=np.int64,
                )
            )
            for start in starts
        ]
    )
    return BleuProbe(
        prompts=prompts,
        references=references,
        prompt_tokens=int(prompt_tokens),
        continuation_tokens=int(continuation_tokens),
    )


@torch.inference_mode()
def evaluate_probe(
    model: nn.Module,
    probe: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    if not is_xla_device(device):
        # Preserve the historical CPU/CUDA/MPS metric path exactly. This keeps
        # validation-checkpoint selection comparable to existing runs.
        losses: list[float] = []
        correct = 0
        top5_correct = 0
        total = 0
        for x_cpu, y_cpu in probe:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            logits, loss = model(x, y)
            if loss is None:
                raise RuntimeError(
                    "evaluation forward pass did not return loss"
                )
            losses.append(float(loss.detach().cpu()))
            correct += int(
                (logits.argmax(dim=-1) == y).sum().detach().cpu()
            )
            top_k = min(5, int(logits.shape[-1]))
            top5_correct += int(
                (
                    logits.topk(top_k, dim=-1).indices
                    == y.unsqueeze(-1)
                )
                .any(dim=-1)
                .sum()
                .detach()
                .cpu()
            )
            total += int(y.numel())
        model.train(was_training)
        mean_loss = float(np.mean(losses))
        return {
            "loss": mean_loss,
            "perplexity": float(math.exp(mean_loss)),
            "bits_per_token": float(mean_loss / math.log(2.0)),
            "accuracy": correct / max(1, total),
            "top5_accuracy": top5_correct / max(1, total),
        }

    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    top5_correct = torch.zeros((), dtype=torch.int64, device=device)
    total = 0
    batches = 0
    for x_cpu, y_cpu in probe:
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("evaluation forward pass did not return loss")
        loss_sum = loss_sum + loss.detach().float()
        correct = correct + (logits.argmax(dim=-1) == y).sum()
        top_k = min(5, int(logits.shape[-1]))
        top5_correct = top5_correct + (
            logits.topk(top_k, dim=-1).indices == y.unsqueeze(-1)
        ).any(dim=-1).sum()
        total += int(y.numel())
        batches += 1
        # XLA is lazy. Execute each fixed-shape batch without transferring
        # scalars to the host, allowing the compiled evaluation graph to be reused.
        mark_step(device)
    if batches == 0:
        raise RuntimeError("evaluation probe is empty")
    synchronize(device)
    mean_loss = float((loss_sum / batches).detach().cpu())
    correct_value = int(correct.detach().cpu())
    top5_correct_value = int(top5_correct.detach().cpu())
    model.train(was_training)
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(mean_loss)),
        "bits_per_token": float(mean_loss / math.log(2.0)),
        "accuracy": correct_value / max(1, total),
        "top5_accuracy": top5_correct_value / max(1, total),
    }


def _cpu_bleu_model(model) -> nn.Module:
    """Build a CPU copy for BLEU when the live model is on TPU/XLA.

    Greedy decoding changes sequence length at every token and would otherwise
    trigger a series of XLA compilations. BLEU is a post-training secondary
    audit, so the small CPU copy avoids that accelerator-specific overhead
    without affecting training, checkpoint selection, or WeightWatcher
    measurements.
    """

    synchronize(model.lm_head.weight.device)
    cpu_model = type(model)(model.cfg).cpu()
    cpu_model.load_state_dict(tree_to_cpu(model.state_dict()))
    return cpu_model


@torch.inference_mode()
def evaluate_bleu(
    model,
    probe: BleuProbe,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Greedy fixed-continuation BLEU diagnostic on held-out segments.

    This is not a translation benchmark. It measures exact lexical overlap
    between deterministic model continuations and the held-out continuation.
    It is evaluated only after training for the final and validation-selected
    checkpoints. On TPU/XLA, decoding is intentionally performed on a CPU
    snapshot because changing sequence lengths are a poor fit for repeated XLA
    compilation.
    """

    try:
        import tiktoken
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise RuntimeError(
            "BLEU evaluation requires tiktoken and sacrebleu; install the "
            "experiment dependencies with `python -m pip install -e .`"
        ) from exc

    evaluation_model = model
    evaluation_device = device
    if is_xla_device(device):
        evaluation_model = _cpu_bleu_model(model)
        evaluation_device = torch.device("cpu")

    was_training = evaluation_model.training
    evaluation_model.eval()
    encoder = tiktoken.get_encoding("gpt2")
    hypotheses: list[str] = []
    references: list[str] = []
    continuation_correct = 0
    continuation_total = 0
    continuation_exact = 0
    for start in range(0, len(probe.prompts), int(batch_size)):
        prompts = probe.prompts[
            start : start + int(batch_size)
        ].to(evaluation_device)
        generated = evaluation_model.generate_greedy(
            prompts,
            probe.continuation_tokens,
        )
        continuation = generated[
            :,
            -probe.continuation_tokens :,
        ].detach().cpu()
        reference_batch = probe.references[
            start : start + int(batch_size)
        ]
        continuation_correct += int(
            (continuation == reference_batch).sum().item()
        )
        continuation_total += int(reference_batch.numel())
        continuation_exact += int(
            (continuation == reference_batch).all(dim=-1).sum().item()
        )
        for predicted_tokens, reference_tokens in zip(
            continuation,
            reference_batch,
            strict=True,
        ):
            hypotheses.append(encoder.decode(predicted_tokens.tolist()))
            references.append(encoder.decode(reference_tokens.tolist()))
    evaluation_model.train(was_training)

    bleu = BLEU(tokenize="13a", effective_order=True)
    score = bleu.corpus_score(hypotheses, [references])
    return {
        "bleu": float(score.score),
        "bleu_examples": float(len(hypotheses)),
        "bleu_sys_len": float(score.sys_len),
        "bleu_ref_len": float(score.ref_len),
        "continuation_token_accuracy": (
            continuation_correct / max(1, continuation_total)
        ),
        "continuation_exact_match": (
            continuation_exact / max(1, len(hypotheses))
        ),
    }
