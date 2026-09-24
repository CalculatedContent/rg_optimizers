from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


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
        [torch.from_numpy(np.asarray(data[start : start + block_size], dtype=np.int64)) for start in starts]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                np.asarray(data[start + 1 : start + 1 + block_size], dtype=np.int64)
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
    starts = torch.randint(len(data) - total - 1, (int(examples),), generator=generator).tolist()
    prompts = torch.stack(
        [torch.from_numpy(np.asarray(data[start : start + prompt_tokens], dtype=np.int64)) for start in starts]
    )
    references = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[start + prompt_tokens : start + prompt_tokens + continuation_tokens],
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
    losses: list[float] = []
    correct = 0
    total = 0
    for x_cpu, y_cpu in probe:
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("evaluation forward pass did not return loss")
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(dim=-1) == y).sum().detach().cpu())
        total += int(y.numel())
    model.train(was_training)
    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(20.0, mean_loss))),
        "accuracy": correct / max(1, total),
    }


@torch.inference_mode()
def evaluate_bleu(
    model,
    probe: BleuProbe,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Greedy fixed-continuation BLEU diagnostic on preregistered test segments.

    This is not a translation benchmark. It measures exact lexical overlap
    between deterministic model continuations and the held-out continuation.
    """
    try:
        import tiktoken
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise RuntimeError(
            "BLEU evaluation requires tiktoken and sacrebleu; run scripts/setup_mac.sh"
        ) from exc

    was_training = model.training
    model.eval()
    encoder = tiktoken.get_encoding("gpt2")
    hypotheses: list[str] = []
    references: list[str] = []
    for start in range(0, len(probe.prompts), int(batch_size)):
        prompts = probe.prompts[start : start + int(batch_size)].to(device)
        generated = model.generate_greedy(prompts, probe.continuation_tokens)
        continuation = generated[:, -probe.continuation_tokens :].detach().cpu()
        reference_batch = probe.references[start : start + int(batch_size)]
        for predicted_tokens, reference_tokens in zip(continuation, reference_batch, strict=True):
            hypotheses.append(encoder.decode(predicted_tokens.tolist()))
            references.append(encoder.decode(reference_tokens.tolist()))
    model.train(was_training)

    bleu = BLEU(tokenize="13a", effective_order=True)
    score = bleu.corpus_score(hypotheses, [references])
    return {
        "bleu": float(score.score),
        "bleu_examples": float(len(hypotheses)),
        "bleu_sys_len": float(score.sys_len),
        "bleu_ref_len": float(score.ref_len),
    }
