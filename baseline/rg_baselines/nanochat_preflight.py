"""One-step device preflight for the pinned nanochat model and optimizer.

This is intentionally independent of the dataset/tokenizer pipeline. It builds a
tiny model from the pinned upstream source, initializes it exactly through
``GPT.init_weights()``, runs one forward/backward pass, and executes the native
combined Muon/AdamW optimizer step under the same compile policy used by the
selected device. It catches source-patch drift and unsupported optimizer
operations before a long data preparation or training run begins.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .nanochat_portable import (
    DISABLE_COMPILE_ENV,
    MPS_FALLBACK_ENV,
    compile_enabled_for_device,
    ensure_checkout,
)
from .nanochat_reference import detect_device_type

_MARKER = "NANOCHAT_PREFLIGHT_JSON="


def run_device_preflight(
    checkout_dir: str | Path,
    *,
    device_type: str = "auto",
) -> dict[str, Any]:
    """Run one real pinned-model/optimizer step in a fresh Python process."""

    resolved = (
        detect_device_type()
        if str(device_type).lower() == "auto"
        else str(device_type).lower()
    )
    if resolved not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device_type: {resolved!r}")
    checkout = ensure_checkout(Path(checkout_dir))

    code = r'''
import json
import torch
from nanochat.gpt import GPT, GPTConfig

torch.manual_seed(1234)
device = torch.device(__DEVICE__)
config = GPTConfig(
    sequence_len=16,
    vocab_size=64,
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    n_embd=64,
    window_pattern="L",
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
optimizer = model.setup_optimizer(
    unembedding_lr=0.008,
    embedding_lr=0.30,
    matrix_lr=0.020,
    weight_decay=0.01,
    scalar_lr=0.50,
)
x = torch.randint(0, config.vocab_size, (2, config.sequence_len), device=device)
y = torch.randint(0, config.vocab_size, (2, config.sequence_len), device=device)
loss = model(x, y)
if not torch.isfinite(loss):
    raise RuntimeError(f"non-finite preflight loss: {loss}")
loss.backward()
optimizer.step()
if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
    raise RuntimeError("non-finite parameter after pinned optimizer step")
print("NANOCHAT_PREFLIGHT_JSON=" + json.dumps({
    "device": str(device),
    "loss": float(loss.detach().cpu()),
    "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    "optimizer_groups": int(len(optimizer.param_groups)),
}))
'''.replace("__DEVICE__", repr(resolved))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout) + os.pathsep + env.get("PYTHONPATH", "")
    env["NANOCHAT_DTYPE"] = "float32"
    env[DISABLE_COMPILE_ENV] = (
        "0" if compile_enabled_for_device(resolved) else "1"
    )
    if resolved == "mps":
        env[MPS_FALLBACK_ENV] = "1"
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=checkout,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "pinned nanochat device preflight failed:\n" + process.stdout
        )
    marker_lines = [
        line for line in process.stdout.splitlines() if line.startswith(_MARKER)
    ]
    if len(marker_lines) != 1:
        raise RuntimeError(
            "pinned nanochat preflight did not emit exactly one result marker:\n"
            + process.stdout
        )
    result = json.loads(marker_lines[0][len(_MARKER) :])
    result.update(
        {
            "compile_enabled": compile_enabled_for_device(resolved),
            "checkout": str(checkout),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one pinned nanochat model/optimizer device step"
    )
    parser.add_argument("--checkout", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
    )
    args = parser.parse_args()
    result = run_device_preflight(args.checkout, device_type=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
