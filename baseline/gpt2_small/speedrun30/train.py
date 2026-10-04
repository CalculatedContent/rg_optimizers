"""Bounded TPU port of the pinned llm.c GPT-2/FineWeb reference.

No WeightWatcher, per-tensor diagnostics, preflight, or checkpoint upload loop.
The unmodified upstream model lives in vendor/. Hardware adaptation is here.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request

import numpy as np
import torch
from vendor import llmc_train_gpt2 as reference

HERE = Path(__file__).resolve().parent
BATCH = 64                     # Global microbatch, 8 sequences per TPU chip
CONTEXT = 1024
TOTAL_BATCH = 524288           # Same tokens/update as the reference
ACCUM = TOTAL_BATCH // (BATCH * CONTEXT)
TOTAL_STEPS = 19560            # Published Oct 13 2024 reference log
WARMUP = 700
PEAK_LR = 0.0006
VAL_TOKENS = 10 * 2**20
SOURCE_COMMIT = "7ecd8906afe6ed7a2b2cdb731c042f26d525b820"


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def learning_rate(step):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    ratio = min(1., (step - WARMUP) / (TOTAL_STEPS - WARMUP))
    return PEAK_LR * 0.5 * (1 + math.cos(math.pi * ratio))


class DeadlineReached(Exception):
    pass


class FineWeb:
    """Download exact pretokenized benchmark shards lazily; never use Edu data."""
    def __init__(self, cache, deadline):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((HERE / "data_manifest.json").read_text())
        self.deadline = deadline
        self.receipts = {}

    def array(self, name):
        info = self.manifest["files"][name]
        path = self.cache / name
        receipt = path.with_suffix(".verified.json")
        trusted = False
        if path.exists() and receipt.exists():
            try:
                saved = json.loads(receipt.read_text())
                trusted = (saved == info and path.stat().st_size == info["size"])
            except (OSError, ValueError):
                pass
        if not trusted:
            url = ("https://huggingface.co/datasets/" + self.manifest["repo"] +
                   "/resolve/" + self.manifest["revision"] + "/" + name)
            print("Downloading benchmark shard: " + name, flush=True)
            temporary = path.with_suffix(".partial")
            digest = hashlib.sha256()
            if time.time() >= self.deadline:
                raise DeadlineReached()
            with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as out:
                while True:
                    if time.time() >= self.deadline:
                        raise DeadlineReached()
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
            if temporary.stat().st_size != info["size"] or digest.hexdigest() != info["sha256"]:
                raise RuntimeError("Benchmark download differs from pinned data: " + name)
            temporary.replace(path)
            write_json(receipt, info)
        self.receipts[name] = info
        with path.open("rb") as f:
            header = np.fromfile(f, dtype="<i4", count=256)
        if len(header) != 256 or header[0] != 20240520 or header[1] != 1:
            raise RuntimeError("Not a GPT-2 uint16 llm.c data shard: " + name)
        if 1024 + 2 * int(header[2]) != path.stat().st_size:
            raise RuntimeError("Incorrect benchmark shard length: " + name)
        return np.memmap(path, mode="r", dtype="<u2", offset=1024, shape=(int(header[2]),))


class TrainStream:
    """Same sequential shard traversal as the upstream Python reference."""
    def __init__(self, source, batch=BATCH, context=CONTEXT):
        self.source = source
        self.batch, self.context = batch, context
        self.names = sorted(x for x in source.manifest["files"] if "_train_" in x)
        self.shard = 0
        self.position = 0
        self.tokens = source.array(self.names[0])

    def next_batch(self):
        count = self.batch * self.context
        if self.position + count + 1 > len(self.tokens):
            self.shard = (self.shard + 1) % len(self.names)
            self.tokens = self.source.array(self.names[self.shard])
            self.position = 0
        buf = torch.from_numpy(np.array(self.tokens[self.position:self.position+count+1], dtype=np.int64))
        self.position += count
        return buf[:-1].reshape(self.batch, self.context), buf[1:].reshape(self.batch, self.context)


def optimizer_for(model, device):
    groups = [
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 0.1},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    # Stock PyTorch AdamW. Capturable keeps bias-correction steps on the TPU;
    # otherwise Python step constants can trigger a fresh XLA graph each update.
    return torch.optim.AdamW(groups, lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8,
                             foreach=False, fused=False, capturable=device.type == "xla")


def clip_gradients(model):
    # Equivalent global L2 clipping without a mixed-dtype torch.stack operation.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    norm = sum(g.detach().float().square().sum() for g in grads).sqrt()
    scale = torch.clamp(1.0 / (norm + 1e-6), max=1.0)
    for grad in grads:
        grad.mul_(scale)
    return norm


class Runtime:
    def __init__(self, kind):
        self.tpu = kind == "tpu"
        if self.tpu:
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr
            import torch_xla.distributed.spmd as xs
            xr.use_spmd()
            if xr.global_runtime_device_count() != 8 or xr.addressable_runtime_device_count() != 8:
                raise RuntimeError("This reference runner requires the existing single-host 8-chip TPU.")
            self.xm, self.xs = xm, xs
            self.mesh = xs.Mesh(np.arange(8), (8,), ("data",))
            self.device = xm.xla_device()
        else:
            self.device = torch.device("cpu")

    def put(self, tensor):
        value = tensor.to(self.device)
        if self.tpu:
            self.xs.mark_sharding(value, self.mesh, ("data",) + (None,) * (value.ndim-1))
        return value

    def replicate(self, tensor):
        if self.tpu:
            self.xs.mark_sharding(tensor, self.mesh, (None,) * tensor.ndim)

    def step(self, wait=False):
        if self.tpu:
            self.xm.mark_step()
            if wait:
                self.xm.wait_device_ops()

    def autocast(self):
        return torch.autocast("xla", dtype=torch.bfloat16) if self.tpu else nullcontext()


def reference_bracket(step):
    rows = json.loads((HERE / "reference_val.json").read_text())
    lo = max((r for r in rows if r["step"] <= step), key=lambda r:r["step"])
    hi = next((r for r in rows if r["step"] >= step), rows[-1])
    return {"published_lower": lo, "published_upper": hi,
            "comparison": "reference curve; different seed/order and hardware, no automatic correctness verdict"}


@torch.no_grad()
def evaluate(model, tokens, rt, root, step, deadline, count=VAL_TOKENS):
    model.eval()
    total = 0.
    evaluated = 0
    size = BATCH * CONTEXT
    for offset in range(0, count, size):
        if time.time() >= deadline:
            break
        buf = torch.from_numpy(np.array(tokens[offset:offset+size+1], dtype=np.int64))
        x = rt.put(buf[:-1].reshape(BATCH, CONTEXT))
        y = rt.put(buf[1:].reshape(BATCH, CONTEXT))
        with rt.autocast():
            _, loss = model(x, y, return_logits=False)
        rt.step()
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite validation loss")
        total += value * size
        evaluated += size
    model.train()
    row = {"kind": "validation", "step": step, "tokens_seen": step*TOTAL_BATCH,
           "evaluation_tokens": evaluated, "full_benchmark_evaluation": evaluated == VAL_TOKENS,
           "val_nll": total/evaluated if evaluated else None,
           "val_perplexity": math.exp(min(total/evaluated, 700)) if evaluated else None,
           **reference_bracket(step)}
    with (root/"metrics.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    write_json(root/"latest_validation.json", row)
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--device", choices=("tpu", "cpu"), default="tpu")
    a = p.parse_args()
    root = a.root
    source = FineWeb(a.cache, a.deadline-240)
    start = time.time()
    step = 0
    write_json(root/"status.json", {"status":"preparing_data", "deadline_unix":a.deadline})
    val = source.array("fineweb_val_000000.bin")
    stream = TrainStream(source)
    rt = Runtime(a.device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    reference.FLASH = 0  # Upstream mathematical attention; CUDA flash kernels unavailable.
    model = reference.GPT(reference.GPTConfig()).to(rt.device)
    for tensor in (*model.parameters(), *model.buffers()):
        rt.replicate(tensor)
    optimizer = optimizer_for(model, rt.device)
    manifest = {"source_repo":"karpathy/llm.c", "source_commit":SOURCE_COMMIT,
                "model":"unchanged upstream GPT-2 124M, 12 layers, 12 heads, 768 width",
                "parameters":sum(p.numel() for p in model.parameters()),
                "batch_tokens":TOTAL_BATCH, "microbatch":BATCH, "accumulation":ACCUM,
                "context":CONTEXT, "optimizer":"torch.optim.AdamW", "peak_lr":PEAK_LR,
                "betas":[0.9,0.95], "epsilon":1e-8, "weight_decay":0.1,
                "warmup_updates":WARMUP, "cosine_schedule_updates":TOTAL_STEPS,
                "final_lr_fraction":0., "validation_tokens":VAL_TOKENS,
                "data_repo":source.manifest["repo"], "data_revision":source.manifest["revision"],
                "numerics":"FP32 weights/moments, BF16 autocast on TPU",
                "differences_from_cuda_log":["PyTorch/XLA instead of CUDA C", "sequential Python loader instead of C shuffle", "different RNG/rounding; no bitwise equivalence claim"],
                "time_limit_is_partial_run":True, "torch":torch.__version__}
    write_json(root/"manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    rt.step(wait=True)
    # A small initial probe avoids spending the training budget on step-zero evaluation.
    evaluate(model, val, rt, root, 0, a.deadline-240, count=2**20)
    write_json(root/"status.json", {"status":"training", "step":0, "deadline_unix":a.deadline})
    try:
        while step < TOTAL_STEPS and time.time() < a.deadline-240:
            began = time.monotonic()
            optimizer.zero_grad(set_to_none=False)
            loss_sum = 0.
            for micro in range(ACCUM):
                if time.time() >= a.deadline-240:
                    raise DeadlineReached()
                x, y = stream.next_batch()
                with rt.autocast():
                    _, loss = model(rt.put(x), rt.put(y), return_logits=False)
                (loss/ACCUM).backward()
                loss_sum = loss_sum + loss.detach()/ACCUM
                # Bound the lazy graph at each microbatch; this is not a host read.
                rt.step()
            for param in model.parameters():
                if param.grad is not None:
                    rt.replicate(param.grad)
            clip_gradients(model)
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = torch.tensor(lr).to(rt.device) if rt.tpu else lr
            optimizer.step()
            rt.step(wait=True)
            step += 1
            elapsed = time.monotonic()-began
            row = {"kind":"train", "step":step, "tokens_seen":step*TOTAL_BATCH,
                   "train_nll":float(loss_sum.cpu()), "lr":lr, "seconds":elapsed,
                   "tokens_per_second":TOTAL_BATCH/elapsed, "elapsed_seconds":time.time()-start}
            if not math.isfinite(row["train_nll"]):
                raise RuntimeError("Nonfinite scalar training loss")
            with (root/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
            write_json(root/"status.json", {"status":"training", **row})
            if step <= 5 or step % 10 == 0:
                print(json.dumps(row), flush=True)
            if step % 250 == 0:
                evaluate(model, val, rt, root, step, a.deadline-240)
    except DeadlineReached:
        pass
    # Only one final model checkpoint; no optimizer/checkpoint uploading in the hot loop.
    write_json(root/"status.json", {"status":"final_evaluation", "step":step})
    result = evaluate(model, val, rt, root, step, a.deadline-45)
    if time.time() < a.deadline-30:
        print("Saving final model checkpoint", flush=True)
        rt.step(wait=True)
        weights = {name:tensor.detach().cpu() for name,tensor in model.state_dict().items()}
        tmp = root/"model_final.tmp"
        torch.save({"model":weights, "step":step, "model_config":vars(model.config),
                    "manifest":manifest, "validation":result, "resumable":False}, tmp)
        tmp.replace(root/"model_final.pt")
    write_json(root/"data_receipts.json", source.receipts)
    write_json(root/"status.json", {"status":"finished_partial_reference_run", "step":step,
               "tokens_seen":step*TOTAL_BATCH, "validation":result,
               "full_training_recipe_completed":step == TOTAL_STEPS})


if __name__ == "__main__":
    main()
