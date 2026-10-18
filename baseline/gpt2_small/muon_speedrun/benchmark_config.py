"""Pinned original GPT-2/FineWeb speedrun baseline, not the evolving modded record.

The Muon arm is an explicit optimizer substitution. See BENCHMARK.md for the
TPU port differences; this identifier must accompany every new comparison.
"""
import math
import os

LLMC_PROFILE = 'llmc_reference'
SGD_SPEEDRUN_PROFILE = 'sgd_speedrun'
PROFILE = os.environ.get('RG_GPT2_PROFILE', LLMC_PROFILE)
BENCHMARK = ('stock-gpt2-124m-fineweb10b-sgdm-layerlr-v1' if PROFILE == SGD_SPEEDRUN_PROFILE else 'llmc-gpt2-124m-fineweb10b-2024-10-13-upstream-tpu-v2')
SOURCE_COMMIT = '7ecd8906afe6ed7a2b2cdb731c042f26d525b820'
TOTAL_STEPS = 19560
BATCH_TOKENS = 524288
VAL_TOKENS = 10485760
WARMUP = 700
WARMDOWN = 0
LR_SCHEDULE = 'cosine_to_floor_0.10'
PEAK_ADAM_LR = 0.0006
# Published GPT-2 Small SGDM center from tml-epfl/why-weight-decay.
SGD_PEAK_LR = 0.3
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 3e-5
SGD_NESTEROV = False
GRAD_CLIP = 1.0
MEASUREMENT_INTERVAL = 250
SGD_COSINE_FLOOR = 0.10
SGD_EMBEDDING_MULTIPLIER = 0.60
SGD_OTHER_MULTIPLIER = 0.75
SGD_BLOCK_MULTIPLIERS = (0.75, 0.80, 0.85, 0.90, 1.20, 1.35, 1.45, 1.40, 1.30, 1.15, 1.00, 0.90)
RESUMABLE_CHECKPOINT_INTERVAL = 2000


def lr_factor(step):
    if step < 0:
        raise ValueError('Training update must be nonnegative')
    if step < WARMUP:
        return (step + 1) / WARMUP
    ratio = min(1.0, (step - WARMUP) / (TOTAL_STEPS - WARMUP))
    cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return SGD_COSINE_FLOOR + (1.0 - SGD_COSINE_FLOOR) * cosine


def protocol():
    return dict(profile=PROFILE, benchmark=BENCHMARK, source_repo='karpathy/llm.c',
                source_commit=SOURCE_COMMIT, steps=TOTAL_STEPS,
                batch_tokens=BATCH_TOKENS, tokens=TOTAL_STEPS * BATCH_TOKENS,
                warmup_updates=WARMUP, warmdown_updates=WARMDOWN, lr_schedule=LR_SCHEDULE,
                adam_peak_lr=PEAK_ADAM_LR,
                sgd_peak_lr=SGD_PEAK_LR, sgd_momentum=SGD_MOMENTUM,
                sgd_weight_decay=SGD_WEIGHT_DECAY, sgd_nesterov=SGD_NESTEROV, adam_betas=[0.9, 0.95],
                adam_eps=1e-8, matrix_weight_decay=0.1,
                gradient_clip_norm=GRAD_CLIP,
                measurement_interval=MEASUREMENT_INTERVAL,
                validation_tokens=VAL_TOKENS,
                data_order='upstream Python sequential shards; discard incomplete microbatch tails; wrap at corpus end',
                implementation='TPU SPMD; BF16 activations; FP32 weights/states; unchanged upstream packed QKV',
                muonclip_variant='LR 0.02, constant momentum 0.95, RMS scale 0.2, decay 0.1, five NS steps on packed matrices; per-head causal QK clip threshold 100, balance 0.5',
                muon_variant='LR 0.04, momentum 0.85 to 0.95 over 500 updates, five NS steps, no hidden decay; baseline LR schedule and clipping')
