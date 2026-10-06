import numpy as np
import torch
from torch import Tensor

NGRAM_VOCAB_SIZE = 84_602_880
NGRAM_DIM = 768
# The shared +-1 sign pool. A power of two, so `& (rows - 1)` is the non-negative remainder of the
# wrapped int32 hash (values ~500-15000 gave similar results in the bigram record).
NGRAM_SIGN_POOL_ROWS = 8192

# Row hashes (record #360): multiply each token by a large odd constant, xor, reduce mod the channel
# size. int32 products wrap; torch's `%` is Python-signed, so the result is non-negative.
BIGRAM_ROW_MULS = (36313, 27191)            # x[t], x[t-1]
TRIGRAM_ROW_MULS = (17351, 60961, 45259)    # x[t], x[t-1], x[t-2]
# Sign-pool hashes (record #360), independent of the row hashes; computed in the embedding kernel
# (perf/kernels/ngram_embed.py). Position t reads pool row
#   bigram:  (m[1] * x[t-1] ^ m[0] * x[t]) & (pool_rows - 1), row 0 at t < 1
#   trigram: (m[2] * x[t-2] ^ m[1] * x[t-1] ^ m[0] * x[t]) & (pool_rows - 1), row 0 at t < 2
BIGRAM_SIGN_MULS = (48271, 30011)           # x[t], x[t-1]
TRIGRAM_SIGN_MULS = (58699, 39779, 26801)   # x[t], x[t-1], x[t-2]

# Adam for the table (record #360). lr and weight decay are Adam's scheduled base values times these
# multipliers; beta1 is 0 for the whole run.
NGRAM_LR_MUL = 70.0
NGRAM_WD_MUL = 5.0
NGRAM_ADAM_BETA2 = 0.95
# From this step the table updates every 4th step instead of every 2nd (record #360). One event then
# stands for two: beta2 is squared, so the second moment decays at the same rate per step, and the
# weight-decay multiplier doubles.
NGRAM_ADAM_PERIOD4_START = 336
NGRAM_WD_MUL_PERIOD4 = 10.0
# The longest cycle, in steps: the cache holds one cycle's rows (two per token per step).
MAX_CYCLE_STEPS = 4
# A cycle's gradient is summed per cache slot in fp16, travels between ranks in bf16 (the table's dtype),
# and is summed per row by the owner in bf16 (record #360's precisions).
GRAD_ACCUM_DTYPE = torch.float16
GRAD_WIRE_DTYPE = torch.bfloat16
GRAD_MERGE_DTYPE = torch.bfloat16

assert NGRAM_VOCAB_SIZE % 2 == 0 and NGRAM_VOCAB_SIZE < 2 ** 31
assert NGRAM_SIGN_POOL_ROWS & (NGRAM_SIGN_POOL_ROWS - 1) == 0
# The last period-2 event (an odd step) is then followed by a whole period-4 cycle.
assert NGRAM_ADAM_PERIOD4_START % 4 == 0


# -----------------------------------------------------------------------------
# Hashing

def ngram_row_ids(x: Tensor) -> Tensor:
    """Token ids [T] (host) -> table row ids [2T] int32: out[:T] bigram rows, out[T:] trigram rows.

    Positions without enough history get the channel's reserved last row. Note the bigram hash at
    t = 1 reads the reserved id, not x[0], as its previous token (it is computed in place after
    out[0] is set, exactly as in record #360).
    """
    x = x.to(torch.int32)
    half = NGRAM_VOCAB_SIZE // 2
    bigram_mod, trigram_mod = half - 1, NGRAM_VOCAB_SIZE - half - 1
    n = x.numel()
    out = torch.empty(2 * n, dtype=torch.int32)
    bigram, trigram = out[:n], out[n:]
    bigram.copy_(x)
    bigram[0] = bigram_mod
    bigram[1:] = torch.bitwise_xor(BIGRAM_ROW_MULS[0] * bigram[1:], BIGRAM_ROW_MULS[1] * bigram[:-1]) % bigram_mod
    trigram[0] = trigram[1] = half + trigram_mod
    trigram[2:] = torch.bitwise_xor(
        torch.bitwise_xor(TRIGRAM_ROW_MULS[0] * x[2:], TRIGRAM_ROW_MULS[1] * x[1:-1]),
        TRIGRAM_ROW_MULS[2] * x[:-2],
    ) % trigram_mod + half
    return out


def sorted_unique_rows(row_id_arrays: list[np.ndarray]) -> np.ndarray:
    """The sorted-unique row ids of several steps' [2T] arrays (a cycle's want list), int32.

    A sort and a neighbour compare rather than np.unique: numpy 2 routes integer unique through a
    hash set, which record #360 measured at 15-60 ms on a cycle's ids against 2-6 ms for the sort.
    """
    rows = np.sort(np.concatenate(row_id_arrays))
    if rows.size:
        rows = rows[np.r_[True, rows[1:] != rows[:-1]]]
    return rows.astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# Update cadence

def is_update_step(step: int) -> bool:
    """Every odd step (with the other Adam params) until NGRAM_ADAM_PERIOD4_START, then steps = 3 mod 4."""
    if step >= NGRAM_ADAM_PERIOD4_START:
        return step % 4 == 3
    return step % 2 == 1


def adam_beta2_and_wd_mul(step: int) -> tuple[float, float]:
    if step >= NGRAM_ADAM_PERIOD4_START:
        return NGRAM_ADAM_BETA2 ** 2, NGRAM_WD_MUL_PERIOD4
    return NGRAM_ADAM_BETA2, NGRAM_WD_MUL


