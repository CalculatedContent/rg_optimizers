# October 6 leaderboard model — TPU port capacity audit

**Status: NOT READY TO TRAIN.** This folder currently contains an offline
capacity audit and a frozen port specification. The TPU trainer is not yet
implemented, and no TPU execution or convergence has been verified. There is
deliberately no `run` command. Passing `check-capacity` is not training approval.

The [CUDA reference](../nanogpt_leaderboard_2026_10_06/README.md), including its
entire `vendor/` directory and launcher, remains unchanged. The proposed port
targets its exact upstream commit
`4ea6b937337a4889b8cfe3f38a93d120048d8f71` (ANVIL2, record #92).
This is the 11-block leaderboard model with the full hashed n-gram table,
not stock GPT-2. Reducing that table or substituting MuonClip/AdamW for ANVIL
would create a different experiment.

## Hardware decision comes first

The repository's existing `baseline/gpt2_small/scripts/tpu_environment.sh`
sets `TPU_ACCELERATOR_TYPE=v5litepod-8`. That is v5e-8. It is not an appropriate
allocation for this full-size leaderboard port.

The BF16 n-gram table is **84,602,880 × 768**, or **129,950,023,680 bytes =
121.025 GiB**. Its upstream row metadata adds 0.946 GiB: one FP32 second
moment, one int32 last-event value and one int32 row-map value per row.
There is no full per-element Adam state or dense table gradient.

| Allocation | Physical chips | HBM/chip | Table + row state/chip | HBM left/chip | Capacity decision |
|---|---:|---:|---:|---:|---|
| v5e-8 / v5litepod-8 | 8 | ≤16 GiB* | 15.246 GiB | ≤0.754 GiB | Reject |
| v4-8 | 4 | 32 GiB | 30.493 GiB | 1.507 GiB | Reject |
| v5p-8 | 4 | 95 GiB | 30.493 GiB | 64.507 GiB | Single-host candidate |
| v5p-16 | 8 | 95 GiB | 15.246 GiB | 79.754 GiB | Multi-host candidate |
| v4-32 | 16 | 32 GiB | 7.623 GiB | 24.377 GiB | Multi-host candidate |

*Google documents v5e HBM as 16 GB. The audit optimistically treats this as
16 GiB; even that upper bound fails. v4/v5p allocation names count TensorCores:
**v5p-8 has four chips, not eight**. Runtime device counts must also be checked
on the actual allocation instead of inferred from the allocation's suffix.

The capacity gate reserves a provisional **16 GiB per chip** beyond the table
and row state. This is a planning allowance, not measured peak memory. It must
cover dense weights/optimizer state, activations, row caches, sparse exchanges
and XLA temporaries. Compiler partitioning, accidental replication, temporary
table copies, validation and export can still make a candidate fail.

Read-only commands from the repository root (standard library only):

```bash
python3 baseline/experiments/nanogpt_leaderboard_2026_10_06_tpu/capacity.py plan
python3 baseline/experiments/nanogpt_leaderboard_2026_10_06_tpu/capacity.py verify-reference
python3 baseline/experiments/nanogpt_leaderboard_2026_10_06_tpu/capacity.py check-capacity --accelerator-type v5p-8
# Returns exit code 2 for the existing small allocation:
python3 baseline/experiments/nanogpt_leaderboard_2026_10_06_tpu/capacity.py check-capacity --accelerator-type v5litepod-8
```

Each command verifies the frozen source hashes and schedule first. These commands
do not contact Google Cloud, allocate hardware, inspect live HBM or allocate tensors.

## Port requirements still to implement

1. Select and inspect the actual TPU allocation. Prefer a single-host v5p-8
   for initial development if available. Use a matching, explicitly tested
   torch/torch_xla pair and record libtpu and runtime versions. Do not install
   the CUDA reference's cu128/FlashAttention dependencies on the TPU.
2. Use XLA SPMD with `torch_xla.runtime.use_spmd()` and explicit row-axis
   sharding. SPMD exposes one logical device; do not combine it with the
   pasted `torch_xla.launch(_mp_fn)` MPMD sketch, manual gradient all-reduces,
   or NCCL. Physical chip count and the reference's eight logical data streams
   are different concepts. Preserve those streams' document segmentation and
   n-gram history when mapping them onto four or sixteen chips.
3. Port the model and optimizers into this sibling folder. Remove CUDA graphs,
   Triton, patched FlashAttention-3 and FP8 caches from the new implementation.
   Preserve mixed-width attention, paired heads, partial key offsets, QK norm,
   YaRN/windows, XSA, smear, value embeddings, MUDD and residual topology.
   Attention must be document-aware and memory-bounded: constructing a dense
   attention mask over the entire packed token stream is not a viable substitute.
4. Preserve signed bigram/trigram hashing, sparse row pulls, touched-row gradient
   accumulation and lazy row-Adam decay/cadence. Never make the full table an
   ordinary autograd embedding parameter or gather it into host RAM. Verify
   XLA's gather/scatter lowering and actual per-device storage before training.
5. Preserve ANVIL's rails/equalizer and auxiliary Adam schedules, embedding
   untying, sampled softmax, MTP/prefix losses, canonical validation mask and
   final tail averaging. A BF16 port changes the FP8 numerical path; label
   that change and check forward, backward and optimizer math. The upstream
   `eval()` path is not a substitute for its training loss.
6. Preserve the complete schedule in `port_contract.json`: 1,194 updates,
   boundaries `[0, 320, 681, 1107, 1174, 1194]`, batch tokens
   `[131072, 262144, 393216, 327680, 131072]`, 328,663,040 training tokens.
   Use static-shape stage variants and bounded row routing. Five stages do
   not by themselves guarantee only five compiled graphs: optimizer cadence,
   sampled loss and validation introduce additional variants. Measure recompiles.
7. Reuse the unchanged reference data downloader, then hash all nine training
   shards and the validation shard. The upstream HF revision is not pinned;
   a filename alone does not establish the same data. Record an immutable
   shared data manifest before comparing reference and port. Compare final
   loss on the full 10,485,760 validation tokens, with identical masking.
8. Export dense tail-averaged state and every n-gram shard with global row
   offsets, outside training timing. Adapt the reference `weight_export.py`
   receipt/hash/coverage protocol to XLA local shards. Write
   `WEIGHTS_COMPLETE.json` only after all shards are verified. Provide at
   least 160 GiB of free disk across the export destination; check per-host
   placement on multi-host allocations. This is a WeightWatcher analysis
   export, not an optimizer resume checkpoint.

To download the data using the reference's existing entry point:

```bash
python3 baseline/experiments/nanogpt_leaderboard_2026_10_06/experiment.py prepare --data-root /data/nanogpt-leaderboard
```

The CUDA reference's `check` and `run` must continue to reject a TPU.

## Acceptance and results

Future runs belong under this folder's ignored `results/` directory (or an
explicit persistent results destination). Each run must retain the port commit,
source pin, numerical differences, data hashes, actual hardware/runtime versions,
stage token counts, validation result, memory/compile measurements and export
receipts. Do not commit a successful-looking result or completion marker without
the corresponding run.

Acceptance requires a fresh full TPU run with finite FineWeb validation
cross-entropy **≤3.28 over exactly 10,485,760 tokens**, plus verified complete
weight export. CPU checks and capacity estimates cannot establish this.
The H100 record's 39.9-second timing is not a TPU target or estimate. Compilation
time and training wall time are currently unknown; even a claim that compilation
must exceed seven minutes would be speculative.

## Sources and checks

- [Google TPU v5p specifications and topology](https://docs.cloud.google.com/tpu/docs/v5p)
- [Google TPU v5e specifications](https://docs.cloud.google.com/tpu/docs/v5e)
- [Google TPU v4 specifications and topology](https://docs.cloud.google.com/tpu/docs/v4)
- [PyTorch/XLA SPMD guide](https://docs.pytorch.org/xla/master/perf/spmd_basic.html)
- [Pinned upstream table state](../nanogpt_leaderboard_2026_10_06/vendor/track_1_short/ngram_table.py)
- [Pinned upstream model](../nanogpt_leaderboard_2026_10_06/vendor/track_1_short/model/gpt.py)

```bash
python3 -m pytest -q baseline/experiments/nanogpt_leaderboard_2026_10_06_tpu/tests/test_capacity.py
```

The tests cover reference tampering, the fixed schedule/token budget, hardware
chip counts, small-allocation rejection and the distinction between capacity
eligibility and training readiness. They do not test a TPU trainer.
