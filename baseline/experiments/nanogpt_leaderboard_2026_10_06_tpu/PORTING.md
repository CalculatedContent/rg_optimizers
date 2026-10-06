# Port provenance and numerical contract

The sibling CUDA source is immutable. `experiment.py verify` verifies its 58 vendored files and metadata before execution. `tpu_port/provenance.json` records source paths and hashes for derived model, schedule, loader and supporting math; the MIT license is included. Handwritten `ops.py`, `host_table.py`, `optimizer.py`, `tail.py` and `runtime.py` replace CUDA execution infrastructure.

## Preserved decisions

All learned dense shapes and the full n-gram table are retained, including unused/padded bank entries. Attention layers 0, 1, 2, 3, 5, 8 and 10 retain their 64/128 QK/V widths and paired-head indexing. Rotary/YaRN, key offset, windows, value embeddings, smear, MUDD, gates, ReLU-squared MLPs, softcap and document packing derive from the pin.

Five stages, 1,194 updates, candidate counts, sum-reduced training gradients, MTP tail-normalizer behavior, prefix targets and final canonical masking are retained. The final window extends to 20 × 128 without another YaRN frequency update. Embedding untie, sparse cadence and last unflushed sparse gradients follow the source. Tail averages and norm restoration precede evaluation/export.

ANVIL retains both velocity rails, six polynomial maps, lookahead, lane equalization, shape-dependent rates, frozen MLP matrices and cautious decay. Auxiliary Adam retains per-parameter settings and cadence. CPU sparse row-wise Adam replays missed second-moment decay events for touched rows and uses the source's squared-learning-rate decay.

## Runtime and numerical differences

- Eight PJRT MPMD processes replace CUDA ranks. Gloo exchanges host n-gram rows; XLA averages dense gradients. This is not SPMD.
- Full n-gram weights and scalar-per-row state stay in host RAM. Only active rows enter HBM. Exchanges happen each step, without CUDA graph overlap or row-prefetch pipelines.
- BF16 projection/MLP matmuls replace FP8 caches. FP32 softmax and bounded attention slabs replace patched FlashAttention. Checkpointing recomputes activations in backward.
- ANVIL polynomial recurrence uses FP32 and XLA highest matmul precision: BF16 recurrence was numerically unstable in an XLA CPU rank-one-gradient test. The same six maps and rails are retained; compute cost and rounding differ from CUDA.
- Dense optimizer state is replicated. ANVIL uses an FP32 master with ordinary BF16 rounding, instead of CUDA packed-mantissa/high-half truncation. Fused arithmetic and reduction orders differ.
- Value embeddings use ordinary BF16 parameter-gradient accumulation instead of persistent FP16 atomics. Host sparse accumulation uses FP16 then BF16 owner exchange with a different reduction order.
- Matching torch/torch_xla 2.9.0 packages replace the CUDA software stack. The H100 timing region and warmup are not reproduced.
- Each measured step includes synchronization and host work. Python slab loops can create large graphs at full validation size. Compile time and throughput remain unmeasured on TPU.

These differences require a convergence experiment. Preflight establishes only execution with finite loss for tested stages on that hardware. A full run is accepted only when data/source manifests, full validation-token count, target NLL and complete weight export all pass. No bitwise equivalence or leaderboard performance claim is made.
