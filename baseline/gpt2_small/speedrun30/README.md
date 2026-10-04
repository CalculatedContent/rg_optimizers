# GPT-2 / FineWeb reference, bounded to 30 minutes

Run this before interpreting the custom MuonClip experiment as a reproduction.
This is a TPU port of the original GPT-2/FineWeb baseline behind the NanoGPT
speedrun's 3.28 target. It is a timed, potentially partial reference run, not a
promise to finish the benchmark in 30 minutes. Karpathy's older OpenWebText
nanoGPT recipe is a different benchmark and is not used here.

## Run from Cloud Shell

From a clean checkout of the pushed commit:

```bash
python3 baseline/gpt2_small/speedrun30/cloudshell.py start
```

The launcher requests a final save from the current MuonClip service, waits up to
90 seconds, and then stops that service if necessary. Its previously saved
checkpoints, diagnostics, cloud archives and FineWeb-Edu corpus are retained.
The new job uses the existing eight-chip TPU and installed PyTorch 2.6/XLA 2.6.
There is no environment installation, reallocation or automatic restart.

The new service has a **30-minute limit including benchmark-data downloads,
compilation, training, evaluation, final save and backup**. Normal training ends
earlier to reserve finalization time; a systemd deadline kills stalled work.
Stopping the old run and checking out the source happen before this clock starts.
The TPU allocation itself continues to exist after the job ends.

```bash
python3 baseline/gpt2_small/speedrun30/cloudshell.py status
```

`SPEEDRUN30_LATEST.json` on the mounted disk records the service, output directory,
commit and deadline. `RUN_STATUS.json` distinguishes finish, failure and timeout.
No completion message should be interpreted as proof of benchmark reproduction.

## Pinned reference

- Model source: [karpathy/llm.c at 7ecd8906](https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/train_gpt2.py).
  `vendor/llmc_train_gpt2.py` is the upstream file, with its MIT license included.
  The GPT model and its initialization are imported directly, not reimplemented.
- Recipe: [GPT-2 124M reproduction launcher](https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/scripts/run_gpt2_124M.sh).
- Training curve: [published October 13 2024 llm.c run](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2024-10-13_llmc).
  `reference_val.json` preserves its validation rows and exact step/token counts.
- Data: `kjj0/fineweb10B-gpt2`, revision
  `889765ea1f903759787add96995d81171b632d0c`, the GPT-2-tokenized FineWeb shards
  used by the speedrun's official cached-data downloader.
  All 104 shard filenames, byte lengths and SHA256 hashes are pinned in
  `data_manifest.json`. Download only the validation shard and training shards
  reached by this run; do not cycle a small subset. Cached files go under
  `/mnt/disks/rg-data/benchmark-fineweb10B-889765ea`.

| Setting | This reference run |
|---|---|
| Model | GPT-2 124M, 12 layers, 12 heads, width 768, tied embeddings |
| Tokenizer / context | GPT-2 / 1,024 |
| Global batch | 524,288 tokens per optimizer update |
| TPU microbatch | 64 sequences globally, accumulated 8 times |
| Optimizer | Stock PyTorch AdamW, betas 0.9/0.95, epsilon 1e-8 |
| Weight decay | 0.1 on matrices, zero on vectors/biases |
| Gradient clipping | Global L2 norm 1.0 |
| Learning rate | 0.0006, 700-update warmup, cosine decay to zero |
| Schedule horizon | 19,560 updates, as in the published curve |
| Full validation | First 10,485,760 validation tokens, fixed context 1,024 |
| Measurements | Scalar train loss/throughput; validation every 250 updates and at end |
| Checkpoint | One final model-only checkpoint; no automatic resume |

The dataset contains 10,255,324,043 training tokens; integer division by the
reference global batch gives 19,560 updates. The wall-clock limit truncates this
schedule; it does not shorten warmup or compress decay to force a lower loss.
The first validation at step zero uses only 1,048,576 tokens to save time and is
explicitly labelled a partial evaluation. Later evaluations use the full
10,485,760-token benchmark unless the deadline interrupts them. A partial
evaluation is never labelled a full benchmark score.

## Hardware changes and limits of comparison

One XLA SPMD process partitions batches over eight TPU chips and replicates
parameters and global gradients. It does not multiply/divide the loss by eight
a second time. BF16 autocast is used for compute with FP32 weights/moments.
The upstream mathematical-attention option is selected because CUDA flash
kernels are unavailable. Stock AdamW uses XLA-supported capturable state and a
tensor learning rate so its changing step/LR do not become host graph constants.
Lazy graphs are submitted at microbatch boundaries and after each optimizer
update; there are no per-tensor finite scans or diagnostic host transfers.

The model, tokenizer, validation file, validation token count, global batch and
optimizer hyperparameters are controlled. This remains a **port**, not bitwise
CUDA reproduction: CUDA's kernels/rounding and C shuffled loader differ from
this PyTorch sequential loader and initialization. Matching CPU reference
updates does not by itself prove TPU correctness. `latest_validation.json`
includes the nearest lower/upper published observations at equal tokens; it
does not interpolate an invented expected loss or issue an automatic pass/fail.

Published curve examples (use only after a full validation on the pinned file):

| Updates | Training tokens | Published validation NLL |
|---:|---:|---:|
| 250 | 131,072,000 | 6.1710 |
| 500 | 262,144,000 | 5.3743 |
| 1,000 | 524,288,000 | 4.3170 |
| 19,560 | 10,255,073,280 | 3.2722 |

## Output and verification

The run writes `metrics.jsonl`, `latest_validation.json`, `manifest.json`, status,
and a final `model_final.pt` when finalization fits within the time cap. A failure
or hard timeout can prevent final evaluation/checkpoint creation; earlier logs
remain on the disk. One bounded final cloud backup is attempted through object
permissions. Check its exit result or `CLOUD_BACKUP_VERIFIED.json`; a local file
alone is not evidence of successful upload.

No WeightWatcher, alpha fits, per-matrix checks, tensor snapshots, preflight, or
periodic checkpoint uploads run in this experiment. It monitors ordinary scalar
loss and aborts on nonfinite scalar loss.

Local CPU tests compare several accumulated updates to the upstream AdamW
implementation, compare shard traversal/target shifting to its data loader,
reject corrupted downloads, exercise the external timeout, and prevent a
duplicate launch. The first live TPU run is still required.
