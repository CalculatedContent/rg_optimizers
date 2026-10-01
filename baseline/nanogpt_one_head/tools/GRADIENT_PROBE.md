# Read-only CPU gradient probe

Run gradient_probe.py using the training VM's existing Python dependencies. It loads
model.py directly from its own checkout and never initializes XLA. Use a separate
worktree; do not update or reinstall the active training checkout.

Example (from this directory):

```bash
nice -n 10 python3 -u gradient_probe.py \
  --run-dir /mnt/disks/rg-data/muonclip-extended/segments/segment_000001/muon_clip/seed_1337 \
  --local-steps 210000 220000 250000 \
  --data-root /mnt/disks/rg-data/rg-nanogpt-one-head/data \
  --output /mnt/disks/rg-data/gradient-probe-2360-2400 \
  --threads 2 --batches 16 --batch-size 2 --repeats 3
```

These local steps correspond to cumulative 2.36M, 2.37M and 2.40M for segment 1.
The script requires permanent epoch checkpoints at those steps and lists recent
available filenames if any are missing. Output must be a new directory outside the
training run. All checkpoints and corpora are read only. Only load trusted checkpoint
files: the project's checkpoints use PyTorch pickle serialization.

Each repeat evaluates 8192 training and 8192 validation token predictions at context
256. Identical sampled windows are used at every checkpoint. Three repeats expose
sampling sensitivity; they are not independent training runs. CPU work can compete
for host resources, so use low priority and two threads while TPU training continues.
No runtime estimate is guaranteed; each split prints progress when complete.

The ZIP contains per-parameter gradient norms, RMS, gradient-to-weight ratios,
gradient energy fractions, train-validation cosine/dot products, sampled losses,
accuracies and provenance (corpus hashes and exact window starts). Tied embedding
and output-head parameters are counted once. Add --save-gradients only if full
gradient tensors are needed; this substantially enlarges the archive.

A negative train-validation cosine means raw gradient descent on these sampled
training windows has an adverse first-order validation direction. This is NOT the
actual MuonClip direction: the probe does not restore momentum, replay training
batches, or simulate optimizer updates. Do not interpret it as causal proof. Probe
norms average sampled gradients and are not directly comparable to logged single
training-update norms. Test data is never read.

Validation: python tools/test_gradient_probe.py (from the nanoGPT project root).
Tests cover accumulation equivalence, sign convention, tied weights, cumulative
steps, fixed batches across checkpoints, checkpoint immutability and archive output.
