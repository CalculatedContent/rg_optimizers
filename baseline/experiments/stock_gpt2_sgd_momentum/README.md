# Stock GPT-2 Small / FineWeb10B — classical SGD + momentum

This experiment reuses the **unchanged, SHA256-verified upstream GPT-2 Small model** and the same pinned FineWeb10B data and training protocol as the AdamW/MuonClip control. No rotary embeddings, RMSNorm, six-head attention, U-Net skips, or untied head.

Architecture: 12 layers, 12 heads, width 768, MLP 3072, context 1024, vocabulary 50257, 124439808 unique parameters; packed QKV, learned positions, affine LayerNorm, GPT-2 GELU, tied token/output embeddings.

## Candidate SGD recipe (not yet empirically optimized)

- Optimizer: `torch.optim.SGD`, classical momentum **0.90**, Nesterov **off**, dampening 0, weight decay **0** on all parameters.
- Peak learning rate **0.01** is an initial screening candidate, **not a proven best value**. Do not label it optimal until a validation-only sweep on the actual GPT-2 model/data/hardware is complete.
- 700 warmup updates, cosine decay to zero; global gradient norm clipping at 1.0.
- 19560 optimizer updates, 524288 tokens/update, microbatch 64 sequences with 8 accumulation steps.
- Same FineWeb10B revision, fixed validation tokens, seed, spectral checkpoints, and stopping budget as the existing controls.
- Zero weight decay intentionally isolates SGD + momentum and avoids additional weight-scale effects; it differs from the AdamW control's matrix weight decay.

## Run

```bash
python3 -m unittest baseline/gpt2_small/muon_speedrun/test_sgd_momentum.py
python3 baseline/gpt2_small/muon_speedrun/cloudshell.py start --node YOUR_TPU_NODE --optimizer sgd_momentum --hours 12
```

The launcher checks the existing TPU lease and performs the full-model TPU forward/backward/optimizer preflight before training. No TPU allocation or training is initiated by this change.

## Qualification before calling these hyperparameters best

On the fixed architecture/data/batch/seed/horizon, sweep peak LR `[0.001, 0.003, 0.01, 0.03, 0.1]` and momentum `[0.9, 0.95]`; reject nonfinite/divergent runs, rank by held-out **validation NLL only**, then repeat the leading candidates over three seeds and a full 19560-update horizon. Preserve the untouched test set until the recipe is locked. These sweeps are a **proposed protocol**, not completed experiments.

Compare loss, perplexity and WeightWatcher alpha against the **stock-GPT-2** controls only; historical modified-architecture speedruns are not comparable.
