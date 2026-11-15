# Experiment03: small GPT, AdamW magnitude graft and SGDM

**Setup and numerical preflight passed. The first three training runs are a gated pilot; confirmatory results are not yet available.**

This is a separate Tiny Shakespeare experiment. It does not overwrite MNIST or Fashion-MNIST.

## Fixed comparison

- nanoGPT, upstream commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, two blocks, width256, four heads, context128, batch32, 1,629,440 parameters, dropout0.
- Native AdamW (base LR0.0005), magnitude-only AdamW graft (Adam LR0.0005; SGDM target LR0.05), and SGDM (LR0.05, momentum0.9). These are preset candidate rates, not claims of exhaustive optimizer tuning.
- The exact graft preserves Adam moments, bias correction and direction. Only the eight physical attention/MLP matrix update magnitudes are replaced by momentum targets computed from that same run's post-clipping gradient history. Packed QKV is grafted jointly. Embeddings, the tied output head, biases and LayerNorm retain native Adam updates in the graft arm. The SGDM reference uses SGDM for all parameters.
- Global gradient clipping at1 in all arms. Same per-step multiplicative matrix decay independent of optimizer LR. Warmup100 steps; cosine to10% peak at10,000 steps.
- Shared initialization and sampled training windows within each seed; contiguous disjoint train/validation/test ranges80/10/10. Windows never cross split boundaries. Test is evaluated only at the fixed confirmatory endpoint.

## Pilot and continuation gate

Three 1,000-step pilot runs use seed41000 and LR multiplier1. They are excluded from confirmatory statistics. Every arm must pass saved-checkpoint verification, remain finite and nonconstant, and reduce validation cross-entropy by at least0.1nat. Alpha alignment is **not** a gate.

Only then may the fixed 45-run grid proceed: seeds41001–41005 × LR multipliers0.5/1/2 × three arms, 10,000 steps each. Three jobs at a time; global maximum96hours from the pilot start; each training job is bounded to4.5hours with time reserved for preservation. Pilot throughput must support these limits. Failures stop or remain explicit; no automatic rate substitutions, replacement seeds or dataset changes.

GitHub Actions provides independent execution from the transient interactive workspace. The single launch file triggers one workflow. Do not edit it or rerun successful jobs to create replacement observations. Standard public-repository runners are used; no paid GPU/larger runner is requested. Artifacts retain checkpoints and diagnostics for90days; durable archival should be completed during that window.

## Monitoring and checks

WeightWatcher0.7.7 default fits every500steps, including initialization and final state. Eight physical matrices are primary; six additional Q/K/V views are secondary. Record alpha, |alpha−2|, fit support/KS distance, spectral norm and layer variance. No alpha-targeted cutoff selection. Save raw spectra and validation trajectories.

Every50steps, record gradient norm, momentum norm, native Adam magnitude, SGDM target, applied magnitude, direction cosine and update gain. All minibatch hashes and training losses are recorded. Keep initialization, step500 and the latest rolling500-step checkpoint, including optimizer state. Verify exact three-step replay from both initialization and step500, restored validation predictions/loss, raw spectra/alpha, final test measurements and file hashes.

Pilot preflight independently checks native PyTorch optimizers, own-history SGDM magnitude, unchanged Adam moments and nongrafted parameters, and the numerical SGDM identity-graft negative control. SGDM is an empirical reference, not an assumption that its alpha must equal2.

Final reports preserve usable, failed and incomplete outcomes. Use five paired seeds for95% intervals; learning-rate settings are not independent seeds. Upward alpha movement, proximity to2 and lower held-out loss are separate outcomes. The report and plots are checked into this folder automatically after the workflow finishes.

[Protocol](Protocol.json) · [Local preflight](Local_Preflight.json) · [Source/data provenance](Provenance.json) · [Workflow](../../../.github/workflows/gpt-experiment03.yml)
