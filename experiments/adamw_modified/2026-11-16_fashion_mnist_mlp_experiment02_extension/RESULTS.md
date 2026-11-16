# Fashion-MNIST fixed continuation: epoch 20 → 60

Completed: 0/120. Full verification passed: **False**.

## Verification failure before continuation

On 16 November 2026 at 00:49:43 UTC (07:49:43 Vietnam time), the run stopped during parent checkpoint replay, before any epoch21 continuation. **0/120 models were extended; no new training checkpoints or epoch60 outcomes exist.**

All five parent seed archives and their 120 saved checkpoints passed their SHA256 checks. Frozen source checks and the fresh-state optimizer contract passed. The parent audit then failed `assert loss == float(recorded['loss'][k])` in `run.py:121` while replaying recorded epoch2 minibatches from saved epoch1 state. The log did not record the two loss values, their difference, or the failing specification, so the root cause and numerical size of the mismatch remain undetermined. File integrity is not equivalent to replay verification. No completed Parent_Validation, Continuation_Freeze, resumed-state contract or continuation gate was produced.

This is a reproducibility/preflight failure, not a model collapse or evidence about the optimizer hypothesis. The previous epoch20 results remain valid as previously reported; this attempt adds no new comparison. Frozen training was not changed, no checks were relaxed and no rerun was launched. GPT remains on hold.

The workflow already published the 120-row incomplete status register and failure summary. There are no new outcome plots to inspect. Both returned artifacts are preserved together in the evidence bundle linked in [Artifact_Receipts.json](verification/Artifact_Receipts.json). The small seed31001 artifact contains restoration spectra and audit/source files, not continued training checkpoints. Original epoch20 checkpoint archives remain separately preserved.



This extension was chosen after observing mixed epoch20 results. It is an exploratory follow-up with its epoch60 endpoint fixed before continuation. Each model retains its own saved model, optimizer and RNG state. Five independent seeds; eight settings averaged within seed for overall paired 95% t intervals. No alpha, accuracy or validation-based endpoint selection. All outcomes and unsupported fits retained. FC3 is exploratory because it has only10 eigenvalues. The test set was observed previously.

Peak rates, architecture, data/split and exact magnitude-only graft are unchanged. Only the future schedule is new: cosine decay from10% of the original peak at20 to1% at60, without a warmup restart. This comparison tests a specified continuation, not a universal optimizer or RG claim.

[Parent epoch20 results](../2026-11-15_fashion_mnist_mlp_experiment02_clean/RESULTS.md) · [All model statuses](results/model_status.csv) · [Epoch20/60 endpoints](results/endpoints20_and60.csv) · [Paired effects](results/paired_effects.csv) · [Within-arm changes](results/within_arm_changes.csv) · [Effect changes](results/effect_changes.csv) · [Verification](results/Verification.json)

**Partial result: no complete five-seed inference is claimed.**

