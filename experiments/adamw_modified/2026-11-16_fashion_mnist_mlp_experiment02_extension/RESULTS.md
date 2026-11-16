# Fashion-MNIST fixed continuation: epoch 20 → 60

Completed: 0/120. Full verification passed: **False**.

This extension was chosen after observing mixed epoch20 results. It is an exploratory follow-up with its epoch60 endpoint fixed before continuation. Each model retains its own saved model, optimizer and RNG state. Five independent seeds; eight settings averaged within seed for overall paired 95% t intervals. No alpha, accuracy or validation-based endpoint selection. All outcomes and unsupported fits retained. FC3 is exploratory because it has only10 eigenvalues. The test set was observed previously.

Peak rates, architecture, data/split and exact magnitude-only graft are unchanged. Only the future schedule is new: cosine decay from10% of the original peak at20 to1% at60, without a warmup restart. This comparison tests a specified continuation, not a universal optimizer or RG claim.

[Parent epoch20 results](../2026-11-15_fashion_mnist_mlp_experiment02_clean/RESULTS.md) · [All model statuses](results/model_status.csv) · [Epoch20/60 endpoints](results/endpoints20_and60.csv) · [Paired effects](results/paired_effects.csv) · [Within-arm changes](results/within_arm_changes.csv) · [Effect changes](results/effect_changes.csv) · [Verification](results/Verification.json)

**Partial result: no complete five-seed inference is claimed.**

