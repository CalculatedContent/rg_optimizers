# Fashion-MNIST epoch20-to60 extension

Continue the 120 completed models from the clean Fashion-MNIST pass, retaining every model's optimizer history and RNG state. The user authorized this extension after the mixed epoch20 findings: the magnitude graft improved accuracy but worsened cross-entropy and mean absolute distance from α=2.

This is a fixed future endpoint chosen after seeing the parent results, so it is an exploratory extension rather than an independent confirmation. No new seeds, settings, architectures or datasets are added. GPT remains on hold.

The three arms are native AdamW, the exact magnitude-only graft, and SGDM. The graft retains Adam direction/moments/bias updates; only each matrix update's Frobenius magnitude is supplied by SGDM momentum on that run's own gradient history. The original kernel is unchanged.

- Resume all five seeds 31001–31005, eight settings, and three arms from epoch20.
- Train exactly40 additional epochs to a fixed primary endpoint at60. Save intermediate checkpoints at21 and40.
- Continue the learning-rate factor smoothly from0.1 at20 to0.01 at60 with cosine annealing. No warmup restart.
- Record validation everyepoch and WeightWatcher every5epochs, test only the new final endpoint. No early stopping or checkpoint selection.
- Verify all parent hashes, optimizer state, restored logits/spectra and old checkpoint replay. Three central continuations gate numerical correctness only and count once within120.
- Report all outcomes, including functional failures and unsupported fits. FC3 remains exploratory. Distinguish rising α from smaller|α−2|.
- Use five seed means for paired95%t intervals; settings are repeated conditions, not independent seeds.
- Four-hour training ceiling and five-hour workflow ceiling. Stop after this extension for review.

Parent run: https://github.com/CalculatedContent/rg_optimizers/actions/runs/43711325491

Parent results: [clean epoch20 pass](../2026-11-15_fashion_mnist_mlp_experiment02_clean/RESULTS.md).

Files: [frozen protocol](Protocol.json), [runner](run.py), [artifact restoration](restore.py), [analysis](report.py), [checkpoint preservation](archive.py). Final reports and plots are written only inside this experiment folder. Parent MNIST and Fashion files stay unchanged.
