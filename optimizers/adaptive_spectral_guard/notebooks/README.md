# AdaptiveSpectralGuard notebooks

Run this notebook for the current experiment:

```text
MNIST_MLP3_AdamW_vs_AdaptiveSpectralGuard_StabilizedV2_30Epochs.ipynb
```

It is fail-fast: it clears cached package modules, imports the local repository
copy, checks `STABILIZED_V2_API`, runs a low-confidence alpha-below-two
preflight, and validates the controller after every epoch.

The following notebook is the original V1 experiment and is retained only for
comparison:

```text
MNIST_MLP3_AdamW_vs_AdaptiveSpectralGuard_30Epochs.ipynb
```
