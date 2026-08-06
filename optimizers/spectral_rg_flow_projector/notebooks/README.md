# MNIST optimizer benchmark notebooks

These notebooks run the same MLP3/MNIST spectral RG-flow experiment with three
different base optimizers:

- `MNIST_MLP3_AdamW_vs_SpectralRGFlowProjector.ipynb`
- `MNIST_MLP3_Adam_vs_SpectralRGFlowProjector.ipynb`
- `MNIST_MLP3_SGD_Momentum_vs_SpectralRGFlowProjector.ipynb`

Each notebook starts the baseline and wrapped models from the same state and
feeds them the same minibatches. The wrapper is applied only after the selected
base optimizer has completed its step.

The SGD notebook uses classical PyTorch momentum:

```python
torch.optim.SGD(
    parameters,
    lr=0.05,
    momentum=0.9,
    dampening=0.0,
    weight_decay=1e-4,
    nesterov=False,
)
```

It is not a Muon experiment. There is no orthognalized matrix update and no
Newton--Schulz iteration.

Run the notebooks from either the repository root or this optimizer folder.
They locate `rg_spectral_flow` automatically and write results beneath
`runs_mnist_spectral_rg_flow/<optimizer>/`.
