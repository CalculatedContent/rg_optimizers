# Full Matrix-Log RG optimizer

This experiment extends the scalar trace-log tracker to the full retained matrix-log geometry and wraps the same SGD + momentum baseline used by `baseline/notebooks/MNIST_MLP3_SGD_Momentum_Baseline.ipynb`.

For retained rank `m`, define `ell_i = log(D_R s_i^2 / ||W||_F^2)` and

`Phi_R(W) = (1/(2m)) sum_i ell_i^2 = (1/(2m)) ||log X_tilde_R||_F^2`.

The frozen-support gradient is

`N_full = (2/m) U_R diag(ell_i/s_i) V_R^T - (2 mean(ell)/||W||_F^2) W`.

For the completed base optimizer displacement `delta_W`, let `d=<N_full,delta_W>`. Only inward motion (`d<0`) is removed:

`delta_W_RG = delta_W - projection_strength * min(d/||N_full||^2,0) N_full`.

This is stronger than scalar Trace-Log: an anisotropic spectrum can have `sum ell_i = 0` while `Phi_R > 0`.

The slow WeightWatcher outer loop uses `detX_num` and `num_pl_spikes`; the cached working rank is their midpoint. The local correction is applied to the completed SGD displacement at a configurable step cadence.

## Notebook

`notebooks/MNIST_MLP3_SGD_Momentum_vs_FullMatrixLogRG.ipynb` mirrors the baseline recipe: `784 -> 512 -> 512 -> 10`, SGD + Nesterov, peak LR `0.05`, floor `5e-4`, two-epoch warm-up, momentum `0.90`, matrix weight decay `1e-4`, and gradient clipping `1.0`.

It first sweeps `projection_strength in {0.25,0.5,1.0}`, `max_correction_ratio in {0.05,0.10,0.25}`, and `apply_every_steps in {1,5,25}` on validation data, then runs the selected point over the same three baseline seeds. Test performance is monitoring-only.

## Tests

```bash
cd optimizers/full_matrix_log_rg
PYTHONPATH=. python -m unittest discover -s tests -v
```
