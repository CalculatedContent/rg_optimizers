# Experiment 01: complete results

All 135 runs reached epoch 120. Every table retains all nine preset batch/rate settings.

## Average over the nine settings

Each entry is a mean [95% Student t interval]. Average the nine outcomes within each seed first, then use the five seed averages. The test set is fixed; these intervals describe seed variability, are not multiplicity-adjusted, and do not include uncertainty from resampling the dataset.

| Arm | FC1 α | FC2 α | Mean hidden α | Mean hidden \|α−2\| | Test accuracy (%) |
|---|---:|---:|---:|---:|---:|
| AdamW | 1.818 [1.778, 1.858] | 1.801 [1.796, 1.807] | 1.810 [1.788, 1.832] | 0.192 [0.172, 0.211] | 98.214 [98.189, 98.238] |
| Magnitude graft | 2.163 [2.089, 2.237] | 1.928 [1.902, 1.954] | 2.045 [2.009, 2.081] | 0.170 [0.128, 0.211] | 88.620 [88.548, 88.692] |
| SGD + momentum | 2.110 [2.097, 2.122] | 1.991 [1.985, 1.997] | 2.050 [2.044, 2.056] | 0.194 [0.191, 0.197] | 98.317 [98.298, 98.336] |

## Paired average effects

Positive Δα means upward movement. Negative Δmean|α−2| means closer to 2. Positive Δaccuracy means higher accuracy. These are distinct outcomes.

| Treatment − reference | ΔFC1 α | ΔFC2 α | Δmean \|α−2\| | Δaccuracy (pp) |
|---|---:|---:|---:|---:|
| Magnitude graft − AdamW | +0.345 [+0.269, +0.421] | +0.127 [+0.099, +0.154] | -0.022 [-0.075, +0.031] | -9.594 [-9.689, -9.498] |
| SGD + momentum − AdamW | +0.292 [+0.248, +0.336] | +0.190 [+0.178, +0.201] | +0.002 [-0.020, +0.025] | +0.103 [+0.069, +0.137] |
| Magnitude graft − SGD + momentum | +0.053 [-0.024, +0.130] | -0.063 [-0.087, -0.039] | -0.024 [-0.063, +0.015] | -9.697 [-9.772, -9.622] |

## All settings: magnitude graft minus native AdamW

Each row uses the same five paired seeds. Entries are paired mean differences [95% t interval].

| Batch | LR multiplier | ΔFC1 α | ΔFC2 α | Δmean \|α−2\| | Δaccuracy (pp) |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.5 | +0.253 [+0.141, +0.365] | +0.173 [+0.152, +0.193] | -0.112 [-0.169, -0.056] | +0.172 [+0.079, +0.265] |
| 128 | 1 | -0.065 [-0.161, +0.031] | +0.259 [+0.236, +0.282] | -0.079 [-0.125, -0.033] | +0.026 [-0.097, +0.149] |
| 128 | 2 | +0.514 [-0.275, +1.304] | -0.280 [-0.527, -0.033] | +0.250 [-0.079, +0.578] | -87.074 [-87.187, -86.961] |
| 256 | 0.5 | +0.555 [+0.445, +0.665] | +0.183 [+0.176, +0.191] | +0.020 [-0.025, +0.066] | +0.020 [-0.083, +0.123] |
| 256 | 1 | +0.293 [+0.256, +0.330] | +0.148 [+0.132, +0.163] | -0.145 [-0.174, -0.116] | +0.124 [+0.034, +0.214] |
| 256 | 2 | -0.059 [-0.143, +0.025] | +0.201 [+0.135, +0.267] | -0.065 [-0.106, -0.025] | +0.080 [-0.118, +0.278] |
| 512 | 0.5 | +0.741 [+0.693, +0.790] | +0.175 [+0.154, +0.196] | +0.095 [+0.065, +0.125] | +0.066 [-0.032, +0.164] |
| 512 | 1 | +0.622 [+0.561, +0.683] | +0.150 [+0.129, +0.172] | -0.001 [-0.030, +0.028] | +0.144 [-0.059, +0.347] |
| 512 | 2 | +0.250 [+0.114, +0.386] | +0.131 [+0.111, +0.150] | -0.160 [-0.239, -0.080] | +0.098 [-0.040, +0.236] |

## Per-setting arm means

Entries below are mean ± seed SD, not confidence intervals. FC3 has only ten eigenvalues and is exploratory; it is excluded from hidden-layer means and distance metrics.

| Batch | LR | Arm | FC1 α | FC2 α | FC3 α (exploratory) | Mean \|α−2\| | Accuracy (%) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 128 | 0.5 | AdamW | 1.855 ± 0.109 | 1.764 ± 0.007 | 3.272 ± 0.397 | 0.198 ± 0.037 | 98.186 ± 0.038 |
| 128 | 0.5 | Magnitude graft | 2.108 ± 0.020 | 1.937 ± 0.011 | 3.171 ± 0.117 | 0.085 ± 0.010 | 98.358 ± 0.093 |
| 128 | 0.5 | SGD + momentum | 2.077 ± 0.023 | 1.904 ± 0.009 | 2.959 ± 0.149 | 0.086 ± 0.012 | 98.514 ± 0.046 |
| 128 | 1 | AdamW | 1.873 ± 0.082 | 1.759 ± 0.015 | 3.666 ± 0.181 | 0.184 ± 0.041 | 98.384 ± 0.067 |
| 128 | 1 | Magnitude graft | 1.808 ± 0.037 | 2.018 ± 0.012 | 3.122 ± 0.295 | 0.105 ± 0.019 | 98.410 ± 0.074 |
| 128 | 1 | SGD + momentum | 1.666 ± 0.016 | 2.014 ± 0.018 | 2.932 ± 0.272 | 0.175 ± 0.008 | 98.468 ± 0.076 |
| 128 | 2 | AdamW | 1.801 ± 0.070 | 1.995 ± 0.015 | 3.419 ± 0.354 | 0.105 ± 0.034 | 98.424 ± 0.091 |
| 128 | 2 | Magnitude graft | 2.315 ± 0.606 | 1.716 ± 0.202 | 1.702 ± 0.337 | 0.355 ± 0.282 | 11.350 ± 0.000 |
| 128 | 2 | SGD + momentum | 1.732 ± 0.067 | 2.192 ± 0.026 | 2.565 ± 0.224 | 0.230 ± 0.026 | 97.940 ± 0.028 |
| 256 | 0.5 | AdamW | 1.834 ± 0.079 | 1.779 ± 0.012 | 3.066 ± 0.132 | 0.194 ± 0.041 | 98.094 ± 0.029 |
| 256 | 0.5 | Magnitude graft | 2.389 ± 0.017 | 1.962 ± 0.011 | 2.964 ± 0.229 | 0.214 ± 0.006 | 98.114 ± 0.079 |
| 256 | 0.5 | SGD + momentum | 2.492 ± 0.016 | 2.010 ± 0.024 | 2.930 ± 0.254 | 0.255 ± 0.012 | 98.256 ± 0.024 |
| 256 | 1 | AdamW | 1.783 ± 0.038 | 1.762 ± 0.008 | 3.457 ± 0.260 | 0.228 ± 0.017 | 98.220 ± 0.041 |
| 256 | 1 | Magnitude graft | 2.075 ± 0.019 | 1.910 ± 0.008 | 3.117 ± 0.284 | 0.083 ± 0.008 | 98.344 ± 0.056 |
| 256 | 1 | SGD + momentum | 2.067 ± 0.022 | 1.900 ± 0.005 | 2.964 ± 0.171 | 0.083 ± 0.012 | 98.446 ± 0.023 |
| 256 | 2 | AdamW | 1.864 ± 0.069 | 1.788 ± 0.025 | 3.343 ± 0.349 | 0.174 ± 0.024 | 98.382 ± 0.074 |
| 256 | 2 | Magnitude graft | 1.806 ± 0.072 | 1.989 ± 0.031 | 3.088 ± 0.352 | 0.108 ± 0.037 | 98.462 ± 0.144 |
| 256 | 2 | SGD + momentum | 1.647 ± 0.019 | 2.029 ± 0.010 | 2.737 ± 0.177 | 0.191 ± 0.011 | 98.474 ± 0.056 |
| 512 | 0.5 | AdamW | 1.812 ± 0.038 | 1.802 ± 0.016 | 2.944 ± 0.176 | 0.193 ± 0.021 | 97.968 ± 0.020 |
| 512 | 0.5 | Magnitude graft | 2.554 ± 0.012 | 1.977 ± 0.012 | 2.828 ± 0.262 | 0.288 ± 0.008 | 98.034 ± 0.076 |
| 512 | 0.5 | SGD + momentum | 2.789 ± 0.019 | 1.989 ± 0.009 | 2.747 ± 0.167 | 0.401 ± 0.013 | 98.044 ± 0.099 |
| 512 | 1 | AdamW | 1.763 ± 0.042 | 1.799 ± 0.021 | 3.029 ± 0.238 | 0.219 ± 0.025 | 98.032 ± 0.076 |
| 512 | 1 | Magnitude graft | 2.385 ± 0.017 | 1.950 ± 0.017 | 2.933 ± 0.115 | 0.218 ± 0.009 | 98.176 ± 0.098 |
| 512 | 1 | SGD + momentum | 2.472 ± 0.019 | 1.993 ± 0.009 | 3.030 ± 0.282 | 0.241 ± 0.011 | 98.246 ± 0.053 |
| 512 | 2 | AdamW | 1.777 ± 0.101 | 1.761 ± 0.012 | 3.286 ± 0.260 | 0.231 ± 0.051 | 98.232 ± 0.100 |
| 512 | 2 | Magnitude graft | 2.027 ± 0.043 | 1.892 ± 0.014 | 2.774 ± 0.365 | 0.071 ± 0.022 | 98.330 ± 0.070 |
| 512 | 2 | SGD + momentum | 2.049 ± 0.017 | 1.886 ± 0.007 | 2.933 ± 0.178 | 0.082 ± 0.007 | 98.464 ± 0.054 |

## Low-accuracy outcomes

The following is a descriptive count with an explicit threshold; no run is excluded from any estimate.

| Arm | Runs with test accuracy <50% | Minimum test accuracy | Maximum test accuracy |
|---|---:|---:|---:|
| AdamW | 0/45 | 97.95% | 98.58% |
| Magnitude graft | 5/45 | 11.35% | 98.65% |
| SGD + momentum | 0/45 | 97.91% | 98.58% |

## Fit support and interpretation

268 of 270 final hidden-layer fits satisfy the prespecified support rule: at least 50 eigenvalues, at least 20 tail eigenvalues, KS D ≤0.2. [Common-support comparisons](results/common_support_pairs.csv) retain the exact paired eligible layers.

WeightWatcher 0.7.7 uses its default tail selection, with no α-targeted cutoff. [Layer measurements](results/all_layer_measurements.csv) include fit support, tail size, KS D, norms, and fixed-rank estimates. [Accuracy correlations](results/accuracy_correlations.csv) are descriptive within-arm correlations; the 45 setting/seed points are not 45 independent seeds.

This intervention tests the consequence of changing update magnitude while retaining Adam direction and local moment history. Spectral shifts alone do not establish an RG Jacobian condition or a universal optimizer advantage. Coupling/Jacobian calculations can be performed from the saved states in subsequent analysis.

