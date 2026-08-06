"""Shared color-blind-safe plotting conventions for all baseline notebooks."""

from __future__ import annotations

# Okabe-Ito-inspired palette.  These colors are fixed across every notebook so
# the same layer/quantity never changes meaning between optimizers.
PERFORMANCE_COLORS = {
    "train": "#0072B2",
    "test": "#D55E00",
}
LAYER_COLORS = {
    "fc1": "#0072B2",
    "fc2": "#E69F00",
    "fc3": "#009E73",
}
LAYER_MARKERS = {
    "fc1": "o",
    "fc2": "s",
    "fc3": "^",
}
METRIC_COLORS = {
    "detX_num": "#CC79A7",
    "num_pl_spikes": "#56B4E9",
    "ERG_gap": "#D55E00",
    "m_midpoint": "#0072B2",
    "trace_log_midpoint_per_eval": "#D55E00",
    "trace_log_midpoint_total": "#CC79A7",
    "mean_gradient_norm_before_clip": "#0072B2",
    "max_gradient_norm_before_clip": "#D55E00",
    "parameter_l2_norm": "#009E73",
    "train_time_sec": "#0072B2",
    "evaluation_time_sec": "#E69F00",
    "weightwatcher_time_sec": "#CC79A7",
}
SEED_TRACE_ALPHA = 0.16
CI_BAND_ALPHA = 0.18
MEAN_LINE_WIDTH = 2.2
SEED_LINE_WIDTH = 0.9
ERROR_CAP_SIZE = 3.0
