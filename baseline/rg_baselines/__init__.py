"""Clean MLP3/MNIST optimizer baselines."""
from .config import BaselineConfig
from .diagnostics import SpectralCheckpoint,measure_weightwatcher_checkpoint,spectral_metrics_from_esd
from .model import MLP3
from .muon import SGDMomentumMuon,zeropower_via_newton_schulz_5
from .optimizers import build_optimizer,optimizer_group_rows
from .plotting import plot_all
from .results import BaselineResult,validate_result
from .runner import run_baseline
__all__=["BaselineConfig","BaselineResult","MLP3","SGDMomentumMuon","SpectralCheckpoint",
"build_optimizer","measure_weightwatcher_checkpoint","optimizer_group_rows","plot_all","run_baseline",
"spectral_metrics_from_esd","validate_result","zeropower_via_newton_schulz_5"]
