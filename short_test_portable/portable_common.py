"""Reuse frozen CPU data/evaluation semantics with explicit device transfers."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'short_test'))
import common as original
from common import *  # Re-export frozen data/model/RNG/WeightWatcher interfaces.
import runtime

class EvaluationModel:
    """Full-tail evaluation stays identical; return metrics' tensors to CPU."""
    def __init__(self, model):
        self.model = model
    @property
    def training(self):
        return self.model.training
    def eval(self):
        self.model.eval()
    def train(self, mode=True):
        self.model.train(mode)
    def __call__(self, x, y):
        logits, loss = self.model(x.to(runtime.DEVICE), y.to(runtime.DEVICE))
        return logits.cpu(), loss.cpu()

def full_eval(model, data):
    return original.full_eval(EvaluationModel(model), data)

def sampled_loss(model, data, seed=10001):
    return original.sampled_loss(EvaluationModel(model), data, seed)

def spectrum(state):
    return original.spectrum(runtime.cpu_tree(state))
