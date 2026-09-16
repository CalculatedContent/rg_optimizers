"""Regression for the actual legacy writer's omitted optimizer field."""
from pathlib import Path
import copy
import sys
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import diagnose_gradients as d
import run as driver


def test_missing_optimizer_keeps_fingerprint_identity():
    m={'profile':{'family':'adamw'},'recipe':'repository'}
    original=copy.deepcopy(m); fingerprint=driver.digest(m)
    assert d.manifest_optimizer(m)=='adamw'
    assert m==original and driver.digest(m)==fingerprint


def test_muon_without_top_level_field():
    assert d.manifest_optimizer({'profile':{'family':'muon'}})=='muon'


@pytest.mark.parametrize('manifest',[{}, {'optimizer':'muon','profile':{'family':'adamw'}},
                                     {'profile':{'family':'sgd'}}])
def test_ambiguous_or_inconsistent_identity_is_rejected(manifest):
    with pytest.raises(ValueError): d.manifest_optimizer(manifest)
