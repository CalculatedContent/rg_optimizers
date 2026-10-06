import importlib.util
from pathlib import Path
import subprocess

import pytest

spec=importlib.util.spec_from_file_location('watchdog',Path(__file__).parents[1]/'scripts/validation_watchdog.py')
watchdog=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class Child:
    pid=123
    def __init__(self,ignores_term=False):
        self.ignores_term=ignores_term; self.calls=[]
    def terminate(self): self.calls.append('terminate')
    def kill(self): self.calls.append('kill')
    def wait(self,timeout):
        if self.ignores_term and self.calls==['terminate']:
            raise subprocess.TimeoutExpired('test-child',timeout)
        return 0


@pytest.mark.parametrize('ignores_term',[False,True])
def test_phase_cutoff_terminates_child(ignores_term,monkeypatch):
    monkeypatch.setattr(watchdog.time,'time',lambda:100)
    child=Child(ignores_term)
    with pytest.raises(RuntimeError,match='Validation incomplete'):
        watchdog.wait_for_child(child,'adamw through step 4',99)
    assert child.calls==(['terminate','kill'] if ignores_term else ['terminate'])


def test_successful_child_is_not_stopped(monkeypatch):
    monkeypatch.setattr(watchdog.time,'time',lambda:100)
    child=Child()
    assert watchdog.wait_for_child(child,'adamw',200)==0
    assert child.calls==[]
