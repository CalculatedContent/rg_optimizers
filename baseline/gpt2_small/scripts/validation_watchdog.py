"""Bound validation phases even if an XLA operation never returns."""
import subprocess
import time


def wait_for_child(child, label, deadline):
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=20)
            raise RuntimeError(f'{label}: phase time limit exceeded; child stopped. Validation incomplete.')
        try:
            return child.wait(timeout=min(30, remaining))
        except subprocess.TimeoutExpired:
            print(f'WAIT {label}: process {child.pid} alive; completion not confirmed; '
                  f'{max(0, deadline-time.time()):.0f}s to phase cutoff', flush=True)
