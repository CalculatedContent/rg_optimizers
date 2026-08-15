from __future__ import annotations

"""Instrumented launcher for Muon/MuonClip update-SVD frame capture."""

import sys
from typing import Any, Sequence

from .muon_svd_capture import MuonUpdateSVDCapture

_ACTIVE_RECORDER: MuonUpdateSVDCapture | None = None
_INSTALLED = False


def _requested_optimizer(argv: Sequence[str]) -> str | None:
    for index, value in enumerate(argv):
        if value == "--optimizer" and index + 1 < len(argv):
            return str(argv[index + 1]).lower()
        if value.startswith("--optimizer="):
            return value.split("=", 1)[1].lower()
    return None


def install_muon_svd_capture() -> None:
    """Install opt-in capture without changing Muon/MuonClip update kernels."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import engine as engine_module
    from . import training as training_module
    from . import train_loop as train_loop_module

    original_execute = train_loop_module.execute_training_loop
    original_optimizer_step = train_loop_module.optimizer_step
    original_worker_module = training_module._mps_worker_module

    def execute_training_loop(**kwargs: Any):
        global _ACTIVE_RECORDER
        recorder = MuonUpdateSVDCapture.from_training_state(
            cfg=kwargs["cfg"],
            model=kwargs["model"],
            handles=kwargs["handles"],
            optimizer_name=kwargs["optimizer_name"],
            run_dir=kwargs["run_dir"],
            device=kwargs["device"],
        )
        if recorder is not None:
            recorder.next_step = int(kwargs["start_step"]) + 1
        previous = _ACTIVE_RECORDER
        _ACTIVE_RECORDER = recorder
        try:
            return original_execute(**kwargs)
        finally:
            _ACTIVE_RECORDER = previous

    def optimizer_step(handles) -> None:
        recorder = _ACTIVE_RECORDER
        active = False
        if recorder is not None:
            active = recorder.capture_before(step=int(recorder.next_step))
        try:
            original_optimizer_step(handles)
        except Exception:
            if active and recorder is not None:
                recorder.abort()
            raise
        if recorder is not None:
            if active:
                recorder.capture_after()
            recorder.next_step = int(recorder.next_step) + 1

    def worker_module(optimizer_name: str) -> str:
        if str(optimizer_name) in {"muon", "muon_clip"}:
            return "rg_nanogpt_one_head.muon_svd_runner"
        return original_worker_module(optimizer_name)

    train_loop_module.execute_training_loop = execute_training_loop
    train_loop_module.optimizer_step = optimizer_step
    # engine imported execute_training_loop by value, so update that binding too.
    engine_module.execute_training_loop = execute_training_loop
    training_module._mps_worker_module = worker_module

    _INSTALLED = True


def main() -> None:
    requested = _requested_optimizer(sys.argv[1:])
    if requested == "muon_clip":
        # Install the repository's existing MuonClip implementation only for a
        # MuonClip run. Ordinary Muon therefore retains its original attention
        # forward path and optimizer partition exactly.
        from .muonclip import install_muonclip_extension

        install_muonclip_extension()
    install_muon_svd_capture()
    from .training import main as training_main

    training_main()


if __name__ == "__main__":
    main()
