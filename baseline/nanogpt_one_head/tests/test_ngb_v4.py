from __future__ import annotations

from pathlib import Path
import runpy

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "ngb"
    / "tests"
    / "test_ngb.py"
)
_NAMESPACE = runpy.run_path(str(_SOURCE))
for _name, _value in _NAMESPACE.items():
    if _name.startswith("test_"):
        globals()[_name] = _value
