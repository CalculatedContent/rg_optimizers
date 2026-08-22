"""Execute every code cell without Jupyter to provide a lightweight CI smoke test."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/portfolio-pgd-matplotlib")

ROOT = Path(__file__).resolve().parents[1]


def execute(path: Path) -> None:
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(previous)
    print(f"PASS {path.name}")


def main() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        execute(path)


if __name__ == "__main__":
    main()
