"""CPU reproduction using the archived, unchanged Shakespeare training functions.

This convenience launcher was added for the paper source bundle. It did not
produce the primary GitHub cohort. It adds isolated outputs and guarded checkpoint
retention, without changing model, optimizer, gradient, RNG, or evaluation code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "modified"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["SHORT_GPT_DEVICE"] = "cpu"
sys.path.insert(0, str(SOURCE))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    manifest = json.loads((SOURCE / "frozen_source/source_manifest.json").read_text())
    for item in manifest["files"]:
        path = SOURCE / "frozen_source" / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"Frozen source changed: {item['path']}")
    expected = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
    if digest(SOURCE / "data/input.txt") != expected:
        raise ValueError("Corpus SHA256 differs")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true", help="Check source/corpus hashes without imports or training")
    p.add_argument("--preflight", action="store_true", help="Run the original local three-update replay preflight")
    p.add_argument("--arm", choices=("adamw", "sgdm", "graft"))
    p.add_argument("--seed", type=int, choices=range(1340, 1345))
    p.add_argument("--output", type=Path, help="New output root; never point at archived cohort directories")
    p.add_argument("--until-step", type=int, choices=(2000, 20000), default=20000)
    a = p.parse_args()
    manifest = verify_sources()
    if a.verify_only:
        print(json.dumps(dict(verified=True, files=len(manifest['files']), data=True)))
        return
    import run_additional_seed as launcher
    packages = launcher.verify()
    run = launcher.run
    run.runtime.DEVICE = "cpu"
    run.runtime.initialize()
    if a.preflight:
        run.RATIO = 400.0
        run.preflight(run.base.load_data(SOURCE / "data/input.txt"))
        return
    if a.arm is None or a.seed is None or a.output is None:
        p.error("Training requires --arm, --seed and --output")
    output = a.output.expanduser().resolve()
    if output == HERE or HERE in output.parents or output == SOURCE or SOURCE in output.parents:
        p.error("Choose an output outside data/protocol, preserving the bundled source")
    folder = output / a.arm / f"seed_{a.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with run.previous.lock(folder / "reproduction.lock"):
        run.RATIO = 400.0 if a.arm == "graft" else 100.0
        data = run.base.load_data(SOURCE / "data/input.txt")
        design = run.design()
        if a.arm == "graft":
            design["recipes"]["graft"]["reference"] = dict(run.SGDM, peak_lr=0.4)
            design.update(phase="confirmation", study_seed=a.seed,
                          available_splits=sorted(data),
                          driver_sha256=run.base.sha(SOURCE / "frozen_source/short_test_adamw_graft/reference_lr_study.py"))
        design_path = folder / "design.json"
        run.previous.freeze(design_path, design)
        identity = dict(arm=a.arm, seed=a.seed, design_sha256=run.base.sha(design_path))
        state_root = folder / "state"
        job = state_root / a.arm / f"seed_{a.seed}"
        last = job / "last.pt"

        def checkpoint_step(path):
            snapshot = run.torch.load(path, map_location="cpu", weights_only=False)
            if snapshot["identity"] != identity:
                raise ValueError(f"Checkpoint identity differs: {path}")
            return snapshot["progress"]["step"]

        current = checkpoint_step(last) if last.exists() else 0
        for target in (2000, 20000):
            if target > a.until_step:
                continue
            saved = folder / f"checkpoint_{target}.pt"
            if saved.exists() and checkpoint_step(saved) != target:
                raise ValueError(f"Retained checkpoint has incorrect update count: {saved}")
            if current > target:
                if not saved.exists():
                    raise ValueError(f"Cannot reconstruct missing checkpoint {target} from update {current}")
                continue
            if current < target:
                run.trial(state_root, data, a.arm, a.seed, run.SCHEDULE, target, identity["design_sha256"])
                current = checkpoint_step(last)
            if current != target:
                raise ValueError(f"Expected update {target}, got {current}; archive not changed")
            if not saved.exists():
                shutil.copyfile(last, saved)
        paths = [folder / f"checkpoint_{t}.pt" for t in (2000, 20000)
                 if (folder / f"checkpoint_{t}.pt").exists()]
        archive = folder / "checkpoints.zip"
        temporary = folder / "checkpoints.zip.tmp"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
            for path in paths:
                z.write(path, path.name)
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:
                raise ValueError("Checkpoint ZIP integrity failure")
        temporary.replace(archive)
        run.base.write_json(folder / "checkpoint_hashes.json",
                            {path.name: dict(sha256=digest(path), bytes=path.stat().st_size,
                                             step=checkpoint_step(path)) for path in paths})
        run.base.write_json(folder / "execution.json", dict(
            seed=a.seed, arm=a.arm, execution_origin="independent_local_reproduction",
            recorded_at=datetime.now(timezone.utc).isoformat(), packages=packages,
            platform=platform.platform(), runtime=run.runtime.environment(),
            torch_build=run.torch.__config__.show(),
            torch_cpu_capability=run.torch.backends.cpu.get_cpu_capability(),
            numerical_threads=run.torch.get_num_threads(),
            source_manifest=manifest, wrapper_sha256=digest(__file__),
            note="Independent rerun; do not pool with published GitHub confirmation seeds."))
        print(json.dumps(dict(output=str(folder), completed_updates=current,
                              retained_checkpoints=[x.name for x in paths])))


if __name__ == "__main__":
    main()
