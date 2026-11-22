"""Optional: regenerate replay inputs from seed ZIPs, evaluated logits, and labels."""
import argparse
from contextlib import contextmanager
import io
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd

from audit_measurements import COHORTS, KEY, ROOT
from rg_gap_audit import cohort_name


@contextmanager
def nested_zip(archive, name):
    with tempfile.TemporaryFile() as tmp:
        with archive.open(name) as source:
            shutil.copyfileobj(source, tmp, length=1024 * 1024)
        with zipfile.ZipFile(tmp) as inner:
            yield inner


def collect_spectra(archive, cohort, wanted, recovered):
    pattern = r'runs/b(\d+)_lr([\d.]+)/(\w+)/seed_(\d+)/spectra/epoch_(\d+)\.npz'
    for name in archive.namelist():
        if name.endswith(".zip"):
            with nested_zip(archive, name) as inner:
                collect_spectra(inner, cohort, wanted, recovered)
            continue
        match = re.fullmatch(pattern, name)
        if not match:
            continue
        batch, rate, arm, seed, epoch = match.groups()
        key = (cohort, int(seed), int(batch), float(rate), arm, int(epoch))
        if key in wanted:
            with np.load(io.BytesIO(archive.read(name))) as data:
                values = np.concatenate([data[layer] for layer in ("fc1","fc2","fc3")])
            if key in recovered:
                np.testing.assert_array_equal(values, recovered[key])
            recovered[key] = values


def export_spectra(archives, output):
    index = pd.read_csv(ROOT/"optimizer_checkpoints.csv").query('layer=="fc1"').set_index(["cohort",*KEY]).index
    wanted = set(index)
    recovered = {}
    for path in sorted(archives.glob("*.zip")):
        cohort = cohort_name(path.name)
        if cohort in COHORTS.values():
            with zipfile.ZipFile(path) as archive:
                collect_spectra(archive, cohort, wanted, recovered)
    if set(recovered) != wanted:
        raise ValueError(f"Missing {len(wanted-set(recovered))} paper checkpoints")
    for slug, cohort in COHORTS.items():
        keys = index[index.get_level_values("cohort")==cohort].sort_values()
        columns = {name:keys.get_level_values(name).to_numpy() for name in KEY}
        columns["arm"] = np.asarray(columns["arm"], dtype="U16")
        np.savez_compressed(output/f"spectra_{slug}.npz", eigenvalues=np.stack([recovered[k] for k in keys]), **columns)


def export_calibration(scores, labels, output):
    rows = []
    for run in pd.read_csv(ROOT/"confidence_run_pairs.csv").itertuples():
        dataset = "digits" if run.cohort.startswith("Digits") else "fashion"
        y = np.load(labels/f"{dataset}-y.npy")
        for epoch in (60,120):
            name = f"{run.cohort}__seed_{run.seed}__{run.setting}__{run.optimizer}__{epoch}.npy"
            logits = np.load(scores/name).astype(float)
            logits -= logits.max(axis=1, keepdims=True)
            confidence = 1/np.exp(logits).sum(axis=1)
            correct = logits.argmax(axis=1)==y
            selected = confidence >= 0.99
            rows.append([run.cohort,run.seed,run.setting,run.optimizer,epoch,len(y),
                         correct.sum(),selected.sum(),correct[selected].sum(),confidence[selected].sum()])
    columns = ["cohort","seed","setting","optimizer","epoch","n","correct","n99","correct99","confidence_sum99"]
    pd.DataFrame(rows,columns=columns).to_csv(output/"calibration_counts.csv",index=False)


def export_paired(archives, labels, output):
    for seed in [33001,33002]:
        path = archives/f"Fashion-AdamW-Paired120-Seed{seed}.zip"
        data = {"labels":np.load(labels/"fashion-y.npy")}
        with zipfile.ZipFile(path) as archive:
            [name] = [n for n in archive.namelist() if n.endswith(".zip")]
            with nested_zip(archive,name) as inner:
                prefix = f"runs/unaugmented/seed_{seed}/"
                for epoch in (60,120):
                    for split in ("test","validation"):
                        name = prefix + f"{split}_logits_{epoch:04d}.npz"
                        with np.load(io.BytesIO(inner.read(name))) as saved:
                            data[f"{split}_{epoch}"] = saved["logits"]
                            if split=="test":
                                np.testing.assert_array_equal(data["labels"],saved["labels"])
                    with np.load(io.BytesIO(inner.read(prefix+f"spectra/epoch_{epoch:03d}.npz"))) as saved:
                        for layer in ("fc1","fc2"):
                            data[f"{layer}_{epoch}"] = saved[layer]
        np.savez_compressed(output/f"fashion_scores_{seed}.npz", **data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archives","scores","labels","output"]:
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    export_spectra(args.archives,args.output)
    export_calibration(args.scores,args.labels,args.output)
    export_paired(args.archives,args.labels,args.output)
