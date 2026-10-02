#!/usr/bin/env bash
set -euo pipefail
AUDIT_NODE=ww-long-20260930-232752-node
AUDIT_PROJECT=tpu-builders-504820
AUDIT_ZONE=us-west4-a
EXPORT_COMMAND=$(cat <<'REMOTE'
python3 - <<'PY'
from pathlib import Path
import datetime, io, json, tarfile
base = Path("/mnt/disks/rg-data")
roots = [base / "muonclip-extended", base / "muonclip-spmd-long"]
if not roots[0].is_dir():
    raise SystemExit("Continuation root missing: run this on the training TPU VM.")
out = base / "muonclip_overnight_metrics.tgz"
info = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": [], "skipped": [], "checkpoints": [],
        "note": "Live per-file snapshot; not atomic across files. No weights included."}
with tarfile.open(str(out) + ".partial", "w:gz") as archive:
    def add(name, data):
        member = tarfile.TarInfo(name)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    for root in roots:
        if not root.exists():
            info["skipped"].append({"path": str(root), "reason": "root missing"})
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            name = str(path.relative_to(base))
            try:
                if path.suffix == ".pt":
                    stat = path.stat()
                    info["checkpoints"].append({"path": name, "bytes": stat.st_size,
                                                "mtime": stat.st_mtime})
                elif path.suffix in {".csv", ".yaml", ".yml", ".json"}:
                    data = path.read_bytes()
                    trimmed = 0
                    if path.suffix == ".csv" and data and not data.endswith(b"\n"):
                        end = data.rfind(b"\n") + 1
                        trimmed = len(data) - end
                        data = data[:end]
                    add(name, data)
                    info["files"].append({"path": name, "bytes": len(data),
                                          "trailing_bytes_removed": trimmed})
            except FileNotFoundError:
                info["skipped"].append({"path": name, "reason": "removed during export"})
    info["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    add("export_info.json", json.dumps(info, indent=2).encode())
Path(str(out) + ".partial").replace(out)
print(f"Exported {len(info['files'])} metadata files to {out}")
PY
REMOTE
)
gcloud compute tpus tpu-vm ssh "$AUDIT_NODE" --project="$AUDIT_PROJECT" --zone="$AUDIT_ZONE" --command="$EXPORT_COMMAND"
gcloud compute tpus tpu-vm scp "$AUDIT_NODE:/mnt/disks/rg-data/muonclip_overnight_metrics.tgz" "$HOME/muonclip_overnight_metrics.tgz" --project="$AUDIT_PROJECT" --zone="$AUDIT_ZONE"
echo "Upload $HOME/muonclip_overnight_metrics.tgz for analysis."
if ! cloudshell download "$HOME/muonclip_overnight_metrics.tgz"; then
  echo "Automatic download failed. Use Cloud Shell Download with: $HOME/muonclip_overnight_metrics.tgz"
fi
