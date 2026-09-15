#!/usr/bin/env python3
"""Compare matched AdamW and qualified high-alpha Muon membership analyses.

This is descriptive for one paired seed. It refuses a Muon run whose saved
spectra fail the alpha gate and verifies that the two source runs used the same
seed, data identity, model initialization, condition, stage, and hardware.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_membership(path: Path):
    path = Path(path).expanduser().resolve()
    table_path = path / "membership_stats.csv"
    meta_path = path / "membership_metadata.json"
    if not table_path.is_file() or not meta_path.is_file():
        raise ValueError(f"not a membership output directory: {path}")
    meta = json.loads(meta_path.read_text())
    audit = Path(meta["source_audit"])
    protocol = json.loads((audit / "protocol.json").read_text())
    run_dir = Path(protocol["source_run"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return path, pd.read_csv(table_path), meta, protocol, manifest


def compare(adamw_dir, muon_dir, gate_path, output):
    a_dir, a, a_meta, a_protocol, am = load_membership(Path(adamw_dir))
    m_dir, m, m_meta, m_protocol, mm = load_membership(Path(muon_dir))
    if am.get("profile", {}).get("family") != "adamw":
        raise ValueError("first analysis is not AdamW")
    if mm.get("profile", {}).get("family") != "muon":
        raise ValueError("second analysis is not Muon")
    required = ["condition", "stage", "seed", "data_sha256", "initial_model_sha256", "source_model"]
    mismatches = [k for k in required if am.get(k) != mm.get(k)]
    if am.get("device") != mm.get("device"):
        mismatches.append("device")
    if mismatches:
        raise ValueError(f"runs are not a matched optimizer pair: {mismatches}")
    gate = json.loads(Path(gate_path).expanduser().resolve().read_text())
    if not gate.get("passed"):
        raise ValueError("Muon run did not pass the high-alpha gate")
    if str(Path(gate.get("run_dir", "")).resolve()) != str(Path(m_protocol["source_run"]).resolve()):
        raise ValueError("alpha gate does not refer to the Muon source run")

    cols = ["step", "baseline_corrected_membership_effect", "auc_baseline_corrected",
            "raw_fresh_minus_seen_nll"]
    left = a[cols].rename(columns={c: f"adamw_{c}" for c in cols if c != "step"})
    right = m[cols].rename(columns={c: f"muon_{c}" for c in cols if c != "step"})
    joined = left.merge(right, on="step", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("no common saved checkpoints")
    joined["muon_minus_adamw_membership_effect"] = (
        joined["muon_baseline_corrected_membership_effect"] - joined["adamw_baseline_corrected_membership_effect"])
    joined["muon_minus_adamw_auc"] = joined["muon_auc_baseline_corrected"] - joined["adamw_auc_baseline_corrected"]
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    joined.to_csv(out / "optimizer_membership_comparison.csv", index=False)
    final = joined.iloc[-1]
    lines = [
        "# AdamW versus qualified high-alpha Muon membership comparison",
        "",
        "One matched seed; descriptive optimizer contrast only. The Muon arm passed the pre-registered spectral gate, but this is not a multi-seed optimizer inference.",
        "",
        f"AdamW source: `{a_protocol['source_run']}`",
        f"Muon source: `{m_protocol['source_run']}`",
        f"Muon gate threshold: {gate['threshold']:.4f}; minimum observed raw/clip alpha: {gate['min_alpha_both_fits']:.4f}",
        "",
        "## Final common checkpoint",
        "",
        f"- step: {int(final.step)}",
        f"- AdamW corrected membership effect: {final.adamw_baseline_corrected_membership_effect:.6f} nat/token",
        f"- Muon corrected membership effect: {final.muon_baseline_corrected_membership_effect:.6f} nat/token",
        f"- Muon minus AdamW effect: {final.muon_minus_adamw_membership_effect:.6f} nat/token",
        f"- AdamW corrected AUC: {final.adamw_auc_baseline_corrected:.4f}",
        f"- Muon corrected AUC: {final.muon_auc_baseline_corrected:.4f}",
        f"- Muon minus AdamW AUC: {final.muon_minus_adamw_auc:.4f}",
        "",
        "See `optimizer_membership_comparison.csv` for the full matched trajectory.",
    ]
    (out / "comparison_report.md").write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adamw", required=True, help="AdamW membership output directory")
    parser.add_argument("--muon", required=True, help="Muon membership output directory")
    parser.add_argument("--muon-gate", required=True, help="full_alpha_gate.json from Muon control")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        out = compare(args.adamw, args.muon, args.muon_gate, args.output)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Finished. Read: {out / 'comparison_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
