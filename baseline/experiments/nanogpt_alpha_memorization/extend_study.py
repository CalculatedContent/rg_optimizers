"""Safely extend completed alpha-memorization runs into a new study root.

Preserves model, optimizer, RNG, presentation counts, and attempted-update state from
checkpoint_latest.pt. The source study is read-only. The extension gets a new protocol
and fingerprint, and ordinary am_train.train() resumes from the copied state.
"""
from __future__ import annotations
import argparse, copy, json, shutil
from pathlib import Path
import torch
from am_data import digest
from am_runtime import atomic_json, atomic_save
from am_train import train

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--arm",required=True,choices=["adamw","muon","muon_qkclip"])
    ap.add_argument("--from-step",required=True,type=int)
    ap.add_argument("--to-step",required=True,type=int)
    ap.add_argument("--seeds",required=True,nargs="+",type=int)
    ap.add_argument("--checkpoint-every",type=int,default=1000)
    ap.add_argument("--behavior-every",type=int,default=1000)
    ap.add_argument("--device",choices=["mps","cpu","cuda"],default="mps")
    args=ap.parse_args()

    source=Path(args.root).resolve()
    if not source.is_dir(): raise SystemExit(f"Missing source study: {source}")
    cfg=json.loads((source/"protocol.json").read_text())
    if args.arm not in cfg.get("arms",[]): raise SystemExit(f"{args.arm} not declared in source protocol")
    if int(cfg["steps"])!=args.from_step: raise SystemExit(f"Source protocol steps={cfg['steps']}, expected {args.from_step}")
    if args.to_step<=args.from_step: raise SystemExit("--to-step must exceed --from-step")
    for seed in args.seeds:
        if seed not in cfg["seeds"]: raise SystemExit(f"Seed {seed} not in source protocol")

    ext_cfg=copy.deepcopy(cfg)
    ext_cfg["name"]=cfg.get("name","study")+f"_extension_to_{args.to_step}"
    ext_cfg["steps"]=args.to_step
    ext_cfg["seeds"]=list(args.seeds)
    ext_cfg["checkpoint_every"]=args.checkpoint_every
    ext_cfg["behavior_every"]=args.behavior_every
    ext_cfg["extension"]={"source_root":str(source),"source_step":args.from_step,"preserve_optimizer_rng":True}

    ext_root=source/"extensions"/f"{args.arm}_{args.from_step}_to_{args.to_step}"
    ext_root.mkdir(parents=True,exist_ok=True)
    protocol_file=ext_root/"protocol.json"
    if protocol_file.exists():
        if json.loads(protocol_file.read_text())!=ext_cfg: raise SystemExit(f"Existing extension protocol differs: {protocol_file}")
    else: atomic_json(protocol_file,ext_cfg)

    print(f"Source study (READ ONLY): {source}",flush=True)
    print(f"Extension study:          {ext_root}",flush=True)

    for seed in args.seeds:
        src=source/args.arm/f"seed_{seed}"
        dst=ext_root/args.arm/f"seed_{seed}"
        latest=src/"checkpoint_latest.pt"
        manifest_file=src/"manifest.json"
        if not latest.exists() or not manifest_file.exists(): raise SystemExit(f"Missing source checkpoint/manifest for seed {seed}")
        saved=torch.load(latest,map_location="cpu",weights_only=True)
        if int(saved["step"])!=args.from_step: raise SystemExit(f"Seed {seed} latest step={saved['step']}, expected {args.from_step}")

        dst.mkdir(parents=True,exist_ok=True)
        new_manifest=copy.deepcopy(json.loads(manifest_file.read_text()))
        new_manifest["protocol"]=ext_cfg
        new_manifest.pop("fingerprint",None)
        new_manifest["fingerprint"]=digest(new_manifest)

        dst_manifest=dst/"manifest.json"
        dst_latest=dst/"checkpoint_latest.pt"
        if not dst_manifest.exists() and not dst_latest.exists():
            boot=copy.deepcopy(saved)
            boot["fingerprint"]=new_manifest["fingerprint"]
            atomic_json(dst_manifest,new_manifest)
            atomic_save(dst_latest,boot)
            src_model=src/f"model_{args.from_step:08d}.pt"
            if src_model.exists(): shutil.copy2(src_model,dst/f"model_{args.from_step:08d}.pt")
            for name in ("data_manifest.json","dataset.json","canaries.json","audit.json"):
                p=src/name
                if p.exists(): shutil.copy2(p,dst/name)
        elif not (dst_manifest.exists() and dst_latest.exists()):
            raise SystemExit(f"Partial bootstrap for seed {seed}; inspect {dst}")

        print(f"\nExtending {args.arm} seed={seed}: {args.from_step} -> {args.to_step}",flush=True)
        result=train(ext_cfg,ext_root,args.arm,seed,args.device,resume=True)
        print(json.dumps(result),flush=True)

    print(f"\nExtension complete: {ext_root}",flush=True)
    print("Original study was not modified.",flush=True)

if __name__=="__main__":
    main()
