"""Restart only the missing tuned-graft seeds using the frozen archive recipe."""
import argparse, hashlib, importlib.metadata, json, math, os, shutil, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"short_test_adamw_graft"))
import run
import report
import torch
ARCHIVE=ROOT/"short_test_adamw_graft/results/2026-11-23_reference_lr_seed1340/raw_tuned_results.zip"
DESIGN="confirmation/rate_0.4/design.json"
RESULT="confirmation/rate_0.4/graft/seed_1340/result.json"
def archive_json(name):
    with zipfile.ZipFile(ARCHIVE) as z:
        return json.loads(z.read(name))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=range(1340,1345),required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--compatibility",action="store_true")
    a=p.parse_args()
    assert a.compatibility == (a.seed == 1340)
    run.runtime.DEVICE="cpu"
    run.runtime.initialize()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frozen=archive_json(DESIGN)
    for name,version in frozen["packages"].items():
        assert importlib.metadata.version(name)==version,(name,version,importlib.metadata.version(name))
    for path,digest in frozen["sources"].items():
        assert run.base.sha(ROOT/path)==digest,path
    assert frozen["recipes"]["graft"]["reference"]["peak_lr"]==0.4
    assert frozen["recipes"]["adamw"]["peak_lr"]==0.001
    assert frozen["schedule"]==run.SCHEDULE
    run.RATIO=0.4/run.ADAM["peak_lr"]
    cfg=run.design()
    cfg["recipes"]["graft"]["reference"]=dict(run.SGDM,peak_lr=0.4)
    cfg.update(phase="confirmation",study_seed=a.seed,available_splits=["test","train","validation"],frozen_archive_sha256=run.base.sha(ARCHIVE))
    data=run.base.load_data(a.data)
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=a.output/"design.json"
    run.previous.freeze(manifest,cfg)
    dh=run.base.sha(manifest)
    folder=a.output/"graft"/f"seed_{a.seed}"
    if a.compatibility:
        run.preflight(data)
        run.trial(a.output,data,"graft",a.seed,run.SCHEDULE,500,dh)
        rec=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
        old=next(x for x in archive_json(RESULT)["history"] if x["step"]==500)
        new=rec["progress"]["history"][-1]
        olda=report.extract_spectrum(old["alpha"])[0][report.ALPHA]
        newa=report.extract_spectrum(new["alpha"])[0][report.ALPHA]
        audit=dict(old_loss=old["validation_loss"],new_loss=new["validation_loss"],old_alpha=olda,new_alpha=newa)
        audit["pass"]=abs(audit["old_loss"]-audit["new_loss"])<=1e-6 and abs(olda-newa)<=1e-5
        run.base.write_json(a.output/"compatibility.json",audit)
        assert audit["pass"],audit
        return
    # Save the intermediate weights and full held-out metrics before continuing.
    if not (folder/"step_2000.pt").exists():
        run.trial(a.output,data,"graft",a.seed,run.SCHEDULE,2000,dh)
        rec=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
        assert rec["progress"]["step"]==2000
        torch.save(rec["model"],folder/"step_2000.pt")
        model,_,_,_=run.setup("graft",a.seed)
        model.load_state_dict(rec["model"])
        run.base.write_json(folder/"step_2000.json",run.measure(model,data,rec["progress"]["history"][-1]))
    run.trial(a.output,data,"graft",a.seed,run.SCHEDULE,20000,dh)
    rec=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
    assert rec["progress"]["step"]==20000
    torch.save(rec["model"],folder/"step_20000.pt")
    result=json.loads((folder/"result.json").read_text())
    assert [r["step"] for r in result["history"]]==list(range(0,20001,500))
    for item in result["history"]:
        summary,_=report.extract_spectrum(item["alpha"])
        assert summary["valid_matrix_fits"]==18,(item["step"],summary)
    run.base.write_json(folder/"verified.json",dict(seed=a.seed,steps=20000,measurements=41,matrices=18,reference_lr=0.4))
if __name__=="__main__":
    main()
