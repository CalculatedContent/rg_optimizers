"""Frozen-checkpoint cross-fitted validation temperature diagnostic; no training."""
import json, pathlib, csv, math
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from model import build
from run import sha, whash, P
D=pathlib.Path(__file__).resolve().parent
def ce(z,y): return float(np.mean(np.logaddexp(0,z)-y*z))
def fit(z,y):
    r=minimize_scalar(lambda t:ce(z*np.exp(-t),y),bounds=(-6,6),method="bounded")
    assert r.success
    return float(np.exp(r.x))
def metrics(z,y):
    p=expit(z); pred=z>0
    ece=sum(float(np.mean(mask))*abs(float(p[mask].mean()-y[mask].mean())) for i in range(15) if (mask:=((p>=i/15)&((p<(i+1)/15) if i<14 else (p<=1)))).any())
    return dict(ce=ce(z,y),accuracy=float(np.mean(pred==y)),auc=float(roc_auc_score(y,z)),brier=float(np.mean((p-y)**2)),ece15=ece)
def main():
    torch.set_num_threads(1)
    base=D.parent/"magic"; dp=base/"prepared/data.npz"; mf=json.loads((base/"prepared/manifest.json").read_text())
    assert sha(dp)==mf["prepared_sha256"]==P["primary_prepared_sha256"]
    with np.load(dp) as data:
        x,y,tr,va=(data[k] for k in ("x","y","train","val"))
    xx=torch.from_numpy(((x[va]-x[tr].mean(0))/np.maximum(x[tr].std(0),1e-12)).astype(np.float32)); yy=y[va].astype(int)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=3470).split(xx,yy))
    out=D/"calibration";out.mkdir(exist_ok=True)
    rows=[]
    for rp in sorted((D/"results/grid").glob("*/result.json")):
        r=json.loads(rp.read_text())
        for h in r["history"]:
            p=rp.parent/h["checkpoint_file"];assert sha(p)==h["checkpoint_sha256"]
            c=torch.load(p,map_location="cpu",weights_only=True); assert whash(c["model"])==h["weights_sha256"]
            m=build();m.load_state_dict(c["model"]);m.eval()
            with torch.inference_mode(): logits=torch.cat([m(b) for b in xx.split(1024)]).double().numpy()
            z=logits[:,1]-logits[:,0]; raw=metrics(z,yy)
            assert abs(raw["ce"]-h["metrics"]["validation"]["ce"])<1e-4
            oof=np.empty_like(z); ts=[]
            for a,b in folds:
                t=fit(z[a],yy[a]);ts.append(t);oof[b]=z[b]/t
            cal=metrics(oof,yy);assert raw["accuracy"]==cal["accuracy"]
            t=fit(z,yy); full=metrics(z/t,yy)
            assert abs(raw["auc"]-full["auc"])<1e-12
            rows.append(dict(case=r["case"],seed=r["seed"],step=h["step"],lr=r["config"]["lr"],alpha=h["spectral"]["raw"]["alpha"],temperature_full=t,temperature_folds=json.dumps(ts),logit_rms=float(np.sqrt(np.mean(z*z))),**{f"gain_{i}":float(m[i].gain.detach()) for i in (0,2,4)},**{"raw_"+k:v for k,v in raw.items()},**{"oof_"+k:v for k,v in cal.items()},**{"full_"+k:v for k,v in full.items()}))
            assert whash(m.state_dict())==h["weights_sha256"]
    assert rows
    with (out/"calibration.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/"summary.json").write_text(json.dumps(dict(n_checkpoints=len(rows),protocol="five-fold cross-fitted temperatures on original validation only; no neural updates; full-fit CE descriptive only; cross-fold AUC may vary",rows=rows),indent=2))
    lines=["# Frozen checkpoint temperature calibration","Cross-fitted validation CE is held-out within validation; no test access or neural updates.","","| Case | Seed | Step | Alpha | Raw CE | Cross-fit CE | Full T | Accuracy |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["step"]==20000:lines.append(f'| {r["case"]} | {r["seed"]} | {r["step"]} | {r["alpha"]} | {r["raw_ce"]:.4f} | {r["oof_ce"]:.4f} | {r["temperature_full"]:.3f} | {r["raw_accuracy"]:.4f} |')
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
