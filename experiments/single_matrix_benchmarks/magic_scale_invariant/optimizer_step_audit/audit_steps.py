"""Counterfactual single optimizer step on disposable copies of immutable 20k states.
No resumed training, no test evaluation, no new model selection.
"""
import argparse, csv, hashlib, json, math, pathlib, sys, traceback, statistics
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=pathlib.Path(__file__).resolve().parent
SEEDS=(3470,3471)
CONF=tuple(range(3472,3477))
CASES=("s0","s1","s2","s3","a0","a1","a2")
TRACKS=("sgdm_wd0","adamw_wd0")
STEPS=(.25,4.)
PRIMARY="09dbaa4e095ff21fc8b1e130798d247b75c15913"
NORMALIZED="e6ec5fb97bc3db0130937dd49e710399dbc1dbec"
DATA="c42ba2691997b884dae5d738bcb2b5bb5a5b07fbca0f0c83afae12ce76812d73"
MANIFEST="26cc2d30485301f72087e88a59c0b044f374b4987abe1e176a308a32e2cc2473"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def whash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def norm(x):return float(torch.linalg.vector_norm(x.double()))
def cos(x,y):
    den=norm(x)*norm(y)
    return float((x.double()*y.double()).sum())/den if den>0 else None
def make_model(arch,build):
    return build() if arch=="normalized" else nn.Sequential(nn.Linear(10,784),nn.ReLU(),nn.Linear(784,784),nn.ReLU(),nn.Linear(784,2))
def opt_for(m,cfg):
    if cfg["optimizer"]=="sgdm":
        return torch.optim.SGD(m.parameters(),lr=cfg["lr"],momentum=cfg["momentum"],weight_decay=0)
    return torch.optim.AdamW(m.parameters(),lr=cfg["lr"],betas=tuple(cfg["betas"]),eps=cfg["eps"],weight_decay=0)
def batch(data,record,state):
    idx=data["train"] if record["phase"]=="grid" else data["outer_train"]
    x=data["x"][idx];y=data["y"][idx]
    mean=x.mean(0);sd=np.maximum(x.std(0),1e-12)
    np.testing.assert_allclose(mean,record["normalization"]["mean"],rtol=0,atol=1e-12)
    np.testing.assert_allclose(sd,record["normalization"]["sd"],rtol=0,atol=1e-12)
    n=len(idx);offset=int(state["offset"]);perm=state["permutation"]
    assert 0<=offset<=n and len(perm)==n
    gen=torch.Generator();gen.set_state(state["batch_rng"])
    if offset==n:perm=torch.randperm(n,generator=gen);offset=0
    ids=perm[offset:offset+256]
    return torch.from_numpy(((x[ids.numpy()]-mean)/sd).astype(np.float32)),torch.from_numpy(y[ids.numpy()].astype(np.int64))
def trial(arch,build,cfg,state,xx,yy,b,comp,clip):
    m=make_model(arch,build);m.load_state_dict(state["model"]);m.train()
    opt=opt_for(m,cfg);opt.load_state_dict(state["optimizer"])
    # load_state_dict can alias tensors on CPU. Clone all mutable states so trials cannot contaminate one another.
    for s in opt.state.values():
        for key,value in list(s.items()):
            if isinstance(value,torch.Tensor):s[key]=value.clone()
    assert len(opt.param_groups)==1
    group=opt.param_groups[0]
    assert group["weight_decay"]==0 and group["lr"]==cfg["lr"]
    w=m[2].weight
    with torch.no_grad():w.mul_(b)
    if b!=1:
        assert arch=="normalized"
        s=opt.state[w]
        if cfg["optimizer"]=="sgdm":s["momentum_buffer"].div_(b)
        else:s["exp_avg"].div_(b);s["exp_avg_sq"].div_(b*b)
        if comp:
            new=dict(group)
            group["params"]=[p for p in group["params"] if p is not w]
            new["params"]=[w]
            new["lr"]=cfg["lr"]*(b*b if cfg["optimizer"]=="sgdm" else b)
            if cfg["optimizer"]=="adamw":new["eps"]=cfg["eps"]/b
            opt.add_param_group(new)
    before={name:p.detach().clone() for name,p in m.named_parameters()}
    opt.zero_grad(set_to_none=True)
    logits=m(xx);loss=F.cross_entropy(logits,yy);loss.backward()
    g=w.grad.detach().clone()
    allnorm=math.sqrt(sum(norm(p.grad)**2 for p in m.parameters() if p.grad is not None))
    clipfactor=min(1.,1./(allnorm+1e-6)) if clip else 1.
    if clip:torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    d=w.detach()-before["2.weight"]
    w0=before["2.weight"].double();v=d.double();r=norm(w0);u=w0/r
    radial=float((u*v).sum());tangent=v-radial*u
    angle=math.atan2(norm(tangent),r+radial)
    metrics={"loss":float(loss.detach()),"global_grad_norm":allnorm,"clip_factor":clipfactor,
      "fc2_raw_norm":r,"fc2_gradient_norm":norm(g),"fc2_gradient_radial_cos":cos(w0,g),
      "fc2_update_radial_cos":cos(w0,v),"fc2_update_relative_norm":norm(v)/r,
      "fc2_tangential_step":norm(tangent)/r,"fc2_angle_rad":angle,
      "fc2_gradient_update_descent_cos":cos(-g,v),
      "fc2_effective_lr":next(q["lr"] for q in opt.param_groups if any(p is w for p in q["params"]))}
    for i in (0,2,4):
        wi=m[i].weight;delta=wi.detach()-before[f"{i}.weight"]
        metrics[f"fc{i//2+1}_update_norm"]=norm(delta)
        if arch=="normalized":
            gi=m[i].gain;old=before[f"{i}.gain"]
            metrics[f"fc{i//2+1}_gain"]=float(old)
            metrics[f"fc{i//2+1}_gain_delta"]=float(gi.detach()-old)
    s=opt.state[w]
    if cfg["optimizer"]=="adamw":
        step=float(s["step"]);beta2=cfg["betas"][1]
        eps=next(q["eps"] for q in opt.param_groups if any(p is w for p in q["params"]))
        pre=1/(s["exp_avg_sq"].double().sqrt()/math.sqrt(1-beta2**step)+eps)
        qs=torch.quantile(pre.flatten(),torch.tensor([.01,.5,.99],dtype=torch.float64))
        metrics.update(precond_p01=float(qs[0]),precond_p50=float(qs[1]),precond_p99=float(qs[2]),precond_p99_over_p01=float(qs[2]/qs[0]),moment_radial_cos=cos(w0,s["exp_avg"]))
    else:metrics["moment_radial_cos"]=cos(w0,s["momentum_buffer"])
    assert math.isfinite(metrics["loss"]) and math.isfinite(angle)
    return metrics,d.detach().clone(),logits.detach().clone(),g
def run_case(root,arch,phase,case,seed,data,build):
    folder=root/"results"/phase/f"{case}-{seed}"
    report=json.loads((folder/"result.json").read_text());cfg=report["config"]
    h=report["history"][-1];p=folder/"final_resume.pt"
    assert report["status"]=="complete" and h["step"]==report["completed_updates"]==20000
    assert report["phase"]==phase and report["case"]==case and report["seed"]==seed and cfg["weight_decay"]==0
    assert sha(p)==report["final_resume_sha256"]
    state=torch.load(p,map_location="cpu",weights_only=True)
    assert state["step"]==20000 and whash(state["model"])==h["weights_sha256"]
    xx,yy=batch(data,report,state)
    rows=[]
    for clip in (True,False):
        m0,d0,z0,g0=trial(arch,build,cfg,state,xx,yy,1.,False,clip)
        meta=dict(architecture=arch,phase=phase,case=case,seed=seed,optimizer=cfg["optimizer"],lr=cfg["lr"],alpha=h["spectral"]["raw"]["alpha"],D=h["spectral"]["raw"]["D"],tail=h["spectral"]["raw"]["tail"],source_commit=PRIMARY if arch=="ordinary" else NORMALIZED,weights_sha256=h["weights_sha256"],resume_sha256=report["final_resume_sha256"],clipped=clip)
        rows.append(dict(**meta,b=1.,compensated=False,delta_equivariance_error=0.,logit_max_error=0.,gradient_covariance_error=0.,**m0))
        if arch=="normalized":
            for b in STEPS:
                for comp in (False,True):
                    m,d,z,g=trial(arch,build,cfg,state,xx,yy,b,comp,clip)
                    logerr=float((z-z0).abs().max())
                    # Powers-of-two rescalings preserve the represented logits within FP32 roundoff.
                    torch.testing.assert_close(z,z0,rtol=2e-5,atol=2e-5)
                    rows.append(dict(**meta,b=b,compensated=comp,delta_equivariance_error=norm(d.double()/b-d0.double())/max(norm(d0),1e-30),logit_max_error=logerr,gradient_covariance_error=norm(g.double()*b-g0.double())/max(norm(g0),1e-30),**m))
    assert whash(state["model"])==h["weights_sha256"] and sha(p)==report["final_resume_sha256"]
    return rows
def save(rows,errors,out):
    out.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    if fields:
        with (out/"step_diagnostics.csv").open("w") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    baselines=[r for r in rows if r["b"]==1 and r["clipped"]]
    summary=[]
    metrics=("fc2_angle_rad","fc2_update_radial_cos","fc2_gradient_radial_cos","fc2_tangential_step","fc2_update_relative_norm","clip_factor","fc2_gain","fc3_gain","precond_p99_over_p01","moment_radial_cos")
    for arch in ("ordinary","normalized"):
        for phase,cases in (("grid",CASES),("confirm",TRACKS)):
            for case in cases:
                rr=[r for r in baselines if r["architecture"]==arch and r["phase"]==phase and r["case"]==case]
                if not rr:continue
                agg=dict(architecture=arch,phase=phase,case=case,n=len(rr),lr=rr[0]["lr"])
                for k in metrics:
                    vv=[r[k] for r in rr if r.get(k) is not None]
                    if vv:agg[k]=statistics.mean(vv)
                summary.append(agg)
    equiv=[]
    for op in ("sgdm","adamw"):
        for clip in (True,False):
            for b in STEPS:
                for comp in (False,True):
                    rr=[r for r in rows if r["architecture"]=="normalized" and r["optimizer"]==op and r["clipped"]==clip and r["b"]==b and r["compensated"]==comp]
                    if rr:equiv.append(dict(optimizer=op,clipped=clip,b=b,compensated=comp,n=len(rr),median_delta_error=statistics.median(r["delta_equivariance_error"] for r in rr),max_delta_error=max(r["delta_equivariance_error"] for r in rr),max_gradient_covariance_error=max(r["gradient_covariance_error"] for r in rr),max_logit_error=max(r["logit_max_error"] for r in rr)))
    complete=len(baselines)==48 and len(rows)==288 and not errors
    (out/"summary.json").write_text(json.dumps(dict(complete=complete,n_endpoints=len(baselines),n_trials=len(rows),errors=errors,baselines=summary,equivariance=equiv),indent=2,allow_nan=False)+"\n")
    lines=["# MAGIC optimizer-state step audit","Diagnostic on disposable copies of final resume states. No persistent model updates, test evaluation or new training.","",f"Complete: {complete}; endpoints {len(baselines)}/48; trials {len(rows)}/288; errors {len(errors)}.","","## Actual clipped baseline step (mean over seeds)","","| Architecture | Phase | Case | n | LR | FC2 angle (rad) | Update radial cosine | Gradient radial cosine | Clip factor |","|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        def fmt(k):return f'{r[k]:.5g}' if k in r else "—"
        lines.append(f'| {r["architecture"]} | {r["phase"]} | {r["case"]} | {r["n"]} | {r["lr"]} | {fmt("fc2_angle_rad")} | {fmt("fc2_update_radial_cos")} | {fmt("fc2_gradient_radial_cos")} | {fmt("clip_factor")} |')
    lines+=["","## FC2 counterfactual rescaling (normalized models only)","","Expected compensated laws: SGDM LR × b², Adam LR × b and epsilon ÷ b, with FC2-only consistently rescaled state. Other parameters and gains remain fixed at the pre-update point. Relative error measures ||delta_b/b - delta_1||/||delta_1||. Clipping may break exact layerwise covariance; FP32 update rounding can contribute residual.","","| Optimizer | Clip | b | Compensated | n | Median error | Max error |","|---|---|---:|---|---:|---:|---:|"]
    for r in equiv:lines.append(f'| {r["optimizer"]} | {r["clipped"]} | {r["b"]} | {r["compensated"]} | {r["n"]} | {r["median_delta_error"]:.5g} | {r["max_delta_error"]:.5g} |')
    lines+=["","[Full per-state diagnostics](step_diagnostics.csv) · [Machine-readable summary](summary.json)","","Errors: "+str(errors),"","This is a local one-step measurement, not an estimate of a whole training trajectory or an ERG Jacobian."]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n")
    return complete
def main():
    p=argparse.ArgumentParser();p.add_argument("--ordinary",required=True,type=pathlib.Path);p.add_argument("--normalized",required=True,type=pathlib.Path);p.add_argument("--out",default=HERE/"results",type=pathlib.Path);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert not a.out.exists(),"Do not overwrite existing audit"
    dp=a.ordinary/"prepared/data.npz";mp=a.ordinary/"prepared/manifest.json"
    assert sha(dp)==DATA and sha(mp)==MANIFEST and json.loads(mp.read_text())["prepared_sha256"]==DATA
    with np.load(dp,allow_pickle=False) as d:data={k:d[k] for k in ("x","y","train","outer_train")}
    sys.path.insert(0,str(a.normalized.resolve()));from model import build
    rows=[];errors=[]
    for arch,root in (("ordinary",a.ordinary),("normalized",a.normalized)):
        for phase,cases,seeds in (("grid",CASES,SEEDS),("confirm",TRACKS,CONF)):
            for case in cases:
                for seed in seeds:
                    try:
                        rows.extend(run_case(root,arch,phase,case,seed,data,build))
                        print("OK",arch,phase,case,seed,flush=True)
                    except Exception as e:
                        traceback.print_exc();errors.append(dict(architecture=arch,phase=phase,case=case,seed=seed,error=repr(e)))
    if not save(rows,errors,a.out):sys.exit(1)
if __name__=="__main__":main()
