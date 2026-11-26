"""Post hoc MAGIC diagnostic. No optimizer.step, no training, no test inference."""
import copy,csv,hashlib,io,contextlib,json,math,os,pathlib,statistics
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from scipy.optimize import minimize_scalar
import weightwatcher as ww
D=pathlib.Path(__file__).resolve().parent; O=D.parent/"magic"; R=D/"results";R.mkdir(exist_ok=True)
EXPECTED="c42ba2691997b884dae5d738bcb2b5bb5a5b07fbca0f0c83afae12ce76812d73"
ARCHIVE="09dbaa4e095ff21fc8b1e130798d247b75c15913"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def whash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def dump(path,j):
    t=path.with_suffix(".tmp");t.write_text(json.dumps(j,indent=2,allow_nan=False)+"\n");os.replace(t,path)
def net():return nn.Sequential(nn.Linear(10,784),nn.ReLU(),nn.Linear(784,784),nn.ReLU(),nn.Linear(784,2))
def dot(a,b):return float(torch.sum(a.double()*b.double()))
def norms(a):return math.sqrt(dot(a,a))
def quantiles(x):
    return [float(v) for v in np.quantile(x,[.01,.1,.5,.9,.99])] if len(x) else None
def predict_logits(m,xx):
    m.eval()
    with torch.no_grad():
        return torch.cat([m(b) for b in xx.split(1024)])
def margin_logits(logits):
    z=logits.double().numpy()
    return z[:,1]-z[:,0]
def predict(m,xx):
    return margin_logits(predict_logits(m,xx))
def archive_metrics(logits,y):
    # Replay the original run.py evaluator, including float32 softmax ties.
    # Logit-based AUC remains the separate scientific diagnostic.
    yy=torch.as_tensor(y,dtype=torch.int64)
    return dict(ce=float(nn.functional.cross_entropy(logits,yy)),
                accuracy=float((logits.argmax(-1)==yy).double().mean()),
                auc=float(roc_auc_score(yy.numpy(),logits.softmax(-1)[:,1].numpy())))
def metrics(z,y):
    signed=z*(2*y-1);good=signed>0;wrong=~good
    # Torch argmax tie picks class 0; use exact decision convention for accuracy.
    good=((z>0).astype(np.int64)==y);wrong=~good
    losses=np.logaddexp(0.,-signed);p=1/(1+np.exp(-np.clip(np.abs(z),0,700)))
    count=int(wrong.sum())
    bins=np.minimum((p*15).astype(int),14);ece=0
    for k in range(15):
        mask=bins==k
        if mask.any():ece+=float(mask.mean())*abs(float(p[mask].mean())-float(good[mask].mean()))
    return dict(accuracy=float(good.mean()),ce=float(losses.mean()),auc=float(roc_auc_score(y,z)),logit_rms=float(np.sqrt(np.mean(z*z))),mean_abs_logit=float(np.abs(z).mean()),mean_confidence=float(p.mean()),confidence_correct=float(p[good].mean()) if good.any() else None,confidence_incorrect=float(p[wrong].mean()) if count else None,wrong_mean_ce=float(losses[wrong].mean()) if count else None,wrong_ce_contribution=float(losses[wrong].sum()/len(y)),correct_ce_contribution=float(losses[good].sum()/len(y)),wrong_mean_abs_margin=float(np.abs(signed[wrong]).mean()) if count else None,margin_quantiles=quantiles(signed),wrong_margin_quantiles=quantiles(signed[wrong]),ece15=float(ece))
def temperature_crossfit(z,y,folds):
    signed=z*(2*y-1);loss=np.empty(len(y));temps=[]
    for fit,eval_ix in folds:
        obj=lambda a: float(np.logaddexp(0.,-signed[fit]*np.exp(a)).mean())
        res=minimize_scalar(obj,bounds=(-5.,5.),method="bounded")
        assert res.success
        scale=float(np.exp(res.x));temps.append(1/scale);loss[eval_ix]=np.logaddexp(0.,-signed[eval_ix]*scale)
    return {"temperature_mean":float(np.mean(temps)),"crossfit_ce":float(loss.mean()),"fold_temperatures":temps}
def spectrum(eig,k=None):
    x=np.asarray(eig,dtype=np.float64);assert len(x)==784 and np.isfinite(x).all() and np.all(x>=0)
    total=float(x.sum());top=float(x[-1])
    if total==0:return {"trace":0.,"lambda_max":0.,"stable_rank":None}
    mu=total/len(x);l=np.log(np.maximum(x,np.finfo(float).tiny)/mu);cum=np.cumsum(l[::-1]);means=cum/np.arange(1,len(x)+1)
    ecs=int(np.argmin(np.abs(means[7:]))+8)
    a=dict(trace=total,lambda_max=top,mean_eigenvalue=mu,median=float(np.median(x)),stable_rank=total/top,top10_energy=float(x[-10:].sum()/total),ecs_rank=ecs,ecs_meanlog=float(means[ecs-1]))
    if k is not None and 8<=k<=len(x):
        a.update(pl_rank=int(k),rank_gap=ecs-int(k),pl_meanlog=float(means[k-1]),pl_geometric_mean_over_full_mean=float(np.exp(means[k-1])),pl_energy=float(x[-k:].sum()/total),pl_logspan=float(np.log(x[-1]/x[-k])))
        xx=x[-k:];edges=np.geomspace(xx[0],xx[-1],5)
        bins=np.clip(np.searchsorted(edges,xx,side="right")-1,0,3)
        a["pl_relative_band_energy"]=[float(xx[bins==b].sum()/xx.sum()*4) for b in range(4)]
    return a
def delta_fit(w):
    z=nn.Sequential(nn.Linear(784,784,bias=False))
    with torch.no_grad():z[0].weight.copy_(w)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            f=ww.WeightWatcher(z).analyze(min_evals=8,randomize=False,plot=False)
        if len(f)!=1:raise ValueError("not one WW matrix")
        row=json.loads(f.to_json(orient="records"))[0]
        if row.get("status")!="success":return {"status":"failed","row":row}
        return {"status":"success","alpha":row["alpha"],"D":row["D"],"tail":row["num_pl_spikes"],"xmin":row["xmin"]}
    except Exception as ex:return {"status":"failed","error":repr(ex)}
def gradients(m,x,y):
    ps=list(m.parameters());m.eval()
    loss=nn.functional.cross_entropy(m(x),y);g=torch.autograd.grad(loss,ps)
    return float(loss),[t.detach() for t in g]
def gradient_geometry(m,x,y):
    _,g=gradients(m,x,y);ps=list(m.parameters());norm=math.sqrt(sum(dot(v,v) for v in g));out={}
    for i in range(3):
        w,b=ps[2*i:2*i+2];gw,gb=g[2*i:2*i+2];r=dot(w,gw)
        out[f"fc{i+1}"]={"beta_weight_only":-r,"beta_bias":-dot(b,gb),"weight_norm":norms(w),"grad_norm":norms(gw),"normalized_radial_gradient":r/max(norms(w)*norms(gw),1e-30),"fractional_radial_derivative":-r/max(dot(w,w),1e-30)}
        if i<2:out[f"fc{i+1}"]["exact_relu_gauge_residual"]=r+dot(b,gb)-dot(ps[2*i+2],g[2*i+2])
    return {"global_gradient_norm":norm,"clip_coefficient":min(1.,1./(norm+1e-6)),"layers":out}
def next_update(m,record,saved,xx,yy):
    # Reconstruct the next minibatch and hypothetical update mathematically; never optimizer.step.
    ps=list(m.parameters());cfg=record["config"];n=len(yy);off=int(saved["offset"]);perm=saved["permutation"]
    if off==n:
        gen=torch.Generator();gen.set_state(saved["batch_rng"]);perm=torch.randperm(n,generator=gen);off=0
    ix=perm[off:off+256];loss,raw=gradients(m,xx[ix],yy[ix]);gn=math.sqrt(sum(dot(g,g) for g in raw));clip=min(1.,1./(gn+1e-6))
    st=saved["optimizer"];pg=st["param_groups"];assert len(pg)==1 and len(pg[0]["params"])==len(ps);ids=pg[0]["params"]
    out={}
    for i,(w,g,pid) in enumerate(zip(ps,raw,ids)):
        gc=g*clip;state=st["state"][pid];lr=cfg["lr"]
        if cfg["optimizer"]=="sgdm":
            assert "momentum_buffer" in state
            u=-lr*(cfg["momentum"]*state["momentum_buffer"]+gc)
        else:
            b1,b2=cfg["betas"];step=int(state["step"].item())+1
            mt=b1*state["exp_avg"]+(1-b1)*gc
            vt=b2*state["exp_avg_sq"]+(1-b2)*gc.square()
            u=-lr*(mt/(1-b1**step))/(torch.sqrt(vt/(1-b2**step))+cfg["eps"])-lr*cfg["weight_decay"]*w.detach()
        wn=norms(w);un=norms(u);rho=dot(w,u)/max(wn*wn,1e-30)
        shape=u-rho*w.detach();out[f"fc{i//2+1}_{'weight' if i%2==0 else 'bias'}"]={"weight_norm":wn,"update_norm":un,"relative_update_norm":un/max(wn,1e-30),"relative_shape_update":norms(shape)/max(wn,1e-30),"fractional_radial_update":rho,"cosine_negative_gradient":-dot(g,u)/max(norms(g)*un,1e-30),"effective_update_over_gradient_norm":un/max(norms(g),1e-30)}
    return {"next_minibatch_loss":loss,"gradient_norm_before_clipping":gn,"clip_coefficient":clip,"layers":out}
def gaugetest(m,xx,y,z0):
    original=copy.deepcopy(m.state_dict());out={}
    for c in (.5,2.):
        for compensated in (False,True):
            m.load_state_dict(original)
            with torch.no_grad():
                m[2].weight.mul_(c)
                if compensated:m[2].bias.mul_(c);m[4].weight.div_(c)
            z=predict(m,xx);out[f"{'compensated' if compensated else 'fc2_only'}_{c}"]={"max_abs_logit_difference":float(np.max(np.abs(z-z0))),"accuracy":metrics(z,y)["accuracy"],"ce":metrics(z,y)["ce"]}
    m.load_state_dict(original)
    return out
def truncation(m,xx,y,k1,k2):
    state=copy.deepcopy(m.state_dict());u,s,vh=torch.linalg.svd(state["2.weight"].double(),full_matrices=False);out={}
    for name,k in (("pl",k1),("ecs",k2)):
        if k is None or not 1<=k<=784:continue
        m.load_state_dict(state)
        with torch.no_grad():m[2].weight.copy_(((u[:,:k]*s[:k])@vh[:k,:]).float())
        z=predict(m,xx);out[name]={"rank":int(k),**metrics(z,y)}
    m.load_state_dict(state);return out
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    P=json.loads((O/"plan.json").read_text());man=json.loads((O/"prepared/manifest.json").read_text())
    assert sha(O/"prepared/data.npz")==man["prepared_sha256"]==EXPECTED
    assert sha(O/"plan.json")==man["plan_sha256"]
    summary=json.loads((O/"results/analysis/summary.json").read_text())
    assert summary["complete"] and summary["n_grid"]==24 and summary["n_confirm"]==15 and not summary["audit_errors"]
    with np.load(O/"prepared/data.npz") as a:x,y,train,val=[a[k] for k in ("x","y","train","val")]
    mean=x[train].mean(0);sd=np.maximum(x[train].std(0),1e-12)
    xt=torch.from_numpy(((x[train]-mean)/sd).astype(np.float32));yt=torch.from_numpy(y[train].astype(np.int64))
    xv=torch.from_numpy(((x[val]-mean)/sd).astype(np.float32));yv=y[val].astype(np.int64)
    _,groups=np.unique(x[val],axis=0,return_inverse=True)
    folds=list(StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=2217).split(x[val],yv,groups))
    assert all(not set(groups[a])&set(groups[b]) for a,b in folds)
    fixed=torch.randperm(len(yt),generator=torch.Generator().manual_seed(1881))[:1024]
    results=[];errors=[]
    for cfg in P["grid"]:
        for seed in P["dev_seeds"]:
            case=cfg["id"];d=O/"results/grid"/f"{case}-{seed}"
            try:
                j=json.loads((d/"result.json").read_text());assert j["status"]=="complete" and j["case"]==case and j["seed"]==seed and j["config"]==cfg and j["completed_updates"]==20000 and j["start_step"]==0
                assert j["plan_sha256"]==sha(O/"plan.json") and j["data_manifest_sha256"]==sha(O/"prepared/manifest.json") and j["source_commit"]=="9571f954a2070e717123f499ec9f41340a56b9ef"
                assert [h["step"] for h in j["history"]]==P["checkpoints"] and all(set(h["metrics"])=={"train","validation"} for h in j["history"])
                m=net();initial=None;prev_z=None;prev_eig=None;rec=[]
                for h in j["history"]:
                    cp=d/h["checkpoint_file"];assert sha(cp)==h["checkpoint_sha256"]
                    q=torch.load(cp,map_location="cpu",weights_only=True);assert q["step"]==h["step"] and whash(q["model"])==q["weights_sha256"]==h["weights_sha256"]==h["spectral_weights_sha256"]
                    m.load_state_dict(q["model"]);w=m[2].weight.detach().clone()
                    if initial is None:initial=w.clone()
                    delta=w-initial;raw=h["spectral"]["raw"];eig=np.array(h["spectral"]["eigenvalues"])
                    spec=spectrum(eig,raw["tail"]);assert abs(dot(w,w)-spec["trace"])<1e-3*max(1.,spec["trace"])
                    spec["alpha"]=raw["alpha"];spec["D"]=raw["D"]
                    if prev_eig is not None:
                        spec["normalized_log_quantile_drift"]=float(np.abs(np.log(np.maximum(eig/eig.mean(),1e-300))-np.log(np.maximum(prev_eig/prev_eig.mean(),1e-300))).mean())
                    prev_eig=eig
                    lv=predict_logits(m,xv);lt=predict_logits(m,xt)
                    z=margin_logits(lv);v=metrics(z,yv);t=metrics(margin_logits(lt),yt.numpy())
                    for part,a0,logits,labels in (("validation",v,lv,yv),("train",t,lt,yt.numpy())):
                        old=h["metrics"][part];replay=archive_metrics(logits,labels)
                        # Preserve original tolerances but compare like with like; do not
                        # mistake saturation ties in archived probabilities for checkpoint drift.
                        for key,tol in (("ce",3e-6),("accuracy",1/old["n"]+1e-9),("auc",1e-5)):
                            err=abs(replay[key]-old[key])
                            assert err<=tol,(case,seed,h["step"],part,key,replay[key],old[key],err,tol)
                            a0["archive_replay_"+key]=replay[key]
                            a0["archive_replay_"+key+"_abs_error"]=err
                        assert abs(a0["ce"]-old["ce"])<3e-6 and abs(a0["accuracy"]-old["accuracy"])<=1/old["n"]+1e-9,(case,seed,h["step"],part,"diagnostic CE/accuracy",a0["ce"],old["ce"],a0["accuracy"],old["accuracy"])
                        a0["logit_minus_archived_auc"]=a0["auc"]-old["auc"]
                    v.update(temperature_crossfit(z,yv,folds))
                    if prev_z is not None:
                        v["prediction_flip_fraction_since_previous"]=float(((z>0)!=(prev_z>0)).mean())
                        v["logit_cosine_since_previous"]=float(np.dot(z,prev_z)/max(np.linalg.norm(z)*np.linalg.norm(prev_z),1e-30))
                    prev_z=z
                    init={"relative_displacement_norm":norms(delta)/norms(initial),"current_initial_weight_cosine":dot(w,initial)/max(norms(w)*norms(initial),1e-30)}
                    if h["step"]>0:
                        de=np.sort(np.linalg.svd(delta.numpy().astype(np.float64),compute_uv=False)**2)
                        init["delta_spectrum"]=spectrum(de)
                    row={"step":h["step"],"weights_sha256":h["weights_sha256"],"spectral":spec,"initialization":init,"train":t,"validation":v,"gradient_geometry":gradient_geometry(m,xt[fixed],yt[fixed])}
                    if h["step"]==20000:
                        init["delta_ww_fit"]=delta_fit(delta)
                        sfile=d/"final_resume.pt";assert sha(sfile)==j["final_resume_sha256"];saved=torch.load(sfile,map_location="cpu",weights_only=True)
                        assert saved["step"]==20000 and whash(saved["model"])==h["weights_sha256"]
                        row["hypothetical_next_update"]=next_update(m,j,saved,xt,yt)
                        row["scale_interventions"]=gaugetest(m,xv,yv,z)
                        row["truncation"]=truncation(m,xv,yv,raw["tail"],spec["ecs_rank"])
                    assert whash(m.state_dict())==h["weights_sha256"]
                    rec.append(row)
                obj={"case":case,"seed":seed,"config":cfg,"history":rec};dump(R/f"{case}-{seed}.json",obj);results.append(obj)
                e=rec[-1];print(json.dumps({"case":case,"seed":seed,"step":20000,"alpha":e["spectral"]["alpha"],"val_ce":e["validation"]["ce"],"crossfit_ce":e["validation"]["crossfit_ce"],"logit_rms":e["validation"]["logit_rms"],"wrong_confidence":e["validation"]["confidence_incorrect"],"delta_alpha":e["initialization"]["delta_ww_fit"].get("alpha"),"rank_gap":e["spectral"]["rank_gap"],"pl_accuracy":e["truncation"]["pl"]["accuracy"],"ecs_accuracy":e["truncation"]["ecs"]["accuracy"]}),flush=True)
            except Exception as ex:errors.append(f"{case}/{seed}: {ex!r}");print(errors[-1],flush=True)
    def flat(r,h):
        q={"case":r["case"],"seed":r["seed"],"track":r["config"]["track"],"lr":r["config"]["lr"],"momentum":r["config"].get("momentum"),"step":h["step"],"weights_sha256":h["weights_sha256"]}
        for name in ("spectral","initialization","train","validation"):
            for k,v in h[name].items():
                if v is None or isinstance(v,(str,int,float)):q[name+"_"+k]=v
        if "delta_ww_fit" in h["initialization"]:
            for k in ("alpha","D","tail","status"):q["delta_ww_"+k]=h["initialization"]["delta_ww_fit"].get(k)
        for name in ("gradient_geometry","hypothetical_next_update"):
            if name in h:
                for layer,vals in h[name]["layers"].items():
                    for k,v in vals.items():q[name+"_"+layer+"_"+k]=v
        for name in ("truncation","scale_interventions"):
            for sub,vals in h.get(name,{}).items():
                for k,v in vals.items():
                    if v is None or isinstance(v,(int,float,str)):q[name+"_"+sub+"_"+k]=v
        return q
    rows=[flat(r,h) for r in results for h in r["history"]]
    def csvout(path,rows):
        if rows:
            with path.open("w") as f:
                wr=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));wr.writeheader();wr.writerows(rows)
    csvout(R/"trajectory.csv",rows);end=[r for r in rows if r["step"]==20000];csvout(R/"endpoints.csv",end)
    keys=["spectral_alpha","spectral_rank_gap","spectral_pl_meanlog","spectral_stable_rank","spectral_top10_energy","initialization_relative_displacement_norm","initialization_current_initial_weight_cosine","delta_ww_alpha","validation_accuracy","validation_ce","validation_crossfit_ce","validation_temperature_mean","validation_logit_rms","validation_confidence_incorrect","validation_wrong_ce_contribution","validation_ece15","validation_auc","train_ce","truncation_pl_accuracy","truncation_pl_ce","truncation_ecs_accuracy","truncation_ecs_ce","scale_interventions_compensated_2.0_max_abs_logit_difference","scale_interventions_fc2_only_2.0_ce","hypothetical_next_update_fc2_weight_relative_update_norm","hypothetical_next_update_fc2_weight_relative_shape_update","hypothetical_next_update_fc2_weight_fractional_radial_update","hypothetical_next_update_fc2_weight_cosine_negative_gradient"]
    cases=[]
    for cfg in P["grid"]:
        z=[q for q in end if q["case"]==cfg["id"]]
        vals={k:statistics.mean([q[k] for q in z]) if len(z)==2 and all(q.get(k) is not None for q in z) else None for k in keys}
        cases.append({"case":cfg["id"],"config":cfg,"n":len(z),"means":vals})
    out={"complete":len(results)==24 and len(rows)==144 and not errors,"source_archive":ARCHIVE,"source_data_sha256":EXPECTED,"training_updates":0,"test_evaluations":0,"n_runs":len(results),"n_checkpoints":len(rows),"failed_delta_fits":sum(q.get("delta_ww_status")=="failed" for q in end),"errors":errors,"cases":cases,"notes":"Retrospective development only; temperature fit cross-fitted on two duplicate-grouped validation halves; no model training. Next update reconstructed analytically from optimizer state and next minibatch, not applied. Truncated SVD without fine-tuning or rescaling."}
    dump(R/"summary.json",out)
    print(json.dumps(out,indent=2),flush=True)
    if not out["complete"]:raise RuntimeError("Incomplete checkpoint audit; preserve partial records")
if __name__=="__main__":main()
