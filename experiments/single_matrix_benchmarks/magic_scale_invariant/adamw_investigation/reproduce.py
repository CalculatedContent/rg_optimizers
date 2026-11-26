"""Reproduce the matched AdamW spectral endpoint diagnostics from two immutable checkout trees.

Usage:
  python reproduce.py /path/to/ordinary/magic /path/to/invariant/magic_scale_invariant
No training, no checkpoint evaluation, no test selection. Reads existing result.json records.
"""
import csv, json, math, pathlib, statistics, sys
from hashlib import sha1

SEEDS=range(3472,3477)
KS=(78,157,235,314,392)
REFS=("09dbaa4e095ff21fc8b1e130798d247b75c15913","e6ec5fb97bc3db0130937dd49e710399dbc1dbec")

def analyze(root,arch,ref,seed):
    path=root/"results"/"confirm"/f"adamw_wd0-{seed}"/"result.json"
    b=path.read_bytes(); j=json.loads(b); h=j["history"][-1]
    assert j["status"]=="complete" and h["step"]==20000
    assert j["config"]["lr"]==.0001 and j["config"]["weight_decay"]==0
    e=h["spectral"]["eigenvalues"]; assert len(e)==784 and sorted(e)==e
    raw=h["spectral"]["raw"]; assert raw["status"]=="success"
    S=sum(e); p=[x/S for x in e]
    k=raw["tail"]; t=e[-k:]; xmin=raw["ww_row"]["xmin"]
    m=sum(math.log(x/xmin) for x in t)/k
    def fixed(k):
        t=e[-k:]; return 1+1/(sum(math.log(x/t[0]) for x in t)/k)
    r=dict(arch=arch,seed=seed,source_commit=ref,source_blob=sha1(f"blob {len(b)}\0".encode()+b).hexdigest(),
       weights_sha256=h["weights_sha256"],alpha=raw["alpha"],D=raw["D"],tail=k,
       tail_energy=sum(t)/S,top1_energy=p[-1],top10_energy=sum(p[-10:]),
       effective_rank=math.exp(-sum(x*math.log(x) for x in p if x>0)),
       participation=1/sum(x*x for x in p),raw_trace=S,relative_xmin=xmin/(S/len(e)),
       trace_log_tail=m-1,mle_alpha=1+1/m,
       accuracy=h["metrics"]["test"]["accuracy"],ce=h["metrics"]["test"]["ce"],
       auc=h["metrics"]["test"]["auc"],train_acc=h["metrics"]["train"]["accuracy"])
    r.update({f"alpha_top_{k}":fixed(k) for k in KS})
    return r,p

def main():
    a,b=(pathlib.Path(x) for x in sys.argv[1:])
    A=[analyze(a,"ordinary",REFS[0],s) for s in SEEDS]
    B=[analyze(b,"invariant",REFS[1],s) for s in SEEDS]
    for (ar,ap),(br,bp) in zip(A,B):
        d=sum(abs(x-y) for x,y in zip(ap,bp))
        ar["paired_sorted_spectral_l1"]=br["paired_sorted_spectral_l1"]=d
    for k,v in A[0][0].items():
        if k=="seed" or not isinstance(v,(int,float)):continue
        av=[r[k] for r,p in A];bv=[r[k] for r,p in B]
        d=[y-x for x,y in zip(av,bv)]
        print(k,statistics.mean(av),statistics.mean(bv),statistics.mean(d),2.776445*statistics.stdev(d)/math.sqrt(5))
    rows=[r for r,p in A+B]
    with open("reproduced_endpoints.csv","w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
