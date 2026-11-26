"""Small same-split tree baseline; validation selection only."""
import json,pathlib
import numpy as np
from sklearn.metrics import log_loss,roc_auc_score,roc_curve
from xgboost import XGBClassifier
from run import D,P,sha
def metrics(m,x,y):
    p=m.predict_proba(x);fpr,tpr,_=roc_curve(y,p[:,1],drop_intermediate=False)
    return {"accuracy":float((p.argmax(1)==y).mean()),"ce":float(log_loss(y,p,labels=[0,1])),"auc":float(roc_auc_score(y,p[:,1])),"gamma_efficiency":{str(t):float(tpr[fpr<=t].max()) for t in P["roc_fpr"]}}
def main():
    out=D/"results/xgb"
    if out.exists():raise FileExistsError("Refuse overwrite")
    out.mkdir(parents=True)
    mf=D/"prepared/manifest.json";dp=D/"prepared/data.npz"
    assert json.loads(mf.read_text())["prepared_sha256"]==sha(dp)
    with np.load(dp) as d:x,y,train,val,outer,test=[d[k] for k in ("x","y","train","val","outer_train","test")]
    b=P["xgb"]
    def make(depth,lr):
        return XGBClassifier(max_depth=depth,learning_rate=lr,n_estimators=b["n_estimators"],random_state=b["seed"],n_jobs=1,tree_method="hist",objective="binary:logistic",eval_metric="logloss",subsample=1,colsample_bytree=1)
    rows=[]
    for depth in b["max_depth"]:
        for lr in b["learning_rate"]:
            m=make(depth,lr);m.fit(x[train],y[train]);rows.append({"max_depth":depth,"learning_rate":lr,"validation":metrics(m,x[val],y[val])})
    best=min(rows,key=lambda r:(-r["validation"]["accuracy"],r["validation"]["ce"],r["max_depth"],r["learning_rate"]))
    (out/"frozen_selection.json").write_text(json.dumps({"selected":best,"scores":rows},indent=2)+"\n")
    m=make(best["max_depth"],best["learning_rate"]);m.fit(x[outer],y[outer]);m.save_model(out/"model.ubj")
    result={"selected":best,"scores":rows,"test":metrics(m,x[test],y[test]),"model_sha256":sha(out/"model.ubj"),"selection_sha256":sha(out/"frozen_selection.json"),"data_manifest_sha256":sha(mf),"plan_sha256":sha(D/"plan.json"),"note":"six development fits and one refit, no test selection"}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
