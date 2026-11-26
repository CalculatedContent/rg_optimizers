"""Deterministic uncalibrated predictive-confidence diagnostics; no fitting."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve
def metrics_from_logits(logits,yy,fprs):
    probs=logits.softmax(-1).numpy()
    target=yy.numpy()
    pred=logits.argmax(-1).numpy()
    good=(pred==target)
    p=probs[:,1].astype(np.float64)
    confidence=probs.max(axis=1).astype(np.float64)
    fpr,tpr,_=roc_curve(target,p,drop_intermediate=False)
    bins=np.minimum((confidence*15).astype(np.int64),14)
    ece=0.; details=[]
    for k in range(15):
        mask=bins==k
        count=int(mask.sum())
        if count:
            conf=float(confidence[mask].mean());acc=float(good[mask].mean())
            ece+=count/len(target)*abs(conf-acc)
            details.append({"bin":k,"n":count,"confidence":conf,"accuracy":acc})
        else:details.append({"bin":k,"n":0,"confidence":None,"accuracy":None})
    return {"ce":float(torch.nn.functional.cross_entropy(logits,yy)),"accuracy":float(good.mean()),"auc":float(roc_auc_score(target,p)),"gamma_efficiency":{str(t):float(tpr[fpr<=t].max()) for t in fprs},"n":len(target),"brier":float(np.mean((p-target)**2)),"ece15":float(ece),"mean_confidence":float(confidence.mean()),"confidence_correct":float(confidence[good].mean()) if good.any() else None,"confidence_incorrect":float(confidence[~good].mean()) if (~good).any() else None,"high_confidence_fraction":float((confidence>=.9).mean()),"high_confidence_error_fraction":float(((confidence>=.9)&(~good)).mean()),"calibration_bins":details}
