"""Descriptive equal-epoch comparison; no checkpoint loading or test evaluation."""
import argparse,hashlib,json,math
from pathlib import Path
import pandas as pd
from step_accounting import accounting

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--original',type=Path,required=True)
    cli.add_argument('--augmented',type=Path,required=True)
    args=cli.parse_args()
    out=Path(__file__).resolve().parent
    a=pd.read_csv(args.original).rename(columns={'validation_loss':'validation_ce','validation_accuracy_pct':'validation_accuracy'})
    b=pd.read_csv(args.augmented)
    rows=[]
    for label,frame in [('original_unaugmented',a),('newer_augmented',b)]:
        subset=frame[(frame.arm=='adamw')&(frame.batch_size==256)&(frame.multiplier==1)&(frame.epoch==60)].copy()
        assert len(subset)==5 and subset.seed.nunique()==5
        subset['cohort']=label
        rows.append(subset[['cohort','seed','arm','batch_size','multiplier','epoch','validation_accuracy','validation_ce','alpha_fc1','alpha_fc2']])
    data=pd.concat(rows,ignore_index=True)
    data.to_csv(out/'comparison_epoch60.csv',index=False)
    summary=data.groupby('cohort')[['validation_accuracy','validation_ce','alpha_fc1','alpha_fc2']].mean()
    sp={'batch_size':256,'multiplier':1,'arm':'adamw'}
    original=accounting(sp,60)
    f=lambda e:e/5 if e<=5 else .1+.45*(1+math.cos(math.pi*(e-5)/115))
    meta={'original_sha256':hashlib.sha256(args.original.read_bytes()).hexdigest(),'augmented_sha256':hashlib.sha256(args.augmented.read_bytes()).hexdigest(),'same_epoch':60,'same_updates':12900,'same_training_examples':3300000,'different_seeds':True,'different_schedules':True,'different_augmentation':True,'paired_comparison':False,'new_test_evaluation':False,'original_adam_lr':.00001,'augmented_adam_lr':.001*f(60),'original_sum_adam_lr':original['sum_optimizer_layer_lrs'][0],'augmented_sum_adam_lr':.001*215*sum(f(e) for e in range(1,61)),'cohort_means':summary.to_dict(orient='index')}
    (out/'verification/Equal_Epoch_Comparison.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(summary.to_string())
    return meta

if __name__=='__main__':main()
