"""Label saved Experiment 01 outcomes and plot their evidence; never train.

Usage: python build_model_status.py EXPERIMENT_DIRECTORY OUTPUT_DIRECTORY
Input files are the existing repository CSVs and checkpoint-diagnostic JSONs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ARMS = ['adamw', 'magnitude_graft', 'sgdm']
LABELS = {'adamw': 'AdamW', 'magnitude_graft': 'Magnitude graft', 'sgdm': 'SGD + momentum'}
COLORS = {'adamw': '#2867A6', 'magnitude_graft': '#CD6B27', 'sgdm': '#35815B'}
METRICS = ['alpha_fc1', 'alpha_fc2', 'mean_abs_alpha_minus_2', 'test_accuracy_pct']
REP = 'replications/2026-11-14_batch128_seed31006/'
INPUTS = ['results/endpoints.csv', 'results/run_disposition.csv', 'results/stable_paired_effects.csv',
          'results/stable_summary.csv', 'results/paired_summaries.csv', 'verification/Low_Accuracy_Diagnostics.json',
          REP + 'results/Endpoints.csv', REP + 'verification/Replication_Verification.json',
          'verification/Final_Verification.json', 'frozen/Protocol.json']


def key(row):
    return (int(row['seed']), int(row['batch_size']), float(row['multiplier']), row['arm'])


def interval(values):
    x = np.asarray(values, dtype=float)
    assert len(x) == 5 and np.isfinite(x).all()
    mean = x.mean(); sd = x.std(ddof=1); half = t.ppf(.975, 4) * sd / np.sqrt(5)
    return {'mean': float(mean), 'seed_sd': float(sd), 'ci_low': float(mean-half), 'ci_high': float(mean+half), 'n': 5}


def model_registry(src, out):
    original = pd.read_csv(src / 'results/endpoints.csv')
    replication = pd.read_csv(src / REP / 'results/Endpoints.csv')
    disposition = pd.read_csv(src / 'results/run_disposition.csv')
    diag = json.loads((src / 'verification/Low_Accuracy_Diagnostics.json').read_text())
    verification = json.loads((src / REP / 'verification/Replication_Verification.json').read_text())
    original_verification = json.loads((src / 'verification/Final_Verification.json').read_text())
    protocol = json.loads((src / 'frozen/Protocol.json').read_text())
    assert verification['passed'] and verification['completed_runs'] == 3
    assert original_verification['passed'] and original_verification['completed_runs'] == 135
    assert original_verification['checkpoints_hashed'] == 675 and verification['checkpoints_hashed'] == 15
    assert original_verification['source_sha256'] == verification['source_sha256']
    assert protocol['seeds'] == [31001, 31002, 31003, 31004, 31005]
    assert protocol['source_sha256'] == verification['source_sha256'] == '1bfd7032e00ad87d729670197925fb40e19bc802a8eb716db2571e56b538d3d8'
    assert len(original) == 135 and len(replication) == 3 and len(disposition) == 138
    combined = pd.concat([original.assign(cohort='original'), replication.assign(cohort='replication')], ignore_index=True)
    assert not combined.duplicated(['seed', 'batch_size', 'multiplier', 'arm']).any()
    observations = {key(r): r for r in combined.to_dict('records')}
    evidence = {key(d['spec']): d for d in diag['runs'] + verification['activity_checks']}
    rows = []
    for row in disposition.to_dict('records'):
        k = key(row); observed = observations[k]
        for metric in METRICS + ['validation_accuracy_pct', 'train_accuracy_pct']:
            assert np.isclose(row[metric], observed[metric], rtol=0, atol=1e-10), (k, metric)
        failed = row['outcome'] == 'failed_hyperparameter_setting'
        if failed:
            d = evidence[k]
            assert d['hidden_test_activity']['fc2']['fraction_test_examples_with_zero_hidden_vector'] == 1
            assert d['prediction_counts'] == [0, 10000, 0, 0, 0, 0, 0, 0, 0, 0]
            assert np.isclose(row['test_accuracy_pct'], 11.35)
            reason = 'Confirmed collapse: FC2 is zero on every test input; always predicts digit 1.'
        else:
            assert row['outcome'] == 'trained_without_observed_collapse'
            assert np.isfinite([row['test_accuracy_pct'], row['validation_accuracy_pct']]).all()
            reason = 'Trained classifier; no observed collapse in saved results. Not a claim of optimal tuning.'
        caution = ''
        if row['arm'] == 'sgdm' and row['batch_size'] == 128 and row['multiplier'] == 2:
            caution = 'Functional control, but weaker accuracy at this high rate; excluded only for matched-setting balance.'
        if row['arm'] == 'adamw':
            caution = 'Strong accuracy does not rule out the late validation-loss increase documented in the results report.'
        run_id = f"b{int(row['batch_size'])}_lr{row['multiplier']:g}/{row['arm']}/seed_{int(row['seed'])}"
        row.update(run_id=run_id, model_status='FAILED' if failed else 'USABLE', status_reason=reason,
                   tuning_caution=caution, include_in_stable_comparison=row['presentation_role'] == 'stable_comparison',
                   prespecified_central_comparison=row['cohort'] == 'original' and row['batch_size'] == 256 and row['multiplier'] == 1,
                   hidden_fit_status='both_hidden_fits_supported' if row['supported_hidden_layers'] == 2 else 'at_least_one_hidden_fit_unsupported',
                   artifact_reference='ARTIFACTS.md' if row['cohort'] == 'original' else REP + 'ARTIFACTS.md')
        rows.append(row)
    registry = pd.DataFrame(rows)
    assert registry.model_status.value_counts().to_dict() == {'USABLE': 132, 'FAILED': 6}
    assert registry.include_in_stable_comparison.sum() == 120
    assert registry.prespecified_central_comparison.sum() == 15
    assert registry[registry.include_in_stable_comparison].supported_hidden_layers.eq(2).all()
    columns = ['run_id','cohort','seed','batch_size','multiplier','arm','model_status','status_reason',
               'presentation_role','include_in_stable_comparison','prespecified_central_comparison',
               'tuning_caution','hidden_fit_status','alpha_fc1','alpha_fc2','alpha_head',
               'mean_abs_alpha_minus_2','train_accuracy_pct','validation_accuracy_pct','test_accuracy_pct','artifact_reference']
    registry[columns].to_csv(out / 'results/model_status.csv', index=False)
    groups = []
    for (cohort, batch, multiplier, arm), g in registry.groupby(['cohort','batch_size','multiplier','arm'], sort=True):
        nfail = int(g.model_status.eq('FAILED').sum())
        groups.append({'cohort':cohort,'batch_size':batch,'multiplier':multiplier,'arm':arm,'n':len(g),
                       'usable_models':int(g.model_status.eq('USABLE').sum()),'failed_models':nfail,
                       'model_status':'FAILED' if nfail == len(g) else 'USABLE' if nfail == 0 else 'MIXED',
                       'in_stable_comparison':bool(g.include_in_stable_comparison.all()),
                       'test_accuracy_mean_pct':g.test_accuracy_pct.mean(),
                       'test_accuracy_seed_sd_pp':g.test_accuracy_pct.std(ddof=1),
                       'validation_accuracy_mean_pct':g.validation_accuracy_pct.mean(),
                       'alpha_fc1_mean':g.alpha_fc1.mean(),'alpha_fc2_mean':g.alpha_fc2.mean(),
                       'mean_abs_alpha_minus_2':g.mean_abs_alpha_minus_2.mean()})
    settings = pd.DataFrame(groups)
    settings.to_csv(out / 'results/model_setting_status.csv', index=False)
    return registry, settings


def paired_results(src, registry, out):
    original = registry[registry.cohort == 'original']
    stable = original[original.include_in_stable_comparison]
    central = original[(original.batch_size == 256) & (original.multiplier == 1)]
    saved_stable = pd.read_csv(src / 'results/stable_paired_effects.csv')
    saved_all = pd.read_csv(src / 'results/paired_summaries.csv')
    estimates = []
    for scope, frame in [('central', central), ('stable', stable)]:
        seed_means = frame.groupby(['seed','arm'])[METRICS].mean()
        assert len(seed_means) == 15
        for arm in ['magnitude_graft','sgdm']:
            differences = seed_means.xs(arm, level='arm') - seed_means.xs('adamw', level='arm')
            for metric in METRICS:
                result = interval(differences[metric])
                saved = saved_stable if scope == 'stable' else saved_all[(saved_all.scope == 'setting') & (saved_all.batch_size == 256) & (saved_all.multiplier == 1)]
                saved = saved[(saved.treatment == arm) & (saved.reference == 'adamw') & (saved.metric == metric)]
                assert len(saved) == 1
                for field in ('mean','seed_sd','ci_low','ci_high'):
                    assert np.isclose(result[field], saved.iloc[0][field], rtol=0, atol=1e-10)
                estimates.append({'scope':scope,'treatment':arm,'reference':'adamw','metric':metric,**result})
    estimates = pd.DataFrame(estimates)
    estimates.to_csv(out / 'results/status_review_paired_effects.csv', index=False)
    return estimates


def status_plot(settings, out):
    original = settings[settings.cohort == 'original']
    order = sorted(set(zip(original.batch_size, original.multiplier)))
    grid = np.ones((len(order),3)); annotations = {}
    for i,(batch,multiplier) in enumerate(order):
        for j,arm in enumerate(ARMS):
            r=original[(original.batch_size==batch)&(original.multiplier==multiplier)&(original.arm==arm)].iloc[0]
            grid[i,j] = 0 if r.model_status == 'FAILED' else 1
            annotations[i,j] = f"{r.model_status}  {int(r.failed_models if r.model_status=='FAILED' else r.usable_models)}/5\n{r.test_accuracy_mean_pct:.2f}% ± {r.test_accuracy_seed_sd_pp:.2f}"
    fig,ax=plt.subplots(figsize=(10.6,8.8))
    ax.imshow(grid,cmap=ListedColormap(['#F8D6D3','#E3F1EC']),vmin=0,vmax=1,aspect='auto')
    for (i,j),label in annotations.items():
        ax.text(j,i,label,ha='center',va='center',fontsize=11,color='#822F28' if grid[i,j]==0 else '#173B30',linespacing=1.5)
    ax.set_xticks(range(3),[LABELS[a] for a in ARMS],fontsize=12)
    ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=12)
    ax.set_yticks(range(len(order)),[f'Batch {b}   LR ×{m:g}' for b,m in order],fontsize=11)
    ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,len(order),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=3);ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Which trained models are usable?',fontsize=18,y=.98,weight='bold')
    fig.text(.5,.925,'Original 135 runs · cell values are test accuracy mean ± seed SD',ha='center',fontsize=11)
    fig.text(.5,.115,'USABLE = trained classifier with no observed collapse; it does not mean optimally tuned or α-aligned.',ha='center',fontsize=9)
    fig.text(.5,.075,'Separate seed 31006 repeat at batch 128 / LR ×2:',ha='center',fontsize=10,weight='bold')
    fig.text(.5,.045,'AdamW: USABLE 98.52%     Graft: FAILED 11.35%     SGDM: USABLE 98.01%',ha='center',fontsize=10)
    fig.subplots_adjust(left=.21,right=.98,bottom=.16,top=.85)
    fig.savefig(out/'plots/Model_Status_By_Setting.png',dpi=170,facecolor='white');plt.close(fig)


def paired_plot(estimates, out):
    rows=[('central','magnitude_graft'),('central','sgdm'),('stable','magnitude_graft'),('stable','sgdm')]
    labels=['Central setting — graft','Central setting — SGDM','Eight stable settings — graft','Eight stable settings — SGDM']
    fig,axs=plt.subplots(1,2,figsize=(12.8,5.6))
    for ax,metric,title in zip(axs,['mean_abs_alpha_minus_2','test_accuracy_pct'],['Change in mean hidden |α − 2|','Change in test accuracy (percentage points)']):
        for i,(scope,arm) in enumerate(rows):
            r=estimates[(estimates.scope==scope)&(estimates.treatment==arm)&(estimates.metric==metric)].iloc[0]
            ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o',color=COLORS[arm],capsize=5,markersize=7,lw=2)
            ax.annotate(f"{r['mean']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]",(r['mean'],i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9,color=COLORS[arm])
        ax.axvline(0,color='#4B5563',ls='--',lw=1)
        ax.axhline(1.5,color='#CBD5E1',lw=1)
        ax.set_yticks(range(4),labels if ax is axs[0] else ['']*4)
        ax.set_ylim(3.6,-.7);ax.set_title(title,fontsize=12,pad=12);ax.grid(axis='x',alpha=.15)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    axs[0].set_xlim(-.20,.06);axs[0].set_xlabel('Negative = closer to 2')
    axs[1].set_xlim(-.06,.31);axs[1].set_xlabel('Positive = higher accuracy')
    fig.suptitle('Paired effects relative to AdamW',fontsize=18,weight='bold',y=.98)
    fig.text(.5,.9,'Central: batch 256 / LR ×1. Stable subset: eight settings, selected after observing the failure.',ha='center',fontsize=10)
    fig.text(.5,.025,'Bars: descriptive paired 95% t intervals over five seeds; stable-setting differences are averaged within seed first.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.22,right=.98,bottom=.17,top=.78,wspace=.2)
    fig.savefig(out/'plots/Paired_Alignment_And_Accuracy.png',dpi=170,facecolor='white');plt.close(fig)


def report(registry,settings,estimates,out):
    lines=['# Model status and results summary','',
        '**138 completed runs: 132 usable classifiers and six confirmed hyperparameter-setting failures.**','',
        '`USABLE` means the saved model learned classification and has no observed collapse. `FAILED` means checkpoint diagnostics confirm collapsed FC2 activations and constant predictions. These are functional labels, not a ranking by α, a claim of optimal tuning, or certification of the theory. The older `complete` labels in execution logs mean the run reached epoch 120; a completed run can still produce a failed model.','',
        '| Model status / comparison role | Count | Use |','|---|---:|---|',
        '| USABLE, matched stable subset | 120 | Main comparison: eight shared settings × five paired seeds × three arms |',
        '| USABLE, controls at failed setting | 12 | Keep as controls; outside the shared stable subset, including two fresh-seed controls |',
        '| FAILED, collapsed graft | 6 | Hyperparameter failure register; not a usable classifier |','',
        '[Every model and its status](results/model_status.csv) · [Setting-level summary](results/model_setting_status.csv) · [Original run dispositions](results/run_disposition.csv)','',
        '## Which settings work?','',
        'All arms are usable at batch 128 with LR ×0.5 and ×1, and in all tested batch 256 and batch 512 settings. At batch 128/LR ×2, AdamW and SGDM remain usable; the graft fails in all six tested seeds. SGDM at that high-rate setting has weaker accuracy than at its lower batch-128 rates, so functional does not mean preferred. AdamW also shows the late validation-loss increase documented in the results report.','',
        '![Model status by setting](plots/Model_Status_By_Setting.png)','',
        'The table reports the original five seeds only. Accuracy is mean ± seed SD, not a confidence interval. The fresh-seed repeat is separate.','',
        '| Batch | LR multiplier | AdamW | Magnitude graft | SGDM |','|---:|---:|---|---|---|']
    original=settings[settings.cohort=='original']
    for (batch,multiplier),g in original.groupby(['batch_size','multiplier']):
        g=g.set_index('arm')
        cells=[f"{g.loc[a,'model_status']}: {g.loc[a,'test_accuracy_mean_pct']:.3f}% ± {g.loc[a,'test_accuracy_seed_sd_pp']:.3f}" for a in ARMS]
        lines.append(f'| {batch} | {multiplier:g} | '+' | '.join(cells)+' |')
    lines += ['', '## What did the intervention show?', '',
        'The intervention changes only matrix-update magnitudes. Adam direction, moments and bias updates remain in place; each SGDM target uses the graft run’s own gradient history.','',
        'At the prespecified central setting, the graft moves both hidden-layer mean α values closer to 2 and raises mean test accuracy from 98.220% to 98.344%. Across the eight shared stable settings, average distance to 2 improves, while the overall accuracy advantage over AdamW remains unresolved.','',
        '| Graft − AdamW | Mean hidden distance change | Accuracy change (pp) |','|---|---:|---:|']
    for scope,label in [('central','Central: batch 256/LR ×1'),('stable','Eight stable settings (post hoc)')]:
        subset=estimates[(estimates.scope==scope)&(estimates.treatment=='magnitude_graft')].set_index('metric')
        cells=[]
        for metric in ['mean_abs_alpha_minus_2','test_accuracy_pct']:
            r=subset.loc[metric];cells.append(f"{r['mean']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', 'Intervals are descriptive paired 95% Student t intervals over five seeds. For the stable subset, first average eight within-seed differences. The nine settings are not independent seeds. The sixth seed is a separate failure replication and contributes to no five-seed interval. The post-hoc stable subset is conditional on successful training and preserves the original full-grid analysis.','',
        '![Paired alpha alignment and accuracy](plots/Paired_Alignment_And_Accuracy.png)','',
        '## Provenance and model retrieval','',
        'Every row includes its exact run ID, cohort, settings, metrics, fit-support label, comparison role and checkpoint-artifact reference. The original and replication checkpoint-verification records were checked for this review; training and checkpoint replay were not repeated. All source data, training code, checkpoints and earlier plots are preserved.','',
        'Reproduce this status report from the repository with `python execution/build_model_status.py . .` from the experiment folder. This reads saved results and writes these presentation files; it never trains.','',
        '[Detailed stable results](RESULTS.md) · [Failure report](HYPERPARAMETER_FAILURES.md) · [Original full-grid results](FULL_GRID_RESULTS.md) · [Original checkpoints](ARTIFACTS.md) · [Replication checkpoints]('+REP+'ARTIFACTS.md) · [Status review audit](verification/Model_Status_Audit.json)','']
    (out/'MODEL_STATUS.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    for folder in ('results','plots','verification'):(args.output/folder).mkdir(parents=True,exist_ok=True)
    registry,settings=model_registry(args.source,args.output)
    estimates=paired_results(args.source,registry,args.output)
    status_plot(settings,args.output);paired_plot(estimates,args.output);report(registry,settings,estimates,args.output)
    audit={'passed':True,'total_runs':len(registry),'usable_models':132,'failed_models':6,'stable_comparison_runs':120,
           'usable_controls_outside_stable_subset':12,'original_seeds':[31001,31002,31003,31004,31005],
           'replication_seed':31006,'classification_uses_alpha':False,'all_failures_match_saved_checkpoint_diagnostics':True,
           'all_endpoint_metrics_match_original_sources':True,'paired_intervals_reproduce_saved_results':True,
           'existing_verification_reviewed':True,'checkpoint_replay_performed_this_review':False,'training_performed':False,
           'input_sha256':{f:hashlib.sha256((args.source/f).read_bytes()).hexdigest() for f in INPUTS}}
    (args.output/'verification/Model_Status_Audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
