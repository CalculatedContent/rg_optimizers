"""Descriptive seed variation, endpoints kept distinct from validation selection."""
import argparse
import json
from pathlib import Path
import statistics
from common import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=[json.loads(p.read_text()) for p in sorted((a.output/'runs').glob('*/result.json'))]
    if not records:return
    summary=dict(completed=len(records),planned=10,seed_statistics={})
    lines=[f'Short GPT seed replication: {len(records)}/10 completed.',
           'Seed variation only. Cap stops do not prove convergence. Same previously inspected test partition.',
           'Uncertainty is across seeds; it does not cover dataset/domain shift. Partial summaries are provisional.','']
    for choice in ['endpoint','validation_selected']:
        stats={}
        for metric in ['alpha','train_loss','validation_loss','test_loss']:
            v=[r[choice]['alpha']['mean_alpha'] if metric=='alpha' else r[choice]['metrics'][metric.removesuffix('_loss')]['loss_nats_per_character'] for r in records]
            v=[x for x in v if x is not None]
            stats[metric]=dict(n=len(v),mean=statistics.mean(v) if v else None,sd=statistics.stdev(v) if len(v)>1 else None,
                               min=min(v) if v else None,max=max(v) if v else None)
        summary['seed_statistics'][choice]=stats
        lines.append(choice+': '+json.dumps(stats))
    summary['confirmed_plateaus']=sum(r['practical_validation_plateau_confirmed'] for r in records)
    summary['total_training_seconds']=sum(r['timing_seconds']['training_updates'] for r in records)
    summary['records']=records
    write_json(a.output/'results.json',summary)
    (a.output/'Results.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for r in records:
        h=[x for x in r['history'] if 'validation' in x]
        axes[0].plot([x['step'] for x in h],[x['validation']['loss_nats_per_character'] for x in h],label=str(r['seed']),lw=1)
    for split,color in [('train','#337a9b'),('validation','#ad6031'),('test','#438346')]:
        rr=[r for r in records if r['endpoint']['alpha']['mean_alpha'] is not None]
        axes[1].scatter([r['endpoint']['alpha']['mean_alpha'] for r in rr],[r['endpoint']['metrics'][split]['loss_nats_per_character'] for r in rr],label=split,color=color)
    axes[0].set(xlabel='Optimizer updates',ylabel='Full validation loss (nats/character)',title=f'{len(records)}/10 seeds')
    axes[1].set(xlabel='Mean alpha, all 18 unique matrices',ylabel='Endpoint loss (nats/character)',title='Individual seed measurements')
    axes[1].axvline(2,color='gray',ls='--',lw=1)
    for ax in axes:ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    fig.savefig(a.output/'Seed_Results.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
