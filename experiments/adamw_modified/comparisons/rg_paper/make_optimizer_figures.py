"""Rebuild the four manuscript plates. Run in the directory containing the two CSVs.
Pass the original RG paper PNG as the first argument (see the appendix link)."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image
root=Path(__file__).resolve().parent
data=pd.read_csv(root/'optimizer_checkpoints.csv')
assert (data.epoch>0).all()
dataset_names = {'Digits sweep': 'Digit MNIST', 'Fashion original': 'Fashion-MNIST',
                 'Fashion continuation': 'Fashion-MNIST'}
data['dataset'] = data.cohort.map(dataset_names)
assert data.dataset.notna().all()
rows = ['Digit MNIST', 'Fashion-MNIST']
epoch_colors = {20: '#70aed4', 30: '#24649a', 60: '#8256a1', 90: '#d17a17', 120: '#933a28'}
assert set(data.epoch) == set(epoch_colors)
arms=['sgdm','adamw','magnitude_graft']
names=['SGD + momentum','AdamW','Modified AdamW']
blue,orange,green='#2467a0','#cb571d','#127f42'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'font.size':9,'axes.edgecolor':'#8e9ba6','axes.axisbelow':True})
keys=['seed','batch','rate']
def winner(v):
    v=v.assign(correct=np.rint(v.test_accuracy_pct*100).astype(int),distance=(v.alpha_model-2).abs())
    assert np.allclose(v.correct,v.test_accuracy_pct*100)
    return v.sort_values(['correct','distance','epoch','seed','batch','rate'],ascending=[False,True,True,True,True,True]).iloc[0]
def scatter(ax,v,x,y):
    for epoch, color in epoch_colors.items():
        points = v[v.epoch.eq(epoch)]
        ax.scatter(points[x], points[y], s=17, c=color, alpha=.78,
                   linewidths=.2, edgecolors='white', zorder=2)

def mark(ax,x,y):
    ax.scatter([x],[y],s=95,marker='^',c=green,edgecolor='white',lw=.8,zorder=8)

def epoch_key(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], label=str(epoch), marker='o', linestyle='none',
                      color=color, markersize=6) for epoch,color in epoch_colors.items()]
    handles.append(Line2D([], [], label='Best accuracy', marker='^', linestyle='none',
                          color=green, markersize=8))
    fig.legend(handles=handles, title='Completed training epochs', loc='lower center',
               bbox_to_anchor=(.51, .003), ncol=6, frameon=False,
               handlelength=1, columnspacing=1.45, handletextpad=.3, fontsize=9)
def grid(ax):
    ax.grid(axis='y', linewidth=.5, color='#dde4e8')
    ax.tick_params(length=2.5, pad=2)

def style(ax):
    grid(ax)
    ax.axvline(2, linestyle=':', linewidth=1.1, color='#384551')
def accuracy_plate():
    fig = plt.figure(figsize=(7.9, 6.05))
    totals = dict(plotted=0, collapsed=0, above_axis=0)
    for i, dataset in enumerate(rows):
        top = .897 - i*.399
        fig.text(.02, top+.070, rows[i], fontsize=12, weight='bold', color='#173e56')
        ylimits, yticks = [
            ((95.7, 98.95), [96, 97, 98]),
            ((85.4, 90.55), [86, 87, 88, 89, 90]),
        ][i]
        for j, arm in enumerate(arms):
            records = data.query('dataset == @dataset and arm == @arm and layer == "fc1"')
            best = winner(records)
            # Only the diagnosed constant-prediction checkpoints are training failures.
            failed = ((records.cohort == 'Digits sweep') & (records.arm == 'magnitude_graft') &
                      (records.batch == 128) & (records.rate == 2) &
                      np.isclose(records.test_accuracy_pct, 11.35))
            above = records.alpha_model > 2.8
            assert not (failed & above).any()
            plotted = records[~(failed | above)]
            totals['plotted'] += len(plotted)
            totals['collapsed'] += int(failed.sum())
            totals['above_axis'] += int(above.sum())
            assert best.name in plotted.index
            assert plotted.alpha_model.between(1.68, 2.8).all()
            assert plotted.test_accuracy_pct.between(*ylimits).all()
            height = .278
            ax = fig.add_axes([.094+j*.308, top-height, .270, height])
            ax.set(xlim=(1.68,2.8), xticks=[1.8,2,2.2,2.4,2.6,2.8], ylim=ylimits, yticks=yticks)
            grid(ax)
            ax.axvline(2, linestyle=':', linewidth=1.2, color='#384551', zorder=0.9)
            ax.set_title(f'{names[j]} · {best.test_accuracy_pct:.2f}%', size=9.5, weight='bold', pad=3)
            scatter(ax, plotted, 'alpha_model', 'test_accuracy_pct')
            mark(ax, best.alpha_model, best.test_accuracy_pct)
            if j == 0:
                ax.set_ylabel('Test accuracy (%)', fontsize=9)
    assert totals == dict(plotted=1134, collapsed=17, above_axis=19), totals
    fig.text(.51,.143,'Weighted average layer α',ha='center',va='bottom',size=10.5)
    epoch_key(fig)
    return fig

with PdfPages(root/'optimizer_figures.pdf') as pdf:
    fig = accuracy_plate()
    pdf.savefig(fig)
    plt.close(fig)
    fig=plt.figure(figsize=(7.9,7.76))
    for i,dataset in enumerate(rows):
        top = .922 - i*.420
        fig.text(.02,top+.049,dataset,fontsize=12,weight='bold',color='#173e56')
        for j,arm in enumerate(arms):
            s = data.query('dataset==@dataset and arm==@arm')
            win = winner(s.query('layer=="fc1"'))
            for k,layer in enumerate(['fc1','fc2']):
                v = s[s.layer.eq(layer)]
                ax = fig.add_axes([.096+j*.307,top-.147-.163*k,.266,.147])
                style(ax)
                ax.axhline(0,c='#54626d',lw=.8)
                ax.set(xlim=(1.25,4.6),xticks=[1.5,2,3,4],ylim=(-147,147),yticks=[-100,0,100])
                assert v.alpha_layer.between(1.25,4.6).all() and v.gap.abs().max()<147
                scatter(ax,v,'alpha_layer','gap')
                mask=np.logical_and.reduce([v[col].eq(win[col]) for col in ['cohort',*keys,'epoch']])
                t=v[mask]
                assert len(t)==1
                mark(ax,t.alpha_layer.item(),t.gap.item())
                if not k:
                    ax.set_title(names[j],size=10,weight='bold',pad=3)
                    ax.tick_params(labelbottom=False)
                if j==0:
                    ax.set_ylabel(layer.upper()+' gap')
    fig.text(.51,.115,'Layer α',ha='center',size=10.5)
    epoch_key(fig)
    pdf.savefig(fig)
    plt.close(fig)
    img=Image.open(sys.argv[1]);size=img.size
    fig=plt.figure(figsize=(8,8*size[1]/size[0]));ax=fig.add_axes([0,0,1,1]);ax.imshow(img);ax.axis('off');pdf.savefig(fig,dpi=440);plt.close(fig)
    g=pd.read_csv(root/'muon_30_epochs.csv');fig,axs=plt.subplots(1,2,figsize=(7.45,2.65),layout='constrained')
    for ax,metric,yl in zip(axs,['alpha','ERG_gap'],['Layer α','ERG gap (eigenvalues)']):
        for layer,color,line in [('fc1','#993c08','-'),('fc2','#cc640c','--'),('fc3','#855c31',':')]:
            v=g[g.layer.eq(layer)&g.metric.eq(metric)].sort_values('epoch')
            ax.plot(v.epoch,v['mean'],color=color,linestyle=line,lw=1.6,label=layer.upper());ax.fill_between(v.epoch,v.minimum,v.maximum,color=color,alpha=.15)
        ax.axhline(2 if metric=='alpha' else 0,ls=':',lw=1,c='#384551');ax.set(xlabel='Epoch',ylabel=yl,xlim=(0,30));ax.legend(frameon=False,fontsize=8);grid(ax)
    pdf.savefig(fig);plt.close(fig)
