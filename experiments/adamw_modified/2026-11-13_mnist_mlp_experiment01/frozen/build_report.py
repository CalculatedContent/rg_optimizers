"""Render a compact technical record after experiment completion and verification."""
import sys,json,html
from pathlib import Path
sys.path.insert(0,'/tmp/adamw_shape_deps')
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
OUT=Path(__file__).resolve().parent
LABEL={'adamw':'AdamW','magnitude_graft':'Magnitude graft','sgdm':'SGD + momentum'}

def main():
    audit=json.loads((OUT/'Final_Verification.json').read_text());assert audit['passed']
    o=pd.read_csv(OUT/'Overall_Summaries.csv');p=pd.read_csv(OUT/'Paired_Summaries.csv');c=pd.read_csv(OUT/'Setting_Summaries.csv')
    l=pd.read_csv(OUT/'All_Layer_Measurements.csv');e=pd.read_csv(OUT/'Endpoints.csv')
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    pdfmetrics.registerFont(TTFont('DV',font));pdfmetrics.registerFont(TTFont('DVB',bold))
    styles=getSampleStyleSheet()
    for n in ['Normal','BodyText','Title','Heading1','Heading2']:styles[n].fontName='DV'
    styles['Title'].fontName='DVB';styles['Title'].fontSize=21;styles['Title'].leading=26
    styles['Heading1'].fontName='DVB';styles['Heading1'].fontSize=14;styles['Heading1'].leading=18
    styles['Normal'].fontSize=9;styles['Normal'].leading=13
    styles.add(ParagraphStyle(name='SmallDV',fontName='DV',fontSize=7.4,leading=10))
    story=[]
    def para(text):story.append(Paragraph(text,styles['Normal']));story.append(Spacer(1,7))
    def head(text):story.append(Paragraph(text,styles['Heading1']));story.append(Spacer(1,7))
    def image(name,width=506):
        from PIL import Image as PI
        im=PI.open(OUT/name);story.append(Image(str(OUT/name),width=width,height=width*im.height/im.width));story.append(Spacer(1,8))
    def table(rows,widths):
        data=[[Paragraph(str(x),styles['SmallDV']) for x in row] for row in rows]
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff6')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8090a0')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d4dce3'))]))
        story.append(tab);story.append(Spacer(1,9))
    def metric(arm,m):return o[(o.arm==arm)&(o.metric==m)].iloc[0]
    def effect(m):return p[(p.scope=='overall')&(p.treatment=='magnitude_graft')&(p.reference=='adamw')&(p.metric==m)].iloc[0]
    def ci(r):return f'{r["mean"]:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]'
    d=effect('mean_abs_alpha_minus_2');acc=effect('test_accuracy_pct');a1=effect('alpha_fc1');a2=effect('alpha_fc2')
    story.append(Paragraph('AdamW magnitude replacement<br/>Experiment 01: MNIST',styles['Title']))
    para('Charles Martin research record | 13 November 2026 | Fixed protocol, five paired seeds, no CNNs')
    para(f'<b>Completed:</b> 135 runs, 120 epochs each, on a fully connected 784→256→256→10 ReLU MLP. Training took {audit["training_wall_seconds"]/3600:.2f} hours. The experiment has stopped; no next dataset or architecture has started.')
    head('Measured outcome')
    para(f'Across the nine preset batch-size/learning-rate settings, replacing only the Adam matrix-update magnitude changed FC1 α by {ci(a1)} and FC2 α by {ci(a2)}. The paired change in mean hidden-layer distance |α−2| was {ci(d)}. The paired test-accuracy change was {ci(acc)} percentage points. Brackets are descriptive 95% t intervals over five independent seed averages.')
    rows=[['Optimizer','FC1 α','FC2 α','Mean |α−2|','Test accuracy (%)']]
    for arm in ['adamw','magnitude_graft','sgdm']:
        vals=[]
        for m in ['alpha_fc1','alpha_fc2','mean_abs_alpha_minus_2','test_accuracy_pct']:
            r=metric(arm,m);vals.append(f'{r["mean"]:.3f} ± {r.seed_sd:.3f}')
        rows.append([LABEL[arm],*vals])
    table(rows,[112,90,90,104,110])
    para('The table averages settings equally within each seed before computing the mean and seed SD. It does not pool layers or hyperparameter settings as independent replicates. The separate setting tables retain all variation across batch size and learning rate.')
    head('Intervention and controls')
    para('Let mₜ=0.9mₜ₋₁+0.1gₜ and let Aₜ be the usual bias-corrected Adam matrix update. The graft uses Dₜ = ||−η<sub>S</sub>mₜ/0.1||<sub>F</sub> Aₜ/||Aₜ||<sub>F</sub>. Adam’s direction, moment history, ε and bias updates remain in place. The SGDM magnitude is computed from the graft run’s own gradients, not from a separate SGDM trajectory. The reference arms are native AdamW and SGD with momentum 0.9.')
    para('A scalar magnitude replacement changes the trajectory but does not establish an exact RG symmetry or volume-preserving map. An upward shift in α and movement toward α=2 are measured separately because overshooting 2 can increase the distance.')
    story.append(PageBreak());head('Every setting: layer α')
    image('Layer_Alpha_All_Settings.png')
    para('Primary spectral outcomes use FC1 and FC2, each with 256 eigenvalues. The output head has only 10 eigenvalues; its exponent is saved and reported as exploratory, and is excluded from the primary hidden-layer averages. WeightWatcher chooses the tail without targeting α=2.')
    story.append(PageBreak());head('Paired effects and accuracy')
    image('Paired_Effects_All_Settings.png')
    image('Accuracy_vs_Alpha.png')
    para('Every dot is a fixed-epoch endpoint. No checkpoint was selected for its test performance or α. Accuracy-versus-spectrum correlations in the data archive are descriptive within each optimizer; shared seeds make an independent-run p-value inappropriate.')
    story.append(PageBreak());head('Training trajectory and exact configuration')
    image('Central_Setting_Trajectories.png')
    table([
        ['Item','Prespecified value'],
        ['Data','MNIST: 55,000 training, 5,000 validation, 10,000 test. Original saved split; input pixels divided by 255; no augmentation.'],
        ['Seeds','31001, 31002, 31003, 31004, 31005. Same initialization and per-epoch example order across all arms and settings.'],
        ['Batch sizes / multipliers','128, 256, 512 / 0.5, 1, 2. Full factorial: 5 × 3 × 3 × 3 = 135 runs.'],
        ['Peak learning rates','Adam: 0.001 in each layer. SGDM magnitude and SGDM baseline: [0.04, 0.2, 0.8] for FC1, FC2, FC3. Multiply every rate by the chosen multiplier. Graft bias rates follow Adam.'],
        ['Schedule / moments','120 epochs; 5-epoch linear warmup; cosine to 10% of peak. β₁=0.9, β₂=0.999, ε=10⁻⁸. SGDM has no Nesterov correction or dampening.'],
        ['Decay control','Each parameter receives factor (1−10⁻⁵s)^(actual batch size/256), with schedule s. Decay per processed example is equal across batch sizes and arms. This holds shrinkage fixed when LR changes; it is not a fixed raw weight_decay hyperparameter sweep.'],
        ['Monitoring','Default WeightWatcher fits and raw squared singular values every 10 epochs. Full checkpoints at epochs 0, 30, 60, 90, 120. Per-step update summaries for the first two and each tenth batch.'],
    ],[112,394])
    story.append(PageBreak());head('Verification and statistical limits')
    aa=audit['analysis_audit'];pf=json.loads((OUT/'Preflight_Audit.json').read_text())
    para(f'All 135 final model hashes match their saved checkpoints. All {audit["checkpoints_hashed"]} checkpoints have file hashes. Saved logits reproduce every final accuracy count; the maximum cross-entropy discrepancy is {aa["max_saved_logit_cross_entropy_difference"]:.2g}. Six selected runs reproduce their raw spectra exactly after restoration and replay the next eleven training batches, including three exact saved loss/update-statistic comparisons per run.')
    para(f'Independent PyTorch reference checks passed for AdamW and SGDM. The graft’s float32 preflight norm error was below 10⁻⁷; Adam-direction cosine error was below 10⁻¹². Over {aa["recorded_graft_magnitudes_checked"]:,} logged matrix updates, the largest magnitude-matching error was {aa["max_recorded_graft_magnitude_relative_error"]:.2g}. Scalar weight dilation changed fitted α by at most {pf["radial_scaling_max_alpha_error"]:.2g}. Frobenius norms use double-precision accumulation; model parameters remain float32.')
    para(f'At the endpoint, {aa["final_hidden_supported"]}/{aa["final_hidden_total"]} hidden-layer fits meet the prespecified sensitivity criterion: at least 50 total eigenvalues, at least 20 tail eigenvalues, and KS D≤0.2. Raw fits, support flags, fixed-rank estimates and matched-support comparisons are retained. The main endpoint does not delete fits based on the direction of the effect.')
    para('For each metric and setting, differences are paired by seed. The overall comparison averages the nine setting-level differences within a seed, then uses a Student t interval with four degrees of freedom. These intervals quantify seed variation conditional on this fixed dataset, architecture and hyperparameter grid. They are not confidence intervals over architectures or datasets, and are not adjusted for multiple endpoints. Five seeds provide limited precision.')
    para('A change in the exponent after a magnitude-only intervention is evidence about this optimizer recipe and training trajectory. It does not isolate a universal RG correction, prove that AdamW necessarily drives α below 2, or show that higher test accuracy must occur whenever α moves closer to 2. The SGDM control is evaluated empirically; its spectral drift is not assumed to vanish.')
    head('All endpoint settings')
    rows=[['Batch','LR ×','Arm','FC1 α','FC2 α','Mean |α−2|','Acc. (%)']]
    for b in [128,256,512]:
        for m in [.5,1.,2.]:
            for arm in ['adamw','magnitude_graft','sgdm']:
                g=e[(e.batch_size==b)&(e.multiplier==m)&(e.arm==arm)]
                rows.append([b,m,LABEL[arm],*[f'{g[q].mean():.3f}' for q in ['alpha_fc1','alpha_fc2','mean_abs_alpha_minus_2','test_accuracy_pct']]])
    table(rows,[38,40,100,76,76,90,86])
    para('Full per-seed measurements and setting-specific seed SDs/intervals are provided in Endpoints.csv, Setting_Summaries.csv and Paired_Summaries.csv. Code, dataset hashes, saved split, optimizer states, spectra and sampled update statistics are included in the reproducibility archives.')
    def footer(canvas,doc):
        canvas.setFont('DV',7);canvas.setFillColor(colors.HexColor('#657386'))
        canvas.drawString(44,25,'Experiment 01 | MNIST magnitude-only AdamW graft | 13 Nov 2026')
        canvas.drawRightString(551,25,str(doc.page))
    pdf=OUT/'Experiment01_MNIST_Technical_Report.pdf'
    SimpleDocTemplate(str(pdf),pagesize=(595,842),rightMargin=44,leftMargin=44,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    record={'experiment':'01','status':'completed_and_verified','dataset':'MNIST','architecture':[784,256,256,10],
       'runs':135,'seeds':[31001,31002,31003,31004,31005],'epochs':120,
       'training_hours':audit['training_wall_seconds']/3600,
       'graft_minus_adamw':{m:effect(m).to_dict() for m in ['alpha_fc1','alpha_fc2','mean_abs_alpha_minus_2','test_accuracy_pct','test_loss']},
       'verification_passed':True,'stop_for_user_review':True,'next_experiment_started':False}
    import study as s
    s.writej(OUT/'Research_Record_Experiment01.json',record)
    print(str(pdf))

if __name__=='__main__':main()
