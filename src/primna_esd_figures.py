"""Publication figures: all plotted data and captions are exported alongside PDFs."""
from __future__ import annotations
import math,textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from primna_esd_runtime import seed,ci,atomic_json
from primna_esd_data import DATASETS,MISSING,FAULTS,SPECS
from primna_esd_models import MODELS,ML,SHORT
from primna_esd_uncertainty import METHODS,CONDITIONS

COLORS=dict(zip(MODELS,['#666666','#A6A6A6','#0072B2','#009E73','#D55E00','#CC79A7']))
SHORT_DS={'Air Quality':'Air','Appliances Energy':'Energy'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def savefig(ctx,fig,stem,title,caption,data):
    fig.suptitle(title,fontsize=12,fontweight='bold')
    if ctx.cfg['profile']!='paper':fig.text(.5,-.115,'PILOT - engineering validation only',ha='center',fontsize=8)
    paths={'pdf':ctx.root/'figures_pdf'/(stem+'.pdf'),'png':ctx.root/'figures_png'/(stem+'.png')}
    dp=ctx.root/'plot_data'/(stem+'.csv');data.to_csv(dp,index=False,float_format='%.12g')
    try:
        with ctx.substage('save PDF | '+stem):
            fig.savefig(paths['pdf'],bbox_inches='tight',pad_inches=.16,metadata={'Title':title,'Subject':caption})
        with ctx.substage('save PNG | '+stem,dpi=ctx.cfg['png_dpi']):
            fig.savefig(paths['png'],bbox_inches='tight',pad_inches=.16,dpi=ctx.cfg['png_dpi'])
    finally:plt.close(fig)
    return {'stem':stem,'type':'figure','pdf':str(paths['pdf'].relative_to(ctx.root)),'plot_data':str(dp.relative_to(ctx.root)),
            'title':title,'caption':caption}


def table(ctx,stem,title,df,caption):
    df.to_csv(ctx.root/'tables'/(stem+'.csv'),index=False,float_format='%.12g')
    try:latex=df.to_latex(index=False,float_format=lambda x:f'{x:.4g}',escape=True)
    except ImportError:
        # Pandas delegates LaTeX to optional Jinja2 in recent releases. Keep the
        # notebook self-contained when that presentation dependency is absent.
        def esc(value):
            s='NA' if pd.isna(value) else f'{value:.4g}' if isinstance(value,(float,np.floating)) else str(value)
            for a,b in [('\\','\\textbackslash{}'),('&','\\&'),('%','\\%'),('$','\\$'),('#','\\#'),('_','\\_'),('{','\\{'),('}','\\}')]:s=s.replace(a,b)
            return s
        cols='l'*len(df.columns);lines=['\\begin{tabular}{'+cols+'}','\\toprule',' & '.join(map(esc,df.columns))+' \\\\','\\midrule']
        lines+=[' & '.join(esc(v) for v in row)+' \\\\' for row in df.itertuples(index=False,name=None)]
        latex='\n'.join(lines+['\\bottomrule','\\end{tabular}'])
    (ctx.root/'tables'/(stem+'.tex')).write_text(latex)
    path=ctx.root/'tables'/(stem+'.pdf');pages=math.ceil(len(df)/12) or 1
    with PdfPages(path) as pdf:
        for page in range(pages):
            d=df.iloc[page*12:(page+1)*12];fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis('off')
            fig.suptitle(title+f'  |  {page+1}/{pages}',fontsize=14,y=.96)
            def fmt(v):
                if pd.isna(v):return 'NA'
                if isinstance(v,(float,np.floating)):return f'{v:.3g}'
                return '\n'.join(textwrap.wrap(str(v),width=max(14,112//max(1,len(df.columns)))))
            headers=['\n'.join(textwrap.wrap(str(c),width=13)) for c in df.columns]
            cells=[[fmt(v) for v in row] for row in d.itertuples(index=False,name=None)]
            tab=ax.table(cellText=cells,colLabels=headers,cellLoc='center',loc='upper center',bbox=[0,.20,1,.72])
            tab.auto_set_font_size(False);tab.set_fontsize(9)
            for (r,c),cell in tab.get_celld().items():
                cell.set_linewidth(.3);cell.PAD=.08
                if r==0:cell.set_facecolor('#E9EDF2');cell.set_text_props(weight='bold');cell.set_height(.075)
                elif r%2==0:cell.set_facecolor('#F7F8FA')
            fig.text(.05,.085,textwrap.fill(caption,145),ha='left',va='bottom',fontsize=9)
            status=('Core cohorts are retrospective; independent external confirmation is reported separately.'
                    if ctx.cfg.get('run_external_confirmation') else 'Retrospective evaluation only; no untouched confirmatory claim.')
            fig.text(.05,.035,status+' NA denotes undefined or unavailable, never zero.',fontsize=8)
            fig.subplots_adjust(left=.04,right=.96,top=.91,bottom=.04);pdf.savefig(fig);plt.close(fig)
    return {'stem':stem,'type':'table','pdf':str(path.relative_to(ctx.root)),'plot_data':'tables/'+stem+'.csv','title':title,'caption':caption}


def newaxes(nrows=1,ncols=2,height=4.6):return plt.subplots(nrows,ncols,figsize=(10,height),squeeze=False,layout='constrained')


def legend_models(fig,models=MODELS):
    fig.legend(handles=[Line2D([0],[0],color=COLORS[m],marker='o',label=SHORT[m]) for m in models],
               loc='lower center',bbox_to_anchor=(.5,-.035),ncol=len(models),frameon=False)


def heatmaps(df,value,rows,columns,title):
    fig,axes=newaxes(height=4.4);v=df[value].to_numpy();v=v[np.isfinite(v)]
    lo=min(0,float(v.min())) if len(v) else 0;hi=max(.01,float(v.max())) if len(v) else 1
    for ax,ds in zip(axes.flat,DATASETS):
        mat=df[df.dataset==ds].pivot_table(index='model',columns=columns[0],values=value).reindex(index=rows,columns=columns[1])
        im=ax.imshow(mat,aspect='auto',cmap='viridis',vmin=lo,vmax=hi)
        ax.set_xticks(range(len(mat.columns)),mat.columns,rotation=35,ha='right');ax.set_yticks(range(len(rows)),[SHORT[x] for x in rows]);ax.set_title(ds)
        for i in range(len(rows)):
            for j in range(len(mat.columns)):
                val=mat.iloc[i,j];ax.text(j,i,'NA' if not np.isfinite(val) else f'{val:.2f}',ha='center',va='center',fontsize=8,
                  color='white' if np.isfinite(val) and val<(lo+hi)/2 else 'black')
    fig.colorbar(im,ax=list(axes.flat),shrink=.75,label=title)
    return fig


def stress_figures(ctx,rq,result,fitted):
    outputs=[];d=result['summary'];kind='missingness' if rq==1 else 'sensor_faults'
    mechanisms=MISSING if rq==1 else FAULTS;show=mechanisms[:3] if rq==1 else ['Gaussian','Calibration bias','Spikes']
    fig,axes=newaxes(2,3,7.3)
    for row,ds in enumerate(DATASETS):
        for col,mech in enumerate(show):
            ax=axes[row,col]
            for m in MODELS:
                g=d[(d.dataset==ds)&(d.mechanism==mech)&(d.model==m)].sort_values('severity')
                ax.plot(g.severity,g.relative_change*100,label=SHORT[m],color=COLORS[m],marker='o',markersize=3)
                ax.fill_between(g.severity,g.relative_low*100,g.relative_high*100,color=COLORS[m],alpha=.10)
            ax.axhline(0,color='black',linewidth=.6);ax.set_title(ds+'\n'+mech);ax.set_xlabel('Nominal severity');ax.set_ylabel('RMSE change (%)');ax.grid(alpha=.2)
    legend_models(fig)
    outputs.append(savefig(ctx,fig,f'RQ{rq}_F1_response_curves',f'RQ{rq}. Severity response, with paired uncertainty',
      'Curves are root mean squared errors pooled over corruption replicas relative to the same clean model. Bands: 95% replicate plus clock-day block bootstrap on the fixed historical test period. Nominal severity has mechanism-specific meaning. Datasets are not pooled.',d))
    a=result['audc'];fig=heatmaps(a[a.mechanism!='Descriptive equal-weight scenario mean'],'AUDC',MODELS,('mechanism',mechanisms),'Normalized area under relative-degradation curve')
    outputs.append(savefig(ctx,fig,f'RQ{rq}_F2_vulnerability_fingerprint',f'RQ{rq}. Mechanism-specific vulnerability',
      'Both panels use an identical color normalization. Negative areas are retained and indicate lower error in this stress setting, not universal superiority. All models, including naive baselines, are shown.',a))
    if rq==1:
        data=result['injection_audit'].groupby(['dataset','mechanism','severity'],as_index=False).agg(achieved=('achieved_new_missing_fraction','mean'),low=('achieved_new_missing_fraction',lambda x:x.quantile(.025)),high=('achieved_new_missing_fraction',lambda x:x.quantile(.975)))
        fig,axes=newaxes()
        for ax,ds in zip(axes.flat,DATASETS):
            for mech,g in data[data.dataset==ds].groupby('mechanism'):
                ax.plot(g.severity*100,g.achieved*100,'o-',label=mech);ax.fill_between(g.severity*100,g.low*100,g.high*100,alpha=.10)
            ax.plot([0,30],[0,30],':',color='black',label='Nominal = achieved');ax.set_title(ds);ax.set_xlabel('Nominal missingness (%)');ax.set_ylabel('Newly missing observed cells (%)');ax.grid(alpha=.2)
        axes[0,1].legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
        outputs.append(savefig(ctx,fig,'RQ1_F3_injection_manipulation_check','RQ1. Was the requested missingness actually achieved?',
          'Denominator is originally observed raw cells over every clock origin, not only origins with an observed future label. Channel rounding, protected MAR context, and conditional risk change realized rates. Bands are empirical corruption-replicate 2.5-97.5 percentiles, not confidence intervals.',data))
    else:
        data=result['tails'].groupby(['dataset','model','mechanism'],as_index=False).agg(tail95=('tail95','mean'),MAE=('MAE','mean'))
        fig,axes=newaxes()
        for ax,ds in zip(axes.flat,DATASETS):
            for m in MODELS:
                g=data[(data.dataset==ds)&(data.model==m)];ax.scatter(g.MAE,g.tail95,label=SHORT[m],color=COLORS[m],s=35)
            ax.set_xlabel('Mean absolute error ('+SPECS[ds]['unit']+')');ax.set_ylabel('95th-percentile absolute error');ax.set_title(ds);ax.grid(alpha=.2)
        legend_models(fig)
        outputs.append(savefig(ctx,fig,'RQ2_F3_tail_error_exposure','RQ2. Typical error and tail-error exposure are different',
          'Each point is a model/fault combination at the largest registered severity, averaged over replicas. This diagnostic distinguishes typical error from the tail of the error distribution; it is not an uncertainty interval or a worst-case guarantee.',data))
    clean=pd.read_csv(ctx.root/'results'/'P2_clean_summary.csv').query("split == 'test'")
    tab=a[a.mechanism=='Descriptive equal-weight scenario mean'].merge(clean[['dataset','model','RMSE']],on=['dataset','model'])
    worst=d.groupby(['dataset','model'],as_index=False).relative_change.max().rename(columns={'relative_change':'Worst scenario change'})
    tab=tab.merge(worst,on=['dataset','model'])[['dataset','model','RMSE','AUDC','AUDC_low','AUDC_high','Worst scenario change']]
    tab.columns=['Dataset','Model','Clean RMSE','Descriptive mean AUDC','AUDC 95% low','AUDC 95% high','Worst scenario change']
    tab['Model']=tab.Model.replace(SHORT)
    outputs.append(table(ctx,f'RQ{rq}_T1_accuracy_and_degradation',f'RQ{rq}. Absolute accuracy and relative degradation',tab,
       'Equal weights across the prespecified mechanisms, not learned weights. AUDC is normalized by the maximum severity. Worst scenario is the maximum pooled relative RMSE change. Clean RMSE units differ by dataset: Air mg/m3; Energy Wh per 10 minutes. Relative changes and AUDC are fractions (0.10 = 10%). Detailed paired block-sign tests and Holm adjustments are separate result files.'))
    return outputs


def explanation_figures(ctx,result,fitted):
    out=[];d=result['stability'];p=result['importance']
    conditions=list(dict.fromkeys(d.scenario));a=d.groupby(['dataset','model','scenario'],as_index=False).Spearman.mean()
    fig,axes=newaxes(height=4.8)
    for ax,ds in zip(axes.flat,DATASETS):
        mat=a[a.dataset==ds].pivot(index='model',columns='scenario',values='Spearman').reindex(index=ML,columns=conditions)
        im=ax.imshow(mat,vmin=-1,vmax=1,aspect='auto',cmap='coolwarm');ax.set_title(ds)
        ax.set_xticks(range(len(conditions)),conditions,rotation=40,ha='right');ax.set_yticks(range(len(ML)),[SHORT[m] for m in ML])
        for i in range(len(ML)):
            for j in range(len(conditions)):ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=list(axes.flat),label='Signed-importance Spearman correlation',shrink=.75)
    out.append(savefig(ctx,fig,'RQ3_F1_rank_stability','RQ3. Stability compared with repeat-only explanation noise',
      'Grouped permutation importance is signed. Clean repeat control changes permutation draws without corrupting the data. Other conditions use paired permutation draws. Both panels share [-1, 1]. Values are conditional on the fixed, contiguous explanation windows.',a))
    selected=[];fig,axes=newaxes(height=5)
    for ax,ds in zip(axes.flat,DATASETS):
        model=fitted[ds]['deployment_model'];g=p[(p.dataset==ds)&(p.model==model)&p.scenario.isin(['Clean','MCAR 20%'])].groupby(['scenario','group']).importance.mean().unstack(0)
        top=g.Clean.abs().nlargest(min(8,len(g))).index;g=g.loc[top].sort_values('Clean');g['dataset']=ds;g['model']=model;selected.append(g.reset_index())
        yy=np.arange(len(g));ax.barh(yy-.18,g.Clean,height=.36,label='Clean');ax.barh(yy+.18,g['MCAR 20%'],height=.36,label='MCAR 20%');ax.axvline(0,color='black',lw=.6)
        ax.set_yticks(yy,g.index);ax.set_xlabel('Signed increase in RMSE after permutation');ax.set_title(ds+' / '+SHORT[model]);ax.legend(frameon=False)
    out.append(savefig(ctx,fig,'RQ3_F2_signed_importance_changes','RQ3. Which evidence changes, and in which direction?',
      'Deployment model is selected using development CV only. Displayed groups are selected by absolute clean importance; negative importance is preserved. All lags of a sensor are permuted together within contiguous evaluation windows. This is grouped permutation evidence, not SHAP or a causal explanation.',pd.concat(selected)))
    fig,axes=newaxes();markers=['o','s','^','v','D','x']
    for ax,ds in zip(axes.flat,DATASETS):
        for m in ML:
            g=d[(d.dataset==ds)&(d.model==m)&(d.scenario!='Clean repeat control')]
            ax.scatter(g.relative_RMSE_change*100,g.Spearman,color=COLORS[m],label=SHORT[m],s=23,alpha=.6)
        ax.axvline(0,color='black',lw=.6);ax.set_ylim(-1.03,1.03);ax.set_title(ds);ax.set_xlabel('Prediction RMSE change (%)');ax.set_ylabel('Explanation-rank stability');ax.grid(alpha=.2)
    legend_models(fig,ML)
    out.append(savefig(ctx,fig,'RQ3_F3_prediction_explanation_dissociation','RQ3. Prediction and explanation can deteriorate differently',
       'Each point is a prespecified corruption replica. The figure diagnoses dissociation; it is not a pooled independent-sample correlation test. Clean-repeat controls are reported separately.',d))
    contrasts=result['contrasts'].copy();contrasts.dataset=contrasts.dataset.replace(SHORT_DS);contrasts.model=contrasts.model.replace(SHORT)
    out.append(table(ctx,'RQ3_T1_explanation_metrics','RQ3. Degradation beyond clean-repeat explanation noise',contrasts,
      'Every entry is a paired degraded-minus-clean-repeat contrast using the same model and replica. Negative rank or agreement contrasts indicate degradation beyond the explainer control. Intervals bootstrap paired replicas; p-values use paired sign flips and Holm correction within dataset/model/metric.'))
    return out


def drift_figures(ctx,result):
    out=[];c=result['calibration'];fig,axes=newaxes()
    symbols={'Page-Hinkley':'o','CUSUM':'s','EWMA':'^','Windowed KS':'D'}
    for ax,ds in zip(axes.flat,DATASETS):
        for method,g in c[c.dataset==ds].groupby('detector'):
            ax.scatter(g.observed_policy_far,g.observed_far_upper95,marker=symbols[method],s=32,label=method,alpha=.7)
            z=g[g.provisional_selected];ax.scatter(z.observed_policy_far,z.observed_far_upper95,s=160,facecolors='none',edgecolors='black')
            z=g[g.selected];ax.scatter(z.observed_policy_far,z.observed_far_upper95,s=45,facecolors='black',edgecolors='black')
        ax.axvline(ctx.cfg['target_far'],color='black',ls='--');ax.axhline(ctx.cfg['target_far'],color='black',ls='--')
        ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=1);ax.set_title(ds)
        ax.set_xlabel('Observed policy alarms / 1,000 steps');ax.set_ylabel('One-sided 95% FAR upper bound / 1,000');ax.grid(alpha=.2)
    axes[0,1].legend(frameon=False,loc='upper left',bbox_to_anchor=(1.02,1))
    out.append(savefig(ctx,fig,'RQ4_F1_calibration_feasibility','RQ4. Feasibility is a constraint, not a best-effort label',
      'Rings mark provisional candidates; filled markers require the exact one-sided event-rate upper bound to meet the target. Zero observed alarms are not treated as proof of a sufficiently low future false-alarm rate.',c))
    d=result['timelines'];fig,axes=newaxes(2,1,6)
    for ax,ds in zip(axes.flat,DATASETS):
        g=d[(d.dataset==ds)&(d.kind=='absolute_error')].sort_values('index');period=SPECS[ds]['period']
        sm=g.value.rolling(period,min_periods=1).mean();ax.plot((g['index']+1)/period,sm,label='Mean absolute error, one-day trailing window')
        point=g.injection_index.iloc[0]/period;ax.axvline(point,color='black',ls='--',label='Injected onset (evaluation only)')
        top=max(float(sm.max()),1e-6)
        for j,(method,aa) in enumerate(d[(d.dataset==ds)&(d.kind=='alarm')].groupby('detector')):
            ax.scatter((aa['index']+1)/period,np.full(len(aa),top*(1.06+.07*j)),marker=symbols[method],label=method,s=20)
        ax.set_title(ds);ax.set_xlabel('Days from test start');ax.set_ylabel('Absolute error ('+SPECS[ds]['unit']+')');ax.grid(alpha=.2)
        ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    out.append(savefig(ctx,fig,'RQ4_F2_alarm_timelines','RQ4. Residual evolution and all alarms, including early alarms',
      'Prespecified abrupt-mean replica 0. Alarm timestamps correspond to residual availability. Pre-injection alarms are not filtered out. Real chronological data have no verified natural-drift labels; these are alarms around a simulated measurement shift.',d))
    a=result['adaptation'];fig,axes=newaxes()
    conditions=list(dict.fromkeys(a.drift));plotrows=[]
    for ax,ds in zip(axes.flat,DATASETS):
        policies=['Periodic','Random budget-matched','Alarm-triggered','Guarded alarm']
        offsets=np.linspace(-.22,.22,len(policies))
        for j,policy in enumerate(policies):
            vals=[];los=[];his=[]
            for cond in conditions:
                g=a[(a.dataset==ds)&(a.policy==policy)&(a.drift==cond)];x=100*g.RMSE_difference/g.frozen_RMSE
                rng=np.random.default_rng(seed(ds,policy,cond,'adapt_ci'));bs=np.mean(rng.choice(x,(ctx.cfg['bootstrap'],len(x))),axis=1);lo,hi=ci(bs)
                pt=x.mean();vals.append(pt);los.append(max(0,pt-lo));his.append(max(0,hi-pt));plotrows.append({'dataset':ds,'policy':policy,'condition':cond,'RMSE_change_percent':pt,'low':lo,'high':hi,'replicates':len(g)})
            ax.errorbar(np.arange(len(conditions))+offsets[j],vals,yerr=[los,his],fmt='o',capsize=3,label=policy)
        ax.axhline(0,color='black',ls='--');ax.set_xticks(range(len(conditions)),conditions,rotation=35,ha='right');ax.set_title(ds);ax.set_ylabel('Full-stream RMSE change vs frozen (%)');ax.grid(alpha=.2)
    axes[0,1].legend(frameon=False)
    out.append(savefig(ctx,fig,'RQ4_F3_non_oracle_adaptation','RQ4. Does deployable adaptation beat frozen and periodic baselines?',
       'Every prediction precedes its outcome. The random policy receives the alarm policy\'s refit-attempt budget, separating timing value from compute value. Guarded candidates use trailing observed labels. Negative is better; comparison with periodic updating is also saved numerically.',pd.DataFrame(plotrows)))
    r=result['operating_results'];tab=r.groupby(['dataset','detector','drift'],as_index=False).agg(enabled=('enabled','first'),pre_FAR=('pre_injection_far','mean'),post_alarm=('post_injection_alarm','mean'),strict_alarm=('alarm_without_prior_alarm','mean'),clean_post=('matched_clean_post_alarm','mean'),delay=('delay_conditional','median'))
    gate=c[c.provisional_selected][['dataset','detector','observed_far_upper95','far_claim_supported']]
    tab=tab.merge(gate,on=['dataset','detector'],how='left');tab.dataset=tab.dataset.replace(SHORT_DS)
    tab.columns=['Dataset','Detector','Injected shift','Enabled','Pre alarms/1k','Post alarm rate','No-prior-alarm rate','Matched clean rate','Conditional residual lag','FAR upper 95%','FAR target supported']
    out.append(table(ctx,'RQ4_T1_operating_characteristics','RQ4. Operating characteristics and matched clean control',tab,
       'Rates are averages over simulated corruptions on one historical test period. Post-onset alarm is not proof of causal detection; compare the same-window clean alarm rate. Residual lag counts updates after the first affected residual, not elapsed time from sensor onset; zero means the arrival of that residual. Misses have undefined lag, never zero. Full censoring information is saved separately.'))
    return out


def uncertainty_figures(ctx,result,fitted):
    out=[];s=result['summary'];fig,axes=newaxes();marker={'Clean':'o','MCAR 20%':'s','Gaussian 10%':'^','Abrupt drift':'D'}
    col={METHODS[0]:'#0072B2',METHODS[1]:'#D55E00'}
    for ax,ds in zip(axes.flat,DATASETS):
        mod=fitted[ds]['deployment_model'];g=s[(s.dataset==ds)&(s.model==mod)]
        for row in g.itertuples():
            ax.errorbar(row.width,row.coverage,xerr=[[max(0,row.width-row.width_low)],[max(0,row.width_high-row.width)]],
                yerr=[[max(0,row.coverage-row.coverage_low)],[max(0,row.coverage_high-row.coverage)]],fmt=marker[row.condition],color=col[row.method],capsize=2)
        ax.axhline(ctx.cfg['coverage'],color='black',ls='--');ax.set_ylim(max(0,min(ctx.cfg['coverage'],s.coverage_low.min())-.04),1.02);ax.set_title(ds+' / '+SHORT[mod]);ax.set_xlabel('Mean interval width ('+SPECS[ds]['unit']+')');ax.set_ylabel('Observed coverage');ax.grid(alpha=.2)
    fig.legend([Line2D([0],[0],marker='o',color=col[m]) for m in METHODS],METHODS,loc='lower center',bbox_to_anchor=(.5,-.03),ncol=2,frameon=False)
    fig.legend([Line2D([0],[0],marker=marker[c],color='black',ls='') for c in CONDITIONS],CONDITIONS,loc='lower center',bbox_to_anchor=(.5,-.075),ncol=4,frameon=False)
    out.append(savefig(ctx,fig,'RQ5_F1_coverage_width','RQ5. Empirical coverage versus interval width',
       'Both methods use the identical calibration-B observations after a temporal gap from local-scale fitting. Bands use replicate plus clock-day resampling. Serial dependence and distribution shift preclude an unconditional exchangeability guarantee; the dashed line is nominal, not guaranteed.',s))
    d=result['curves'];fig,axes=newaxes();used=[]
    for ax,ds in zip(axes.flat,DATASETS):
        mod=fitted[ds]['deployment_model'];g=d[(d.dataset==ds)&(d.model==mod)&(d.replicate==0)];used.append(g)
        for k,cond in enumerate(CONDITIONS):
            local=g[(g.condition==cond)&(g.method==METHODS[1])&g.kind.str.startswith('attainable')].sort_values('accepted_fraction')
            line,=ax.plot(local.accepted_fraction,local.selective_RMSE,'o-',markersize=3,label=cond)
            random=g[(g.condition==cond)&(g.method==METHODS[1])&g.kind.str.startswith('random')]
            ax.plot(random.accepted_fraction,random.selective_RMSE,'--',color=line.get_color(),alpha=.5)
            glob=g[(g.condition==cond)&(g.method==METHODS[0])&g.kind.str.startswith('attainable')]
            ax.scatter(glob.accepted_fraction,glob.selective_RMSE,marker='x',color=line.get_color(),s=70)
        ax.set_title(ds+' / '+SHORT[mod]);ax.set_xlabel('Actually attainable accepted fraction');ax.set_ylabel('Selective RMSE');ax.grid(alpha=.2)
    axes[0,1].legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    out.append(savefig(ctx,fig,'RQ5_F2_tie_safe_risk_coverage','RQ5. Selective prediction without arbitrary tie breaking',
       'Prespecified replica 0. Solid dots: attainable local-width thresholds. Cross at 100%: constant global width has no individual ranking. Dashed lines: sqrt(expected MSE) for random selection. Operational thresholds are fixed on policy data; these curves are descriptive, not test-tuned policies.',pd.concat(used)))
    decision=result['decision_utility'];fig,axes=newaxes();used=[]
    event_q=float(min(decision.event_quantile.unique(),key=lambda x:abs(x-.90)))
    fn_cost=float(min(decision.false_negative_cost_ratio.unique(),key=lambda x:abs(x-5.0)))
    markers={'Point forecast':'o',METHODS[0]:'s',METHODS[1]:'D'}
    for ax,ds in zip(axes.flat,DATASETS):
        mod=fitted[ds]['deployment_model'];g=decision[(decision.dataset==ds)&(decision.model==mod)&
          (decision.event_quantile==event_q)&(decision.false_negative_cost_ratio==fn_cost)&
          (decision.abstention_cost==.25)]
        a=g.groupby(['dataset','model','condition','method','review_capacity'],as_index=False).agg(cost=('expected_cost','mean'),abstention=('abstention_rate','mean'),
          cost_low=('expected_cost',lambda x:x.quantile(.025)),cost_high=('expected_cost',lambda x:x.quantile(.975)));used.append(a)
        a=a[(a.method!='Point forecast') | (a.review_capacity==a.review_capacity.min())]
        for k,(cond,q) in enumerate(a.groupby('condition')):
            for row in q.itertuples():
                ax.errorbar(row.abstention,row.cost,yerr=[[max(0,row.cost-row.cost_low)],[max(0,row.cost_high-row.cost)]],
                  fmt=markers[row.method],color=plt.cm.tab10(k),capsize=2,label=cond if row.method=='Point forecast' else None)
        ax.set_title(ds+' / '+SHORT[mod]);ax.set_xlabel('Abstention/review fraction');ax.set_ylabel('Expected decision cost per origin');ax.grid(alpha=.2)
    axes[0,1].legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    fig.legend([Line2D([0],[0],marker=markers[m],color='black',ls='',label=m) for m in markers],
      list(markers),loc='lower center',bbox_to_anchor=(.5,-.04),ncol=3,frameon=False)
    out.append(savefig(ctx,fig,'RQ5_F3_environmental_decision_tradeoff','RQ5. Uncertainty changes environmental decision cost and review demand',
       f'High-event thresholds are frozen at the policy {event_q:.0%} quantile. The panel fixes abstention cost at 0.25, varies feasible review capacity, and uses a false-negative cost ratio of {fn_cost:g}. The full cost/capacity grid is saved; these are scenarios, not stakeholder utilities.',pd.concat(used)))
    dep={ds:fitted[ds]['deployment_model'] for ds in DATASETS}
    tab=s[s.apply(lambda r:r.model==dep[r.dataset],axis=1)][['dataset','model','condition','method','coverage','width','accepted','high_error_given_accepted']].copy()
    du=pd.concat(used,ignore_index=True).rename(columns={'cost':'Decision cost','abstention':'Review fraction'})
    du=du.sort_values('review_capacity').groupby(['dataset','model','condition','method'],as_index=False).first()
    du=du[['dataset','model','condition','method','Decision cost','Review fraction']]
    tab=tab.merge(du,on=['dataset','model','condition','method'],how='left')
    tab.dataset=tab.dataset.replace(SHORT_DS);tab.model=tab.model.replace(SHORT);tab.method=tab.method.replace({METHODS[0]:'Global',METHODS[1]:'Local-scale'})
    tab.columns=['Dataset','Model','Condition','Method','Coverage','Mean width','Acceptance','High error / accepted','Decision cost','Review fraction']
    out.append(table(ctx,'RQ5_T1_uncertainty_decision','RQ5. Calibration, warning, and decision utility are distinct endpoints',tab,
       f'Deployment models only. Decision columns use the policy {event_q:.0%} event threshold and false-negative cost ratio {fn_cost:g}; the complete prespecified sensitivity grid is in RQ5_decision_utility.csv. Costs are scenario assumptions, not inferred stakeholder preferences.'))
    return out


def compute_figures(ctx,result):
    out=[];d=result['measurements'];a=d.groupby(['dataset','model','stage'],as_index=False).agg(seconds=('seconds_per_operation','mean'),s_low=('seconds_per_operation',lambda x:x.quantile(.025)),s_high=('seconds_per_operation',lambda x:x.quantile(.975)),energy=('energy_kWh_per_operation','mean'),energy_kind=('energy_kind','first'),RSS=('RSS_peak_MB','max'),RSS_delta=('RSS_delta_MB','max'),repeats=('replicate','nunique'))
    clean=pd.read_csv(ctx.root/'results'/'P2_clean_summary.csv').query("split == 'test'")
    ref=clean[clean.model=='Persistence'][['dataset','RMSE']].rename(columns={'RMSE':'reference_RMSE'})
    use=a[a.stage=='Training'].merge(clean[['dataset','model','RMSE']],on=['dataset','model']).merge(ref,on='dataset');use['nRMSE']=use.RMSE/use.reference_RMSE
    fig,axes=newaxes()
    for ax,ds in zip(axes.flat,DATASETS):
        q=use[use.dataset==ds].copy();pareto=[]
        for r in q.itertuples():
            ax.errorbar(r.seconds,r.nRMSE,xerr=[[max(0,r.seconds-r.s_low)],[max(0,r.s_high-r.seconds)]],fmt='o',color=COLORS[r.model],capsize=3)
            ax.annotate(SHORT[r.model],(r.seconds,r.nRMSE),xytext=(5,4),textcoords='offset points')
            dominated=((q.seconds<=r.seconds)&(q.nRMSE<=r.nRMSE)&((q.seconds<r.seconds)|(q.nRMSE<r.nRMSE))).any()
            if not dominated:pareto.append((r.seconds,r.nRMSE))
        if pareto:
            pp=np.array(sorted(pareto));ax.plot(pp[:,0],pp[:,1],':',color='black',label='Mean-point nondominated set')
        ax.axhline(1,color='black',lw=.6);ax.set_xscale('log');ax.set_title(ds);ax.set_xlabel('Training seconds per fit');ax.set_ylabel('RMSE / persistence RMSE');ax.grid(alpha=.2)
    out.append(savefig(ctx,fig,'RQ6_F1_accuracy_runtime_frontier','RQ6. Accuracy and measured training cost',
      'Costs use one host, fixed model seeds, excluded warmups and randomized repeat order. Horizontal bars are the empirical 2.5-97.5 percentiles across timing repeats, not cross-hardware confidence intervals. The dotted nondominated set is computed from means only.',use))
    fig,axes=newaxes();energy_rows=a[a.stage=='Training'].copy()
    have_energy=np.isfinite(energy_rows.energy).any()
    for ax,ds in zip(axes.flat,DATASETS):
        q=energy_rows[energy_rows.dataset==ds].set_index('model').reindex(ML)
        for j,m in enumerate(ML):
            r=q.loc[m]
            if have_energy:
                if np.isfinite(r.energy):
                    marker='o' if str(r.energy_kind).startswith('counter:') else '^'
                    ax.scatter(j,r.energy*1000,s=70,marker=marker,color=COLORS[m])
                else:ax.text(j,.5,'Unavailable',ha='center',va='center',rotation=90,transform=ax.get_xaxis_transform(),fontsize=9)
            else:
                ax.scatter(j,r.RSS,s=60,marker='o',color=COLORS[m]);ax.annotate(f'+{r.RSS_delta:.1f} MB',(j,r.RSS),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(len(ML)),[SHORT[m] for m in ML]);ax.set_title(ds);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Energy per fit (Wh)' if have_energy else 'Measured process peak RSS (MB)')
        if not have_energy:ax.set_ylim(bottom=0);ax.margins(y=.15)
    if have_energy:
        fig.legend([Line2D([0],[0],marker='o',color='black',ls=''),Line2D([0],[0],marker='^',color='black',ls='')],
          ['CPU package counter (host scope)','CodeCarbon estimated/mixed'],loc='lower center',bbox_to_anchor=(.5,-.03),ncol=2,frameon=False)
    else:fig.text(.5,-.035,'Energy unavailable: displaying measured memory; labels show peak increase over initial RSS.',ha='center',fontsize=9)
    title='RQ6. Energy evidence is limited by exposed instrumentation' if have_energy else 'RQ6. Memory remains measurable when energy telemetry is unavailable'
    out.append(savefig(ctx,fig,'RQ6_F2_energy_evidence_and_provenance',title,
       'Missing telemetry remains unavailable, never zero or assumed model-specific wattage. CPU package counters are host-scoped; CodeCarbon fallback values are estimates. When all energy readings are unavailable this panel displays measured process peak RSS and increments, not a fabricated energy comparison. Process RSS includes prior allocations.',energy_rows))
    fig,axes=newaxes()
    for ax,ds in zip(axes.flat,DATASETS):
        for j,stage in enumerate(['Training','Inference / 1000']):
            q=a[(a.dataset==ds)&(a.stage==stage)].set_index('model').reindex(ML)
            ax.errorbar(np.arange(len(ML))+(j-.5)*.18,q.seconds,yerr=[np.maximum(0,q.seconds-q.s_low),np.maximum(0,q.s_high-q.seconds)],fmt='o',capsize=3,label=stage)
        ax.set_yscale('log');ax.set_xticks(range(len(ML)),[SHORT[m] for m in ML]);ax.set_title(ds);ax.set_ylabel('Seconds (log scale)');ax.grid(axis='y',alpha=.2)
    axes[0,1].legend(frameon=False)
    out.append(savefig(ctx,fig,'RQ6_F3_training_inference_cost','RQ6. Training and inference are distinct operational costs',
       'Inference is normalized per 1,000 predictions. Time covers computation, whereas energy covers the tracked operation batch and may also include instrumentation overhead. No arbitrary robustness-per-watt composite is formed.',a))
    tab=a.copy();tab.dataset=tab.dataset.replace(SHORT_DS);tab.model=tab.model.replace(SHORT);tab.energy_kind=tab.energy_kind.map(lambda x:'Counter CPU' if x.startswith('counter:') else 'Unavailable' if x.startswith('unavailable') else 'Estimated/mixed')
    tab=tab[['dataset','model','stage','repeats','seconds','energy','energy_kind','RSS','RSS_delta']]
    tab.columns=['Dataset','Model','Operation','Repeats','Time (s)','Energy (kWh)','Energy provenance','Peak RSS MB','RSS increase MB']
    out.append(table(ctx,'RQ6_T1_compute_and_measurement_scope','RQ6. Measured cost and energy qualification',tab,
       'Peak RSS is process resident memory, not model-only allocation. A zero RSS increment can mean previously allocated memory was reused. Energy estimates are not converted into carbon claims. Detailed operation counts, warnings and hardware records are included.'))
    return out


def all_figures(ctx):
    outputs=[]
    outputs+=stress_figures(ctx,1,ctx.results['RQ1'],ctx.results['P2'])
    outputs+=stress_figures(ctx,2,ctx.results['RQ2'],ctx.results['P2'])
    outputs+=explanation_figures(ctx,ctx.results['RQ3'],ctx.results['P2'])
    outputs+=drift_figures(ctx,ctx.results['RQ4'])
    outputs+=uncertainty_figures(ctx,ctx.results['RQ5'],ctx.results['P2'])
    outputs+=compute_figures(ctx,ctx.results['RQ6'])
    for rq in range(1,7):assert sum(o['stem'].startswith('RQ'+str(rq)+'_') for o in outputs)==4
    atomic_json(ctx.root/'manifests'/'publication_artifacts.json',outputs)
    captions=[]
    for o in outputs:captions.append('## '+o['stem']+'\n'+o['title']+'\n\n'+o['caption']+'\n')
    (ctx.root/'FIGURE_TABLE_CAPTIONS.md').write_text('\n'.join(captions))
    return outputs
