"""Independent cross-domain replication on public environmental time series.

This phase is never labelled prospectively confirmatory by software. Genuine
confirmation requires an externally timestamped protocol and author attestation.
"""
from __future__ import annotations
import io, json, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from primna_esd_runtime import atomic_json, digest, filehash, seed, metrics, ci
from primna_esd_data import features, corrupt, scenario, cv_folds
from primna_esd_models import factory, grid, fit_logged
from primna_esd_uncertainty import calibrate_intervals, conformal_quantile
from primna_esd_figures import savefig, table, newaxes

EXTERNAL_SPECS={
 'Beijing Aotizhongxin':{
   'target':'CO','frequency':'1h','period':24,'lags':[1,2,3,6,12,23,24],
   'windows':[3,6,24],'unit':'ug/m3','filename':'PRSA_Data_Aotizhongxin_20130301-20170228.csv',
   'url':'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip',
   'doi':'10.24432/C5RK5G','domain':'air quality'},
 'Household Power':{
   'target':'Global_active_power','frequency':'10min','period':144,'lags':[1,2,3,6,12,18,36,143,144],
   'windows':[6,18,144],'unit':'kW','filename':'household_power_consumption.txt',
   'url':'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip',
   'doi':'10.24432/C58K54','domain':'energy demand'}
}
EXTERNAL_NAMES=list(EXTERNAL_SPECS)
EXT_MODELS=['Persistence','Seasonal naive','Ridge','HistGradientBoosting']
EXT_SHORT={'Persistence':'Persist.','Seasonal naive':'Seasonal','Ridge':'Ridge','HistGradientBoosting':'HGB'}
EXT_COLORS={'Persistence':'#666666','Seasonal naive':'#A6A6A6','Ridge':'#0072B2','HistGradientBoosting':'#D55E00'}
EXT_CONDITIONS=['Clean','MCAR 20%','Block 20%','Gaussian 10%','Abrupt drift']


def _download_or_find(ctx,name):
    spec=EXTERNAL_SPECS[name];explicit=ctx.cfg.get('external_data_paths',{}).get(name)
    if explicit:
        p=Path(explicit)
        if not p.exists():raise FileNotFoundError(p)
        return p
    for root in [Path('/kaggle/input'),ctx.root/'data']:
        if root.exists():
            hits=sorted(root.rglob(spec['filename']))
            if hits:return hits[0]
    if not ctx.cfg['allow_download']:
        raise FileNotFoundError('Attach '+spec['filename']+' as Kaggle input or enable Internet.')
    ctx.log('Downloading official UCI external-confirmation archive: %s',spec['url'])
    archive=ctx.root/'data'/('external_'+digest(name)[:10]+'.zip')
    with urllib.request.urlopen(spec['url'],timeout=180) as r:archive.write_bytes(r.read())
    with zipfile.ZipFile(archive) as outer:
        direct=[n for n in outer.namelist() if Path(n).name==spec['filename']]
        if direct:
            dest=ctx.root/'data'/spec['filename'];dest.write_bytes(outer.read(direct[0]));return dest
        for member in outer.namelist():
            if member.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
                    hits=[n for n in inner.namelist() if Path(n).name==spec['filename']]
                    if hits:
                        dest=ctx.root/'data'/spec['filename'];dest.write_bytes(inner.read(hits[0]));return dest
    raise RuntimeError('Official archive did not contain '+spec['filename'])


def _parse(path,name):
    if name=='Beijing Aotizhongxin':
        d=pd.read_csv(path,na_values=['NA','?'])
        t=pd.to_datetime(d[['year','month','day','hour']],errors='raise')
        d=d.drop(columns=['No','year','month','day','hour','wd','station'],errors='ignore')
        d=d.apply(pd.to_numeric,errors='raise')
    else:
        d=pd.read_csv(path,sep=';',na_values=['?',''],low_memory=False)
        t=pd.to_datetime(d.pop('Date').astype(str)+' '+d.pop('Time').astype(str),format='%d/%m/%Y %H:%M:%S',errors='raise')
        d=d.apply(pd.to_numeric,errors='coerce');d.index=pd.DatetimeIndex(t)
        d=d.resample('10min').mean();t=d.index
    d.index=pd.DatetimeIndex(t);d=d.sort_index()
    if d.index.duplicated().any():raise ValueError(name+': duplicate timestamps')
    spec=EXTERNAL_SPECS[name];grid_index=pd.date_range(d.index[0],d.index[-1],freq=spec['frequency'])
    if not d.index.isin(grid_index).all():raise ValueError(name+': timestamps do not align with the declared grid')
    return d.reindex(grid_index)


def _splits(index,history):
    n=len(index);cuts=[int(n*x) for x in (.55,.70,.85)];gap=history+1
    out={'development':index[:cuts[0]],'calibration':index[cuts[0]+gap:cuts[1]],
         'policy':index[cuts[1]+gap:cuts[2]],'test':index[cuts[2]+gap:-1]}
    if min(map(len,out.values()))<200:raise ValueError('External series too short after temporal gaps.')
    return out


def _bundle(ctx,name,path):
    raw=_parse(path,name);spec=dict(EXTERNAL_SPECS[name]);columns=list(raw.columns)
    X=features(raw,spec,columns);y=raw[spec['target']].shift(-1)
    naive=pd.DataFrame({'Persistence':raw[spec['target']],
                        'Seasonal naive':raw[spec['target']].shift(spec['period']-1)},index=raw.index)
    splits=_splits(raw.index,max(spec['lags']+spec['windows']))
    dev=raw.loc[splits['development'],columns];med=dev.median();std=dev.std().replace(0,np.nan)
    iqr=(dev.quantile(.75)-dev.quantile(.25)).replace(0,np.nan)
    st=pd.DataFrame({'median':med.fillna(0),'std':std.fillna(iqr).fillna(1),'iqr':iqr.fillna(std).fillna(1)})
    groups={}
    for col in X:
        group='Calendar' if col.startswith('time::') else col.split('::')[-1]
        groups.setdefault(group,[]).append(col)
    bundle={'dataset':name,'raw':raw,'X':X,'y':y,'naive':naive,'spec':spec,'columns':columns,
            'stats':st,'groups':groups,'splits':splits,'input_sha256':filehash(path),
            'all_measurements_missing_rows':int(raw.isna().all(axis=1).sum())}
    bundle['data_key']=digest({'file':bundle['input_sha256'],'split_boundaries':{k:[str(v[0]),str(v[-1])] for k,v in splits.items()},'features':list(X)})
    return bundle


def _fit_external(ctx,b):
    dev=b['splits']['development'];observed=dev[b['y'].loc[dev].notna()]
    fit={'models':{},'params':{},'cv_scores':{},'fallbacks':{},'clean':{}}
    for name in EXT_MODELS[:2]:
        v=b['naive'].loc[observed,name].median();fit['fallbacks'][name]=float(v if np.isfinite(v) else b['y'].loc[observed].median())
    cvrows=[]
    for name in EXT_MODELS[2:]:
        configs=grid(name,ctx.cfg['profile']);scores=[]
        for config_id,params in enumerate(configs):
            fold_scores=[]
            for fold,tr,va in cv_folds(b,ctx.cfg['external_cv_folds']):
                label=f'External | {b["dataset"]} | {name} | config {config_id+1} | fold {fold+1}'
                def work(name=name,params=params,tr=tr,va=va,fold=fold,label=label):
                    est=factory(name,params,seed(ctx.cfg['seed'],'external',b['dataset'],name,fold),ctx.cfg['threads'])
                    est,ws,secs=fit_logged(ctx,est,b['X'].loc[tr],b['y'].loc[tr],label)
                    return {'RMSE':metrics(b['y'].loc[va],est.predict(b['X'].loc[va]))['RMSE'],'warnings':ws,'seconds':secs}
                r=ctx.task('EXTERNAL_CV',label,{'data':b['data_key'],'params':params,'fold':fold},work)
                fold_scores.append(r['RMSE']);cvrows.append({'dataset':b['dataset'],'model':name,'config':config_id,'fold':fold,'params':json.dumps(params),**r})
            scores.append(float(np.mean(fold_scores)))
        best=int(np.argmin(scores));params=configs[best];fit['params'][name]=params;fit['cv_scores'][name]=scores[best]
        label=f'External | {b["dataset"]} | {name} | final fit'
        def final(name=name,params=params,label=label):
            est=factory(name,params,seed(ctx.cfg['seed'],'external',b['dataset'],name,'final'),ctx.cfg['threads'])
            est,ws,secs=fit_logged(ctx,est,b['X'].loc[observed],b['y'].loc[observed],label)
            return est,ws,secs
        est,ws,secs=ctx.task('EXTERNAL_FIT',label,{'data':b['data_key'],'params':params},final)
        fit['models'][name]=est
    fit['deployment_model']=min(EXT_MODELS[2:],key=lambda n:(fit['cv_scores'][n],n))
    return fit,pd.DataFrame(cvrows)


def _predict(b,fit,X,naive):
    out=[]
    for name in EXT_MODELS:
        if name in fit['models']:p=fit['models'][name].predict(X)
        else:p=naive[name].fillna(fit['fallbacks'][name]).to_numpy(float)
        if not np.isfinite(p).all():raise ValueError('Nonfinite external prediction: '+name)
        out.append(p)
    return np.column_stack(out)


def _decision_rows(ctx,b,fit,ca,condition,rep,y,p,halves):
    pol=b['splits']['policy'];policy_y=b['y'].loc[pol].dropna().to_numpy();rows=[];valid=np.isfinite(y)
    for event_q in ctx.cfg['decision_event_quantiles']:
        threshold=float(np.quantile(policy_y,event_q));truth=y>=threshold
        for fn_ratio in ctx.cfg['decision_fn_cost_ratios']:
            for method,half in [('Point forecast',None)]+list(halves.items()):
                if half is None:action=(p>=threshold).astype(int);abstain=np.zeros(len(p),bool)
                else:
                    low=p-half;high=p+half;action=np.full(len(p),-1);action[low>=threshold]=1;action[high<threshold]=0;abstain=action<0
                fp=(action==1)&(~truth)&valid;fn=(action==0)&truth&valid;ab=abstain&valid
                cost=(fp.sum()+fn_ratio*fn.sum()+ctx.cfg['decision_abstain_cost']*ab.sum())/max(1,valid.sum())
                rows.append({'dataset':b['dataset'],'condition':condition,'replicate':rep,'method':method,
                  'event_quantile':event_q,'event_threshold':threshold,'false_negative_cost_ratio':fn_ratio,
                  'expected_cost':cost,'false_negative_rate':fn.sum()/max(1,(truth&valid).sum()),
                  'false_positive_rate':fp.sum()/max(1,((~truth)&valid).sum()),'abstention_rate':ab.sum()/max(1,valid.sum())})
    return rows


def external_suite(ctx):
    protocol={'target_journal':'Environment Systems and Decisions','status':'independent retrospective replication; not preregistered confirmation',
      'author_declares_outcomes_previously_inspected':bool(ctx.cfg['external_outcomes_previously_inspected']),
      'datasets':{k:{'doi':v['doi'],'domain':v['domain'],'target':v['target']} for k,v in EXTERNAL_SPECS.items()},
      'primary_endpoints':['test RMSE relative to persistence','natural-missing versus complete-origin error','condition-specific RMSE change','empirical interval coverage','decision cost and abstention'],
      'decision_sensitivity':{'event_quantiles':ctx.cfg['decision_event_quantiles'],'false_negative_cost_ratios':ctx.cfg['decision_fn_cost_ratios'],'abstention_cost':ctx.cfg['decision_abstain_cost']}}
    atomic_json(ctx.root/'manifests'/'external_confirmation_protocol.json',protocol)
    bundles={};fits={};audit=[];cv=[];clean=[];stress=[];intervals=[];decisions=[];natural_missing=[]
    for name in EXTERNAL_NAMES:
        path=_download_or_find(ctx,name);b=ctx.task('EXTERNAL_DATA',name+' external bundle',{'sha256':filehash(path),'spec':EXTERNAL_SPECS[name]},lambda name=name,path=path:_bundle(ctx,name,path))
        bundles[name]=b;fit,cvr=_fit_external(ctx,b);fits[name]=fit;cv.append(cvr)
        idx=b['splits']['test'];y=b['y'].loc[idx].to_numpy();P=_predict(b,fit,b['X'].loc[idx],b['naive'].loc[idx]);fit['clean']['test']=P
        for j,m in enumerate(EXT_MODELS):
            met=metrics(y,P[:,j]);clean.append({'dataset':name,'model':m,'deployment_model':fit['deployment_model'],**met,'relative_to_persistence':met['RMSE']/metrics(y,P[:,0])['RMSE']})
        deploy_clean=P[:,EXT_MODELS.index(fit['deployment_model'])]
        missing_origin=b['raw'].loc[idx,b['columns']].isna().any(axis=1).to_numpy();observed_y=np.isfinite(y)
        for state,mask in [('Fully observed origin',~missing_origin),('Natural missing origin',missing_origin)]:
            use=mask&observed_y
            estimable=bool(use.sum()>=ctx.cfg['natural_missing_min_n'])
            natural_missing.append({'dataset':name,'state':state,'n':int(use.sum()),'estimable':estimable,
              'RMSE':metrics(y[use],deploy_clean[use])['RMSE'] if estimable else np.nan,
              'MAE':metrics(y[use],deploy_clean[use])['MAE'] if estimable else np.nan})
        ca=ctx.task('EXTERNAL_CAL',name+' external interval calibration',
          {'data':b['data_key'],'model':fit['deployment_model'],'coverage':ctx.cfg['coverage']},
          lambda:calibrate_intervals(ctx,b,fit,fit['deployment_model']))
        for condition in EXT_CONDITIONS:
            reps=1 if condition=='Clean' else ctx.cfg['external_reps']
            for rep in range(reps):
                label=f'External | {name} | {condition} | replicate {rep+1}/{reps}'
                def work(condition=condition,rep=rep):
                    if condition=='Clean':X=b['X'].loc[idx];N=b['naive'].loc[idx]
                    else:X,N,_=scenario(b,condition,rep)
                    predall=_predict(b,fit,X,N)
                    scale=np.maximum(ca['scale_model'].predict(X),ca['scale_floor'])
                    return {'predictions':predall,'scale':scale}
                result_rep=ctx.task('EXTERNAL_STRESS',label,
                  {'data':b['data_key'],'condition':condition,'replicate':rep,'model':fit['deployment_model']},work)
                predall=result_rep['predictions'];scale=result_rep['scale']
                deploy=predall[:,EXT_MODELS.index(fit['deployment_model'])]
                halves={'Global split conformal':np.full(len(deploy),ca['global_q']),
                        'Locally scaled conformal':ca['local_q']*scale}
                for j,m in enumerate(EXT_MODELS):
                    met=metrics(y,predall[:,j]);base=metrics(y,P[:,j])['RMSE']
                    stress.append({'dataset':name,'condition':condition,'replicate':rep,'model':m,**met,'clean_RMSE':base,'relative_change':met['RMSE']/base-1})
                for method,half in halves.items():
                    valid=np.isfinite(y);intervals.append({'dataset':name,'condition':condition,'replicate':rep,'method':method,
                      'coverage':float(np.mean(np.abs(y[valid]-deploy[valid])<=half[valid])),'mean_width':float(np.mean(2*half[valid]))})
                decisions.extend(_decision_rows(ctx,b,fit,ca,condition,rep,y,deploy,halves))
        audit.append({'dataset':name,'doi':EXTERNAL_SPECS[name]['doi'],'input_sha256':b['input_sha256'],'rows':len(b['raw']),
          'development_start':b['splits']['development'][0],'test_end':b['splits']['test'][-1],'deployment_model':fit['deployment_model']})
    result={'protocol':protocol,'data_audit':pd.DataFrame(audit),'cv':pd.concat(cv,ignore_index=True),'clean':pd.DataFrame(clean),
      'stress':pd.DataFrame(stress),'intervals':pd.DataFrame(intervals),'decisions':pd.DataFrame(decisions),
      'natural_missing':pd.DataFrame(natural_missing)}
    for k,v in result.items():
        if isinstance(v,pd.DataFrame):ctx.table('EXTERNAL_'+k,v)
    return result


def external_figures(ctx,result):
    out=[];clean=result['clean'];stress=result['stress'];intervals=result['intervals'];dec=result['decisions'];natural=result['natural_missing']
    fig,axes=newaxes()
    for ax,ds in zip(axes.flat,EXTERNAL_NAMES):
        g=clean[clean.dataset==ds]
        ax.bar(np.arange(len(g)),g.relative_to_persistence,color=[EXT_COLORS[m] for m in g.model])
        ax.axhline(1,color='black',lw=.8);ax.set_xticks(range(len(g)),[EXT_SHORT[m] for m in g.model],rotation=20)
        ax.set_ylabel('Test RMSE / persistence RMSE');ax.set_title(ds);ax.grid(axis='y',alpha=.2)
    out.append(savefig(ctx,fig,'EV_F1_external_clean_skill','External confirmation: clean forecasting skill',
      'Independent UCI cohorts were specified before outcome inspection. Values below one outperform persistence. Model selection uses development-only temporal CV; no core-study test result selects the external model.',clean))
    agg=stress.groupby(['dataset','condition','model'],as_index=False).agg(change=('relative_change','mean'),low=('relative_change',lambda x:x.quantile(.025)),high=('relative_change',lambda x:x.quantile(.975)))
    fig,axes=newaxes()
    for ax,ds in zip(axes.flat,EXTERNAL_NAMES):
        q=agg[(agg.dataset==ds)&(agg.condition!='Clean')]
        for m,g in q.groupby('model'):
            x=np.arange(len(g));ax.errorbar(x,g.change*100,yerr=[np.maximum(0,(g.change-g.low)*100),np.maximum(0,(g.high-g.change)*100)],fmt='o-',capsize=2,label=EXT_SHORT[m],color=EXT_COLORS[m])
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(len(g)),g.condition,rotation=28,ha='right');ax.set_ylabel('RMSE change from clean (%)');ax.set_title(ds);ax.grid(alpha=.2)
    axes[0,1].legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
    out.append(savefig(ctx,fig,'EV_F2_external_stress_replication','External confirmation: mechanism-specific degradation',
      'Points average registered perturbation replicas; bars are empirical 2.5–97.5 percentiles across replicas. Conditions retain distinct physical/statistical meanings and are not pooled into one universal severity score.',agg))
    q=dec[(dec.event_quantile==.9)&(dec.method=='Locally scaled conformal')]
    agg=q.groupby(['dataset','condition','false_negative_cost_ratio'],as_index=False).agg(cost=('expected_cost','mean'),abstention=('abstention_rate','mean'))
    fig,axes=newaxes()
    for ax,ds in zip(axes.flat,EXTERNAL_NAMES):
        for condition,g in agg[agg.dataset==ds].groupby('condition'):
            ax.plot(g.abstention,g.cost,'o-',label=condition)
        ax.set_xlabel('Abstention rate');ax.set_ylabel('Expected decision cost / observation');ax.set_title(ds+' (90th-percentile event)');ax.grid(alpha=.2)
    axes[0,1].legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
    out.append(savefig(ctx,fig,'EV_F3_external_decision_tradeoff','External confirmation: uncertainty-aware decision trade-off',
      'Event thresholds are frozen from policy outcomes. Curves vary the registered false-negative/false-positive cost ratio; abstention cost is fixed in configuration. This is a sensitivity analysis, not a claimed universal social valuation.',agg))
    im=intervals.groupby(['dataset','condition','method'],as_index=False).agg(coverage=('coverage','mean'),coverage_low=('coverage',lambda x:x.quantile(.025)),coverage_high=('coverage',lambda x:x.quantile(.975)),width=('mean_width','mean'))
    dm=dec.groupby(['dataset','condition','method'],as_index=False).agg(cost=('expected_cost','mean'),abstention=('abstention_rate','mean'))
    nm=natural.pivot(index='dataset',columns='state',values=['n','RMSE']).reset_index()
    nm.columns=['dataset']+[f'{a}: {b}' for a,b in nm.columns[1:]]
    if {'RMSE: Natural missing origin','RMSE: Fully observed origin'}<=set(nm.columns):
        nm['Natural/complete RMSE ratio']=nm['RMSE: Natural missing origin']/nm['RMSE: Fully observed origin']
    tabdf=im.merge(dm,on=['dataset','condition','method'],how='left').merge(nm,on='dataset',how='left')
    out.append(table(ctx,'EV_T1_external_confirmation_endpoints','External confirmation endpoints',tabdf,
      'External cohorts are independent public time series, not additional corruption replicas of the core datasets. Coverage is empirical. Decision costs are averaged over the registered sensitivity grid. Natural-missingness columns compare real incomplete sensor origins with fully observed origins and are descriptive, because missingness is not randomized.'))
    return out
