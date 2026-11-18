"""Independent multi-site Beijing evaluation for the failure-aware audit.

All twelve stations are retained.  Local and leave-one-station-out (LOSO)
models use fixed, declared hyperparameters so the held-out station never enters
model or parameter selection.  Stations, rather than synthetic perturbation
replicas, are the unit of external generalisation.
"""
from __future__ import annotations
import io, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from primna_esd_runtime import digest,filehash,seed,metrics,ci
from primna_esd_data import features,scenario
from primna_esd_models import factory,fit_logged
from primna_esd_figures import savefig,table

URL='https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
DOI='10.24432/C5RK5G'
TARGET='CO'
MODELS=['Ridge','HistGradientBoosting']
PARAMS={
 'Ridge':{'alpha':10.0},
 'HistGradientBoosting':{'max_iter':250,'max_leaf_nodes':31,'learning_rate':.05,'l2_regularization':.5}}
SPEC={'target':TARGET,'frequency':'1h','period':24,'lags':[1,2,3,6,12,23,24],
      'windows':[3,6,24],'unit':'ug/m3'}
CONDITIONS=['MCAR 20%','Block 20%','Gaussian 10%','Abrupt drift']
COLORS={'Ridge':'#0072B2','HistGradientBoosting':'#D55E00'}


def station_paths(ctx):
    explicit=ctx.cfg.get('beijing_multisite_root')
    roots=[Path(explicit)] if explicit else [Path('/kaggle/input'),ctx.root/'data'/'beijing_multisite']
    for root in roots:
        if root.exists():
            hits=sorted(root.rglob('PRSA_Data_*_20130301-20170228.csv'))
            if len(hits)==12:return hits
    if not ctx.cfg['allow_download']:
        raise FileNotFoundError('Attach the official UCI Beijing Multi-Site archive or enable Kaggle Internet.')
    ctx.log('Downloading official UCI Beijing multi-site archive: %s',URL)
    target=ctx.root/'data'/'beijing_multisite';target.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(URL,timeout=180) as response:payload=response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        members=[n for n in outer.namelist() if Path(n).name.startswith('PRSA_Data_') and n.endswith('.csv')]
        if not members:
            nested=[n for n in outer.namelist() if n.lower().endswith('.zip')]
            for n in nested:
                with zipfile.ZipFile(io.BytesIO(outer.read(n))) as inner:
                    for m in inner.namelist():
                        if Path(m).name.startswith('PRSA_Data_') and m.endswith('.csv'):
                            (target/Path(m).name).write_bytes(inner.read(m))
        else:
            for m in members:(target/Path(m).name).write_bytes(outer.read(m))
    hits=sorted(target.glob('PRSA_Data_*_20130301-20170228.csv'))
    if len(hits)!=12:raise RuntimeError(f'Expected 12 Beijing station files, found {len(hits)}.')
    return hits


def _bundle(path):
    station=path.name.split('PRSA_Data_',1)[1].split('_2013',1)[0]
    d=pd.read_csv(path,na_values=['NA','?'])
    t=pd.to_datetime(d[['year','month','day','hour']],errors='raise')
    d=d.drop(columns=['No','year','month','day','hour','wd','station'],errors='ignore').apply(pd.to_numeric,errors='raise')
    d.index=pd.DatetimeIndex(t);d=d.sort_index()
    if d.index.duplicated().any():raise ValueError(station+': duplicate timestamps')
    grid=pd.date_range(d.index[0],d.index[-1],freq='1h');d=d.reindex(grid)
    columns=list(d.columns);X=features(d,SPEC,columns);y=d[TARGET].shift(-1)
    n=len(d);cuts=[int(n*x) for x in (.60,.80)];gap=max(SPEC['lags']+SPEC['windows'])+1
    splits={'development':d.index[:cuts[0]],'validation':d.index[cuts[0]+gap:cuts[1]],
            'test':d.index[cuts[1]+gap:-1]}
    med=d.loc[splits['development']].median();std=d.loc[splits['development']].std().replace(0,np.nan)
    iqr=(d.loc[splits['development']].quantile(.75)-d.loc[splits['development']].quantile(.25)).replace(0,np.nan)
    stats=pd.DataFrame({'median':med.fillna(0),'std':std.fillna(iqr).fillna(1),'iqr':iqr.fillna(std).fillna(1)})
    groups={}
    for col in X:groups.setdefault('Calendar' if col.startswith('time::') else col.split('::')[-1],[]).append(col)
    b={'dataset':'Beijing '+station,'station':station,'raw':d,'X':X,'y':y,'columns':columns,
       'spec':SPEC,'stats':stats,'groups':groups,'splits':splits,'input_sha256':filehash(path)}
    b['data_key']=digest({'file':b['input_sha256'],'station':station,'split':{k:[str(v[0]),str(v[-1])] for k,v in splits.items()}})
    return b


def _observed(b,split):
    idx=b['splits'][split];return idx[b['y'].loc[idx].notna()]


def _scoreable_test(b,split):
    """Rows where both the forecast target and persistence baseline are observed."""
    idx=b['splits'][split]
    target_ok=b['y'].loc[idx].notna()
    persistence_ok=b['raw'].loc[idx,TARGET].notna()
    return idx[target_ok & persistence_ok]


def _fit(ctx,name,X,y,label,tag):
    est=factory(name,PARAMS[name],seed(ctx.cfg['seed'],'Beijing',name,tag),ctx.cfg['threads'])
    est,ws,secs=fit_logged(ctx,est,X,y,label);return est,ws,secs


def run_multisite(ctx):
    paths=station_paths(ctx)
    bundles={}
    for path in paths:
        raw_hash=filehash(path)
        b=ctx.task('BEIJING_DATA','Beijing '+path.stem,{'sha256':raw_hash,'spec':SPEC},lambda path=path:_bundle(path))
        bundles[b['station']]=b
    stations=sorted(bundles);fits={};audit=[];rows=[];stress=[]
    for holdout in stations:
        b=bundles[holdout]
        target_observed=_observed(b,'test')
        test=_scoreable_test(b,'test')
        if len(test)==0:
            raise RuntimeError(f'{holdout}: no test rows have both an observed target and persistence baseline.')
        yt=b['y'].loc[test].to_numpy()
        persist=b['raw'].loc[test,TARGET].to_numpy()
        if not np.all(np.isfinite(yt)):
            raise RuntimeError(f'{holdout}: nonfinite test targets after scoreability filtering.')
        if not np.all(np.isfinite(persist)):
            raise RuntimeError(f'{holdout}: nonfinite persistence values after scoreability filtering.')
        excluded=len(target_observed)-len(test)
        ctx.log('Beijing %s | target-observed=%d | persistence-scoreable=%d | excluded-missing-persistence=%d',
                holdout,len(target_observed),len(test),excluded)
        base=metrics(yt,persist)['RMSE']
        if not np.isfinite(base) or base<=0:
            raise RuntimeError(f'{holdout}: invalid persistence RMSE {base!r}.')
        for design in ['Local','LOSO']:
            if design=='Local':
                tr=_observed(b,'development');X=b['X'].loc[tr];y=b['y'].loc[tr]
            else:
                xs=[];ys=[]
                for s in stations:
                    if s==holdout:continue
                    q=bundles[s];tr=_observed(q,'development');xs.append(q['X'].loc[tr]);ys.append(q['y'].loc[tr])
                X=pd.concat(xs);y=pd.concat(ys)
                # Equal-site cap prevents the largest station or memory limits from dominating.
                max_n=ctx.cfg['multisite_train_cap'];take=np.linspace(0,len(X)-1,min(max_n,len(X)),dtype=int);X=X.iloc[take];y=y.iloc[take]
            for name in MODELS:
                label=f'Beijing {holdout} | {design} | {name}'
                est,ws,secs=ctx.task('BEIJING_FIT',label,{'holdout':holdout,'design':design,'model':name,
                  'inputs':{s:bundles[s]['data_key'] for s in stations},'params':PARAMS[name]},
                  lambda name=name,X=X,y=y,label=label,design=design:_fit(ctx,name,X,y,label,holdout+design))
                fits[(holdout,design,name)]=est;pred=est.predict(b['X'].loc[test]);m=metrics(yt,pred)
                rows.append({'station':holdout,'design':design,'model':name,'n_test':m['n'],'RMSE':m['RMSE'],
                  'MAE':m['MAE'],'R2':m['R2'],'AE_q95':m['AE_q95'],'persistence_RMSE':base,
                  'RMSE_ratio_to_persistence':m['RMSE']/base,'fit_seconds':secs,'warnings':' | '.join(ws)})
        for name in MODELS:
            est=fits[(holdout,'LOSO',name)];clean=metrics(yt,est.predict(b['X'].loc[test]))['RMSE']
            for condition in CONDITIONS:
                for rep in range(ctx.cfg['multisite_stress_reps']):
                    Xp,_,_=scenario(b,condition,rep,scope='test');pred=est.predict(Xp.loc[test]);m=metrics(yt,pred)
                    stress.append({'station':holdout,'model':name,'condition':condition,'replicate':rep,
                      'RMSE':m['RMSE'],'relative_change':m['RMSE']/clean-1})
        audit.append({'station':holdout,'doi':DOI,'input_sha256':b['input_sha256'],'rows':len(b['raw']),
          'test_start':test[0],'test_end':test[-1],
          'test_observed_labels':len(target_observed),'test_scoreable_labels':len(test),
          'test_excluded_missing_persistence':excluded})
    clean=pd.DataFrame(rows);stress=pd.DataFrame(stress);audit=pd.DataFrame(audit)
    # The site bootstrap is descriptive uncertainty across the twelve observed
    # stations; it is not treated as a random sample of all cities.
    summary=[];rng=np.random.default_rng(seed(ctx.cfg['seed'],'site bootstrap'))
    for (design,name),g in clean.groupby(['design','model']):
        x=g.RMSE_ratio_to_persistence.to_numpy();draw=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(ctx.cfg['bootstrap'])])
        summary.append({'design':design,'model':name,'stations':len(x),'mean_ratio':x.mean(),
          'median_ratio':np.median(x),'fraction_better_than_persistence':np.mean(x<1),
          'site_boot_low':np.quantile(draw,.025),'site_boot_high':np.quantile(draw,.975),
          'worst_station_ratio':x.max()})
    summary=pd.DataFrame(summary)
    for name,df in [('BEIJING_multisite_audit',audit),('BEIJING_multisite_clean',clean),
                    ('BEIJING_multisite_stress',stress),('BEIJING_multisite_summary',summary)]:ctx.table(name,df)
    return {'audit':audit,'clean':clean,'stress':stress,'summary':summary}


def multisite_figures(ctx,result):
    out=[];clean=result['clean'];stress=result['stress'];summary=result['summary']
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.8),constrained_layout=True)
    for ax,design in zip(axes,['Local','LOSO']):
        q=clean[clean.design==design]
        for name,g in q.groupby('model'):
            g=g.sort_values('station');ax.plot(g.station,g.RMSE_ratio_to_persistence,'o-',label=name,color=COLORS[name])
        ax.axhline(1,color='black',ls='--');ax.set_title(design+' training');ax.tick_params(axis='x',rotation=55)
        ax.set_ylabel('RMSE / persistence RMSE');ax.grid(alpha=.2)
    axes[1].legend(frameon=False)
    out.append(savefig(ctx,fig,'EV_F1_multisite_skill','External replication across all 12 Beijing stations',
      'Every point is a genuinely held-out station test result. LOSO models are trained on the other eleven stations with fixed hyperparameters. Values below one beat persistence; failures are retained.',clean))
    agg=stress.groupby(['model','condition','station'],as_index=False).relative_change.mean()
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.8),constrained_layout=True)
    for ax,(name,g) in zip(axes,agg.groupby('model')):
        groups=[g[g.condition==c].relative_change.to_numpy()*100 for c in CONDITIONS]
        ax.boxplot(groups,tick_labels=CONDITIONS,showfliers=True);ax.axhline(0,color='black',lw=.8)
        ax.set_title(name+' LOSO');ax.set_ylabel('RMSE change from clean (%)');ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
    out.append(savefig(ctx,fig,'EV_F2_multisite_stress','Between-station distribution of degradation',
      'Boxes summarize the twelve station-level mean effects; corruption replicates are averaged within station before cross-station display. Stations, not simulated replicas, are the generalisation unit.',agg))
    pivot=clean.pivot_table(index=['station','model'],columns='design',values='RMSE_ratio_to_persistence').reset_index()
    pivot['LOSO_minus_local']=pivot.LOSO-pivot.Local
    fig,ax=plt.subplots(figsize=(10.2,4.8),constrained_layout=True)
    for name,g in pivot.groupby('model'):
        g=g.sort_values('LOSO_minus_local');ax.plot(g.LOSO_minus_local,g.station,'o',label=name,color=COLORS[name])
    ax.axvline(0,color='black',ls='--');ax.set_xlabel('LOSO minus local RMSE ratio');ax.set_ylabel('Held-out station');ax.grid(alpha=.2);ax.legend(frameon=False)
    out.append(savefig(ctx,fig,'EV_F3_transfer_penalty','Cross-site transfer penalty is heterogeneous',
      'Positive values mean that training on other stations transfers worse than station-local training. This isolates transportability rather than mixing stations with corruption replicas.',pivot))
    tabdf=clean.merge(result['audit'][['station','test_observed_labels','test_scoreable_labels','test_excluded_missing_persistence']],on='station').sort_values(['station','design','model'])
    out.append(table(ctx,'EV_T1_multisite_endpoints','External multi-site replication endpoints',tabdf,
      'All twelve stations are shown. Hyperparameters are fixed before held-out-station testing. Model and persistence metrics use the same explicitly reported scoreable rows; targets lacking a persistence value are excluded only from this paired comparison and counted in the audit. Ratios below one beat persistence. The station bootstrap in BEIJING_multisite_summary.csv is descriptive and does not imply that Beijing stations represent all cities.'))
    return out
