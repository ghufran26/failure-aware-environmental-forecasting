"""Raw inputs, frozen historical date boundaries, causal features and fault generators."""
from __future__ import annotations
import io, json, math, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from primna_esd_runtime import digest,filehash,atomic_json,seed

DATASETS=['Air Quality','Appliances Energy']
SPECS={
 'Air Quality':{'target':'CO(GT)','frequency':'1h','period':24,'lags':[1,2,3,6,12,23,24],
   'windows':[3,6,24],'exclude':['NMHC(GT)'],'filename':'AirQualityUCI.csv','unit':'mg/m3',
   'url':'https://archive.ics.uci.edu/static/public/360/air+quality.zip',
   'ranges':{'development':['2004-03-11 18:00','2004-11-17 09:00'],
             'calibration':['2004-11-18 12:00','2004-12-20 14:00'],
             'policy':['2004-12-21 16:00','2005-01-28 23:00'],
             'test':['2005-01-30 01:00','2005-04-04 13:00']}},
 'Appliances Energy':{'target':'Appliances','frequency':'10min','period':144,
   'lags':[1,2,3,6,12,18,36,143,144],'windows':[6,18,144],'exclude':['rv1','rv2'],
   'filename':'energydata_complete.csv','unit':'Wh per 10 min',
   'url':'https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip',
   'ranges':{'development':['2016-01-12 17:00','2016-04-01 12:20'],
             'calibration':['2016-04-02 12:40','2016-04-15 19:40'],
             'policy':['2016-04-16 20:00','2016-04-30 03:00'],
             'test':['2016-05-01 03:20','2016-05-27 17:50']}}
}
MISSING=['MCAR','Block outage','Channel dropout','MAR','MNAR sensitivity']
FAULTS=['Gaussian','Multiplicative','Calibration bias','Spikes','Clipping','Stuck-at']
DRIFTS=['Abrupt mean','Gradual mean','Variance','Recurring']


def input_path(ctx,ds):
    spec=SPECS[ds];explicit=ctx.cfg.get('data_paths',{}).get(ds)
    if explicit:
        p=Path(explicit)
        if not p.exists():raise FileNotFoundError(p)
        return p
    roots=[Path('/kaggle/input'),ctx.root/'data']
    matches=[]
    for root in roots:
        if root.exists(): matches+=sorted(root.rglob(spec['filename']))
    if matches:
        hashes={filehash(p) for p in matches}
        if len(hashes)>1:raise RuntimeError(f'Multiple differing {spec["filename"]} files: set data_paths explicitly.')
        return matches[0]
    if not ctx.cfg['allow_download']:raise FileNotFoundError('Attach '+spec['filename']+' or enable Internet.')
    ctx.log('Downloading official UCI archive: %s',spec['url'])
    archive=ctx.root/'data'/('uci_'+str(spec['period'])+'.zip')
    try:
        with urllib.request.urlopen(spec['url'],timeout=120) as r: archive.write_bytes(r.read())
        with zipfile.ZipFile(archive) as z:
            wanted=[n for n in z.namelist() if Path(n).name==spec['filename']]
            if len(wanted)!=1:raise ValueError('Expected one '+spec['filename'])
            dest=ctx.root/'data'/spec['filename'];dest.write_bytes(z.read(wanted[0]));return dest
    except Exception as exc:
        raise RuntimeError('Official download unavailable. Attach the UCI CSV using Add Input; no synthetic fallback.') from exc


def parse_raw(path,ds):
    spec=SPECS[ds]
    if ds=='Air Quality':
        df=pd.read_csv(path,sep=';',decimal=',').dropna(how='all')
        df=df.loc[:,~df.columns.str.startswith('Unnamed')]
        t=pd.to_datetime(df.pop('Date').astype(str)+' '+df.pop('Time').astype(str),format='%d/%m/%Y %H.%M.%S',errors='coerce')
        df=df.apply(pd.to_numeric,errors='raise').replace(-200,np.nan)
    else:
        df=pd.read_csv(path);t=pd.to_datetime(df.pop('date'),errors='coerce')
        df=df.apply(pd.to_numeric,errors='raise')
    if t.isna().any():raise ValueError('Invalid timestamps: no silent row removal.')
    df.index=pd.DatetimeIndex(t);df=df.sort_index()
    # Official files have no duplicate clock origins; reject conflicts rather than keep-last.
    if df.index.duplicated().any():raise ValueError('Duplicate timestamps require explicit adjudication before forecasting.')
    grid=pd.date_range(df.index[0],df.index[-1],freq=spec['frequency'])
    if not df.index.isin(grid).all():raise ValueError('Timestamps do not align with declared acquisition grid.')
    inserted=len(grid)-len(df);df=df.reindex(grid)
    if np.isinf(df.to_numpy()).any():raise ValueError('Infinite input values.')
    return df,inserted


def features(raw,spec,columns):
    """Features at origin t use only raw measurements at or before t; target is t+1."""
    b=raw[columns];parts=[b.add_prefix('cur::')]
    for lag in spec['lags']:parts.append(b.shift(lag).add_prefix(f'lag{lag}::'))
    dc=[c for c in columns if c==spec['target'] or c.startswith('PT08') or c in {'T','RH','lights','T_out','RH_out'}]
    parts.append(b[dc].diff().add_prefix('diff1::'))
    for w in spec['windows']:
        r=b[spec['target']].rolling(w,min_periods=max(2,w//3))
        parts.append(pd.DataFrame({f'roll{w}_{op}::{spec["target"]}':getattr(r,op)() for op in ['mean','std','min','max']}))
    ts=raw.index;tm={};h=ts.hour+ts.minute/60
    for k,v,p in [('hour',h,24),('dow',ts.dayofweek,7),('doy',ts.dayofyear,365.25)]:
        tm['time::'+k+'_sin']=np.sin(2*np.pi*v/p);tm['time::'+k+'_cos']=np.cos(2*np.pi*v/p)
    parts.append(pd.DataFrame(tm,index=ts));X=pd.concat(parts,axis=1)
    assert X.columns.is_unique
    assert sum(c.startswith('lag') for c in X)==len(columns)*len(spec['lags'])
    return X


def make_bundle(ctx,ds,path):
    raw,inserted=parse_raw(path,ds);spec=SPECS[ds]
    columns=[c for c in raw if c not in spec['exclude']]
    X=features(raw,spec,columns);y=raw[spec['target']].shift(-1)
    naive=pd.DataFrame({'Persistence':raw[spec['target']],
                       'Seasonal naive':raw[spec['target']].shift(spec['period']-1)},index=raw.index)
    ranges={k:(pd.Timestamp(v[0]),pd.Timestamp(v[1])) for k,v in spec['ranges'].items()}
    splits={k:raw.index[(raw.index>=start)&(raw.index<=end)] for k,(start,end) in ranges.items()}
    if any(len(i)<50 for i in splits.values()):raise ValueError('Input does not cover frozen historical intervals.')
    delta=pd.Timedelta(spec['frequency']);history=max(spec['lags']+spec['windows'])
    for a,b in zip(list(splits)[:-1],list(splits)[1:]):
        assert splits[a][-1]+delta < splits[b][0]-history*delta
    dev=raw.loc[:ranges['development'][1],columns]
    med=dev.median();std=dev.std().replace(0,np.nan);iqr=(dev.quantile(.75)-dev.quantile(.25)).replace(0,np.nan)
    st=pd.DataFrame({'median':med.fillna(0),'std':std.fillna(iqr).fillna(1),
                     'iqr':iqr.fillna(std).fillna(1)})
    groups={}
    for col in X:
        group='Calendar' if col.startswith('time::') else col.split('::')[-1]
        groups.setdefault(group,[]).append(col)
    b={'dataset':ds,'raw':raw,'X':X,'y':y,'naive':naive,'spec':spec,'columns':columns,'stats':st,
       'groups':groups,'splits':splits,'input_sha256':filehash(path),'inserted_grid_rows':inserted}
    b['data_key']=digest({'file':b['input_sha256'],'ranges':spec['ranges'],'features':list(X),'columns':columns})
    return b


def acquire(ctx):
    bundles={};audit=[];splits=[]
    for ds in DATASETS:
        path=input_path(ctx,ds);h=filehash(path)
        bundles[ds]=ctx.task('P1',ds+' raw/features',{'input':h,'spec':SPECS[ds]},lambda ds=ds,path=path:make_bundle(ctx,ds,path))
        b=bundles[ds]
        # Cache full raw inputs as CSV alongside the canonical parsed time series.
        dest=ctx.root/'data'/SPECS[ds]['filename']
        if path.resolve()!=dest.resolve():
            import shutil;shutil.copy2(path,dest)
        for split,idx in b['splits'].items():
            splits.append({'dataset':ds,'split':split,'start':idx[0],'end':idx[-1],
                           'clock_origins':len(idx),'observed_future_targets':int(b['y'].loc[idx].notna().sum()),
                           'evaluation_status':'historical inspected cohort; retrospective core audit'})
        for col in b['raw']:
            audit.append({'dataset':ds,'variable':col,'n_rows':len(b['raw']),
              'missing_fraction':b['raw'][col].isna().mean(),'included':col in b['columns'],
              'input_sha256':h,'inserted_grid_rows':b['inserted_grid_rows']})
    ctx.table('P1_data_audit',audit);ctx.table('P1_frozen_split_manifest',splits)
    return bundles


def cv_folds(bundle,n_splits):
    """Equally spaced CLOCK folds before dropping unavailable labels; full temporal gap."""
    idx=bundle['splits']['development'];gap=max(bundle['spec']['lags']+bundle['spec']['windows'])+1
    fold_n=len(idx)//(n_splits+1)
    for f in range(n_splits):
        vstart=(f+1)*fold_n
        tr=idx[:vstart-gap];va=idx[vstart:vstart+fold_n if f<n_splits-1 else len(idx)]
        tr=tr[bundle['y'].loc[tr].notna()];va=va[bundle['y'].loc[va].notna()]
        if len(tr)<50 or len(va)<20:raise ValueError('Insufficient train/validation labels.')
        assert tr[-1]+pd.Timedelta(bundle['spec']['frequency']) < va[0]
        yield f,tr,va


def fit_risk_intercept(z,rate):
    z=np.clip(np.nan_to_num(z),-8,8);lo=-30.;hi=30.
    for _ in range(70):
        mid=(lo+hi)/2;p=1/(1+np.exp(-np.clip(z+mid,-40,40)))
        if p.mean()>rate:hi=mid
        else:lo=mid
    return (lo+hi)/2


def corrupt(bundle,kind,mechanism,severity,rep,scope='test'):
    """Perturb every clock time in the evaluation interval, including unscored origins.
    Existing missing measurements stay missing. Ground-truth y is never modified.
    Nominal severity is mechanism-specific, not a universal physical error percentage.
    """
    raw=bundle['raw'].copy();idx=bundle['splits'][scope];cols=bundle['columns'];st=bundle['stats']
    block=raw.loc[idx,cols];arr=block.to_numpy().copy();original=arr.copy();observed=np.isfinite(arr)
    n,p=arr.shape;mask=np.zeros_like(observed);s=float(severity)
    rng=np.random.default_rng(seed(bundle['dataset'],kind,mechanism,rep)) # severity omitted: paired/nested paths
    med=st['median'].to_numpy()[None,:];scale=st['iqr'].to_numpy()[None,:]
    onset=None;context=None
    if s>0 and kind=='missing':
        if mechanism=='MCAR':mask=rng.random((n,p))<s
        elif mechanism=='Block outage':
            # Every channel loses a contiguous interval of the requested clock fraction.
            length=min(n,max(1,int(round(s*n))))
            anchors=rng.uniform(0,1,p)
            for j in range(p):
                start=int(anchors[j]*max(1,n-length+1));mask[start:start+length,j]=True
        elif mechanism=='Channel dropout':
            order=rng.permutation(p);mask[:,order[:min(p,max(1,int(round(s*p))))]]=True
        elif mechanism in {'MAR','MNAR sensitivity'}:
            dev=bundle['raw'].loc[:bundle['splits']['development'][-1],cols].to_numpy()
            if mechanism=='MAR':
                context=next((c for c in ['RH','RH_out','T','T_out'] if c in cols),cols[0]);j=cols.index(context)
                z=(arr[:,j]-med[0,j])/scale[0,j];zd=(dev[:,j]-med[0,j])/scale[0,j]
                a=fit_risk_intercept(zd,s)
                prob=1/(1+np.exp(-np.clip(np.nan_to_num(z)+a,-40,40)))
                mask=rng.random((n,p))<prob[:,None];mask[:,j]=False
            else:
                z=np.abs((arr-med)/scale);zd=np.abs((dev-med)/scale)
                a=fit_risk_intercept(zd.ravel(),s)
                prob=1/(1+np.exp(-np.clip(np.nan_to_num(z)+a,-40,40)))
                mask=rng.random((n,p))<prob
        else:raise KeyError(mechanism)
        arr[mask & observed]=np.nan
    elif s>0 and kind=='fault':
        order=rng.permutation(p)
        if mechanism=='Gaussian':arr+=s*rng.normal(size=(n,p))*scale
        elif mechanism=='Multiplicative':arr*=1+s/2*rng.normal(size=(n,p))
        elif mechanism=='Calibration bias':
            chosen=order[:max(1,int(np.ceil(.35*p)))];arr[:,chosen]+=2*s*scale[:,chosen]
        elif mechanism=='Spikes':
            hit=rng.random((n,p))<s/2;arr+=hit*rng.choice([-1,1],(n,p))*(2.5+5*s)*scale
        elif mechanism=='Clipping':arr=np.clip(arr,med-(1.5-s)*scale,med+(1.5-s)*scale)
        elif mechanism=='Stuck-at':
            chosen=order[:max(1,int(round(s*p)))];starts=rng.integers(0,max(1,n//2),p)
            for j in chosen:arr[starts[j]:,j]=med[0,j]
        else:raise KeyError(mechanism)
        # Signed electronic measurement errors are permitted; no undocumented clipping to test bounds.
    elif s>0 and kind=='drift':
        onset=max(1,int(.40*n));chosen=rng.permutation(p)[:max(1,int(np.ceil(.35*p)))];t=np.arange(n-onset)
        if mechanism=='Abrupt mean':add=np.ones((len(t),len(chosen)))*s
        elif mechanism=='Gradual mean':add=np.minimum(1,t/max(1,n//4))[:,None]*s*np.ones((1,len(chosen)))
        elif mechanism=='Variance':add=rng.normal(size=(len(t),len(chosen)))*s
        elif mechanism=='Recurring':add=(((t//max(1,n//6))%2)==0)[:,None]*s*np.ones((1,len(chosen)))
        else:raise KeyError(mechanism)
        arr[onset:,chosen]+=add*scale[:,chosen]
    arr[~observed]=np.nan # never turn a naturally missing value into an observed one
    raw.loc[idx,cols]=arr
    X=features(raw,bundle['spec'],cols).loc[idx]
    target=bundle['spec']['target'];P=bundle['spec']['period']
    naive=pd.DataFrame({'Persistence':raw[target], 'Seasonal naive':raw[target].shift(P-1)},index=raw.index).loc[idx]
    changed=np.isfinite(original)&(np.isnan(arr)|(~np.isclose(original,arr,equal_nan=True)))
    changed_abs=np.where(np.isfinite(arr)&observed,abs(arr-original)/scale,np.nan)
    meta={'nominal_severity':s,'achieved_new_missing_fraction':float((np.isnan(arr)&observed).sum()/max(1,observed.sum())),
          'changed_observed_fraction':float(changed.sum()/max(1,observed.sum())),
          'mean_absolute_change_IQR':float(np.nanmean(changed_abs)),'MAR_context_protected':context,
          'injection_onset':onset,'grid_rows':n,'scored_rows':int(bundle['y'].loc[idx].notna().sum())}
    return X,naive,meta


def scenario(bundle,name,rep,scope='test'):
    lookup={'Clean':('none','Clean',0),'MCAR 20%':('missing','MCAR',.2),
      'Block 20%':('missing','Block outage',.2),'Gaussian 10%':('fault','Gaussian',.1),
      'Bias 10%':('fault','Calibration bias',.1),'Abrupt drift':('drift','Abrupt mean',.75)}
    return corrupt(bundle,*lookup[name],rep,scope=scope)
