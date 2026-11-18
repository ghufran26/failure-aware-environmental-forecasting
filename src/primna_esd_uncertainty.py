"""RQ5: fair calibration samples, stable scores, tie-aware decision policies."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from primna_esd_runtime import seed,metrics,ci,block_sums
from primna_esd_data import DATASETS,scenario
from primna_esd_models import ML,MissingAware,fit_logged

CONDITIONS=['Clean','MCAR 20%','Gaussian 10%','Abrupt drift']
METHODS=['Global split conformal','Locally scaled conformal']


def conformal_quantile(scores,coverage):
    x=np.asarray(scores,dtype=float);x=x[np.isfinite(x)];n=len(x)
    if not n:raise ValueError('No finite calibration scores')
    k=math.ceil((n+1)*coverage)
    return float(np.sort(x)[k-1]) if k<=n else float('inf')


def canonical_score(score,step):
    s=np.asarray(score,dtype=float)
    if not np.isfinite(s).all():raise ValueError('Nonfinite width score; calibration sample too small or numerical error.')
    return np.round(s/step)*step


def interval_metrics(y,p,half,score,threshold,high_threshold):
    y=np.asarray(y);p=np.asarray(p);half=np.asarray(half);score=np.asarray(score)
    valid=np.isfinite(y);err=abs(y[valid]-p[valid]);hh=half[valid];ss=score[valid]
    accepted=ss<=threshold;high=err>high_threshold
    constant=np.unique(ss).size==1
    auc=(.5 if constant else float(roc_auc_score(high,ss))) if len(np.unique(high))==2 else np.nan
    return {'n':len(err),'coverage':float(np.mean(err<=hh)),'mean_width':float(np.mean(2*hh)),
       'acceptance':float(accepted.mean()),'accepted_RMSE':float(np.sqrt(np.mean(err[accepted]**2))) if accepted.any() else np.nan,
       'joint_false_reassurance':float(np.mean(high&accepted)),
       'high_error_given_accepted':float(high[accepted].mean()) if accepted.any() else np.nan,
       'high_error_AUROC':auc,'score_constant':constant,'unique_scores':int(np.unique(ss).size)}


def attainable_curve(y,p,score):
    valid=np.isfinite(y);y=np.asarray(y)[valid];p=np.asarray(p)[valid];s=np.asarray(score)[valid]
    thresholds=np.unique(np.quantile(s,np.linspace(0,1,21),method='higher'))
    rows=[]
    for th in thresholds:
        take=s<=th;rows.append({'accepted_fraction':take.mean(),'selective_RMSE':metrics(y[take],p[take])['RMSE'],
                               'threshold':th,'n_accepted':int(take.sum()),'kind':'attainable tied-score threshold'})
    # Expected squared error of uniform random selection is the full-cohort MSE.
    full=metrics(y,p)['RMSE']
    for f in np.linspace(.1,1,10):rows.append({'accepted_fraction':f,'selective_RMSE':full,'threshold':np.nan,
       'n_accepted':int(round(f*len(y))),'kind':'random selection: sqrt(expected MSE)'})
    return pd.DataFrame(rows)


def decision_metrics(y,p,half,event_threshold,fn_cost,abstain_cost,review_capacity=1.0):
    """Three-action monitoring policy: alert, no alert, or abstain/review.

    The event threshold is frozen from policy outcomes. Interval policies alert
    only when the full interval exceeds the threshold and reject only when the
    full interval is below it. Straddling intervals abstain. Costs are a declared
    sensitivity grid, not estimated social preferences.
    """
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);valid=np.isfinite(y)
    truth=y>=event_threshold
    if half is None:
        action=(p>=event_threshold).astype(int);abstain=np.zeros(len(p),dtype=bool)
    else:
        h=np.asarray(half,dtype=float);action=np.full(len(p),-1,dtype=int)
        action[p-h>=event_threshold]=1;action[p+h<event_threshold]=0;abstain=action<0
        # Review capacity is a fraction of valid origins.  If more intervals
        # straddle the threshold than can be reviewed, prioritize the largest
        # threshold overlap and fall back to the point decision for the rest.
        candidates=np.flatnonzero(abstain & valid);limit=int(np.floor(review_capacity*valid.sum()))
        if len(candidates)>limit:
            ambiguity=h[candidates]-np.abs(p[candidates]-event_threshold)
            reviewed=set(candidates[np.argsort(-ambiguity)[:limit]].tolist())
            overflow=np.array([i for i in candidates if i not in reviewed],dtype=int)
            action[overflow]=(p[overflow]>=event_threshold).astype(int);abstain=action<0
    fp=(action==1)&(~truth)&valid;fn=(action==0)&truth&valid;ab=abstain&valid
    n=max(1,int(valid.sum()));npos=max(1,int((truth&valid).sum()));nneg=max(1,int((~truth&valid).sum()))
    return {'expected_cost':float((fp.sum()+fn_cost*fn.sum()+abstain_cost*ab.sum())/n),
      'false_negative_rate':float(fn.sum()/npos),'false_positive_rate':float(fp.sum()/nneg),
      'abstention_rate':float(ab.sum()/n),'review_capacity':float(review_capacity),
      'capacity_binding':bool((abstain&valid).sum()>=int(np.floor(review_capacity*valid.sum())) and review_capacity<1),
      'alert_rate':float(((action==1)&valid).sum()/n)}


def mean_matrix_ci(V,mask,block,B,rng):
    V=np.asarray(V);mask=np.asarray(mask,dtype=bool)
    ss=np.array([block_sums(v,mask,block)[0] for v in V]);nn=block_sums(np.zeros(len(mask)),mask,block)[1]
    vals=[]
    for _ in range(B):
        r=rng.integers(0,len(V),len(V));b=rng.integers(0,len(nn),len(nn));den=nn[b].sum()
        vals.append(ss[r][:,b].sum(axis=1).mean()/den if den else np.nan)
    return float(ss.sum(axis=1).mean()/nn.sum()),*ci(vals)


def calibrate_intervals(ctx,b,fit,model_name):
    model=fit['models'][model_name];ca=b['splits']['calibration'];gap=max(b['spec']['lags']+b['spec']['windows'])+1
    half=len(ca)//2;ia=ca[:half-gap];ib=ca[half:]
    ia=ia[b['y'].loc[ia].notna()];ib=ib[b['y'].loc[ib].notna()]
    if min(len(ia),len(ib))<30:raise ValueError('Calibration subdivisions too short.')
    xa=b['X'].loc[ia];ea=abs(b['y'].loc[ia].to_numpy()-model.predict(xa))
    scale=Pipeline([('impute',MissingAware()),('model',HistGradientBoostingRegressor(loss='absolute_error',
        max_iter=150 if ctx.cfg['profile']=='paper' else 45,max_leaf_nodes=15,learning_rate=.05,
        early_stopping=False,random_state=seed(b['dataset'],model_name,'scale')))])
    scale,ws,_=fit_logged(ctx,scale,xa,ea,'uncertainty scale model')
    floor=max(.10*float(np.median(ea)),1e-6*float(np.std(b['y'].loc[ia])),1e-8)
    xb=b['X'].loc[ib];eb=abs(b['y'].loc[ib].to_numpy()-model.predict(xb));sb=np.maximum(scale.predict(xb),floor)
    qg=conformal_quantile(eb,ctx.cfg['coverage']);ql=conformal_quantile(eb/sb,ctx.cfg['coverage'])
    pol=b['splits']['policy'];xp=b['X'].loc[pol];ep=abs(b['y'].loc[pol].to_numpy()-model.predict(xp));pv=np.isfinite(ep)
    high=float(np.quantile(ep[pv],.9));widths={'Global split conformal':np.full(len(pol),2*qg),
        'Locally scaled conformal':2*ql*np.maximum(scale.predict(xp),floor)}
    policies={}
    for method,u in widths.items():
        step=max(1,float(np.median(u)))*1e-10;s=canonical_score(u,step)
        th=float(np.quantile(s,ctx.cfg['acceptance_target'],method='higher'))
        edges=np.unique(np.quantile(s,np.linspace(0,1,9),method='higher'))
        policies[method]={'score_step':step,'threshold':th,'policy_realized_acceptance':float(np.mean(s<=th)),
                         'constant_score':len(np.unique(s))==1,'bin_edges':edges[1:-1].tolist()}
    return {'scale_model':scale,'scale_floor':floor,'global_q':qg,'local_q':ql,'policies':policies,
       'high_error_threshold':high,'scale_fit_n':len(ia),'common_conformal_n':len(ib),
       'scale_fit_last':str(ia[-1]),'conformal_first':str(ib[0]),'warnings':ws}


def uncertainty_suite(ctx,bundles,fitted):
    parameter_rows=[];raw_metrics=[];summaries=[];curves=[];reliability=[];decisions=[]
    for ds in DATASETS:
        b=bundles[ds];idx=b['splits']['test'];y=b['y'].loc[idx].to_numpy();valid=np.isfinite(y)
        for model_name in ML:
            fit=fitted[ds];model=fit['models'][model_name]
            ca=ctx.task('RQ5CAL',f'RQ5 calibration | {ds} | {model_name}',{'data':b['data_key'],'model':model_name},
                        lambda model_name=model_name:calibrate_intervals(ctx,b,fit,model_name))
            for method in METHODS:
                parameter_rows.append(dict(dataset=ds,model=model_name,method=method,q=ca['global_q'] if method==METHODS[0] else ca['local_q'],
                    scale_floor=ca['scale_floor'],scale_fit_n=ca['scale_fit_n'],common_conformal_n=ca['common_conformal_n'],
                    high_error_threshold=ca['high_error_threshold'],**{k:v for k,v in ca['policies'][method].items() if k!='bin_edges'}))
            for condition in CONDITIONS:
                R=1 if condition=='Clean' else ctx.cfg['uncertainty_reps'];collected={m:[] for m in METHODS}
                for rep in range(R):
                    label=f'RQ5 | {ds} | {model_name} | {condition} | {rep+1}/{R}'
                    def work(condition=condition,rep=rep):
                        X,_,_=scenario(b,condition,rep);p=model.predict(X);sc=np.maximum(ca['scale_model'].predict(X),ca['scale_floor'])
                        return {'prediction':p,'global_half':np.full(len(p),ca['global_q']),'local_half':ca['local_q']*sc}
                    res=ctx.task('RQ5',label,{'data':b['data_key'],'model':model_name,'condition':condition,'rep':rep},work)
                    p=res['prediction']
                    policy_y=b['y'].loc[b['splits']['policy']].dropna().to_numpy()
                    for event_q in ctx.cfg['decision_event_quantiles']:
                        event_threshold=float(np.quantile(policy_y,event_q))
                        for fn_cost in ctx.cfg['decision_fn_cost_ratios']:
                            for abstain_cost in ctx.cfg['decision_abstain_costs']:
                                for capacity in ctx.cfg['decision_review_capacities']:
                                    decisions.append(dict(dataset=ds,model=model_name,condition=condition,replicate=rep,
                                      method='Point forecast',event_quantile=event_q,event_threshold=event_threshold,
                                      false_negative_cost_ratio=fn_cost,abstention_cost=abstain_cost,
                                      **decision_metrics(y,p,None,event_threshold,fn_cost,abstain_cost,capacity)))
                    for method,hkey in zip(METHODS,['global_half','local_half']):
                        hh=res[hkey];policy=ca['policies'][method];u=canonical_score(2*hh,policy['score_step'])
                        met=interval_metrics(y,p,hh,u,policy['threshold'],ca['high_error_threshold'])
                        alpha=1-ctx.cfg['coverage'];err=abs(y-p);wink=2*hh+2/alpha*np.maximum(0,err-hh)
                        met['Winkler']=float(np.mean(wink[valid]));met['RMSE']=metrics(y,p)['RMSE']
                        raw_metrics.append(dict(dataset=ds,model=model_name,condition=condition,replicate=rep,method=method,**met))
                        for event_q in ctx.cfg['decision_event_quantiles']:
                            event_threshold=float(np.quantile(policy_y,event_q))
                            for fn_cost in ctx.cfg['decision_fn_cost_ratios']:
                                for abstain_cost in ctx.cfg['decision_abstain_costs']:
                                    for capacity in ctx.cfg['decision_review_capacities']:
                                        decisions.append(dict(dataset=ds,model=model_name,condition=condition,replicate=rep,
                                          method=method,event_quantile=event_q,event_threshold=event_threshold,
                                          false_negative_cost_ratio=fn_cost,abstention_cost=abstain_cost,
                                          **decision_metrics(y,p,hh,event_threshold,fn_cost,abstain_cost,capacity)))
                        cr=attainable_curve(y,p,u).assign(dataset=ds,model=model_name,condition=condition,replicate=rep,method=method);curves.append(cr)
                        binid=np.searchsorted(np.asarray(policy['bin_edges']),u,side='right')
                        for bn in np.unique(binid):
                            take=(binid==bn)&valid
                            if take.any():reliability.append(dict(dataset=ds,model=model_name,condition=condition,replicate=rep,method=method,
                              policy_bin=int(bn),n=int(take.sum()),sum_width=float((2*hh[take]).sum()),sum_absolute_error=float(err[take].sum())))
                        collected[method].append({'coverage':(err<=hh).astype(float),'width':2*hh,'winkler':wink,
                            'accepted':(u<=policy['threshold']).astype(float),'error':err,'score':u})
                        # Portable row-level arrays, independently readable without unpickling models.
                        outdir=ctx.root/'results'/'RQ5_rows';outdir.mkdir(exist_ok=True)
                        fname=f'{seed(ds,model_name,condition,rep,method):08x}.npz'
                        np.savez_compressed(outdir/fname,origin_ns=idx.asi8,y_true=y,prediction=p,half_width=hh,
                            uncertainty_score=u,accepted=u<=policy['threshold'],dataset=ds,model=model_name,condition=condition,method=method,replicate=rep)
                for method,rec in collected.items():
                    ans={'dataset':ds,'model':model_name,'condition':condition,'method':method,'replicates':len(rec),'n_scored':int(valid.sum())}
                    for field in ['coverage','width','winkler','accepted']:
                        pt,lo,hi=mean_matrix_ci([r[field] for r in rec],valid,b['spec']['period'],ctx.cfg['bootstrap'],np.random.default_rng(seed(ds,model_name,condition,method,field)))
                        ans[field]=pt;ans[field+'_low']=lo;ans[field+'_high']=hi
                    m=pd.DataFrame(raw_metrics);sub=m[(m.dataset==ds)&(m.model==model_name)&(m.condition==condition)&(m.method==method)]
                    for f in ['high_error_AUROC','joint_false_reassurance','high_error_given_accepted','accepted_RMSE','score_constant']:
                        ans[f]=sub[f].mean()
                    summaries.append(ans)
    result={'parameters':pd.DataFrame(parameter_rows),'replicate_metrics':pd.DataFrame(raw_metrics),
      'summary':pd.DataFrame(summaries),'curves':pd.concat(curves,ignore_index=True),
      'reliability_bins':pd.DataFrame(reliability),'decision_utility':pd.DataFrame(decisions)}
    for k,v in result.items():ctx.table('RQ5_'+k,v)
    return result
