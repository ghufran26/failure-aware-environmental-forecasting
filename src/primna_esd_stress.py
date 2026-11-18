"""RQ1/RQ2 common-denominator stress tests and RQ3 explanation stability."""
from __future__ import annotations
import math,time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
from primna_esd_runtime import seed,metrics,paired_block_summary,ci,holm,paired_sign_test
from primna_esd_data import DATASETS,MISSING,FAULTS,corrupt,scenario
from primna_esd_models import MODELS,ML,predict_all


def stress_suite(ctx,bundles,fitted,rq):
    kind='missing' if rq==1 else 'fault';mechanisms=MISSING if rq==1 else FAULTS
    levels=ctx.cfg['missing_levels'] if rq==1 else ctx.cfg['fault_levels']
    records=[];summary=[];audcs=[];tests=[];tails=[];audit=[]
    total=len(DATASETS)*len(mechanisms)*len(levels)*ctx.cfg['perturb_reps']
    completed=0;started=time.monotonic()
    for ds in DATASETS:
        b=bundles[ds];fit=fitted[ds];idx=b['splits']['test'];y=b['y'].loc[idx].to_numpy();clean=fit['clean']['test']
        R=ctx.cfg['perturb_reps'];B=ctx.cfg['bootstrap'];block=b['spec']['period'];all_mechanism_boot=[]
        clean_rmse=[metrics(y,clean[:,j])['RMSE'] for j in range(len(MODELS))]
        ctx.log('RQ%d PLAN | dataset=%s | rows=%d | models=%d | mechanisms=%d | levels=%d | replicas=%d | bootstrap=%d',
                rq,ds,len(idx),len(MODELS),len(mechanisms),len(levels),R,B)
        for mech in mechanisms:
            relative_boot=[];relative_point=[]
            for s in levels:
                ctx.log('RQ%d LEVEL START | %s | %s | severity=%s',rq,ds,mech,s)
                PP=[]
                for rep in range(R):
                    label=f'RQ{rq} | {ds} | {mech} s={s} | replicate {rep+1}/{R}'
                    if s==0:out={'predictions':clean,'audit':{'achieved_new_missing_fraction':0.,'changed_observed_fraction':0.,'mean_absolute_change_IQR':0.}}
                    else:
                        def work(mech=mech,s=s,rep=rep):
                            with ctx.substage(label+' corruption'):
                                X,N,meta=corrupt(b,kind,mech,s,rep)
                            with ctx.substage(label+' six-model prediction',shape=X.shape):
                                columns=[]
                                for model_name in MODELS:
                                    def predict_one(model_name=model_name):
                                        if model_name not in ML:
                                            p=N[model_name].fillna(fit['fallbacks'][model_name]).to_numpy(dtype=float)
                                        else:p=fit['models'][model_name].predict(X)
                                        p=np.asarray(p)
                                        if p.shape!=(len(X),) or not np.isfinite(p).all():
                                            raise ValueError('Invalid prediction shape/values: '+model_name)
                                        return p
                                    columns.append(ctx.task('RQ'+str(rq)+'MODEL',label+' | model='+model_name,
                                        {'data':b['data_key'],'mechanism':mech,'severity':s,'rep':rep,
                                         'model':model_name,'params':fit.get('params',{}).get(model_name)},predict_one))
                                predictions=np.column_stack(columns)
                            return {'predictions':predictions,'audit':meta}
                        out=ctx.task('RQ'+str(rq),label,{'data':b['data_key'],'mechanism':mech,'severity':s,'rep':rep},work)
                    P=out['predictions'];PP.append(P)
                    completed+=1
                    ctx.log('RQ%d PROGRESS | units=%d/%d | elapsed=%.1f min | current=%s',
                            rq,completed,total,(time.monotonic()-started)/60,label)
                    audit.append(dict(dataset=ds,mechanism=mech,severity=s,replicate=rep,**out['audit']))
                    for j,m in enumerate(MODELS):
                        met=metrics(y,P[:,j]);cr=clean_rmse[j]
                        records.append(dict(dataset=ds,mechanism=mech,severity=s,replicate=rep,model=m,**met,clean_RMSE=cr,relative_change=met['RMSE']/cr-1))
                        if s==levels[-1]:tails.append(dict(dataset=ds,mechanism=mech,replicate=rep,model=m,tail95=met['AE_q95'],MAE=met['MAE']))
                PP=np.asarray(PP)
                with ctx.substage(f'RQ{rq} bootstrap | {ds} | {mech} | {s}',draws=B,shape=PP.shape):
                    point,cp,bt,rb,ss,counts=ctx.task('RQ'+str(rq)+'BOOT',f'RQ{rq} bootstrap | {ds} | {mech} | {s}',
                        {'data':b['data_key'],'mechanism':mech,'severity':s,'replicas':R,'draws':B,'block':block},
                        lambda:paired_block_summary(y,PP,clean,block,B,np.random.default_rng(seed(ds,rq,'shared_boot'))))
                relative_point.append(point/cp-1);relative_boot.append(rb)
                for j,m in enumerate(MODELS):
                    lo,hi=ci(bt[:,j]);rl,rh=ci(rb[:,j]);summary.append(dict(dataset=ds,mechanism=mech,severity=s,model=m,
                      pooled_RMSE=point[j],RMSE_low=lo,RMSE_high=hi,relative_change=point[j]/cp[j]-1,relative_low=rl,relative_high=rh,
                      interval='95% replicate + clock-day block bootstrap; fixed historical test period'))
                if s==levels[-1]:
                    ref=MODELS.index('HistGradientBoosting')
                    for j,m in enumerate(MODELS):
                        if j==ref:continue
                        diff=(ss[:,j,:]-ss[:,ref,:]).mean(axis=0)
                        p_sign=ctx.task('RQ'+str(rq)+'SIGN',f'RQ{rq} sign test | {ds} | {mech} | {m}',
                            {'data':b['data_key'],'severity':s,'draws':ctx.cfg['randomization_draws']},
                            lambda:paired_sign_test(diff,ctx.cfg['randomization_draws'],np.random.default_rng(seed(ds,rq,mech,m,'test'))))
                        tests.append({'dataset':ds,'mechanism':mech,'severity':s,'model':m,'reference':'HistGradientBoosting',
                         'MSE_difference':float(diff.sum()/counts.sum()),
                         'p_block_sign':p_sign,
                         'n_clock_day_blocks':len(counts),'assumption':'secondary block-sign randomization; blocks sign-exchangeable; not independent datasets'})
            area=trapezoid(np.array(relative_point),levels,axis=0)/max(levels)
            area_boot=trapezoid(np.array(relative_boot),levels,axis=0)/max(levels)
            all_mechanism_boot.append((area,area_boot))
            for j,m in enumerate(MODELS):
                lo,hi=ci(area_boot[:,j]);audcs.append(dict(dataset=ds,mechanism=mech,model=m,AUDC=area[j],AUDC_low=lo,AUDC_high=hi))
        av=np.mean([x[0] for x in all_mechanism_boot],axis=0);avb=np.mean([x[1] for x in all_mechanism_boot],axis=0)
        for j,m in enumerate(MODELS):
            lo,hi=ci(avb[:,j]);audcs.append(dict(dataset=ds,mechanism='Descriptive equal-weight scenario mean',model=m,AUDC=av[j],AUDC_low=lo,AUDC_high=hi))
    tests=pd.DataFrame(tests);tests['p_Holm']=np.nan
    for ds,g in tests.groupby('dataset'):tests.loc[g.index,'p_Holm']=holm(g['p_block_sign'])
    out={'replicates':pd.DataFrame(records),'summary':pd.DataFrame(summary),'audc':pd.DataFrame(audcs),
         'comparisons':tests,'tails':pd.DataFrame(tails),'injection_audit':pd.DataFrame(audit)}
    for name,df in out.items():ctx.table(f'RQ{rq}_{name}',df)
    return out


EXPLANATION_SCENARIOS=['Clean repeat control','MCAR 20%','Block 20%','Gaussian 10%','Bias 10%','Abrupt drift']


def explanation_positions(n,budget):
    # Two prespecified CONTIGUOUS windows, not evenly spaced isolated observations.
    w=min(n//2,max(20,budget//2));starts=[max(0,n//4-w//2),min(n-w,3*n//4-w//2)]
    a=np.arange(starts[0],starts[0]+w);b=np.arange(starts[1],starts[1]+w)
    if np.intersect1d(a,b).size:return np.arange(min(n,budget)),[min(n,budget)]
    return np.r_[a,b],[w,w]


def permutation_order(lengths,block,rng):
    order=[];offset=0
    for n in lengths:
        parts=[np.arange(i,min(n,i+block))+offset for i in range(0,n,block)]
        order.extend([parts[i] for i in rng.permutation(len(parts))]);offset+=n
    return np.concatenate(order)


def group_importance(model,X,y,groups,repeats,lengths,block,random_seed):
    baseline=metrics(y,model.predict(X))['RMSE'];rows=[]
    for group,cols in groups.items():
        for r in range(repeats):
            rng=np.random.default_rng(seed(random_seed,group,r))
            order=permutation_order(lengths,block,rng);Xp=X.copy();Xp.loc[:,cols]=X.iloc[order][cols].to_numpy()
            rows.append({'group':group,'permutation':r,'importance':metrics(y,model.predict(Xp))['RMSE']-baseline})
    return pd.DataFrame(rows),baseline


def stability(a,b):
    a,b=a.align(b);a=a.astype(float);b=b.astype(float)
    sp=float(stats.spearmanr(a,b).statistic) if a.nunique()>1 and b.nunique()>1 else np.nan
    kt=float(stats.kendalltau(a,b).statistic) if a.nunique()>1 and b.nunique()>1 else np.nan
    k=min(8,max(2,len(a)//3));topa=set(a.index[a>=a.nlargest(k).min()]);topb=set(b.index[b>=b.nlargest(k).min()])
    active=np.abs(a)>max(1e-12,.1*np.median(abs(a)))
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return {'Spearman':sp,'Kendall':kt,'topk_Jaccard':len(topa&topb)/max(1,len(topa|topb)),
      'signed_cosine':float(np.dot(a,b)/den) if den else np.nan,
      'sign_agreement':float(np.mean(np.sign(a[active])==np.sign(b[active]))) if active.any() else np.nan,
      'top_k_requested':k,'top_a_with_ties':len(topa),'top_b_with_ties':len(topb)}


def explanation_suite(ctx,bundles,fitted):
    profiles=[];out=[];windows=[]
    R=ctx.cfg['explanation_reps'];K=ctx.cfg['importance_repeats']
    for ds in DATASETS:
        b=bundles[ds];idx=b['splits']['test'];pos,lengths=explanation_positions(len(idx),ctx.cfg['explanation_rows'])
        ti=idx[pos];y=b['y'].loc[ti].to_numpy();clean=b['X'].loc[ti];block=max(4,b['spec']['period']//6)
        windows.extend([{'dataset':ds,'origin':t} for t in ti])
        for model in ML:
            est=fitted[ds]['models'][model]
            for rep in range(R):
                lab=f'RQ3 | {ds} | {model} | paired clean profile {rep+1}/{R}'
                clean_res=ctx.task('RQ3',lab,{'data':b['data_key'],'model':model,'rep':rep,'role':'clean'},
                    lambda rep=rep:group_importance(est,clean,y,b['groups'],K,lengths,block,seed(ds,rep,'paired_perms')))
                ip,clean_rmse=clean_res;a=ip.groupby('group')['importance'].mean()
                tmp=ip.assign(dataset=ds,model=model,scenario='Clean',replicate=rep);profiles.append(tmp)
                for condition in EXPLANATION_SCENARIOS:
                    lab=f'RQ3 | {ds} | {model} | {condition} | {rep+1}/{R}'
                    def work(condition=condition,rep=rep):
                        if condition=='Clean repeat control':Xp=clean;sr=seed(ds,rep,'independent_clean_perms')
                        else:Xp=scenario(b,condition,rep)[0].loc[ti];sr=seed(ds,rep,'paired_perms')
                        return group_importance(est,Xp,y,b['groups'],K,lengths,block,sr)
                    res=ctx.task('RQ3',lab,{'data':b['data_key'],'model':model,'rep':rep,'condition':condition},work)
                    ip2,pert_rmse=res;bp=ip2.groupby('group')['importance'].mean();met=stability(a,bp)
                    out.append(dict(dataset=ds,model=model,scenario=condition,replicate=rep,**met,
                                    relative_RMSE_change=pert_rmse/clean_rmse-1,n_scored=int(np.isfinite(y).sum())))
                    profiles.append(ip2.assign(dataset=ds,model=model,scenario=condition,replicate=rep))
    stability_df=pd.DataFrame(out)
    # RQ3 asks whether degradation exceeds the explainer's own Monte Carlo
    # variability.  Therefore every degraded profile is paired with the clean
    # repeat from the same dataset/model/replicate; raw stability alone is not
    # an answer to that question.
    metrics_to_compare=['Spearman','Kendall','topk_Jaccard','signed_cosine','sign_agreement']
    control=(stability_df[stability_df.scenario=='Clean repeat control']
             [['dataset','model','replicate']+metrics_to_compare]
             .rename(columns={m:m+'_control' for m in metrics_to_compare}))
    paired=(stability_df[stability_df.scenario!='Clean repeat control']
            .merge(control,on=['dataset','model','replicate'],validate='many_to_one'))
    contrasts=[]
    rng=np.random.default_rng(seed(ctx.cfg['seed'],'RQ3 paired contrasts'))
    for (ds,model,scenario_name),g in paired.groupby(['dataset','model','scenario'],sort=False):
        for metric_name in metrics_to_compare:
            delta=(g[metric_name]-g[metric_name+'_control']).dropna().to_numpy(float)
            if not len(delta):continue
            draws=np.array([np.mean(rng.choice(delta,len(delta),replace=True)) for _ in range(ctx.cfg['bootstrap'])])
            # Two-sided exact sign-flip randomization on paired replica contrasts.
            B=ctx.cfg['randomization_draws'];observed=abs(delta.sum())
            null=np.abs((rng.choice([-1.,1.],size=(B,len(delta)))*delta).sum(axis=1))
            contrasts.append({'dataset':ds,'model':model,'scenario':scenario_name,'metric':metric_name,
              'n_pairs':len(delta),'mean_degraded_minus_control':float(delta.mean()),
              'CI_low':float(np.quantile(draws,.025)),'CI_high':float(np.quantile(draws,.975)),
              'p_signflip':float((1+(null>=observed-1e-12).sum())/(B+1))})
    contrast_df=pd.DataFrame(contrasts);contrast_df['p_Holm']=np.nan
    for (ds,model,metric_name),g in contrast_df.groupby(['dataset','model','metric']):
        contrast_df.loc[g.index,'p_Holm']=holm(g.p_signflip)
    result={'stability':stability_df,'contrasts':contrast_df,
      'importance':pd.concat(profiles,ignore_index=True),'windows':pd.DataFrame(windows)}
    for k,v in result.items():ctx.table('RQ3_'+k,v)
    return result
