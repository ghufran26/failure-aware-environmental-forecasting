"""RQ4: empirical calibration feasibility, matched controls, non-oracle adaptation."""
from __future__ import annotations
from collections import deque
import math,time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2
from primna_esd_runtime import seed,metrics,ci
from primna_esd_data import DATASETS,DRIFTS,corrupt
from primna_esd_models import ML,factory,fit_logged

GRIDS={'Page-Hinkley':[8,18,35,75,150,400,1000],
       'CUSUM':[8,18,40,80,160,400,1000],
       'EWMA':[2,4,8,16,32,64],
       'Windowed KS':[.02,.005,.001,.0001,.00001,.000001]}


class Detector:
    """One-sided error-rise monitoring. KS is a two-adjacent-window KS detector,
    NOT River KSWIN. Thresholds are calibrated empirically; iid p-values are not
    claimed valid under serial dependence. reset/cooldown applies after all alarms.
    """
    def __init__(self,method,param,cooldown=30):
        self.method=method;self.param=param;self.cooldown=cooldown;self.last=-10**9;self.reset()
    def reset(self):self.n=0;self.mean=0.;self.s=0.;self.minimum=0.;self.z=0.;self.buf=deque(maxlen=120)
    def update(self,value,index):
        if not np.isfinite(value):return False
        self.n+=1;alarm=False
        if self.method=='Page-Hinkley':
            self.mean+=(value-self.mean)/self.n;self.s+=value-self.mean-.02;self.minimum=min(self.minimum,self.s)
            alarm=self.n>=30 and self.s-self.minimum>self.param
        elif self.method=='CUSUM':
            self.s=max(0,self.s+value-.5);alarm=self.n>=30 and self.s>self.param
        elif self.method=='EWMA':
            self.z=.12*value+.88*self.z;alarm=self.n>=30 and self.z>self.param
        elif self.method=='Windowed KS':
            self.buf.append(value)
            if len(self.buf)==120 and self.n%10==0:
                a=np.asarray(self.buf);alarm=ks_2samp(a[:80],a[80:],method='asymp').pvalue<self.param and a[80:].mean()>a[:80].mean()
        else:raise KeyError(self.method)
        alarm=alarm and index-self.last>=self.cooldown
        if alarm:self.last=index;self.reset()
        return bool(alarm)


def alarms(z,method,param):
    d=Detector(method,param);return [i for i,v in enumerate(z) if d.update(v,i)]


def normalize(errors,reference):
    ref=np.asarray(reference);ref=ref[np.isfinite(ref)];mu=float(ref.mean());sd=max(float(ref.std()),1e-8)
    return (np.asarray(errors)-mu)/sd,mu,sd


def circular_blocks(x,block,rng):
    x=np.asarray(x);n=len(x);starts=rng.integers(0,n,math.ceil(n/block))
    ii=np.concatenate([(s+np.arange(block))%n for s in starts])[:n];return x[ii]


def select_feasible(table,target):
    # The old implementation treated a zero count as proof that FAR <= target.
    # Here the exact one-sided Poisson upper bound is the evidential gate.  A
    # provisional candidate can still be studied when the available exposure is
    # too short, but it is never labelled as satisfying the operational claim.
    t=table.copy();t['feasible']=(t['null_far_q95']<=target)&(t['observed_policy_far']<=target)
    t['far_claim_supported']=t['observed_far_upper95']<=target
    t['selected']=False;t['provisional_selected']=False
    ok=t[t.feasible & (t.policy_strict_detection>=.5)].sort_values(
        ['policy_strict_detection','policy_median_delay','observed_far_upper95'],ascending=[False,True,True],na_position='last')
    if not ok.empty:
        t.loc[ok.index[0],'provisional_selected']=True
        if bool(ok.iloc[0].far_claim_supported):t.loc[ok.index[0],'selected']=True
    return t


def calibrate(ctx,b,fit):
    model_name=fit['deployment_model'];model=fit['models'][model_name]
    ca=b['splits']['calibration'];po=b['splits']['policy']
    calerr=abs(b['y'].loc[ca].to_numpy()-model.predict(b['X'].loc[ca]))
    polerr=abs(b['y'].loc[po].to_numpy()-model.predict(b['X'].loc[po]))
    z,mu,sd=normalize(polerr,calerr);S=ctx.cfg['detector_sims'];tables=[]
    for method,params in GRIDS.items():
        rec=[]
        for param in params:
            label=f'RQ4 calibration | {b["dataset"]} | {method} | {param}'
            def work(method=method,param=param):
                nulls=[];strict=[];delay=[]
                for r in range(S):
                    rng=np.random.default_rng(seed(b['dataset'],'shared_policy_stream',r))
                    x=circular_blocks(z,b['spec']['period'],rng);nulls.append(len(alarms(x,method,param))*1000/len(x))
                    point=int(.6*len(x));xd=x.copy();xd[point:]+=2.
                    aa=alarms(xd,method,param);pre=[i for i in aa if i<point];post=[i for i in aa if i>=point]
                    strict.append(int(not pre and bool(post)))
                    if not pre and post:delay.append(post[0]-point)
                k=len(alarms(z,method,param));exposure=len(z)
                upper=.5*chi2.ppf(.95,2*(k+1))/exposure*1000
                return {'detector':method,'parameter':param,'null_mean_far':np.mean(nulls),
                   'null_far_q95':np.quantile(nulls,.95),'observed_policy_far':len(alarms(z,method,param))*1000/len(z),
                   'observed_alarm_count':k,'observed_exposure':exposure,'observed_far_upper95':upper,
                   'policy_strict_detection':np.mean(strict),'policy_median_delay':np.median(delay) if delay else np.nan,'n_simulations':S}
            rec.append(ctx.task('RQ4CAL',label,{'data':b['data_key'],'model':model_name,'parameter':param,'method':method},work))
        tables.append(select_feasible(pd.DataFrame(rec),ctx.cfg['target_far']))
    tab=pd.concat(tables,ignore_index=True).assign(dataset=b['dataset'],model=model_name)
    chosen=tab[tab.provisional_selected].sort_values(['policy_strict_detection','policy_median_delay','observed_far_upper95'],ascending=[False,True,True],na_position='last')
    primary=None if chosen.empty else (str(chosen.iloc[0].detector),float(chosen.iloc[0].parameter))
    supported=bool(not chosen.empty and chosen.iloc[0].far_claim_supported)
    return {'table':tab,'mean':mu,'scale':sd,'primary':primary,'far_claim_supported':supported}


def prequential_adapt(ctx,b,fit,X,y,policy,normalizer,detector_spec=None,scheduled_indices=None):
    """Predict first, reveal y(t+1), update next prediction. No onset argument exists.
    Batch predictions between updates are a computational optimization: estimates
    are row-independent and only past labels enter detector/refitting decisions.
    All alarms, including pre-injection alarms, are actionable. A fixed refit budget
    and cooldown are configured before test evaluation.
    """
    n=len(X);y=np.asarray(y,dtype=float);model_name=fit['deployment_model'];current=fit['models'][model_name]
    params=fit['params'][model_name];out=np.empty(n);events=[]
    mu,sd=normalizer;det=Detector(*detector_spec) if detector_spec else None
    window=ctx.cfg['adapt_window'];cooldown=ctx.cfg['adapt_cooldown'];budget=ctx.cfg['max_refits']
    dev=b['splits']['development'];dv=dev[b['y'].loc[dev].notna()][-window:]
    baseX=b['X'].loc[dv];basey=b['y'].loc[dv]
    cursor=0;last_update=-10**9;last_attempt=-10**9
    while cursor<n:
        forecasts=current.predict(X.iloc[cursor:]);did_update=False
        for i in range(cursor,n):
            out[i]=forecasts[i-cursor]
            signal=False
            if np.isfinite(y[i]) and det:signal=det.update((abs(y[i]-out[i])-mu)/sd,i)
            scheduled=(policy=='Periodic' and (i+1)%cooldown==0) or (
                policy=='Random budget-matched' and scheduled_indices is not None and i in scheduled_indices)
            triggered=policy in {'Alarm-triggered','Guarded alarm'} and signal
            eligible=np.flatnonzero(np.isfinite(y[:i+1]))
            if (scheduled or triggered) and i<n-1 and i-last_attempt>=cooldown and len(events)<budget and len(eligible)>=ctx.cfg['min_update_labels']:
                validation=np.array([],dtype=int)
                if policy=='Guarded alarm':
                    nv=min(ctx.cfg['adapt_validation_labels'],max(10,len(eligible)//4))
                    validation=eligible[-nv:];candidate_pool=eligible[:-nv]
                else:candidate_pool=eligible
                recent=candidate_pool[-window:]
                if len(recent)<max(20,ctx.cfg['min_update_labels']//2):continue
                trainX=pd.concat([baseX,X.iloc[recent]]).tail(window)
                trainy=pd.concat([basey,pd.Series(y[recent],index=X.index[recent])]).tail(window)
                # Training includes label of origin i only after its target time arrives.
                update_time=X.index[i]+pd.Timedelta(b['spec']['frequency'])
                assert update_time<=X.index[i+1]
                fresh=factory(model_name,params,seed(ctx.cfg['seed'],b['dataset'],policy,len(events),'adapt'),ctx.cfg['threads'])
                fresh,ws,secs=fit_logged(ctx,fresh,trainX,trainy,'causal '+policy+' candidate')
                accepted=True;base_val=np.nan;candidate_val=np.nan;base_tail=np.nan;candidate_tail=np.nan
                if policy=='Guarded alarm':
                    vy=y[validation];bp=current.predict(X.iloc[validation]);cp=fresh.predict(X.iloc[validation])
                    base_val=metrics(vy,bp)['RMSE'];candidate_val=metrics(vy,cp)['RMSE']
                    base_tail=float(np.quantile(abs(vy-bp),.95));candidate_tail=float(np.quantile(abs(vy-cp),.95))
                    accepted=(candidate_val<=base_val*(1-ctx.cfg['adapt_min_relative_gain']) and
                              candidate_tail<=base_tail*(1+ctx.cfg['adapt_tail_tolerance']))
                events.append({'origin_index':i,'label_available_time':update_time,'first_new_prediction':X.index[i+1],
                               'n_train':len(trainX),'n_validation':len(validation),'fit_seconds':secs,
                               'warnings':' | '.join(ws),'trigger':policy,'accepted':accepted,
                               'current_validation_RMSE':base_val,'candidate_validation_RMSE':candidate_val,
                               'current_validation_AE95':base_tail,'candidate_validation_AE95':candidate_tail})
                last_attempt=i
                if det:det.reset()
                if accepted:
                    current=fresh;last_update=i;cursor=i+1;did_update=True;break
        if not did_update:break
    return {'predictions':out,'events':events}


def drift_suite(ctx,bundles,fitted):
    cal_tables=[];rows=[];adapt=[];events=[];timelines=[];natural=[];disabled=[]
    for ds in DATASETS:
        b=bundles[ds];fit=fitted[ds];cal=calibrate(ctx,b,fit);cal_tables.append(cal['table'])
        idx=b['splits']['test'];y=b['y'].loc[idx].to_numpy();model_name=fit['deployment_model'];model=fit['models'][model_name]
        clean=model.predict(b['X'].loc[idx]);clean_z=(abs(y-clean)-cal['mean'])/cal['scale']
        selections={str(r.detector):float(r.parameter) for r in cal['table'].itertuples() if r.provisional_selected}
        supported_detectors={str(r.detector):bool(r.far_claim_supported) for r in cal['table'].itertuples() if r.provisional_selected}
        for method in GRIDS:
            if method not in selections:disabled.append({'dataset':ds,'detector':method,'status':'disabled: no feasible calibration point'})
        for method,param in selections.items():
            for i in alarms(clean_z,method,param):natural.append({'dataset':ds,'detector':method,'index':i,'time':idx[i]+pd.Timedelta(b['spec']['frequency']),
              'interpretation':'unlabelled natural alarm; not verified environmental/concept drift'})
        for condition in ['No injected shift']+DRIFTS:
            R=ctx.cfg['adaptation_reps'] if condition=='No injected shift' else ctx.cfg['drift_reps']
            for rep in range(R):
                if condition=='No injected shift':X=b['X'].loc[idx];onset=None
                else:X,_,meta=corrupt(b,'drift',condition,.75,rep);onset=meta['injection_onset']
                pred=model.predict(X);z=(abs(y-pred)-cal['mean'])/cal['scale']
                if condition!='No injected shift':
                    for method in GRIDS:
                        param=selections.get(method);aa=[] if param is None else alarms(z,method,param)
                        pre=[i for i in aa if i<onset];post=[i for i in aa if i>=onset]
                        control=[] if param is None else alarms(clean_z,method,param)
                        control_post=[i for i in control if i>=onset]
                        rows.append({'dataset':ds,'model':model_name,'drift':condition,'replicate':rep,'detector':method,
                          'enabled':param is not None,'far_target_supported':supported_detectors.get(method,False),
                          'pre_injection_alarms':len(pre),'pre_injection_far':len(pre)*1000/onset,
                          'post_injection_alarm':bool(post),'alarm_without_prior_alarm':bool(post) and not pre,
                          'matched_clean_post_alarm':bool(control_post),'delay_conditional':post[0]-onset if post else np.nan,
                          'delay_or_censored':post[0]-onset if post else len(idx)-onset,'injection_index':onset,'n_clock':len(idx)})
                        if rep==0 and condition=='Abrupt mean':
                            for i in aa:timelines.append({'dataset':ds,'detector':method,'kind':'alarm','index':i,'value':np.nan,'injection_index':onset})
                    if rep==0 and condition=='Abrupt mean':
                        for i,val in enumerate(abs(y-pred)):timelines.append({'dataset':ds,'detector':'Residual','kind':'absolute_error','index':i,'value':val,'injection_index':onset})
                if rep<ctx.cfg['adaptation_reps']:
                    policy_results={}
                    for pol in ['Frozen','Periodic','Alarm-triggered','Guarded alarm']:
                        label=f'RQ4 adaptation | {ds} | {condition} | {pol} | {rep+1}'
                        if pol=='Frozen':res={'predictions':pred,'events':[]}
                        elif pol in {'Alarm-triggered','Guarded alarm'} and cal['primary'] is None:res={'predictions':pred,'events':[]}
                        else:
                            res=ctx.task('RQ4ADAPT',label,{'data':b['data_key'],'condition':condition,'rep':rep,'policy':pol,'detector':cal['primary']},
                              lambda pol=pol,X=X:prequential_adapt(ctx,b,fit,X,y,pol,(cal['mean'],cal['scale']),cal['primary'] if pol in {'Alarm-triggered','Guarded alarm'} else None))
                        policy_results[pol]=res
                        met=metrics(y,res['predictions']);fmet=metrics(y,pred)
                        accepted_events=sum(bool(e.get('accepted',True)) for e in res['events'])
                        adapt.append(dict(dataset=ds,drift=condition,replicate=rep,policy=pol,**met,
                          frozen_RMSE=fmet['RMSE'],RMSE_difference=met['RMSE']-fmet['RMSE'],refits=accepted_events,refit_attempts=len(res['events']),
                          refit_seconds=sum(e['fit_seconds'] for e in res['events']),
                          detector_disabled=cal['primary'] is None and pol in {'Alarm-triggered','Guarded alarm'}))
                        for ev in res['events']:events.append(dict(dataset=ds,drift=condition,replicate=rep,policy=pol,**ev))
                    # Equal-compute negative control: update at random eligible times
                    # using the same attempt budget as alarm-triggered adaptation.
                    budget_n=len(policy_results['Alarm-triggered']['events'])
                    eligible=np.arange(ctx.cfg['min_update_labels'],max(ctx.cfg['min_update_labels']+1,len(idx)-1),ctx.cfg['adapt_cooldown'])
                    rng=np.random.default_rng(seed(ds,condition,rep,'random_budget'))
                    schedule=set(sorted(rng.choice(eligible,min(budget_n,len(eligible)),replace=False).tolist())) if budget_n else set()
                    pol='Random budget-matched';label=f'RQ4 adaptation | {ds} | {condition} | {pol} | {rep+1}'
                    res=ctx.task('RQ4ADAPT',label,{'data':b['data_key'],'condition':condition,'rep':rep,
                      'policy':pol,'budget':budget_n},lambda X=X,schedule=schedule:prequential_adapt(
                        ctx,b,fit,X,y,pol,(cal['mean'],cal['scale']),None,scheduled_indices=schedule))
                    met=metrics(y,res['predictions']);fmet=metrics(y,pred)
                    adapt.append(dict(dataset=ds,drift=condition,replicate=rep,policy=pol,**met,
                      frozen_RMSE=fmet['RMSE'],RMSE_difference=met['RMSE']-fmet['RMSE'],
                      refits=len(res['events']),refit_attempts=len(res['events']),
                      refit_seconds=sum(e['fit_seconds'] for e in res['events']),detector_disabled=False))
                    for ev in res['events']:events.append(dict(dataset=ds,drift=condition,replicate=rep,policy=pol,**ev))
    adaptation_df=pd.DataFrame(adapt)
    # Incremental value over scheduled updating is more relevant than comparison
    # with a permanently frozen model alone.
    periodic=(adaptation_df[adaptation_df.policy=='Periodic'][['dataset','drift','replicate','RMSE']]
              .rename(columns={'RMSE':'periodic_RMSE'}))
    adaptation_df=adaptation_df.merge(periodic,on=['dataset','drift','replicate'],how='left')
    adaptation_df['RMSE_difference_vs_periodic']=adaptation_df.RMSE-adaptation_df.periodic_RMSE
    out={'calibration':pd.concat(cal_tables,ignore_index=True),'operating_results':pd.DataFrame(rows),
         'adaptation':adaptation_df,'update_events':pd.DataFrame(events),
         'timelines':pd.DataFrame(timelines),'natural_events':pd.DataFrame(natural),'disabled':pd.DataFrame(disabled)}
    for k,v in out.items():ctx.table('RQ4_'+k,v)
    return out
