"""Regression tests target the specific audit failures. Fixtures are unit tests only."""
from __future__ import annotations
import inspect,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from primna_esd_runtime import atomic_json,metrics,filehash
from primna_esd_data import features,SPECS,corrupt
from primna_esd_models import MissingAware,factory
from primna_esd_drift import select_feasible,prequential_adapt
from primna_esd_uncertainty import conformal_quantile,canonical_score,interval_metrics,attainable_curve,decision_metrics


def run_tests(ctx,bundles=None):
    results=[]
    def check(name,fn):
        fn();results.append({'test':name,'passed':True});ctx.log('TEST PASS | %s',name)
    def constant_width():
        y=np.array([0.,1.,9.,3.]);p=np.zeros(4);h=np.full(4,2.);s=canonical_score(2*h,1e-9)
        m=interval_metrics(y,p,h,s,4.,4.)
        assert m['acceptance']==1 and m['high_error_AUROC']==.5 and m['score_constant']
        curve=attainable_curve(y,p,s);a=curve[curve.kind.str.startswith('attainable')]
        assert len(a)==1 and a.accepted_fraction.iloc[0]==1
    check('Constant-width intervals cannot create selective rankings',constant_width)
    def permutation_invariance():
        y=np.arange(6.);p=np.ones(6);s=np.array([1.,1.,2.,2.,3.,3.]);h=s/2
        a=interval_metrics(y,p,h,s,2.,2.);ii=np.array([3,0,5,1,4,2]);b=interval_metrics(y[ii],p[ii],h[ii],s[ii],2.,2.)
        for k in a:assert np.isclose(a[k],b[k],equal_nan=True)
    check('Tied-score decisions are row-order invariant',permutation_invariance)
    check('Finite-sample conformal rank returns infinity when required',lambda: _assert(np.isinf(conformal_quantile([1,2],.99))))
    def detector_infeasible():
        t=pd.DataFrame({'null_mean_far':[.2,.3],'null_far_q95':[5.,6.],'observed_policy_far':[0.,0.],
          'observed_far_upper95':[4.,4.],
          'policy_strict_detection':[1.,.8],'policy_median_delay':[1.,2.]})
        assert not select_feasible(t,1.).selected.any()
    check('Zero alarms do not bypass the rare-FAR uncertainty gate',detector_infeasible)
    def no_oracle():assert all('onset' not in n and 'drift' not in n for n in inspect.signature(prequential_adapt).parameters)
    check('Operational adaptation has no oracle-onset input',no_oracle)
    def causal_features():
        raw=pd.DataFrame({'CO(GT)':np.arange(100.),'T':np.arange(100.)},index=pd.date_range('2020-01-01',periods=100,freq='h'))
        a=features(raw,SPECS['Air Quality'],list(raw));raw.iloc[70:]=999.;b=features(raw,SPECS['Air Quality'],list(raw))
        pd.testing.assert_frame_equal(a.iloc[:70],b.iloc[:70])
        assert sum(c.startswith('lag') for c in a)==len(raw.columns)*len(SPECS['Air Quality']['lags'])
        # For target t+1, the same clock hour yesterday is t+1-24, i.e. lag 23 at origin t.
        assert raw['CO(GT)'].shift(23).iloc[50]==27
    check('Feature construction is nonrecursive and future invariant; seasonal offset',causal_features)
    def imputer():
        t=MissingAware().fit([[1.,np.nan],[3.,np.nan]]);z=t.transform([[np.nan,9.]])
        assert np.allclose(z,[[2.,9.,1.,0.]])
    check('All-feature missing indicators and training-only medians',imputer)
    def denominator():
        try:metrics([1.,2.],[1.,np.nan])
        except ValueError:return
        raise AssertionError('Nonfinite scored prediction silently ignored')
    check('Failed predictions do not shrink the evaluation denominator',denominator)
    def decision_policy():
        y=np.array([0.,10.,0.,10.]);p=np.array([1.,9.,6.,4.]);h=np.array([1.,1.,2.,2.])
        r=decision_metrics(y,p,h,5.,5.,.25)
        assert r['false_negative_rate']==0 and r['false_positive_rate']==0 and r['abstention_rate']==.5
    check('Interval decision policy separates alerts, rejection, and abstention',decision_policy)
    def capacity_policy():
        y=np.array([0.,10.,0.,10.]);p=np.array([4.,6.,4.,6.]);h=np.full(4,3.)
        r=decision_metrics(y,p,h,5.,5.,.25,review_capacity=.25)
        assert r['abstention_rate']<=.25 and r['capacity_binding']
    check('Decision review demand obeys the registered capacity',capacity_policy)
    def adaptation_future():
        class Tiny:
            cfg=dict(ctx.cfg,adapt_window=50,adapt_cooldown=20,max_refits=3,min_update_labels=10)
            def log(self,*args):pass
        oldidx=pd.date_range('2020-01-01',periods=80,freq='h');testidx=pd.date_range(oldidx[-1]+pd.Timedelta('2h'),periods=100,freq='h')
        oldX=pd.DataFrame({'x':np.sin(np.arange(80)/6)},index=oldidx);oy=pd.Series(2*oldX.x+1,index=oldidx)
        X=pd.DataFrame({'x':np.sin(np.arange(100)/6)},index=testidx);y=(2*X.x+2).to_numpy()
        est=factory('Ridge',{'alpha':1.},1,1).fit(oldX,oy)
        b={'dataset':'unit_fixture','X':oldX,'y':oy,'spec':{'frequency':'1h'},'splits':{'development':oldidx}}
        fit={'deployment_model':'Ridge','models':{'Ridge':est},'params':{'Ridge':{'alpha':1.}}}
        a=prequential_adapt(Tiny(),b,fit,X,y,'Periodic',(0,1));yy=y.copy();yy[70:]+=1000
        c=prequential_adapt(Tiny(),b,fit,X,yy,'Periodic',(0,1));assert np.allclose(a['predictions'][:71],c['predictions'][:71])
        assert all(pd.Timestamp(e['label_available_time'])<=pd.Timestamp(e['first_new_prediction']) for e in a['events'])
    check('Prequential predictions precede future-label updates',adaptation_future)
    if bundles:
        for ds,b in bundles.items():
            def actual_data(b=b):
                assert b['raw'].index.is_monotonic_increasing and b['raw'].index.is_unique
                pd.testing.assert_series_equal(b['y'],b['raw'][b['spec']['target']].shift(-1))
                X,N,m=corrupt(b,'missing','MAR',.2,0)
                assert len(X)==len(b['splits']['test']) and m['MAR_context_protected'] is not None
                assert m['scored_rows']==b['y'].loc[X.index].notna().sum()
                assert N.index.equals(X.index)
            check(ds+': actual labels and full-clock corruption denominator',actual_data)
    atomic_json(ctx.root/'manifests'/'regression_tests.json',results)
    return results


def _assert(value):
    if not value:raise AssertionError
