"""Six fixed model families, temporal inner validation, per-fit checkpoints."""
from __future__ import annotations
import json,time,warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits
from primna_esd_runtime import seed,metrics,paired_block_summary,ci
from primna_esd_data import DATASETS,cv_folds

MODELS=['Persistence','Seasonal naive','Ridge','Random Forest','HistGradientBoosting','MLP']
ML=MODELS[2:]
SHORT={'Persistence':'Persist.','Seasonal naive':'Seasonal','Ridge':'Ridge','Random Forest':'RF','HistGradientBoosting':'HGB','MLP':'MLP'}


class MissingAware(TransformerMixin,BaseEstimator):
    """Training-only median imputation and fixed indicators for ALL input columns."""
    def fit(self,X,y=None):
        a=np.asarray(X,dtype=float);self.n_features_in_=a.shape[1]
        self.medians_=pd.DataFrame(a).median().fillna(0).to_numpy()
        return self
    def transform(self,X):
        a=np.asarray(X,dtype=float)
        if a.shape[1]!=self.n_features_in_:raise ValueError('Feature schema mismatch')
        m=np.isnan(a);return np.concatenate([np.where(m,self.medians_,a),m.astype(float)],axis=1)


def grid(name,profile):
    if profile=='pilot':
        return {'Ridge':[{'alpha':10.}],
                'Random Forest':[{'n_estimators':35,'max_depth':10,'min_samples_leaf':2,'max_features':.7}],
                'HistGradientBoosting':[{'max_iter':70,'max_leaf_nodes':15,'learning_rate':.05,'l2_regularization':.1}],
                'MLP':[{'hidden_layer_sizes':(32,16),'alpha':.001,'learning_rate_init':.001,'max_iter':120}]}[name]
    return {
      'Ridge':[{'alpha':a} for a in [.1,1.,10.,100.]],
      'Random Forest':[{'n_estimators':300,'max_depth':d,'min_samples_leaf':l,'max_features':f} for d,l,f in [(12,1,.7),(18,2,.7),(None,4,1.)]],
      'HistGradientBoosting':[{'max_iter':i,'max_leaf_nodes':l,'learning_rate':lr,'l2_regularization':reg} for i,l,lr,reg in [(350,15,.05,.1),(350,31,.05,.5),(450,63,.03,1.)]],
      'MLP':[{'hidden_layer_sizes':h,'alpha':a,'learning_rate_init':lr,'max_iter':i} for h,a,lr,i in [((64,32),.001,.001,700),((128,64),.001,.0005,700),((128,64,32),.01,.0005,800)]]
    }[name]


def factory(name,params,random_seed,n_threads=4):
    p=dict(params)
    if name=='Ridge':return Pipeline([('impute',MissingAware()),('scale',StandardScaler()),('model',Ridge(**p))])
    if name=='Random Forest':return Pipeline([('impute',MissingAware()),('model',RandomForestRegressor(**p,random_state=random_seed,n_jobs=n_threads))])
    if name=='HistGradientBoosting':return Pipeline([('impute',MissingAware()),('model',HistGradientBoostingRegressor(**p,early_stopping=False,random_state=random_seed))])
    if name=='MLP':
        reg=Pipeline([('impute',MissingAware()),('scale',StandardScaler()),('model',MLPRegressor(**p,early_stopping=False,n_iter_no_change=40,random_state=random_seed))])
        return TransformedTargetRegressor(regressor=reg,transformer=StandardScaler())
    raise KeyError(name)


def fit_logged(ctx,model,X,y,label):
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=ctx.cfg['threads']):
        warnings.simplefilter('always');model.fit(X,y)
    messages=[str(w.category.__name__)+': '+str(w.message) for w in ws]
    for w in messages:ctx.log('WARNING %s | %s',label,w)
    return model,messages,time.perf_counter()-start


def predict_all(bundle,fit,X,naive):
    arr=[]
    for model in MODELS:
        if model not in ML:
            p=naive[model].fillna(fit['fallbacks'][model]).to_numpy(dtype=float)
        else:p=fit['models'][model].predict(X)
        if not np.isfinite(p).all():raise ValueError('Nonfinite prediction: '+model)
        arr.append(p)
    return np.column_stack(arr)


def train(ctx,bundles):
    fitted={};cv_records=[];clean_records=[];clean_summary=[];selection=[]
    for ds in DATASETS:
        b=bundles[ds];dev=b['splits']['development'];observed=dev[b['y'].loc[dev].notna()]
        fit={'models':{},'params':{},'cv_scores':{},'fallbacks':{},'clean':{}}
        for name in MODELS[:2]:
            v=b['naive'].loc[observed,name].median();fit['fallbacks'][name]=float(v if np.isfinite(v) else b['y'].loc[observed].median())
        for name in ML:
            configs=grid(name,ctx.cfg['profile']);scores=[]
            for config_id,params in enumerate(configs):
                errors=[]
                for fold,tr,va in cv_folds(b,ctx.cfg['cv_folds']):
                    label=f'{ds} | {name} | config {config_id+1}/{len(configs)} | fold {fold+1}/{ctx.cfg["cv_folds"]}'
                    def work(name=name,params=params,tr=tr,va=va,fold=fold,label=label):
                        est=factory(name,params,seed(ctx.cfg['seed'],ds,name,fold),ctx.cfg['threads'])
                        est,ws,elapsed=fit_logged(ctx,est,b['X'].loc[tr],b['y'].loc[tr],label)
                        pr=est.predict(b['X'].loc[va]);return dict(metrics(b['y'].loc[va],pr),warnings=ws,fit_seconds=elapsed,
                            train_end=str(tr[-1]),validation_start=str(va[0]),n_train=len(tr),n_validation=len(va))
                    out=ctx.task('P2CV',label,{'data':b['data_key'],'params':params,'fold':fold},work)
                    errors.append(out['RMSE']);cv_records.append(dict(dataset=ds,model=name,config=config_id,fold=fold,parameters=json.dumps(params),**out))
                scores.append(float(np.mean(errors)))
            best=int(np.argmin(scores));params=configs[best];fit['params'][name]=params;fit['cv_scores'][name]=scores[best]
            label=f'{ds} | {name} | final fit'
            def finalfit(name=name,params=params,label=label):
                est=factory(name,params,seed(ctx.cfg['seed'],ds,name,'final'),ctx.cfg['threads'])
                est,ws,elapsed=fit_logged(ctx,est,b['X'].loc[observed],b['y'].loc[observed],label)
                return {'model':est,'warnings':ws,'fit_seconds':elapsed}
            out=ctx.task('P2FIT',label,{'data':b['data_key'],'params':params},finalfit)
            fit['models'][name]=out['model']
            selection.append({'dataset':ds,'model':name,'best_config':best,'cv_RMSE':scores[best],'params':json.dumps(params),'warnings':' | '.join(out['warnings'])})
        fit['deployment_model']=min(ML,key=lambda n:(fit['cv_scores'][n],n))
        ctx.log('%s deployment model frozen from development CV only: %s',ds,fit['deployment_model'])
        for split in ['calibration','policy','test']:
            idx=b['splits'][split];y=b['y'].loc[idx].to_numpy();P=predict_all(b,fit,b['X'].loc[idx],b['naive'].loc[idx]);fit['clean'][split]=P
            for j,name in enumerate(MODELS):
                for i,t in enumerate(idx):clean_records.append({'dataset':ds,'split':split,'origin':t,'target_time':t+pd.Timedelta(b['spec']['frequency']),
                    'model':name,'y_true':y[i],'y_pred':P[i,j],'scored':bool(np.isfinite(y[i]))})
            point,_,boot,_,_,_=paired_block_summary(y,P[None,:,:],P,b['spec']['period'],ctx.cfg['bootstrap'],np.random.default_rng(seed(ds,split,'clean_ci')))
            for j,name in enumerate(MODELS):
                lo,hi=ci(boot[:,j]);clean_summary.append(dict(dataset=ds,split=split,model=name,**metrics(y,P[:,j]),RMSE_low=lo,RMSE_high=hi))
        fitted[ds]=fit
    ctx.table('P2_rolling_origin_CV',cv_records);ctx.table('P2_model_selection',selection)
    ctx.table('P2_clean_row_predictions',clean_records);ctx.table('P2_clean_summary',clean_summary)
    return fitted
