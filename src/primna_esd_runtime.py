"""Auditable task cache, logging, statistics and artifact utilities."""
from __future__ import annotations
import contextlib, datetime as dt, gc, hashlib, importlib.metadata, json, logging, os
import platform, shutil, sys, threading, time, traceback, uuid, zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import stats

SCHEMA_VERSION = 'primna-esd-audit-v4.0'


def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x, (Path,pd.Timestamp,dt.datetime)): return str(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    return x


def digest(x):
    return hashlib.sha256(json.dumps(jsonable(x), sort_keys=True, default=str).encode()).hexdigest()


def filehash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(jsonable(value),indent=2,default=str),encoding='utf-8')
    os.replace(tmp,path)


def seed(*parts): return int(digest(parts)[:8],16)


def environment():
    result={'python':platform.python_version(),'platform':platform.platform()}
    for name in ['numpy','pandas','scipy','scikit-learn','joblib','matplotlib','psutil','threadpoolctl','codecarbon']:
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]='unavailable'
    return result


def resource_state(path=None):
    """Small, failure-tolerant resource snapshot for progress and diagnostics."""
    proc=psutil.Process();vm=psutil.virtual_memory()
    state={'rss_mb':round(proc.memory_info().rss/2**20,1),
           'available_ram_mb':round(vm.available/2**20,1),
           'ram_percent':float(vm.percent),'process_threads':proc.num_threads(),
           'cpu_count':psutil.cpu_count(logical=True),'gpu':'not requested (CPU-only design)'}
    if path is not None:
        disk=psutil.disk_usage(str(Path(path)))
        state.update(disk_free_mb=round(disk.free/2**20,1),disk_percent=float(disk.percent))
    return state


class Context:
    def __init__(self, config, source_dir):
        self.cfg=dict(config); self.source_dir=Path(source_dir)
        sources={p.name:p.read_text() for p in self.source_dir.glob('primna_esd_*.py')}
        core={k:v for k,v in sources.items() if not k.endswith('_figures.py')}
        runtime_keys={'start_phase','end_phase','resume_root','work_root','data_paths','external_data_paths','allow_download',
                      'allow_install','heartbeat_seconds','snapshot_each_phase','overwrite_failed_tasks','png_dpi'}
        scientific={k:v for k,v in self.cfg.items() if k not in runtime_keys}
        self.env=environment()
        compute_env={k:v for k,v in self.env.items() if k!='platform'}
        self.key=digest({'schema':SCHEMA_VERSION,'sources':core,'config':scientific,'environment':compute_env})
        self.root=Path(config['work_root'])/('primna_esd_'+config['profile']+'_'+self.key[:12])
        self.root.mkdir(parents=True,exist_ok=True)
        self.cache=self.root/'checkpoints'; self.cache.mkdir(exist_ok=True)
        for n in ['logs','source','manifests','data','results','figures_pdf','figures_png','tables','plot_data']:
            (self.root/n).mkdir(exist_ok=True)
        self.logger=logging.getLogger('primna_esd_'+self.key[:12]); self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fmt=logging.Formatter('%(asctime)s | %(message)s')
        sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); self.logger.addHandler(sh)
        fh=logging.FileHandler(self.root/'logs'/'execution.log'); fh.setFormatter(fmt); self.logger.addHandler(fh)
        self.results={}; self.task_events=[]; self.phase_status={};self._last_event_flush=0.
        evaluation_status='retrospective core audit plus independent external replication; no software-generated confirmatory claim'
        atomic_json(self.root/'manifests'/'run_identity.json',{'schema':SCHEMA_VERSION,'key':self.key,
            'config':self.cfg,'environment':self.env,'source_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
            'target_journal':'Environment Systems and Decisions (Springer Nature)',
            'evaluation_status':evaluation_status})
        for name,s in sources.items(): (self.root/'source'/name).write_text(s)
        atomic_json(self.root/'source'/'config.json',self.cfg)
        self.restore(config.get('resume_root'))
        self.log('Run directory: %s | accelerator: CPU only',self.root)

    def log(self,msg,*args):
        self.logger.info(msg,*args)
        for handler in self.logger.handlers:
            try:handler.flush()
            except Exception:pass

    def resources(self): return resource_state(self.root)

    def record_task_event(self,event):
        self.task_events.append(event)
        # Append one small record instead of rewriting a growing manifest after
        # each model. The conventional JSON manifest is flushed periodically
        # and before snapshots; checkpoint metadata remains the commit marker.
        with (self.root/'logs'/'task_events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(jsonable(event),default=str)+'\n')
        if time.monotonic()-self._last_event_flush>=10:
            atomic_json(self.root/'manifests'/'task_events.json',self.task_events)
            self._last_event_flush=time.monotonic()

    def failure(self,label):
        # Preserve the active traceback immediately. Do not start a potentially
        # multi-GB archive job while handling cancellation or another exception.
        self.log('FAILED %s\n%s',label,traceback.format_exc())
        try:
            atomic_json(self.root/'manifests'/'phase_status.json',self.phase_status)
            atomic_json(self.root/'manifests'/'task_events.json',self.task_events)
        except Exception:
            self.log('Failed to save failure status\n%s',traceback.format_exc())

    @contextlib.contextmanager
    def substage(self,label,**details):
        started=time.perf_counter()
        self.log('SUBSTAGE START %s | details=%s | resources=%s',label,jsonable(details),self.resources())
        try:
            yield
        except BaseException:
            self.log('SUBSTAGE FAILED %s | %.1f s | details=%s\n%s',label,time.perf_counter()-started,
                     jsonable(details),traceback.format_exc())
            raise
        else:
            self.log('SUBSTAGE COMPLETE %s | %.1f s | resources=%s',label,time.perf_counter()-started,self.resources())

    def restore(self,root):
        if not root: return
        root=Path(root)
        candidates=[root] if (root/'manifests'/'run_identity.json').exists() else list(root.rglob('run_identity.json'))
        candidates=[p.parent.parent if p.is_file() else p for p in candidates]
        matched=[]
        for p in candidates:
            ident=json.loads((p/'manifests'/'run_identity.json').read_text())
            if ident.get('key')==self.key: matched.append(p)
        if len(matched)!=1:
            raise RuntimeError('Resume requires exactly one compatible ESD-v4 run. Legacy runs and changed code/config/environment are rejected. Found '+str(len(matched)))
        src=matched[0]
        if src.resolve()==self.root.resolve(): return
        manifest=src/'manifests'/'output_checksums.csv'
        if not manifest.exists():raise RuntimeError('Resume archive lacks checksum manifest. Preserve it; do not treat as a trusted compatible snapshot.')
        entries=pd.read_csv(manifest)
        for row in entries.itertuples():
            p=src/row.path
            if not p.is_file() or filehash(p)!=row.sha256:raise RuntimeError('Resume checksum mismatch: '+str(p))
            rel=Path(row.path)
            if rel.parts[0] in {'source','logs'} or rel.name in {'run_identity.json','task_events.json'}:continue
            dst=self.root/rel
            if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        self.log('Restored verified compatible snapshot: %s',src)

    @contextlib.contextmanager
    def heartbeat(self,label):
        start=time.monotonic(); stop=threading.Event()
        def pulse():
            while not stop.wait(self.cfg.get('heartbeat_seconds',60)):
                self.log('RUNNING %s | elapsed %.1f min | resources=%s',label,(time.monotonic()-start)/60,self.resources())
        t=threading.Thread(target=pulse,daemon=True);t.start()
        try: yield
        finally: stop.set();t.join(timeout=2)

    def task(self,stage,label,inputs,fn):
        # Include cache validation, loading, serialization and hashing, not only fn().
        with self.heartbeat('task lifecycle | '+label):
            return self._task_impl(stage,label,inputs,fn)

    def _task_impl(self,stage,label,inputs,fn):
        key=digest({'run':self.key,'stage':stage,'label':label,'inputs':inputs})
        stem=stage+'_'+key[:24]; path=self.cache/(stem+'.joblib'); meta=self.cache/(stem+'.json')
        expected={'task_key':key,'run_key':self.key,'label':label,'stage':stage}
        if path.exists() or meta.exists():
            reason=None
            try:
                if not (path.exists() and meta.exists()):reason='incomplete checkpoint pair'
                else:
                    m=json.loads(meta.read_text())
                    if not all(m.get(k)==v for k,v in expected.items()):reason='identity mismatch'
                    elif m.get('sha256')!=filehash(path):reason='checksum mismatch'
                    else:
                        self.log('LOAD %s',label)
                        # Load only checkpoints created under this exact run identity.
                        obj=joblib.load(path); self.record_task_event(dict(expected,status='loaded'))
                        return obj
            except Exception as exc:
                reason='metadata/load failure: '+type(exc).__name__+': '+str(exc)
            quarantine=self.cache/'quarantine';quarantine.mkdir(exist_ok=True)
            stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            moved=[]
            for bad in (path,meta):
                if bad.exists():
                    target=quarantine/(bad.name+'.'+stamp+'.invalid');os.replace(bad,target);moved.append(str(target.name))
            self.log('CACHE REBUILD %s | reason=%s | quarantined=%s',label,reason,moved)
        # Metadata is a commit marker: an interrupted half-write is never loaded.
        self.log('START %s',label)
        started=time.perf_counter(); temp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        try:
            obj=fn()
            self.log('SERIALIZE START %s',label)
            joblib.dump(obj,temp,compress=3)
            os.replace(temp,path)
            atomic_json(meta,dict(expected,sha256=filehash(path),seconds=time.perf_counter()-started,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            self.log('SAVED %s | %.1f s',label,time.perf_counter()-started)
            self.record_task_event(dict(expected,status='computed'))
            return obj
        except BaseException:
            self.log('FAILED/INTERRUPTED %s\n%s',label,traceback.format_exc())
            if temp.exists(): temp.unlink()
            raise

    def table(self,name,df):
        if not isinstance(df,pd.DataFrame): df=pd.DataFrame(df)
        p=self.root/'results'/(name+'.csv');p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
        df.to_csv(tmp,index=False,float_format='%.12g');os.replace(tmp,p)
        return df

    def snapshot(self,complete=False):
        # Print only: writing to execution.log while checksumming it invalidates
        # the archive. This heartbeat deliberately never uses self.log().
        stop=threading.Event();started=time.monotonic()
        def pulse():
            while not stop.wait(self.cfg.get('heartbeat_seconds',60)):
                print(f'{dt.datetime.now(dt.timezone.utc).isoformat()} SNAPSHOT RUNNING | '
                      f'{time.monotonic()-started:.1f} s',flush=True)
        thread=threading.Thread(target=pulse,daemon=True);thread.start()
        print('SNAPSHOT START: inventory, hashing, compression, verification',flush=True)
        try:return self._snapshot_impl(complete)
        finally:stop.set();thread.join(timeout=2)

    def _snapshot_impl(self,complete=False):
        atomic_json(self.root/'manifests'/'task_events.json',self.task_events)
        atomic_json(self.root/'manifests'/'phase_status.json',self.phase_status)
        status={'complete':bool(complete),'run_key':self.key,'profile':self.cfg['profile'],
                'evaluation_status':json.loads((self.root/'manifests'/'run_identity.json').read_text())['evaluation_status'],
                'phases':self.phase_status}
        atomic_json(self.root/'RUN_STATUS.json',status)
        publication_dirs={'source','results','figures_pdf','figures_png','plot_data','tables','manifests'}
        top_files={'README_RESULTS.md','FIGURE_TABLE_CAPTIONS.md','RUN_STATUS.json'}
        def included(p):
            rel=p.relative_to(self.root)
            if not p.is_file() or p.name=='output_checksums.csv' or p.name.endswith('.tmp') or p.name in {'.DS_Store'}:return False
            if not complete:return True
            return (rel.parts[0] in publication_dirs or str(rel) in top_files)
        payloads=[p for p in sorted(self.root.rglob('*')) if included(p)]
        records=[]
        for i,p in enumerate(payloads):
            if i%100==0:print(f'SNAPSHOT HASH {i+1}/{len(payloads)} | {p.name}',flush=True)
            records.append({'path':str(p.relative_to(self.root)),'bytes':p.stat().st_size,'sha256':filehash(p)})
        pd.DataFrame(records).to_csv(self.root/'manifests'/'output_checksums.csv',index=False)
        suffix='Outputs' if complete else 'Partial'
        dest=self.root.parent/f'Environment_Systems_and_Decisions_Failure_Aware_Audit_v4_{suffix}.zip'
        tmp=dest.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for i,p in enumerate(payloads+[self.root/'manifests'/'output_checksums.csv']):
                if i%100==0:print(f'SNAPSHOT ZIP {i+1}/{len(payloads)+1} | {p.name}',flush=True)
                z.write(p,str(Path(self.root.name)/p.relative_to(self.root)))
        os.replace(tmp,dest)
        # Validate the delivered archive, not merely the source directory. The
        # checksum manifest intentionally excludes itself to avoid recursion.
        expected={str(Path(self.root.name)/r['path']):r for r in records}
        with zipfile.ZipFile(dest) as z:
            names=set(z.namelist())
            required=set(expected)|{str(Path(self.root.name)/'manifests'/'output_checksums.csv')}
            missing=required-names
            if missing:raise RuntimeError('ZIP validation failed; missing entries: '+str(sorted(missing)[:10]))
            for i,(name,row) in enumerate(expected.items()):
                if i%100==0:print(f'SNAPSHOT VERIFY {i+1}/{len(expected)}',flush=True)
                h=hashlib.sha256()
                with z.open(name) as stream:
                    for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
                if h.hexdigest()!=row['sha256']:
                    raise RuntimeError('ZIP checksum mismatch: '+name)
        print(f'{suffix} ZIP: {dest}', flush=True) # Do not mutate the just-hashed execution log.
        return dest


def scored(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);mask=np.isfinite(y)
    if not np.all(np.isfinite(p[mask])): raise ValueError('Nonfinite predictions on observed targets; no silent denominator reduction.')
    return y[mask],p[mask],mask


def metrics(y,p):
    y,p,_=scored(y,p);e=y-p
    den=np.sum((y-y.mean())**2)
    return {'n':len(y),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(abs(e))),
            'R2':float(1-np.sum(e*e)/den) if den>0 else np.nan,'AE_q95':float(np.quantile(abs(e),.95))}


def block_sums(values,valid,block):
    a=np.asarray(values); a=np.where(valid,a,0)
    starts=np.arange(0,len(a),block)
    return np.add.reduceat(a,starts),np.add.reduceat(np.asarray(valid,dtype=int),starts)


def paired_block_summary(y,predictions,clean,block,B,rng):
    """Two-level resampling: corruption replicates and nonoverlapping clock-time blocks.
    Estimand is sqrt(mean squared error over replicas and observed timestamps), not
    a sample of independent datasets. All models share the same bootstrap draws.
    predictions: (replicate, time, model); clean: (time, model).
    """
    y=np.asarray(y);P=np.asarray(predictions);C=np.asarray(clean)
    valid=np.isfinite(y);counts=block_sums(np.zeros(len(y)),valid,block)[1]
    sums=np.array([[block_sums((p[:,j]-y)**2,valid,block)[0] for j in range(P.shape[2])] for p in P])
    cs=np.array([block_sums((C[:,j]-y)**2,valid,block)[0] for j in range(C.shape[1])])
    point=np.sqrt(sums.sum(axis=2).mean(axis=0)/counts.sum());cpoint=np.sqrt(cs.sum(axis=1)/counts.sum())
    boot=np.empty((B,P.shape[2]));rel=np.empty_like(boot)
    for b in range(B):
        ri=rng.integers(0,len(P),len(P));bi=rng.integers(0,len(counts),len(counts));den=counts[bi].sum()
        if den==0: boot[b]=np.nan;rel[b]=np.nan;continue
        boot[b]=np.sqrt(sums[ri][:,:,bi].sum(axis=2).mean(axis=0)/den)
        cr=np.sqrt(cs[:,bi].sum(axis=1)/den)
        rel[b]=boot[b]/np.maximum(cr,1e-12)-1
    return point,cpoint,boot,rel,sums,counts


def scalar_block_ci(values,block,B,rng):
    a=np.asarray(values,dtype=float);v=np.isfinite(a);ss,nn=block_sums(a,v,block)
    draws=[]
    for _ in range(B):
        i=rng.integers(0,len(ss),len(ss));d=nn[i].sum()
        draws.append(ss[i].sum()/d if d else np.nan)
    return float(np.nanmean(a)),*np.nanquantile(draws,[.025,.975])


def ci(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return tuple(np.quantile(x,[.025,.975])) if len(x) else (np.nan,np.nan)


def holm(p):
    p=np.asarray(p,dtype=float);out=np.full(len(p),np.nan);valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid])];m=len(order)
    if m:out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(m-np.arange(m))))
    return out


def paired_sign_test(block_differences,B,rng):
    d=np.asarray(block_differences,dtype=float);d=d[np.isfinite(d)]
    if not len(d):return np.nan
    observed=abs(d.sum());null=np.abs((rng.choice([-1.,1.],size=(B,len(d)))*d).sum(axis=1))
    return float((1+(null>=observed-1e-12).sum())/(B+1))
