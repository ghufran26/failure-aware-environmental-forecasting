"""RQ6: repeatable runtime/memory and explicit energy provenance; no manual watts."""
from __future__ import annotations
import gc,os,platform,threading,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from primna_esd_runtime import seed,atomic_json,digest
from primna_esd_data import DATASETS
from primna_esd_models import ML,factory


def rapl_packages():
    found={}
    root=Path('/sys/class/powercap')
    if not root.exists():return []
    for p in root.rglob('energy_uj'):
        try:
            name=(p.parent/'name').read_text().strip()
            if not name.startswith('package-'):continue
            value=float(p.read_text());maximum=float((p.parent/'max_energy_range_uj').read_text())
            found[str(p.resolve())]={'path':p,'max':maximum,'name':name}
        except (OSError,ValueError):continue
    return list(found.values())


class Probe:
    def __init__(self,ctx,label):
        self.ctx=ctx;self.label=label;self.process=psutil.Process();self.start_rss=self.process.memory_info().rss
        self.peak=self.start_rss;self.packages=rapl_packages();self.joules=0.;self.rapl_error=None;self.previous={}
        self.stop_event=threading.Event();self.tracker=None;self.energy_kind='unavailable';self.estimate=np.nan
    def _sample(self):
        try:self.peak=max(self.peak,self.process.memory_info().rss)
        except psutil.Error:pass
        for p in self.packages:
            try:
                key=str(p['path']);v=float(p['path'].read_text())
                if key in self.previous:self.joules+=((v-self.previous[key])%p['max'])/1e6
                self.previous[key]=v
            except (OSError,ValueError) as exc:self.rapl_error=str(exc)
    def start(self):
        if self.packages:self.energy_kind='counter: CPU packages (host scope, not whole machine)'
        elif self.ctx.cfg.get('try_codecarbon',True):
            try:
                from codecarbon import OfflineEmissionsTracker
                target=self.ctx.root/'logs'/'energy';target.mkdir(exist_ok=True)
                self.tracker=OfflineEmissionsTracker(country_iso_code='USA',project_name=self.label,
                    output_dir=str(target),output_file=f'{seed(self.label)}.csv',measure_power_secs=1,
                    save_to_file=True,save_to_api=False,log_level='error',tracking_mode='process')
                self.tracker.start();self.energy_kind='CodeCarbon estimated/mixed CPU+RAM; not direct total measurement'
            except Exception as exc:
                self.ctx.log('Energy telemetry unavailable: %s',exc);self.tracker=None
        self._sample()
        def loop():
            while not self.stop_event.wait(.2):self._sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def stop(self):
        self.stop_event.set();self.thread.join();self._sample();e=np.nan
        if self.packages and self.rapl_error is None:e=self.joules/3.6e6
        elif self.tracker is not None:
            try:
                self.tracker.stop();dat=self.tracker.final_emissions_data
                e=float(dat.energy_consumed) # stop() returns emissions, NOT kWh
            except Exception as exc:self.ctx.log('Could not read energy telemetry: %s',exc)
        if not np.isfinite(e) or e<=0:
            e=np.nan;self.energy_kind='unavailable: no valid positive energy readout'
        return {'energy_kWh_total':e,'energy_kind':self.energy_kind,'RSS_peak_MB':self.peak/2**20,
                'RSS_delta_MB':max(0,self.peak-self.start_rss)/2**20,'RAPL_error':self.rapl_error}


def benchmark(ctx,label,fn):
    gc.collect();probe=Probe(ctx,label);probe.start();cost=[];warnings_seen=[];start=time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=ctx.cfg['threads']):
            warnings.simplefilter('always')
            while time.perf_counter()-start<ctx.cfg['benchmark_seconds'] or not cost:
                t=time.perf_counter();result=fn();cost.append(time.perf_counter()-t)
                del result
            warnings_seen=[str(w.category.__name__)+': '+str(w.message) for w in ws]
    finally:measure=probe.stop()
    for w in set(warnings_seen):ctx.log('BENCHMARK WARNING %s: %s',label,w)
    return dict(measure,seconds_per_operation=float(np.mean(cost)),operations=len(cost),
      tracking_elapsed_seconds=time.perf_counter()-start,energy_kWh_per_operation=measure['energy_kWh_total']/len(cost),
      warnings=' | '.join(sorted(set(warnings_seen))))


def compute_suite(ctx,bundles,fitted):
    rows=[];machine={'platform':platform.platform(),'processor':platform.processor(),'logical_CPUs':os.cpu_count(),
      'threads_used':ctx.cfg['threads'],'physical_memory_bytes':psutil.virtual_memory().total,
      'RAPL_packages':[str(x['path']) for x in rapl_packages()],
      'energy_scope_note':'CodeCarbon output is estimated/mixed unless proven otherwise; RAPL covers host CPU package domains only. No carbon-emissions or geographic claims.',
      'comparison_scope':'one execution host; fixed model seed and parameters; randomized benchmark order; warmups excluded'}
    try:machine['host_boot_fingerprint']=digest(Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except OSError:machine['host_boot_fingerprint']='unavailable'
    atomic_json(ctx.root/'manifests'/'hardware_and_energy_provenance.json',machine)
    for ds in DATASETS:
        b=bundles[ds];fit=fitted[ds];ix=b['splits']['development'];ix=ix[b['y'].loc[ix].notna()]
        X=b['X'].loc[ix];y=b['y'].loc[ix];xt=b['X'].loc[b['splits']['test']]
        # Warm up each algorithm outside timed measurements once per session.
        for m in ML:
            with ctx.heartbeat('RQ6 untimed warmup '+ds+' '+m),threadpool_limits(limits=ctx.cfg['threads']):
                warm=factory(m,fit['params'][m],seed(ctx.cfg['seed'],ds,m,'cost'),ctx.cfg['threads']);warm.fit(X,y);warm.predict(xt.iloc[:100]);del warm
        for rep in range(ctx.cfg['compute_reps']):
            order=np.random.default_rng(seed(ds,'compute_order',rep)).permutation([(m,s) for m in ML for s in ['Training','Inference / 1000']])
            for m,stage in order:
                label=f'RQ6 | {ds} | {m} | {stage} | repeat {rep+1}/{ctx.cfg["compute_reps"]}'
                def work(m=m,stage=stage,label=label):
                    if stage=='Training':
                        def fn():
                            est=factory(m,fit['params'][m],seed(ctx.cfg['seed'],ds,m,'cost'),ctx.cfg['threads']);est.fit(X,y);return est
                    else:
                        def fn():return fit['models'][m].predict(xt)
                    rr=benchmark(ctx,label,fn)
                    factor=1000/len(xt) if stage!='Training' else 1.
                    rr['seconds_per_operation']*=factor;rr['energy_kWh_per_operation']*=factor
                    return rr
                # Hardware is part of this task identity; cannot reuse cost readings on another host.
                out=ctx.task('RQ6',label,{'data':b['data_key'],'model':m,'stage':stage,'rep':rep,'hardware':machine},work)
                rows.append(dict(dataset=ds,model=m,stage=stage,replicate=rep,**out))
    result={'measurements':pd.DataFrame(rows),'hardware':machine};ctx.table('RQ6_measurements',result['measurements'])
    return result
