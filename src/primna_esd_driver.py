"""Explicit phases, verified caches, artifact checks, and final ZIP packaging."""
from __future__ import annotations
import importlib.metadata
from pathlib import Path
import pandas as pd

from primna_esd_runtime import Context,atomic_json,filehash,digest
from primna_esd_data import acquire
from primna_esd_models import train
from primna_esd_stress import stress_suite,explanation_suite
from primna_esd_drift import drift_suite
from primna_esd_uncertainty import uncertainty_suite
from primna_esd_compute import compute_suite
from primna_esd_figures import all_figures,stress_figures,explanation_figures,drift_figures,uncertainty_figures,compute_figures
from primna_esd_checks import run_tests
from primna_esd_external import external_suite
from primna_esd_multisite import run_multisite,multisite_figures

PHASES={1:'P1',2:'P2',3:'RQ1',4:'RQ2',5:'RQ3',6:'RQ4',7:'RQ5',8:'RQ6',9:'EXTERNAL'}


def initialize(config,source_dir):
    if config['profile'] not in {'paper','pilot'}:raise ValueError('Use paper or explicitly labelled pilot profile.')
    if not config.get('acknowledge_retrospective',False):
        raise ValueError('Acknowledge that the two core historical test periods were previously inspected.')
    if config['profile']=='paper' and not config.get('run_external_confirmation',False):
        raise ValueError('Paper profile requires independent external confirmation; use pilot only for engineering tests.')
    ctx=Context(config,source_dir)
    required=['numpy','pandas','scipy','scikit-learn','matplotlib','joblib','psutil','threadpoolctl']
    lines=[]
    for package in required:
        try:lines.append(package+'=='+importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:pass
    (ctx.root/'source'/'requirements_minimal.txt').write_text('\n'.join(lines)+'\n')
    return ctx


def stage_work(ctx,number):
    if number==1:
        out=acquire(ctx);run_tests(ctx,out);return out
    if number==2:return train(ctx,ctx.results['P1'])
    b=ctx.results['P1'];f=ctx.results['P2']
    if number in [3,4]:return stress_suite(ctx,b,f,number-2)
    if number==5:return explanation_suite(ctx,b,f)
    if number==6:return drift_suite(ctx,b,f)
    if number==7:return uncertainty_suite(ctx,b,f)
    if number==8:return compute_suite(ctx,b,f)
    if number==9:return {'cross_domain':external_suite(ctx),'multisite':run_multisite(ctx)}
    raise KeyError(number)


def _phase_figures(ctx,number,key):
    if number in [3,4]:return stress_figures(ctx,number-2,ctx.results[key],ctx.results['P2'])
    if number==5:return explanation_figures(ctx,ctx.results[key],ctx.results['P2'])
    if number==6:return drift_figures(ctx,ctx.results[key])
    if number==7:return uncertainty_figures(ctx,ctx.results[key],ctx.results['P2'])
    if number==8:return compute_figures(ctx,ctx.results[key])
    if number==9:return multisite_figures(ctx,ctx.results[key]['multisite'])
    return []


def run_phase(ctx,number):
    if number>ctx.cfg['end_phase']:
        ctx.log('Phase %s not requested (END_PHASE=%s)',number,ctx.cfg['end_phase']);return
    if number==10:return finalize(ctx)
    if number<ctx.cfg.get('start_phase',1):ctx.log('Loading prerequisite phase %d; missing compatible tasks will recompute.',number)
    key=PHASES[number];ctx.log('========== PHASE %d: %s ==========',number,key)
    if number in [1,9]:
        ctx.results[key]=stage_work(ctx,number);ctx.phase_status[str(number)]='complete'
    else:
        identities={ds:b['data_key'] for ds,b in ctx.results['P1'].items()}
        if number==8:
            try:host=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            except OSError:host='unknown host'
            identities['host_boot_fingerprint']=digest(host)
        def compute():
            with ctx.substage('phase scientific computation',phase=number,key=key):
                result=stage_work(ctx,number)
            artifacts=[]
            ctx.log('ARTIFACT HASH START | phase=%d | key=%s',number,key)
            for p in (ctx.root/'results').rglob('*'):
                if p.is_file():
                    rel=str(p.relative_to(ctx.root))
                    if p.name.startswith(key+'_') or key+'_' in str(p.relative_to(ctx.root/'results')):
                        artifacts.append({'path':rel,'sha256':filehash(p)})
            ctx.log('ARTIFACT HASH COMPLETE | phase=%d | files=%d',number,len(artifacts))
            return {'result':result,'artifacts':artifacts}
        try:
            payload=ctx.task('PHASE'+str(number),key+' phase result',identities,compute)
            ctx.log('ARTIFACT VALIDATION START | phase=%d | files=%d',number,len(payload['artifacts']))
            invalid=[a for a in payload['artifacts'] if not (ctx.root/a['path']).exists() or filehash(ctx.root/a['path'])!=a['sha256']]
            if invalid:
                ctx.log('ARTIFACT VALIDATION REBUILD | phase=%d | invalid=%s',number,[a['path'] for a in invalid])
                with ctx.substage('phase artifact reconstruction',phase=number,key=key,invalid=len(invalid)):
                    ctx.results[key]=stage_work(ctx,number)
            else:ctx.results[key]=payload['result']
            ctx.log('ARTIFACT VALIDATION COMPLETE | phase=%d | invalid=%d',number,len(invalid))
            ctx.phase_status[str(number)]='complete'
        except BaseException:
            ctx.phase_status[str(number)]='failed_or_interrupted';ctx.failure('phase '+str(number));raise
    atomic_json(ctx.root/'manifests'/'phase_status.json',ctx.phase_status)
    try:
        with ctx.substage('phase figures and tables',phase=number,key=key), ctx.heartbeat('phase figures '+str(number)):
            arts=_phase_figures(ctx,number,key)
        if arts:atomic_json(ctx.root/'manifests'/(key+'_artifacts.json'),arts)
    except BaseException:
        ctx.phase_status[str(number)]='numerics_complete_figure_failed';ctx.failure('phase figures '+str(number));raise
    ctx.log('PHASE %d COMPLETE. Safe restart: Run All; valid tasks will LOAD.',number)
    if ctx.cfg.get('snapshot_each_phase',True):ctx.snapshot(False)
    return ctx.results[key]


def finalize(ctx):
    missing=[PHASES[i] for i in range(1,10) if PHASES[i] not in ctx.results]
    if missing:raise RuntimeError('Cannot finalize; required phase results not loaded: '+', '.join(missing))
    tests=run_tests(ctx,ctx.results['P1']);artifacts=all_figures(ctx)
    if len(artifacts)!=24:raise AssertionError('Expected exactly four distinct publication artifacts per RQ.')
    ext=multisite_figures(ctx,ctx.results['EXTERNAL']['multisite']);artifacts+=ext
    if len(ext)!=4:raise AssertionError('Expected three external-confirmation figures and one table.')
    atomic_json(ctx.root/'manifests'/'publication_artifacts.json',artifacts)
    (ctx.root/'FIGURE_TABLE_CAPTIONS.md').write_text('\n'.join(
      '## '+a['stem']+'\n'+a['title']+'\n\n'+a['caption']+'\n' for a in artifacts))
    try:import fitz
    except ImportError as exc:raise RuntimeError('PyMuPDF is required for final PDF rendering validation.') from exc
    pdf_checks=[]
    for a in artifacts:
        p=ctx.root/a['pdf'];data=ctx.root/a['plot_data']
        assert p.read_bytes().startswith(b'%PDF') and p.stat().st_size>1000
        assert data.exists() and len(pd.read_csv(data))>0
        doc=fitz.open(p);pages=len(doc);fonts=sum(len(pg.get_fonts()) for pg in doc);doc.close()
        if pages<1 or fonts<1:raise AssertionError('PDF renderer/font validation failed: '+str(p))
        pdf_checks.append({'path':a['pdf'],'bytes':p.stat().st_size,'plot_data':a['plot_data'],'pages':pages,'font_records':fonts})
    assert ctx.results['RQ5']['parameters'].query("method == 'Global split conformal'").constant_score.all()
    pars=ctx.results['RQ4']['calibration']
    far_supported=bool(pars[pars.provisional_selected].far_claim_supported.all()) if pars.provisional_selected.any() else False
    ext_summary=ctx.results['EXTERNAL']['multisite']['summary']
    cross_site_supported=bool((ext_summary.query("design == 'LOSO'").fraction_better_than_persistence>=ctx.cfg['cross_site_success_fraction']).any())
    atomic_json(ctx.root/'manifests'/'final_QA.json',{
      'regression_tests_passed':len(tests),'principal_artifacts':pdf_checks,'retrospective_core':True,
      'external_evaluation_status':'independent retrospective replication; not preregistered confirmation',
      'completed_profile':ctx.cfg['profile'],'technical_package_complete':True,
      'evidence_gates':{'rare_false_alarm_claim_supported':far_supported,
                        'cross_site_replication_gate_supported':cross_site_supported},
      'publication_ready_claim':'not determined by software; requires author and peer review',
      'energy_status':sorted(set(ctx.results['RQ6']['measurements'].energy_kind))})
    readme=("# Failure-aware environmental forecasting audit (ESD v4)\n\n"
      "Target journal: Environment Systems and Decisions (Springer Nature).\n\n"
      "All cohorts are reported as retrospective or independent external replication. The software never upgrades a user declaration into confirmatory evidence. All twelve Beijing stations are evaluated, including leave-one-station-out transfer.\n\n"
      "All fitted preprocessing uses development data. Six research questions are evaluated, each with three vector PDF figures and one publication table. External confirmation adds three figures and one table. Numeric plot sources, CSV/LaTeX tables, captions, source, configuration, checksums, and resumable checkpoints are included.\n\n"
      "Detector feasibility requires both observed-policy FAR and the 95th percentile of dependent-null FAR to meet the target. Guarded alarm adaptation validates every candidate refit on already observed pre-update labels. Failed or disabled detectors remain visible. Coverage is empirical rather than an unconditional exchangeability guarantee.\n\n"
      "RQ5 reports an environmental monitoring decision sensitivity analysis using frozen policy thresholds, false-negative/false-positive cost ratios, and an explicit abstention cost. These costs are transparent scenarios, not claimed stakeholder utilities.\n\n"
      "Energy telemetry may be unavailable or estimated; no carbon claim is generated. Passing software QA establishes reproducibility and technical completeness, not journal acceptance or universal environmental validity.\n\n"
      "Checkpoints are matched to source, scientific configuration, package versions, inputs and task parameters. The final dissemination ZIP contains exactly the payloads listed in its checksum manifest; resumable checkpoints remain in partial phase archives.\n")
    (ctx.root/'README_RESULTS.md').write_text(readme)
    ctx.phase_status['10']='complete'
    return ctx.snapshot(True)
