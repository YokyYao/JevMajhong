"""Shared single-factor recipe, saved-score objective and actual launch checks."""
from collections import defaultdict
import math
from pathlib import Path
import tomllib

from tools.curriculum_basic_repair import MASS,profile as basic_profile,grade
from tools.curriculum_calls_repair import prediction_index
from tools.curriculum_parent import ParentNotReady,rows,require_final_integrity
from tools.curriculum_transfer import ROOT,read,sha

METHOD='max_discard_class_plus_conditional_v1'
COMPONENTS=('objective','standard_action_ce','riichi_max_class_ce','riichi_group_class_ce','conditional_discard_ce','plain_action_ce')


def require_recipe(config,control):
    old=control['curriculum']
    allowed={'run_id','stage','plan_root','out','candidate_rule','changes_vs_v1',
        'riichi_loss_method','control_registration','launch_requires_completed_failed_source_quality'}
    if set(config)-allowed!=set(old)-allowed:raise ValueError('Undeclared recipe fields added or removed')
    for key in set(old)-allowed:
        a,b=config[key],old[key]
        if key in ('reference_checkpoint','reference_evaluation_cache','data_root'):a,b=str((ROOT/a).resolve()),str((ROOT/b).resolve())
        if a!=b:raise ValueError('Single-factor control changed '+key)
    if (config.get('riichi_loss_method')!=METHOD or config.get('launch_requires_completed_failed_source_quality') is not True
        or config.get('stage')!='stage1-riichi-class-control-v1'
        or config.get('candidate_rule')!='registered_final_full_four_epochs_riichi_class_control_v1'
        or config.get('out')==old['out'] or config.get('plan_root')==old['plan_root']
        or control['registered_final_step']!=608 or control['train_records']!=14285
        or old['epochs']!=4 or old['accumulation']!=94 or old['seed']!=2026112002):
        raise ValueError('Explicit fixed608 single-factor fresh control required')


def require_registered(reg):
    c=reg['curriculum'];path=(ROOT/c['control_registration']).resolve();control=read(path)
    if reg['inputs'].get(str(path))!=sha(path):raise ValueError('Original control registration must be frozen')
    require_recipe(c,control)
    if (reg.get('status')!='stage1_riichi_class_control_registered_not_started'
        or reg.get('registered_final_step')!=608 or reg['stage1_loss_weights']!=control['stage1_loss_weights']
        or reg['stage1_loss_mass']!=control['stage1_loss_mass'] or reg['source']!=control['source']
        or reg['source_sha256']!=control['source_sha256']):raise ValueError('Control identity/loss sources changed')
    args=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    old_args=tomllib.loads(Path(control['training_config']['trainer_config']).read_text())
    if {k:v for k,v in args.items() if k!='out'}!={k:v for k,v in old_args.items() if k!='out'}:
        raise ValueError('The control must keep all original trainer arguments except its new output')
    if reg['training_config'].get('loss_method')!=METHOD:raise ValueError('Actual training config lacks registered class loss')
    return control


def cached_parts(question,keys,probabilities,ce):
    source=question['src'].removeprefix('curriculum1:')
    if source not in MASS or question.get('target') is not None:raise ValueError('Six-source hard target required')
    standard=ce(question,keys,probabilities)
    if source.endswith(':plain'):
        if any(v['type']=='reach' for v in question['criteria'].values()):raise ValueError('Plain row offers reach')
        return dict(objective=standard,standard_action_ce=standard,riichi_max_class_ce=0.,riichi_group_class_ce=0.,conditional_discard_ce=0.,plain_action_ce=standard)
    values=dict(zip(keys,probabilities));reach=[k for k in keys if question['criteria'][k]['type']=='reach']
    discards=[k for k in keys if question['criteria'][k]['type']=='dahai']
    if len(reach)!=1 or len(reach)+len(discards)!=len(keys) or not discards:raise ValueError('One reach and ordinary alternatives required')
    yes=question['label']==reach[0]
    if yes!=source.endswith(':riichi_yes'):raise ValueError('Class source and target differ')
    log=lambda v:math.log(max(v,1e-30))
    r=log(values[reach[0]]);dmax=log(max(values[k] for k in discards));dmass=log(sum(values[k] for k in discards))
    softplus=lambda v:max(v,0)+math.log1p(math.exp(-abs(v)))
    maximum=softplus(dmax-r if yes else r-dmax);group=softplus(dmass-r if yes else r-dmass)
    conditional=0. if yes else dmass-log(values[question['label']])
    return dict(objective=maximum+conditional,standard_action_ce=standard,riichi_max_class_ce=maximum,
        riichi_group_class_ce=group,conditional_discard_ce=conditional,plain_action_ce=0.)


def profile(records,predictions,mass,base,ce):
    saved=prediction_index(records,predictions);result=basic_profile(records,predictions,mass,base,ce)
    components=defaultdict(list)
    for row in records:
        q=row['questions']['action'];p=saved[row['_meta']['id']];source=q['src'].removeprefix('curriculum1:')
        components[source].append(cached_parts(q,p['keys'],p['p'],ce))
    if set(components)!=set(MASS) or mass!=MASS:raise ValueError('Actual six-source validation required')
    result['source_loss_components']={s:{k:sum(r[k] for r in values)/len(values) for k in COMPONENTS} for s,values in components.items()}
    result['source_objective']={s:r['objective'] for s,r in result['source_loss_components'].items()}
    result['weighted_class_objective']=sum(mass[s]*r['objective'] for s,r in result['source_loss_components'].items())
    result['weighted_target_ce_is_original_action_ce']=True
    return result


def require_initial_predictions(original,current):
    a={p['id']:p for p in original};b={p['id']:p for p in current}
    if len(a)!=len(original) or len(b)!=len(current) or len(a)!=3266 or set(a)!=set(b):raise ValueError('All3266 distinct initial validation menus required')
    error=0.
    for key,left in a.items():
        right=b[key]
        if left['keys']!=right['keys'] or len(left['p'])!=len(right['p']):raise ValueError('Initial candidate order differs')
        if any(not math.isfinite(v) for v in left['p']+right['p']):raise ValueError('Nonfinite initial prediction')
        error=max(error,max(abs(x-y) for x,y in zip(left['p'],right['p'])))
    if error>=5e-5:raise ValueError('Initial control probabilities differ')
    return dict(menus=len(a),maximum_probability_error=error,tolerance=5e-5)


def require_launch_failure(control,evidence,completion,validation):
    if (control['registered_final_step']!=608 or completion.get('status')!='stage1_fit_and_curves_complete'
        or completion.get('optimizer_steps')!=608 or completion.get('actual_records_seen')!=57140
        or evidence.get('status')!='stage1_final_evidence_audit_passed' or evidence.get('optimizer_steps')!=608
        or evidence.get('real_records_seen')!=57140 or not evidence.get('source_and_data_hashes_unchanged')
        or validation.get('local_step')!=608 or evidence.get('final_summary',{}).get('matches')!=64
        or evidence.get('final_summary',{}).get('seed_groups')!=8):
        raise ParentNotReady('Only an actually completed/audited basic608 final64 can trigger this fresh control')
    reloads=[r for r in evidence.get('saved_reloads',[]) if r['step']==608]
    require_final_integrity(control,completion,evidence,validation,reloads[0]['identity'] if len(reloads)==1 else None)
    from tools.seal_curriculum_basic_repair import require_source_quality
    # The sealer's routine intentionally rejects failed quality. Here a fresh
    # experiment requires that very failure; it does not admit it as a parent.
    calculated=grade(validation)
    if (calculated!=evidence.get('basic_repair_source_quality')
        or calculated['original_aggregate_gate']!=evidence.get('stage1_advancement')):
        raise ParentNotReady('Stored/actual final source diagnostics disagree')
    if calculated['all_source_quality_passed']:
        require_source_quality(control,evidence,validation)
        raise ParentNotReady('Current basic final passes; this conditional control must not run')
    # Validate all source/count relationships even on a failed-quality final.
    by=validation['riichi_by_source'];counts=validation['source_stratum_counts']
    if (control['curriculum'].get('source_specific_parent_quality_required') is not True
        or evidence.get('source_specific_parent_seal_required') is not True
        or set(counts)!=set(MASS) or set(by)!={'legacy','teacher'}
        or any(type(n) is not int or n<=0 for n in counts.values())):raise ParentNotReady('Full actual source evidence required')
    for s,c in by.items():
        if (any(type(c.get(k,0)) is not int or c.get(k,0)<0 for k in ('tp','fn','tn','fp'))
            or c.get('tp',0)+c.get('fn',0)!=counts[s+':riichi_yes']
            or c.get('tn',0)+c.get('fp',0)!=counts[s+':riichi_no']):raise ParentNotReady('Actual failed source counts differ')
    if (any(validation['riichi_type_counts'].get(k,0)!=sum(c.get(k,0) for c in by.values()) for k in ('tp','fn','tn','fp'))
        or any(validation['stratum_counts'][k]!=sum(counts[s+':'+k] for s in by) for k in ('plain','riichi_yes','riichi_no'))):
        raise ParentNotReady('Failed source and pooled populations disagree')
    return dict(status='fresh_control_triggered_by_actual_completed_source_quality_failure',original_source_quality=calculated,
        failed_parent_admitted=False,student_initialization='fresh_fixed_base',prior_final_step=608)


def require_launch_receipt(reg,registered,receipt_path):
    control=require_registered(reg);root=Path(registered).resolve().parent.parent
    if Path(receipt_path).resolve()!=root/'launch-authorization-private.json':raise ParentNotReady('Use this registered run\'s actual conditional launch receipt')
    receipt=read(receipt_path)
    if (receipt.get('status')!='actual_failed_basic_final_authorizes_fresh_control_v1'
        or receipt.get('registered_sha256')!=sha(registered)):
        raise ParentNotReady('Actual fixed-registration launch evidence required')
    for p,h in receipt['bound_files'].items():
        if sha(p)!=h:raise ParentNotReady('Launch input changed: '+p)
    old_root=(ROOT/control['curriculum']['out']).parent;fit=ROOT/control['curriculum']['out']
    evidence=old_root/'final-evidence-private.json';completion=fit/'completion.json';validation=fit/'metrics/validation_metrics.jsonl'
    for p in (evidence,completion,validation):
        if receipt['bound_files'].get(str(p.resolve()))!=sha(p):raise ParentNotReady('Original full final evidence is not bound')
    trigger=require_launch_failure(control,read(evidence),read(completion),rows(validation)[-1])
    if receipt.get('quality_trigger')!=trigger:raise ParentNotReady('Conditional trigger changed')
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    for role in ('fit','audit'):
        path=old_root/(role+'-terminal-scheduler-private.json');archived=read(path)
        if receipt['bound_files'].get(str(path.resolve()))!=sha(path):raise ParentNotReady('Actual original scheduler archive must be bound')
        scheduler_exit(archived['exit'],archived['raw'],old_root/(role+'.slurm'))
    ci_path=root/'public-source-ci-private.json';ci=read(ci_path)
    if (receipt['bound_files'].get(str(ci_path.resolve()))!=sha(ci_path) or ci.get('conclusion')!='success'
        or ci.get('core')!='success' or ci.get('desktop')!='success'):
        raise ParentNotReady('Successful public source CI before launch required')
    for name,digest in ci['published_source_files'].items():
        if sha(ROOT/name)!=digest:raise ParentNotReady('Published checked source changed')
    return trigger
