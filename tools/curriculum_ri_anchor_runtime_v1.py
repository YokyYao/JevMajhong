"""Explicit augmented-data continuation, cohort metrics and completed-control gate."""
from collections import Counter
from contextlib import contextmanager
import copy
import dis
import math
from pathlib import Path
import tomllib
from types import FunctionType

from tools.curriculum_transfer import read,sha
from tools import curriculum_progression_parent_v1 as original_parent
from tools import curriculum_progression_parent_fit_v1 as canonical_profile
from tools import curriculum_progression_fit_v1 as original_profile
from tools import curriculum_progression_runtime_v1 as original_policy
from tools import curriculum_ri_parent_anchor_weighted_v1 as weighted
from tools.curriculum_progression_parent_loss_v1 import MASS,METHOD as VALIDATION_METHOD
from tools.curriculum_complete_legal_class_policy_v1 import select

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'runs/curriculum-ri-balanced-replay-fit-v2'
RUN=ROOT/'runs/curriculum-ri-anchor-fit-v1'
REGISTERED=RUN/'fit-plan/registered.json'
DATA=ROOT/'runs/curriculum-ri-augmented-data-v1/data-v1'
CACHE=ROOT/'runs/curriculum-ri-augmented-data-v1/parent-targets-v1'
OPTIMIZER_CONTROL=ROOT/'runs/curriculum-ri-augmented-data-v1/optimizer-control-v2'
DATA_READY=RUN/'fit-plan/data-receipt.json'
ENTRY=ROOT/'cluster/curriculum-ri-anchor-fit-v1.slurm'
POINTS=[0,649,1298,1947,2596]
PARENT_METHOD='audited_calls388_augmented19_train_only_RI_retention_v1'


def verify_files(files):
    if any(sha(path)!=expected for path,expected in files.items()):
        raise ValueError('Frozen source/input changed')


def prerequisites():
    files={}
    for folder in (DATA,CACHE,OPTIMIZER_CONTROL):
        path=folder/'completion-binding-private.json'
        for name,expected in read(path)['files'].items():
            if name in files and files[name]!=expected:raise ValueError('Conflicting prerequisite identity')
            files[name]=expected
        files[str(path)]=sha(path)
    verify_files(files)
    data,cache,control=(read(folder/'summary-private.json') for folder in (DATA,CACHE,OPTIMIZER_CONTROL))
    if (data['unique_rows']!={'train':92570,'validation':24998} or data['total_optimizer_updates']!=2596
            or data['total_physical_exposures']!=189508 or cache['RI_TRAIN_targets']!=4244
            or cache['student_updates']!=0 or control['actual_optimizer_updates']!=2
            or control['unique_TRAIN_admitted']!=92570 or control['actual_physical_records']!=146
            or control['actual_optimizer_states']!=376 or not control['all_optimizer_moments_and_scheduler_reloaded_exactly']
            or not control['all_saved_trainable_tensors_reloaded_exactly'] or control['maximum_saved_probability_error']!=0
            or not control['this_control_must_not_initialize_a_future_fit']):
        raise ValueError('Complete audited data/cache and actual repaired optimizer control required')
    return files


def curriculum_view(old):
    stale={'step0_parameter_cache','step0_validation_reuse_cache','failed_attempt_registration',
           'failed_optimizer_updates_not_resumed','failure_repair','paired_order_method',
           'retry_from_qualified_calls_not_control_final','all_nineteen_sources_each_mixing_block',
           'each_epoch_without_replacement','mixing_block_records'}
    c={k:copy.deepcopy(v) for k,v in old.items() if k not in stale}
    c.update(run_id='jev-curriculum-ri-anchor-v1',stage='stage3-ri-parent-anchor-v1',
        out=str(RUN.relative_to(ROOT)/'stage3-fit'),plan_root=str(RUN.relative_to(ROOT)/'fit-plan'),
        data_root=str(DATA.relative_to(ROOT)),data_ready=str(DATA_READY),loss_method=weighted.METHOD,
        validation_and_game_steps=POINTS,expected_record_exposures=189508,
        checkpoint_every_steps=59,candidate_rule='registered_final_full2596_augmented_RI_anchor_v1',
        final_optimizer_steps=2596,validation_every_steps=649,game_evaluation_every_steps=649,
        RI_anchor_strength=4.,retention_quality_cohort='original23688',full_zero_validation_inference=True,
        all_originals_each_epoch=True,each_update_four_RI_sources_minimum=1,baseline_nonRI_order_block_records=511)
    return c


def trainer_view(old):
    return dict(old,data=str(DATA/'train-union-private.jsonl'),eval_data=str(DATA/'validation-union-private.jsonl'),
        out=str(RUN/'stage3-fit/training-run'),max_steps=2596,checkpoint_every_steps=59,eval_every_steps=649)


def config_view(old,c,trainer_hash,receipt_hash):
    cfg=dict(old,model_id=c['run_id'],out=c['out'],trainer_config=str(RUN/'fit-plan/trainer.toml'),
        trainer_config_sha256=trainer_hash,data_receipt=str(DATA_READY),data_receipt_sha256=receipt_hash,
        registered_final_step=2596,loss_method=weighted.METHOD)
    cfg['parent_initialization']=dict(old['parent_initialization'],method=PARENT_METHOD,child_data_ready=str(DATA_READY))
    return cfg


def check_recipe(reg,old,args,old_args):
    if (reg['registered_final_step']!=2596 or reg['train_records']!=92570 or reg['validation_records']!=24998
            or reg['physical_train_slots_per_epoch']!=94754 or reg['expected_record_exposures']!=189508
            or reg['course_loss_mass']!=MASS or reg['validation_loss_method']!=VALIDATION_METHOD
            or reg['curriculum']!=curriculum_view(old['curriculum']) or args!=trainer_view(old_args)
            or reg['parent_model']!=old['parent_model'] or reg['source']!=old['source']
            or reg['source_sha256']!=old['source_sha256'] or reg['integration_control_updates_excluded'] is not True):
        raise ValueError('Exact augmented2596 recipe, parent, source and independent optimizer required')
    expected=config_view(old['training_config'],reg['curriculum'],
        reg['training_config']['trainer_config_sha256'],reg['training_config']['data_receipt_sha256'])
    if reg['training_config']!=expected:raise ValueError('Unregistered initialization or training configuration change')
    expected_plans=[str(DATA/f'epoch-{n:02d}-private.json') for n in (1,2)]
    if reg['replay_epoch_plans']!=expected_plans:raise ValueError('Both already frozen complete coverage plans required')


def completed_current(done,evidence,execution,captured):
    proofs=captured['scheduler_proofs']
    if (done['status']!='balanced_replay2472_fit_and_real_curves_complete' or done['optimizer_steps']!=2472
            or done['actual_records_seen']!=180456 or evidence['status']!='balanced_replay2472_final_evidence_audit_passed'
            or evidence['optimizer_steps']!=2472 or evidence['saved_final_reloaded_menus']!=23688
            or execution['status']!='actual_balanced_replay2472_full_evidence_independently_audited'
            or captured['status']!='balanced_replay_actual_terminals_preserved_not_promoted'
            or any(proofs[k]['state']!='COMPLETED' or proofs[k]['successful'] is not True
                or proofs[k]['exit_code']!='0:0' for k in ('fit','audit'))
            or str(execution['actual_fit_job'])!=proofs['fit']['job_id']
            or type(evidence['stage3_advancement']['passed']) is not bool
            or captured['strategy_grade']!=evidence['stage3_advancement']):
        raise ValueError('Complete successful execution/audit and unchanged final strategy grade required')
    return copy.deepcopy(evidence['stage3_advancement'])


def failed_current(done,evidence,execution,captured):
    grade=completed_current(done,evidence,execution,captured)
    if grade['passed'] is not False:raise ValueError('Qualified current final must follow its existing later course')
    return grade


def require_registration(reg):
    if reg['status']!='ri_anchor2596_registered_not_started':raise ValueError('Actual authorized registration required')
    old=read(CONTROL/'fit-plan/registered.json')
    check_recipe(reg,old,tomllib.loads(Path(reg['training_config']['trainer_config']).read_text()),
        tomllib.loads(Path(old['training_config']['trainer_config']).read_text()))
    verify_files(reg['inputs'])
    if sha(DATA/'completion-binding-private.json')!=reg['data_binding_sha256']:
        raise ValueError('Frozen audited union differs')
    counts=read(DATA/'summary-private.json')['source_counts']['train']
    expected={s:92570*MASS[s]/counts[s] for s in MASS}
    if reg['course_loss_weights']!=expected:raise ValueError('Declared unique-row source weights differ')


def require_launch(registered):
    gate=read(RUN/'launch-gate-private.json');verify_files(gate['files'])
    if gate['registered_sha256']!=sha(registered):raise ValueError('Actual gate registration differs')
    grade=failed_current(*(read(gate['evidence_paths'][k]) for k in ('completion','audit','execution','capture')))
    if grade!=gate['unaltered_failed_control_grade']:raise ValueError('Original final failure was changed')


def verify_parent(cfg):
    old=read(CONTROL/'fit-plan/registered.json')
    # Preserve the complete original parent verifier; it validates its own
    # original corpus. This explicit additional bridge admits only the audited
    # augmented child paths rather than asking that verifier to accept them.
    parent=original_parent.verify_parent(old['training_config'])
    if (cfg['parent_initialization']!=dict(old['training_config']['parent_initialization'],
            method=PARENT_METHOD,child_data_ready=str(DATA_READY))
            or cfg['base']!=old['training_config']['base'] or cfg['base_revision']!=old['training_config']['base_revision']):
        raise ValueError('Explicit same-qualified-parent augmented-data bridge required')
    args=tomllib.loads(Path(cfg['trainer_config']).read_text())
    expected=trainer_view(tomllib.loads(Path(old['training_config']['trainer_config']).read_text()))
    if args!=expected or Path(args['init_from']).resolve()!=Path(parent['model']['directory']).resolve():
        raise ValueError('Exact augmented data, actual initializer and fresh optimizer required')
    proof=prerequisites();receipt=read(DATA_READY)
    if sha(DATA_READY)!=cfg['data_receipt_sha256'] or receipt['status']!='balanced_raw_data_ready':
        raise ValueError('Complete augmented input receipt required')
    for split,key in (('train','data'),('validation','eval_data')):
        spec=receipt['files'][split]
        if Path(spec['path']).resolve()!=Path(args[key]).resolve() or sha(spec['path'])!=spec['sha256']:
            raise ValueError('Augmented child source differs')
    sources=RUN/'fit-plan/course-sources-private.json';source_files=read(sources)['files'];verify_files(source_files)
    parent['receipt'].update(proof);parent['receipt'].update(source_files)
    parent['receipt'][str(sources)]=sha(sources);parent['receipt'][str(DATA_READY)]=sha(DATA_READY)
    if REGISTERED.exists():
        registered=read(REGISTERED);verify_files(registered['inputs']);parent['receipt'].update(registered['inputs'])
        parent['receipt'][str(REGISTERED)]=sha(REGISTERED)
    return parent


@contextmanager
def parent_loader(initialization):
    previous=initialization.verified_parent;initialization.verified_parent=verify_parent
    try:yield
    finally:initialization.verified_parent=previous


policy_contract=original_policy.policy_contract


def prediction_parity(parent_rows,parent_predictions,child_rows,child_predictions):
    if len(child_rows)!=24998 or len(child_predictions)!=24998:raise ValueError('Full augmented child VAL required')
    original_ids=set(read(DATA/'cohort-ids-private.json')['validation']['original'])
    original_rows=[r for r in child_rows if r['_meta']['id'] in original_ids]
    original_predictions=[p for p in child_predictions if p['id'] in original_ids]
    proof=original_policy.prediction_parity(parent_rows,parent_predictions,original_rows,original_predictions)
    return dict(proof,child_menus=24998,original_cohort_menus=23688)


def require_step0(identity,reg,proof):
    parity=proof['all_parent_prediction_parity'];tensors=proof['exact_tensors']
    if (not proof['checked_before_first_optimizer_update'] or proof['step0_bundle_identity']!=identity['identity_sha256']
            or proof['parent_bundle_identity']!=reg['curriculum']['parent_identity']
            or not tensors['parent_adapter_exact'] or not tensors['parent_head_exact']
            or tensors['adapter_tensors']!=372 or tensors['head_tensors']!=4
            or parity['parent_menus']!=12479 or parity['child_menus']!=24998
            or parity['original_cohort_menus']!=23688 or not parity['choices_equal']
            or not 0<=parity['max_probability_error']<5e-5):
        raise ValueError('Actual complete parent parity and exact initialization required')


def augmented_profile_function():
    function=original_profile.profile
    references=sum(i.opname=='LOAD_CONST' and i.argval==12479 for i in dis.get_instructions(function))
    if references!=1:raise ValueError('Review the original profile cardinality guard before extending it')
    code=function.__code__.replace(co_consts=tuple(13789 if v==12479 else v for v in function.__code__.co_consts))
    inner=FunctionType(code,dict(function.__globals__),function.__name__,function.__defaults__,function.__closure__)
    outer=canonical_profile.profile
    return FunctionType(outer.__code__,dict(outer.__globals__,previous_profile=inner),
        outer.__name__,outer.__defaults__,outer.__closure__)


def profile(records,predictions,base,ce):
    if len(records)!=24998 or len(predictions)!=24998:raise ValueError('All24998 actual VAL menus required')
    cohorts=read(DATA/'cohort-ids-private.json')['validation']
    original_ids=set(cohorts['original']);new_ids=set(cohorts['supplement'])
    byid={r['_meta']['id']:r for r in records};saved={p['id']:p for p in predictions}
    if (len(original_ids)!=23688 or len(new_ids)!=1310 or original_ids&new_ids
            or set(byid)!=original_ids|new_ids or set(saved)!=set(byid)
            or len(byid)!=len(records) or len(saved)!=len(predictions)):
        raise ValueError('Complete separate original and new VAL cohorts required')
    result=augmented_profile_function()(records,predictions,base,ce)
    old_rows=[r for r in records if r['_meta']['id'] in original_ids]
    old_predictions=[p for p in predictions if p['id'] in original_ids]
    result['original_validation']=canonical_profile.profile(old_rows,old_predictions,{},ce)
    new_counts=Counter();confusion=Counter();correct=0;loss=0.
    for row in records:
        if row['_meta']['id'] not in new_ids:continue
        q=row['questions']['action'];p=saved[row['_meta']['id']];keys=p['keys']
        kinds=[q['criteria'][k]['type'] for k in keys];chosen,_=select(kinds,p['p'])
        correct+=keys[chosen]==q['label'];loss+=ce(q,keys,p['p']);new_counts[q['src']]+=1
        if 'reach' in kinds:
            truth=q['criteria'][q['label']]['type']=='reach';actual=kinds[chosen]=='reach'
            confusion['tp' if truth and actual else 'fn' if truth else 'fp' if actual else 'tn']+=1
    result['new_teacher_validation']=dict(records=1310,stratum_counts=dict(new_counts),
        natural_target_ce=loss/1310,hard_agreement=correct/1310,
        riichi_type_counts={k:confusion[k] for k in ('tp','fn','tn','fp')},
        consumed_validation_not_strength_proof=True)
    result.update(retention_quality_cohort='original23688',canonical_validation_has_TRAIN_anchor=False,
        full_and_original_and_new_cohorts_reported=True)
    return result
