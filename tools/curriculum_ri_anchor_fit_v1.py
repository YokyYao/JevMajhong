"""Finite, completed-control-gated augmented RI training lifecycle."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_ri_anchor_runtime_v1 as contract
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,fields,terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

RUN=contract.RUN;REGISTERED=contract.REGISTERED;ENTRY=contract.ENTRY
CONFIG=ROOT/'configs/training/curriculum-ri-anchor-fit-v1.json'
PREPARED=RUN/'prepared-v1/completion-binding-private.json'
PREVIEW=RUN/'fit-plan/prospective-private.json'
QUEUE=RUN/'fit-queue-private.json'
BRIDGE=RUN/'activation-binding-private.json'
PUBLIC=(
    'tools/curriculum_ri_anchor_runtime_v1.py','tools/curriculum_ri_anchor_fit_v1.py',
    'tools/train_curriculum_ri_anchor_v1.py','tools/evaluate_curriculum_ri_anchor_v1.py',
    'tools/audit_curriculum_ri_anchor_v1.py','tests/core/test_curriculum_ri_anchor_runtime_v1.py',
    'cluster/curriculum-ri-anchor-fit-v1.slurm','configs/training/curriculum-ri-anchor-fit-v1.json',
    'docs/experiments/plans/curriculum-ri-anchor-fit-20261125.md')


def merge(files,extra):
    for name,expected in extra.items():
        if name in files and files[name]!=expected:raise ValueError('Conflicting frozen identity')
        files[name]=expected


def verify_binding(path):
    result=read(path);contract.verify_files(result['files']);return result


def prepare():
    resource()
    if PREPARED.exists() or PREVIEW.exists() or REGISTERED.exists():raise ValueError('Never overwrite preparation/registration')
    settings=read(CONFIG)
    if (settings['full_steps']!=2596 or settings['curve_steps']!=contract.POINTS
            or settings['unique_train']!=92570 or settings['validation']!=24998
            or settings['batch']!=1 or settings['accumulation']!=73
            or settings['anchor_strength']!=4. or settings['checkpoint_every_steps']!=59
            or settings['automatic_release'] is not False or settings['reserved_final_payloads_opened'] is not False):
        raise ValueError('Explicit frozen full-course numerical and publication recipe required')
    old_path=contract.CONTROL/'fit-plan/registered.json';old=read(old_path)
    files=dict(old['inputs']);merge(files,contract.prerequisites());files[str(old_path)]=sha(old_path)
    for name in PUBLIC:files[str(ROOT/name)]=sha(ROOT/name)
    contract.verify_files(files)
    plan=RUN/'fit-plan';plan.mkdir(parents=True)
    data=read(contract.DATA/'summary-private.json')
    receipt=dict(status='balanced_raw_data_ready',requires_masked_soft_target_loss=True,
        files={split:dict(path=str(contract.DATA/f'{split}-union-private.jsonl'),
            sha256=sha(contract.DATA/f'{split}-union-private.jsonl'),records=count)
            for split,count in data['unique_rows'].items()},
        fixed_validation_reference=dict(path=str(Path(tomllib.loads(Path(old['training_config']['trainer_config']).read_text())['eval_data'])),
            sha256=sha(tomllib.loads(Path(old['training_config']['trainer_config']).read_text())['eval_data'])),
        cohort_membership=str(contract.DATA/'cohort-ids-private.json'),
        full_raw_inputs=True,unchanged_original_and_audited_supplement=True,raw_private=True)
    write(contract.DATA_READY,receipt)
    old_args=tomllib.loads(Path(old['training_config']['trainer_config']).read_text())
    args=contract.trainer_view(old_args);trainer=plan/'trainer.toml'
    trainer.write_text(''.join(f'{key} = {json.dumps(value)}\n' for key,value in args.items()))
    c=contract.curriculum_view(old['curriculum'])
    cfg=contract.config_view(old['training_config'],c,sha(trainer),sha(contract.DATA_READY))
    cfg_path=plan/'training-config.json';write(cfg_path,cfg)
    for path in (contract.DATA_READY,trainer,cfg_path):files[str(path)]=sha(path)
    write(plan/'course-sources-private.json',{'files':files})
    files[str(plan/'course-sources-private.json')]=sha(plan/'course-sources-private.json')
    counts=data['source_counts']['train']
    reg=dict(status='prospective_ri_anchor2596_not_authorized',curriculum=c,source=old['source'],
        source_sha256=old['source_sha256'],parent_model=old['parent_model'],training_config=cfg,
        registered_final_step=2596,train_records=92570,validation_records=24998,
        physical_train_slots_per_epoch=94754,expected_record_exposures=189508,
        course_loss_mass=contract.MASS,course_loss_weights={s:92570*contract.MASS[s]/counts[s] for s in contract.MASS},
        validation_loss_method=contract.VALIDATION_METHOD,
        replay_epoch_plans=[str(contract.DATA/f'epoch-{n:02d}-private.json') for n in (1,2)],
        data_binding_sha256=sha(contract.DATA/'completion-binding-private.json'),
        integration_control_updates_excluded=True,inputs=files)
    contract.check_recipe(reg,old,args,old_args)
    sys.path[:0]=[old['source'],old['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from jevmahjong.training import initialization
    from jevmahjong.training.base_lora import target_cross_entropy
    with contract.parent_loader(initialization):parent=initialization.bind_parent(cfg,args)
    if parent['model']['identity_sha256']!=c['parent_identity']:raise ValueError('Wrong qualified initializer')
    validation=[json.loads(line) for line in Path(args['eval_data']).open()]
    old_predictions_path=ROOT/old['curriculum']['out']/'training-run/training_eval/step-000000/development/rows.json'
    files[str(old_predictions_path)]=sha(old_predictions_path)
    original_predictions=read(old_predictions_path);original_ids={p['id'] for p in original_predictions}
    # Structural profile contract only. Uniform new-cohort scores are not a
    # model evaluation, are never training targets and never enter real curves.
    extra=[]
    for row in validation:
        if row['_meta']['id'] in original_ids:continue
        keys=list(row['questions']['action']['criteria'])
        extra.append(dict(id=row['_meta']['id'],keys=keys,p=[1/len(keys)]*len(keys)))
    fixture=contract.profile(validation,original_predictions+extra,{},target_cross_entropy)
    if (sum(fixture['stratum_counts'].values())!=24998
            or sum(fixture['original_validation']['stratum_counts'].values())!=23688
            or fixture['new_teacher_validation']['records']!=1310 or len(extra)!=1310):
        raise ValueError('Complete cohort profile contract failed')
    tests=subprocess.run([sys.executable,'-m','unittest','tests.core.test_curriculum_ri_anchor_runtime_v1'],
        capture_output=True,text=True,timeout=180)
    proof=RUN/'prepared-v1';proof.mkdir()
    (proof/'tests-private.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('New recipe and completed-control gate tests failed')
    write(PREVIEW,reg)
    summary=dict(status='full_ri_anchor2596_standby_prepared_not_registered',TRAIN=92570,VAL=24998,
        full_steps=2596,actual_optimizer_control_updates=2,control_updates_excluded=True,
        explicit_augmented_parent_bridge_and_ancestral_groups_checked=True,
        profile_fixture='actual_original0_cache_plus_uniform_new_not_model_evaluation',
        actual_new_VAL_inference_performed=False,profile_cohort_counts=dict(full=24998,original=23688,new=1310),
        runtime_tests_returncode=0,student_updates=0,student_fit_admitted=False)
    write(proof/'summary-private.json',summary)
    for path in (PREVIEW,proof/'summary-private.json',proof/'tests-private.log'):files[str(path)]=sha(path)
    write(PREPARED,{'files':files});print(json.dumps(summary),flush=True)


def journal_owned(role,job,dependency):
    with (RUN/'owned-submissions-private.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(role=role,job_id=job,dependency=dependency))+'\n')


def submit(role,dependency,limit,cpus=4,memory='48G'):
    result=subprocess.run(['sbatch','--parsable','--job-name=jev-ri-anchor-'+role,
        '--dependency='+dependency,'--kill-on-invalid-dep=yes','--time='+limit,
        '--cpus-per-task='+str(cpus),'--mem='+memory,'--export=ALL,JEV_RI_ANCHOR_PHASE='+role,str(ENTRY)],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    job=result.stdout.strip().split(';')[0]
    if result.returncode or not job.isdigit():raise RuntimeError(result.stderr or result.stdout)
    journal_owned(role,job,dependency);return job


def cancel_owned(jobs):
    for job in jobs:
        actual=scheduler(job)
        if actual.returncode:continue
        value=fields(actual.stdout)
        if value.get('JobId')==str(job) and value.get('Command')==str(ENTRY):
            subprocess.run(['scancel',str(job)],check=True,timeout=20)


def enqueue():
    resource();prepared=verify_binding(PREPARED)
    if BRIDGE.exists():raise ValueError('No duplicate activation submission')
    queue_path=contract.CONTROL/'fit-queue-private.json';old_queue=read(queue_path);capture=old_queue['capture_job']
    actual=scheduler(capture)
    if actual.returncode or fields(actual.stdout).get('Command')!=str(ROOT/'cluster/curriculum-ri-balanced-replay-v2.slurm'):
        raise ValueError('Exact still-queryable current terminal-capture job required before enqueue')
    job=submit('activate','afterany:'+str(capture),'00:10:00',1,'6G')
    try:
        observed=scheduler(job);value=fields(observed.stdout)
        if observed.returncode or value.get('Command')!=str(ENTRY) or 'afterany:'+str(capture)+'(' not in value.get('Dependency',''):
            raise ValueError('Actual bounded activation dependency differs')
        files=dict(prepared['files']);files[str(PREPARED)]=sha(PREPARED);files[str(queue_path)]=sha(queue_path)
        write(BRIDGE,dict(files=files,activation_job=job,current_capture_job=str(capture),
            current_queue=str(queue_path),registered=False,child_fit_jobs_submitted=False,
            current_capture_observed=actual.stdout,activation_observed=observed.stdout))
    except BaseException:
        cancel_owned([job]);raise
    print(json.dumps(dict(status='standby_activation_dependency_bound_not_registered',activation_job=job)),flush=True)


def activate():
    resource();binding=verify_binding(BRIDGE)
    if str(binding['activation_job'])!=os.environ['SLURM_JOB_ID']:raise ValueError('Wrong activation job')
    if REGISTERED.exists() or QUEUE.exists() or (RUN/'activation-outcome-private.json').exists():
        raise ValueError('No duplicate registration or activation')
    capture_receipt=ROOT/'logs'/f"curriculum-ri-balanced-replay-v2-exit-{binding['current_capture_job']}.json"
    receipt=read(capture_receipt)
    if receipt.get('phase')!='capture' or receipt.get('exit_code')!=0:raise ValueError('Actual successful current capture required')
    captured_path=contract.CONTROL/'terminal-v1/terminal-evidence-private.json';captured=read(captured_path)
    capture_binding=contract.CONTROL/'terminal-v1/completion-binding-private.json';verify_binding(capture_binding)
    if any(not captured['scheduler_proofs'][role]['successful'] for role in ('fit','audit')):
        write(RUN/'activation-outcome-private.json',dict(status='skipped_current_infrastructure_failure_needs_repair',
            student_fit_started=False,original_strategy_grade=captured['strategy_grade']))
        return
    paths=dict(completion=str(contract.CONTROL/'stage3-fit/completion.json'),audit=str(contract.CONTROL/'final-evidence-private.json'),
        execution=str(contract.CONTROL/'audit-execution-private.json'),capture=str(captured_path))
    grade=contract.completed_current(*(read(paths[k]) for k in ('completion','audit','execution','capture')))
    if grade['passed']:
        write(RUN/'activation-outcome-private.json',dict(status='skipped_qualified_current_follows_existing_3B',
            student_fit_started=False,original_strategy_grade=grade));return
    prepared=verify_binding(PREPARED);reg=copy.deepcopy(read(PREVIEW))
    reg['status']='ri_anchor2596_registered_not_started';files=dict(prepared['files'])
    for path in (*map(Path,paths.values()),capture_receipt,capture_binding):files[str(path)]=sha(path)
    merge(files,read(capture_binding)['files']);reg['inputs']=files
    write(REGISTERED,reg);contract.require_registration(reg)
    gate_files={str(Path(p)):sha(p) for p in (*paths.values(),str(capture_receipt),str(capture_binding))}
    write(RUN/'launch-gate-private.json',dict(files=gate_files,registered_sha256=sha(REGISTERED),
        evidence_paths=paths,unaltered_failed_control_grade=grade))
    contract.require_launch(REGISTERED)
    jobs=[]
    try:
        activation=os.environ['SLURM_JOB_ID']
        fit=submit('fit','afterok:'+activation,'20:00:00');jobs.append(fit)
        audit=submit('audit','afterok:'+fit,'02:00:00');jobs.append(audit)
        capture=submit('capture','afterany:'+fit+':'+audit,'00:05:00',1,'4G');jobs.append(capture)
        observations={role:scheduler(job) for role,job in (('fit',fit),('audit',audit),('capture',capture))}
        for role,result in observations.items():
            value=fields(result.stdout)
            if result.returncode or value.get('Command')!=str(ENTRY) or value.get('JobState')!='PENDING':
                raise ValueError('Owned finite child job differs')
        if ('afterok:'+activation+'(' not in fields(observations['fit'].stdout).get('Dependency','')
                or 'afterok:'+fit+'(' not in fields(observations['audit'].stdout).get('Dependency','')
                or any('afterany:'+j+'(' not in fields(observations['capture'].stdout).get('Dependency','') for j in (fit,audit))):
            raise ValueError('Actual complete child dependency graph differs')
        queue_files=dict(files)
        for path in (REGISTERED,RUN/'launch-gate-private.json',PREPARED):queue_files[str(path)]=sha(path)
        write(QUEUE,dict(files=queue_files,fit_job=fit,audit_job=audit,capture_job=capture,activation_job=activation,
            observed={k:v.stdout for k,v in observations.items()},no_held_or_released_before_activation_success=True))
        write(RUN/'activation-outcome-private.json',dict(status='registered2596_after_original_failed_strategy',
            queue_sha256=sha(QUEUE),registered_sha256=sha(REGISTERED),original_strategy_grade=grade,
            child_jobs=jobs,control_updates_excluded=True,student_fit_started=False))
    except BaseException:
        cancel_owned(jobs);raise


def bound_job(role):
    resource();queue=verify_binding(QUEUE);outcome=read(RUN/'activation-outcome-private.json')
    if (str(queue[role+'_job'])!=os.environ['SLURM_JOB_ID'] or outcome['queue_sha256']!=sha(QUEUE)
            or outcome['registered_sha256']!=sha(REGISTERED)):
        raise ValueError('Wrong actual job or changed activation proof')
    return queue


def fit():
    bound_job('fit')
    from tools.train_curriculum_ri_anchor_v1 import train
    train(REGISTERED)


def audit():
    queue=bound_job('audit');job=queue['fit_job'];actual=scheduler(job)
    receipt=read(ROOT/'logs'/f'curriculum-ri-anchor-v1-exit-{job}.json')
    if actual.returncode or receipt['phase']!='fit':raise ValueError('Actual complete FIT phase required')
    scheduler_exit(receipt,actual.stdout,ENTRY)
    write(RUN/'fit-terminal-scheduler-private.json',dict(raw=actual.stdout,exit=receipt))
    from tools.audit_curriculum_ri_anchor_v1 import audit as check
    check(REGISTERED,RUN/'final-evidence-private.json')
    write(RUN/'audit-execution-private.json',dict(status='actual_ri_anchor2596_full_evidence_independently_audited',
        actual_fit_job=job,automatic_stage3B=False,automatic_release=False))


def capture():
    queue=bound_job('capture');out=RUN/'terminal-v1';out.mkdir();proofs={}
    for role in ('fit','audit'):
        job=queue[role+'_job'];actual=scheduler(job);raw=actual.stdout
        if actual.returncode:
            if role!='fit':raise ValueError('Actual audit terminal expired before capture')
            raw=read(RUN/'fit-terminal-scheduler-private.json')['raw']
        record=terminal_record(raw,job,ENTRY)
        path=ROOT/'logs'/f'curriculum-ri-anchor-v1-exit-{job}.json';receipt=read(path) if path.exists() else None
        if record['successful']:
            if receipt is None or receipt.get('phase')!=role:raise ValueError('Actual successful phase receipt required')
            scheduler_exit(receipt,raw,ENTRY)
        record['exit_receipt']=receipt;proofs[role]=record;write(out/(role+'-terminal-private.json'),record)
    evidence=RUN/'final-evidence-private.json'
    write(out/'terminal-evidence-private.json',dict(status='ri_anchor2596_actual_terminals_preserved_not_promoted',
        scheduler_proofs=proofs,strategy_grade=read(evidence)['stage3_advancement'] if evidence.exists() else None,
        automatic_stage3B=False,automatic_release=False))
    write(out/'completion-binding-private.json',{'files':{str(p):sha(p) for p in (QUEUE,*out.glob('*-private.json'))}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,
        choices=('prepare','enqueue','activate','fit','audit','capture'))
    arguments=parser.parse_args();globals()[arguments.phase]()
