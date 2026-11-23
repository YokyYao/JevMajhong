"""Real two-update upstream ordering integration and immutable recovery registration."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only,write
from tools.curriculum_transfer import read,sha
from tools.curriculum_checkpoint_evaluation_fence_v1 import CheckpointEvaluationFence,checkpoint_ready
from tools.curriculum_progression_parent_v1 import verify_parent

PUBLIC_FILES=('tools/curriculum_checkpoint_evaluation_fence_v1.py',
    'tools/prepare_curriculum_progression_replay_v4.py','tools/train_curriculum_progression_v3.py',
    'tools/finish_curriculum_progression_audit_v4.py',
    'tests/core/test_curriculum_checkpoint_evaluation_fence_v1.py',
    'cluster/curriculum-progression-v4.slurm','docs/experiments/plans/curriculum-progression-replay-20261123.md')


def toml(path,args):
    with path.open('x') as stream:stream.write(''.join(f'{k} = {json.dumps(v)}\n' for k,v in args.items()))


def mini_integration(reg,out):
    from jevany import train as trainer
    import torch
    torch.set_num_threads(4)
    args=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    out.mkdir(parents=True,exist_ok=False);rows=[];seen=set()
    from jevany.suite import semantic_hash
    for line in Path(args['data']).open():
        row=json.loads(line);q=row['questions']['action']
        if q.get('target') is None and {v['type'] for v in q['criteria'].values()}=={'dahai'}:
            key=semantic_hash(row)
            if key in seen:continue
            seen.add(key);rows.append(row)
            if len(rows)==4:break
    if len(rows)!=4:raise ValueError('Four semantically distinct TRAIN-only control fixtures required')
    data=out/'TRAIN-only-control-test-not-validation.jsonl'
    validation=out/'disjoint-TRAIN-only-control-validation.jsonl'
    for path,subset in ((data,rows[:2]),(validation,rows[2:])):
        with path.open('x') as stream:
            for row in subset:stream.write(json.dumps(row)+'\n')
    args.update(data=str(data),eval_data=str(validation),out=str(out/'toy-run'),epochs=1,max_steps=2,
        batch=1,accum=1,checkpoint_every_steps=1,eval_every_steps=1,eval_before_start=False,
        seed=2026112307,early_stop_patience=0)
    config=out/'toy.toml';toml(config,args)
    events=[];root=out/'toy-run/checkpoints'
    def inspect(step):
        state=torch.load(root/f'step-{step:06d}'/'trainer_state.pt',map_location='cpu',weights_only=False)
        if state['optimizer_step']!=step or any(float(v['step'])!=step for v in state['optimizer']['state'].values()):
            raise ValueError('Native callback preceded a real durable optimizer state')
        events.append(dict(kind='native_callback_after_full_save',step=step))
    fence=CheckpointEvaluationFence(lambda step:checkpoint_ready(root,step),inspect)
    original=dict(batch=trainer.batch_loss,tracker=trainer.start_wandb)
    class Tracker:
        config={}
        id='routing-integration';name=id;project=id;entity=None;url=None
        def log(self,values):
            if 'train/objective' not in values:
                step=values['optimizer_step'];events.append(dict(kind='validation_before_save',step=step))
                fence.validation(step)
        def finish(self):pass
    def batch(*a,**kw):
        fence.flush();events.append(dict(kind='next_optimizer_batch'))
        return original['batch'](*a,**kw)
    trainer.batch_loss=batch;trainer.start_wandb=lambda *a,**kw:Tracker()
    try:trainer.main(['--config',str(config),'--base_load_path',reg['training_config']['base_load_path']])
    finally:trainer.batch_loss=original['batch'];trainer.start_wandb=original['tracker']
    fence.flush()
    callbacks=[e['step'] for e in events if e['kind']=='native_callback_after_full_save']
    if callbacks!=[1,2] or fence.queued!=fence.finished:raise ValueError('Actual periodic/final save ordering failed')
    result=dict(status='real_upstream_two_update_validation_save_and_deferred_callback_integration_passed',
        actual_control_test_updates=2,callbacks=callbacks,actual_durable_optimizer_checked=True,
        final_callback_without_next_batch_checked=True,fixtures_from_TRAIN_only=True,semantically_disjoint_control_inputs=True,
        control_test_not_a_candidate_or_validation_curve=True,events=events)
    write(out/'summary-private.json',result);return result


def prepare():
    gpu_only();old_path=ROOT/'runs/curriculum-progression-v1/fit-plan/registered.json';old=read(old_path)
    exit_path=ROOT/'logs/curriculum-progression-exit-318606.json';exit=read(exit_path)
    if exit['exit_code']!=1 or exit['phase']!='fit':raise ValueError('Preserved actual failed run required')
    old_fit=ROOT/old['curriculum']['out']
    actual_steps=[int(p.name[5:]) for p in (old_fit/'training-run/checkpoints').glob('step-*')]
    if actual_steps!=[0]:raise ValueError('Inspect new recoverable checkpoint before choosing replay')
    failed_lines=[json.loads(line) for line in (old_fit/'metrics/train_metrics.jsonl').open()]
    if len(failed_lines)!=600 or failed_lines[-1]['records_seen']!=43800:
        raise ValueError('Exact original failed600 attempt evidence required')
    if any(sha(p)!=h for p,h in old['inputs'].items()):raise ValueError('Original source/config/data changed')
    sys.path.insert(0,old['source']);sys.path.insert(0,old['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    parent_before=bundle(old['parent_model']['directory'])
    run=ROOT/'runs/curriculum-progression-v4'
    if (run/'fit-plan').exists() or (run/'stage3-fit').exists():raise ValueError('Never overwrite registration or fit')
    for name in PUBLIC_FILES:
        if name.endswith('.py'):compile((ROOT/name).read_text(),str(ROOT/name),'exec')
    preparation_failure=ROOT/'logs/curriculum-progression-v2-exit-318957.json'
    if read(preparation_failure)!=dict(job_id='318957',phase='prepare',exit_code=1):
        raise ValueError('Preserved preparation failure required')
    tracker_failure=ROOT/'logs/curriculum-progression-v3-exit-318969.json'
    if read(tracker_failure)!=dict(job_id='318969',phase='prepare',exit_code=1):
        raise ValueError('Preserved control tracker interface failure required')
    integration=mini_integration(old,run/'routing-integration-v4')
    if bundle(old['parent_model']['directory'])!=parent_before:raise ValueError('Control test changed sealed parent')
    reg=copy.deepcopy(old);c=reg['curriculum'];cfg=reg['training_config']
    c.update(run_id='jev-curriculum-progression-v4',stage='stage3-progression-v4',
        out='runs/curriculum-progression-v4/stage3-fit',plan_root='runs/curriculum-progression-v4/fit-plan',
        step0_parameter_cache=str(old_fit/'training-run/checkpoints/step-000000'),
        step0_validation_reuse_cache=str(old_fit/'training-run/training_eval/step-000000'),
        step0_native_reuse_cache=str(old_fit/'evaluations/step-000000'),
        failed_attempt_registration=str(old_path),failed_optimizer_updates_not_resumed=600,
        failure_repair='defer_native_callback_until_full_checkpoint_save',checkpoint_every_steps=100)
    plan=ROOT/c['plan_root'];plan.mkdir(parents=True)
    args=tomllib.loads(Path(old['training_config']['trainer_config']).read_text())
    args.update(out=str((ROOT/c['out']/'training-run').resolve()),checkpoint_every_steps=100)
    trainer=plan/'trainer.toml';toml(trainer,args)
    cfg.update(model_id=c['run_id'],out=c['out'],trainer_config=str(trainer),trainer_config_sha256=sha(trainer))
    verify_parent(cfg)
    config=plan/'training-config.json';write(config,cfg)
    files=dict(old['inputs'])
    paths=[old_path,exit_path,tracker_failure,ROOT/'logs/curriculum-progression-v3-318969.log',preparation_failure,ROOT/'logs/curriculum-progression-v2-318957.log',trainer,config,old_fit/'metrics/train_metrics.jsonl',
        old_fit/'metrics/validation_metrics.jsonl',old_fit/'validation-curve.json',
        old_fit/'initialization-audit.json',old_fit/'actual-update-proof.json',
        run/'routing-integration-v4/summary-private.json',ROOT/'logs/curriculum-progression-318606.log']
    paths += [ROOT/name for name in PUBLIC_FILES]
    paths += list((old_fit/'training-run/training_eval/step-000000').rglob('*'))
    paths += list((old_fit/'training-run/checkpoints/step-000000').iterdir())
    paths += [old_fit/'evaluations/step-000000'/name for name in ('protocol.json','summary.json','matches-private.jsonl')]
    for path in paths:
        if path.is_file():files[str(path.resolve())]=sha(path)
    reg.update(inputs=files,optimizer_updates=0,new_optimizer=True,replay_from_sealed_parent_not_resume=True,
        failed_attempt_updates_excluded_from_candidate_trace=600,original_budget_and_gradient_recipe_unchanged=True)
    registered=plan/'registered.json';write(registered,reg)
    files=dict(files);files[str(registered.resolve())]=sha(registered)
    if any(sha(p)!=h for p,h in files.items()):raise ValueError('Recovery launch input changed')
    write(run/'launch-binding-private.json',dict(files=files,registered_sha256=sha(registered),
        original_failed_attempt_preserved=True,recovery_mode='replay_same_parent_seed_data_not_resume600',
        zero_baseline_reuse_requires_exact_actual_loaded_tensors=True,
        new_candidate_optimizer_updates=0,job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname()))
    write(run/'prepare-execution-private.json',dict(status='same_recipe2400_replay_registered_after_real_ordering_integration',
        original_failed_job=318606,failed600_weights_not_saved=True,new_optimizer_updates=0,
        routing_control_test_updates=integration['actual_control_test_updates'],
        original_parent_bundle_unchanged=True,actual_registered_final_step=2400,
        existing_step0_measurements_reused_with_numeric_proof_only=True,automatic_release=False))
    print(json.dumps(dict(status='progression_v3_replay_registered',actual_fit_updates=0,control_test_updates=2)),flush=True)


if __name__=='__main__':prepare()


