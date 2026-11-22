"""Fresh fixed-base single-factor RI-loss control after actual final failure."""
from collections import Counter,defaultdict
import gc
import hashlib
import json
import math
import os
import socket
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_riichi_class_loss import annotate,weighted_loss as class_loss
from tools.curriculum_riichi_control import profile as control_profile,require_registered,require_launch_receipt,require_initial_predictions,COMPONENTS
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(registered,launch_receipt):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS'):raise RuntimeError('Scheduled GPU required')
    assert not socket.gethostname().startswith(('lo-','login'))
    manifest=read(registered);c=manifest['curriculum'];sys.path.insert(0,manifest['source'])
    sys.path.insert(0,manifest['training_config']['upstream_source']);os.chdir(ROOT)
    import torch
    torch.set_num_threads(4)
    from jevany import train as trainer
    from jevmahjong.training import base_lora
    from jevmahjong.training.metric_journal import MetricJournal,LocalTracker,export_csv
    if base_lora.package_digest(Path(manifest['source'])/'jevmahjong')!=manifest['source_sha256']:raise ValueError('Frozen source changed')
    def verify():
        for path,h in manifest['inputs'].items():
            if sha(path)!=h:raise ValueError('Registered input changed: '+path)
    verify();control=require_registered(manifest);require_launch_receipt(manifest,registered,launch_receipt)
    out=ROOT/c['out'];metrics=out/'metrics'
    if out.exists():raise ValueError('Existing fit requires inspection, never silent restart')
    # base_lora.train owns creating out; the tracker is created only after it does so.
    pending=[];queue=set();finished=set();tracker_ref=[];last=manifest['registered_final_step']
    original_batch=trainer.batch_loss;original_tracker=trainer.start_wandb
    original_random=trainer.random
    original_loss=base_lora.question_loss;original_profile=base_lora.validation_profile;original_materialize=trainer.materialize
    def materialize(request):return annotate(original_materialize(request),request)
    trainer.materialize=materialize
    initial_verified=False
    from tools.curriculum_course_order import trainer_course_random_proxy
    import tomllib
    trainer_args=tomllib.loads(Path(manifest['training_config']['trainer_config']).read_text())
    train_rows=[json.loads(s) for s in Path(trainer_args['data']).read_text().splitlines() if s]
    categories=[r['questions']['action']['src'].removeprefix('curriculum1:') for r in train_rows]
    trainer.random,epoch_rng=trainer_course_random_proxy(original_random,c['seed'],categories,
        c['batch']*c['accumulation'],out/'epoch-order-private')
    old_requests=trainer.training_requests
    def verified_requests(*args,**kwargs):
        requests,receipt=old_requests(*args,**kwargs)
        if [r['_meta']['id'] for r in requests]!=[r['_meta']['id'] for r in train_rows]:
            raise ValueError('Token admission reordered or dropped training records')
        return requests,receipt
    trainer.training_requests=verified_requests
    def game_eval(step):
        nonlocal initial_verified
        checkpoint=out/'training-run/checkpoints'/f'step-{step:06d}'
        if not checkpoint.exists():return
        if step==0 and not initial_verified:
            from safetensors.torch import load_file
            from jevmahjong.training.initialization import exact_loaded_tensors
            old_fit=ROOT/control['curriculum']['out'];old_checkpoint=old_fit/'training-run/checkpoints/step-000000'
            exact=exact_loaded_tensors(load_file(checkpoint/'adapter_model.safetensors'),
                torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=True)['head'],old_checkpoint)
            if exact['adapter_tensors']!=372 or exact['head_tensors']!=4:raise ValueError('Fresh comparison tensor inventory differs')
            old=read(old_fit/'training-run/training_eval/step-000000/development/rows.json')
            actual=read(out/'training-run/training_eval/step-000000/development/rows.json')
            parity=require_initial_predictions(old,actual)
            write(out/'initial-control-validation-private.json',dict(exact_tensors=exact,validation=parity,
                comparison_only_no_control_parameters_loaded=True,checked_before_first_optimizer_update=True))
        command=[sys.executable,str(ROOT/'tools/evaluate_curriculum.py'),'--registered',str(registered),
            '--checkpoint',str(checkpoint),'--step',str(step),'--groups',str(c['stage_end_seed_groups'] if step==last else c['curve_seed_groups'])]
        reference=out/'evaluations/reference-v8'
        if not reference.exists() and c.get('reference_evaluation_cache'):
            cache=(ROOT/c['reference_evaluation_cache']).resolve()
            stored=read(cache/'summary.json');protocol=read(cache/'protocol.json')
            if (stored['checkpoint_sha256']!=c['reference_identity'] or stored['matches']!=64
                or protocol['seeds']!=list(range(c['curve_seed_start'],c['curve_seed_start']+c['stage_end_seed_groups']))
                or protocol['profiles']!=c['opponent_profiles']):
                raise ValueError('Cached reference grid is not the registered comparison')
            reference.parent.mkdir(parents=True,exist_ok=True)
            reference.symlink_to(cache,target_is_directory=True)
        if not (reference/'summary.json').exists():
            base=[sys.executable,str(ROOT/'tools/evaluate_curriculum.py'),'--registered',str(registered),
                '--checkpoint',str(ROOT/c['reference_checkpoint']),'--step','0','--groups',str(c['stage_end_seed_groups']),'--baseline']
            subprocess.run(base,check=True,timeout=1200)
        subprocess.run(command,check=True,timeout=1200)
        if step==0:
            old_fit=ROOT/control['curriculum']['out'];load=lambda p:[json.loads(s) for s in Path(p).open()]
            old={(r['seed'],r['profile'],r['seat']):r for r in load(old_fit/'evaluations/step-000000/matches-private.jsonl')}
            actual=load(out/'evaluations/step-000000/matches-private.jsonl')
            fields=('rank','net_points','final_scores','hands','wins','dealin_hands','choices')
            if len(actual)!=16 or len(old)!=16 or any(any(r[k]!=old[r['seed'],r['profile'],r['seat']][k] for k in fields) for r in actual):
                raise ValueError('Fresh initial same16 control outcomes differ')
            write(out/'initial-control-games-private.json',dict(matches=16,exact_fresh_control_outcomes=True,
                comparison_only_no_parameters_loaded=True,checked_before_first_optimizer_update=True))
            initial_verified=True
        finished.add(step)
    def flush_queue():
        for step in sorted(queue-finished):game_eval(step)
    def observed_batch(*args,**kwargs):
        flush_queue()
        if not initial_verified:raise ValueError('All initial tensor/prediction/game checks must finish before any update')
        return original_batch(*args,**kwargs)
    def profile(records,rows,weights):
        return control_profile(records,rows,manifest['stage1_loss_mass'],original_profile(records,rows,weights),base_lora.target_cross_entropy)
    class Tracker(LocalTracker):
        def log(self,values):
            step=values['optimizer_step']
            if 'train/objective' in values:
                by=defaultdict(list)
                for row in pending:by[row['source']].append(row)
                if not pending:raise ValueError('Missing actual post-update source losses')
                if set(by)!=set(manifest['stage1_loss_mass']):
                    raise ValueError('Actual optimizer update lacks a declared source/class')
                means={s:{k:sum(r[k] for r in rows)/len(rows) for k in COMPONENTS} for s,rows in by.items()}
                n=sum(len(rows) for rows in by.values())
                common_ce=sum(len(by[s])*means[s]['standard_action_ce']*manifest['stage1_loss_weights'][s] for s in by)/n
                self.journal.append('train',step,loss=values['train/objective'],ce=common_ce,
                    reward=None,lr=values.get('train/lr'),lr_position='post_scheduler_next_update',
                    head_lr_derived=values['train/lr']*c['head_learning_rate']/c['learning_rate'],
                    grad_norm=values.get('train/grad_norm_pre_clip'),records_seen=values.get('train/records_seen'),
                    source_ce={k:means[k]['standard_action_ce'] for k in by},source_objective={k:means[k]['objective'] for k in by},
                    source_loss_components=means,source_counts={k:len(v) for k,v in by.items()},
                    ce_is_original_hard_action_loss=True,loss_method=c['riichi_loss_method'],
                    raw_upstream_metrics=values)
                pending.clear()
            else:
                curve=read(out/'validation-curve.json');v=curve[-1]
                if v['step']!=step:raise ValueError('Validation event step mismatch')
                self.journal.append('validation',step,loss=v['weighted_class_objective'],ce=v['weighted_target_ce'],natural_loss=v['natural_target_ce'],
                    source_objective=v['source_objective'],source_loss_components=v['source_loss_components'],
                    common_weighted_target_ce=v['weighted_target_ce'],loss_method=c['riichi_loss_method'],
                    stratum_ce=v['stratum_ce'],stratum_accuracy=v['stratum_accuracy'],stratum_counts=v['stratum_counts'],
                    riichi_type_counts=v['riichi_type_counts'],
                    riichi_by_source=v['riichi_by_source'],source_stratum_ce=v['source_stratum_ce'],
                    source_stratum_accuracy=v['source_stratum_accuracy'],source_stratum_counts=v['source_stratum_counts'],
                    validation_sha256=manifest['training_config']['data_receipt_sha256'],
                    validation_data_sha256=read(manifest['training_config']['data_receipt'])['files']['validation']['sha256'])
                if step==0 or step%c['game_evaluation_every_steps']==0 or step==last:queue.add(step)
                flush_queue()
    def start_tracker(args,directory):
        journal=MetricJournal(metrics,c['run_id'],c['stage'],c['stage_offset'])
        t=Tracker(journal,c['run_id']);tracker_ref.append(t);return t
    trainer.start_wandb=start_tracker;trainer.batch_loss=observed_batch
    base_lora.question_loss=lambda z,q,d:class_loss(z,q,d,manifest['stage1_loss_weights'],pending)
    base_lora.validation_profile=profile
    try:
        base_lora.train(manifest['training_config']);flush_queue()
        if queue!=finished or not initial_verified:raise ValueError('A registered saved-checkpoint/initial control evaluation is missing')
        actual=read(out/'training-run/training_metrics.json')
        if actual['optimizer_steps']!=last:raise ValueError('Fixed full budget did not complete')
        logs=[json.loads(s) for s in (metrics/'train_metrics.jsonl').read_text().splitlines()]
        if [r['local_step'] for r in logs]!=list(range(1,last+1)):raise ValueError('Missing optimizer metric events')
        if actual['records_seen']!=manifest['train_records']*c['epochs']:raise ValueError('Incorrect registered epoch exposure')
        if len(epoch_rng)!=1 or epoch_rng[0].epochs!=c['epochs']:raise ValueError('Epoch scheduler did not run as registered')
        verify();export_csv(metrics)
        subprocess.run([sys.executable,str(ROOT/'tools/plot_curriculum.py'),'--metrics-dir',str(metrics)],check=True,timeout=180)
        subprocess.run([sys.executable,str(ROOT/'tools/plot_curriculum_fixed_grid.py'),'--metrics-dir',str(metrics)],check=True,timeout=180)
        write(out/'completion.json',dict(status='stage1_fit_and_curves_complete',optimizer_steps=last,
            actual_records_seen=actual['records_seen'],curve_steps=sorted(finished),new_stage2_started=False,automatic_release=False))
    finally:
        trainer.start_wandb=original_tracker;trainer.batch_loss=original_batch
        trainer.random=original_random;trainer.training_requests=old_requests
        base_lora.question_loss=original_loss;base_lora.validation_profile=original_profile
        trainer.materialize=original_materialize

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--registered',type=Path,required=True);p.add_argument('--launch-receipt',type=Path,required=True)
    a=p.parse_args();run(a.registered.resolve(),a.launch_receipt.resolve())
