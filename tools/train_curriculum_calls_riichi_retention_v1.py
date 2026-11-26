"""Continue a qualified max-class basic final, retaining its RI loss explicitly."""
from collections import Counter,defaultdict
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_fit import parent_loader,verify_parent
from tools.curriculum_calls_repair import require_parent_prediction_parity
from tools.curriculum_calls_riichi_retention_v1 import METHOD,annotate,weighted_loss
from tools.curriculum_calls_riichi_fit_v1 import require_recipe,profile as source_profile,summarize,require_parent_kind


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def run(registered):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    reg=read(registered);c=reg['curriculum'];sys.path.insert(0,reg['source'])
    sys.path.insert(0,reg['training_config']['upstream_source']);os.chdir(ROOT)
    import torch
    torch.set_num_threads(4)
    from jevany import train as trainer
    from jevmahjong.training import base_lora,initialization
    from jevmahjong.training.metric_journal import MetricJournal,LocalTracker,export_csv
    from jevmahjong.evaluation.unified_contract import bundle
    from tools.curriculum_course_order import trainer_course_random_proxy
    from tools.curriculum_input_contract import seal_checkpoint
    if base_lora.package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:raise ValueError('Pinned source changed')
    def verify():
        for p,h in reg['inputs'].items():
            if sha(p)!=h:raise ValueError('Registered input changed: '+p)
    verify();verify_parent(reg['training_config'])
    parent_seal=read(c['parent_seal']);require_parent_kind(read(parent_seal['parent_registration']))
    data=read(c['data_ready']);require_recipe(c,data,data['categories']['train'])
    if (reg['status']!='stage2_calls_riichi_retention_registered_not_started' or reg['registered_final_step']!=384
        or reg['training_config'].get('loss_method')!=METHOD):
        raise ValueError('Explicit complete repaired-calls registration required')
    out=ROOT/c['out'];metrics=out/'metrics';last=reg['registered_final_step']
    if out.exists():raise ValueError('Existing fit must never be silently restarted')
    args=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    rows=[json.loads(s) for s in Path(args['data']).read_text().splitlines() if s]
    categories=[r['questions']['action']['src'].removeprefix('curriculum2:') for r in rows]
    originals=dict(random=trainer.random,batch=trainer.batch_loss,tracker=trainer.start_wandb,
        requests=trainer.training_requests,materialize=trainer.materialize,
        loss=base_lora.question_loss,profile=base_lora.validation_profile)
    def materialize(request):return annotate(originals['materialize'](request),request)
    trainer.materialize=materialize
    trainer.random,epoch_rng=trainer_course_random_proxy(originals['random'],c['seed'],categories,c['accumulation'],out/'epoch-order-private')
    pending=[];queue=set();finished=set();contract_queue=set();contracts={}
    warmstart=None
    def flush_contracts():
        if not c.get('riichi_input_schema'):return
        for step in sorted(contract_queue-set(contracts)):
            path=out/'training-run/checkpoints'/f'step-{step:06d}'
            if not (path/'head.pt').exists():continue
            contracts[step]=seal_checkpoint(path,reg,bundle)
            write(out/'input-contracts.json',{str(k):v for k,v in contracts.items()})
    def game_eval(step):
        nonlocal warmstart
        checkpoint=out/'training-run/checkpoints'/f'step-{step:06d}'
        if not checkpoint.exists():return
        flush_contracts()
        if step==0 and warmstart is None:
            from safetensors.torch import load_file
            from jevmahjong.training.initialization import exact_loaded_tensors
            exact=exact_loaded_tensors(load_file(checkpoint/'adapter_model.safetensors'),
                torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=True)['head'],reg['parent_model']['directory'])
            if exact['adapter_tensors']!=372 or exact['head_tensors']!=4:raise ValueError('Unexpected warm-start tensor inventory')
            seal=read(c['parent_seal']);parent_reg=read(seal['parent_registration']);parent_args=tomllib.loads(Path(parent_reg['training_config']['trainer_config']).read_text())
            parent_rows=[json.loads(s) for s in Path(parent_args['eval_data']).open()]
            validation=[json.loads(s) for s in Path(args['eval_data']).open()]
            parent_fit=ROOT/parent_reg['curriculum']['out']
            parent_p=read(parent_fit/'training-run/training_eval'/f"step-{seal['selected_step']:06d}"/'development/rows.json')
            child_p=read(out/'training-run/training_eval/step-000000/development/rows.json')
            parity=require_parent_prediction_parity(parent_rows,parent_p,validation,child_p)
            if parity['checked_basic_menus']!=3266:raise ValueError('Full basic validation parity required')
            warmstart=dict(exact_tensors=exact,basic_validation_prediction_parity=parity,
                checked_before_first_optimizer_update=True)
            write(out/'warmstart-validation-private.json',warmstart)
        for label,source,expected in [('reference-v8',c['reference_evaluation_cache'],c['reference_identity']),
                                     ('reference-parent',c['parent_evaluation_cache'],c['parent_identity'])]:
            target=out/'evaluations'/label;source=(ROOT/source).resolve()
            summary=read(source/'summary.json');protocol=read(source/'protocol.json')
            if (summary['checkpoint_sha256']!=expected or summary['matches']!=64
                or protocol['seeds']!=list(range(c['curve_seed_start'],c['curve_seed_start']+c['stage_end_seed_groups']))
                or protocol['profiles']!=c['opponent_profiles']):raise ValueError('Reference cache grid differs')
            if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source,target_is_directory=True)
        script='evaluate_curriculum_context.py' if c.get('riichi_input_schema') else 'evaluate_curriculum.py'
        subprocess.run([sys.executable,str(ROOT/'tools'/script),'--registered',str(registered),'--checkpoint',str(checkpoint),
            '--step',str(step),'--groups',str(c['stage_end_seed_groups'] if step==last else c['curve_seed_groups'])],check=True,timeout=1200)
        if step==0:
            load_rows=lambda p:[json.loads(s) for s in Path(p).open()]
            parent_games={(r['seed'],r['profile'],r['seat']):r for r in load_rows(ROOT/c['parent_evaluation_cache']/'matches-private.jsonl')}
            actual=load_rows(out/'evaluations/step-000000/matches-private.jsonl')
            fields=('rank','net_points','final_scores','hands','wins','dealin_hands','choices')
            if len(actual)!=16 or any(any(r[k]!=parent_games[r['seed'],r['profile'],r['seat']][k] for k in fields) for r in actual):
                raise ValueError('Exact parent step-zero serving/game reproduction failed')
            write(out/'warmstart-games-private.json',dict(matches=16,exact_parent_outcomes=True,checked_before_first_update=True))
        finished.add(step)
    def flush():
        flush_contracts()
        for step in sorted(queue-finished):game_eval(step)
    def batch(*a,**kw):
        flush()
        if warmstart is None or 0 not in finished:
            raise ValueError('Exact parent validation/game checks must finish before optimizer updates')
        return originals['batch'](*a,**kw)
    def admitted(*a,**kw):
        result=originals['requests'](*a,**kw)
        if [r['_meta']['id'] for r in result[0]]!=[r['_meta']['id'] for r in rows]:raise ValueError('Token admission changed row order/count')
        return result
    def loss(logits,question,device):
        return weighted_loss(logits,question,device,reg['course_loss_weights'],pending)
    def profile(records,predictions,weights):
        return source_profile(records,predictions,reg['course_loss_mass'],originals['profile'](records,predictions,weights),base_lora.target_cross_entropy)
    class Tracker(LocalTracker):
        def log(self,values):
            step=values['optimizer_step']
            if 'train/objective' in values:
                summary=summarize(pending,reg['course_loss_weights'])
                if abs(summary['loss']-values['train/objective'])>1e-5:
                    raise ValueError('Observed class objective differs from actual optimizer loss')
                self.journal.append('train',step,**summary,reward=None,
                    lr=values['train/lr'],lr_position='post_scheduler_next_update',
                    head_lr_derived=values['train/lr']*c['head_learning_rate']/c['learning_rate'],
                    grad_norm=values['train/grad_norm_pre_clip'],records_seen=values['train/records_seen'],
                    loss_method=METHOD,ce_is_original_hard_action_loss=True,
                    raw_upstream_metrics=values)
                pending.clear()
            else:
                point=read(out/'validation-curve.json')[-1]
                if point['step']!=step:raise ValueError('Validation event step differs')
                self.journal.append('validation',step,loss=point['weighted_class_objective'],ce=point['weighted_target_ce'],
                    source_objective=point['source_objective'],source_loss_components=point['source_loss_components'],
                    phase_loss=point['phase_weighted_ce'],loss_method=METHOD,
                    stratum_ce=point['stratum_ce'],stratum_accuracy=point['stratum_accuracy'],stratum_counts=point['stratum_counts'],
                    riichi_type_counts=point['riichi_type_counts'],operation_type_counts=point['operation_type_counts'],
                    riichi_by_source=point['riichi_by_source'],basic_replay=point['basic_replay'],
                    basic_replay_source_quality=point['basic_replay_source_quality'],
                    operation_type_counts_by_source=point['operation_type_counts_by_source'],
                    validation_data_sha256=read(reg['training_config']['data_receipt'])['files']['validation']['sha256'])
                contract_queue.add(step)
                if step==0 or step%c['game_evaluation_every_steps']==0 or step==last:queue.add(step)
                flush()
    trainer.batch_loss=batch;trainer.training_requests=admitted
    trainer.start_wandb=lambda a,d:Tracker(MetricJournal(metrics,c['run_id'],c['stage'],c['stage_offset']),c['run_id'])
    base_lora.question_loss=loss;base_lora.validation_profile=profile
    try:
        with parent_loader(initialization):base_lora.train(reg['training_config'])
        flush()
        actual=read(out/'training-run/training_metrics.json')
        if actual['optimizer_steps']!=last or actual['records_seen']!=reg['train_records']*c['epochs']:
            raise ValueError('Calling course did not finish the registered full exposure')
        if queue!=finished or len(epoch_rng)!=1 or epoch_rng[0].epochs!=c['epochs'] or warmstart is None:raise ValueError('Missing epoch/game/initial parity evidence')
        if c.get('riichi_input_schema') and set(contracts)!={0,last,*range(c['validation_every_steps'],last+1,c['validation_every_steps'])}:
            raise ValueError('A context checkpoint contract is missing')
        verify();export_csv(metrics)
        for name in ('plot_curriculum.py','plot_curriculum_fixed_grid.py'):
            subprocess.run([sys.executable,str(ROOT/'tools'/name),'--metrics-dir',str(metrics)],check=True,timeout=180)
        write(out/'completion.json',dict(status='stage2_calls_riichi_retention_fit_and_curves_complete',loss_method=METHOD,optimizer_steps=last,
            actual_records_seen=actual['records_seen'],curve_steps=sorted(finished),new_optimizer=True,
            parent_identity=c['parent_identity'],automatic_stage3_started=False,automatic_release=False))
    finally:
        trainer.random=originals['random'];trainer.batch_loss=originals['batch'];trainer.start_wandb=originals['tracker']
        trainer.training_requests=originals['requests'];base_lora.question_loss=originals['loss'];base_lora.validation_profile=originals['profile']
        trainer.materialize=originals['materialize']


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--registered',type=Path,required=True);run(p.parse_args().registered.resolve())
