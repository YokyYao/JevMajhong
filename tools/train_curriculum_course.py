"""Continue one accepted adapter/head for expert calls, with full replay."""
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha,curriculum_parent_loader


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def run(registered):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
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
    verify();out=ROOT/c['out'];metrics=out/'metrics';last=reg['registered_final_step']
    if out.exists():raise ValueError('Existing fit must never be silently restarted')
    args=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    rows=[json.loads(s) for s in Path(args['data']).read_text().splitlines() if s]
    categories=[r['questions']['action']['src'].removeprefix('curriculum2:') for r in rows]
    originals=dict(random=trainer.random,batch=trainer.batch_loss,tracker=trainer.start_wandb,
        requests=trainer.training_requests,loss=base_lora.question_loss,profile=base_lora.validation_profile)
    trainer.random,epoch_rng=trainer_course_random_proxy(originals['random'],c['seed'],categories,c['accumulation'],out/'epoch-order-private')
    pending=[];queue=set();finished=set();contract_queue=set();contracts={}
    def flush_contracts():
        if not c.get('riichi_input_schema'):return
        for step in sorted(contract_queue-set(contracts)):
            path=out/'training-run/checkpoints'/f'step-{step:06d}'
            if not (path/'head.pt').exists():continue
            contracts[step]=seal_checkpoint(path,reg,bundle)
            write(out/'input-contracts.json',{str(k):v for k,v in contracts.items()})
    def game_eval(step):
        checkpoint=out/'training-run/checkpoints'/f'step-{step:06d}'
        if not checkpoint.exists():return
        flush_contracts()
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
        finished.add(step)
    def flush():
        flush_contracts()
        for step in sorted(queue-finished):game_eval(step)
    def batch(*a,**kw):flush();return originals['batch'](*a,**kw)
    def admitted(*a,**kw):
        result=originals['requests'](*a,**kw)
        if [r['_meta']['id'] for r in result[0]]!=[r['_meta']['id'] for r in rows]:raise ValueError('Token admission changed row order/count')
        return result
    def loss(logits,question,device):
        kind=question['src'].removeprefix('curriculum2:')
        if kind not in reg['course_loss_weights'] or question.get('target') is not None:
            raise ValueError('This course admits only registered expert hard targets')
        ce=torch.nn.functional.cross_entropy(logits[None],torch.tensor([question['label']],device=device))
        if torch.is_grad_enabled() and logits.requires_grad:pending.append((kind,float(ce.detach())))
        return ce*reg['course_loss_weights'][kind]
    def profile(records,predictions,weights):
        value=originals['profile'](records,predictions,weights);byid={r['_meta']['id']:r for r in records}
        groups=defaultdict(list);hits=defaultdict(list);rc=Counter();operations=defaultdict(Counter)
        for prediction in predictions:
            row=byid[prediction['id']];q=row['questions']['action'];kind=q['src'].removeprefix('curriculum2:')
            groups[kind].append(base_lora.target_cross_entropy(q,prediction['keys'],prediction['p']))
            chosen=prediction['keys'][max(range(len(prediction['p'])),key=prediction['p'].__getitem__)]
            hits[kind].append(chosen==q['label']);actual=q['criteria'][chosen]['type'];truth=q['criteria'][q['label']]['type']
            if kind in ('riichi_yes','riichi_no'):
                yes=kind=='riichi_yes';reach=actual=='reach';rc['tp' if yes and reach else 'fn' if yes else 'fp' if reach else 'tn']+=1
            for op in {v['type'] for v in q['criteria'].values()}&{'chi','pon','ankan','daiminkan','kakan','reach'}:
                yes=truth==op;selected=actual==op;operations[op]['tp' if yes and selected else 'fn' if yes else 'fp' if selected else 'tn']+=1
        if set(groups)!=set(reg['course_loss_mass']):raise ValueError('Missing held-out loss stratum')
        value['phase_weighted_ce']=value['weighted_target_ce']
        value['stratum_ce']={k:sum(v)/len(v) for k,v in groups.items()}
        value['stratum_accuracy']={k:sum(v)/len(v) for k,v in hits.items()}
        value['stratum_counts']={k:len(v) for k,v in groups.items()}
        value['riichi_type_counts']=dict(rc);value['operation_type_counts']={k:dict(v) for k,v in operations.items()}
        value['weighted_target_ce']=sum(reg['course_loss_mass'][k]*value['stratum_ce'][k] for k in groups)
        return value
    class Tracker(LocalTracker):
        def log(self,values):
            step=values['optimizer_step']
            if 'train/objective' in values:
                by=defaultdict(list)
                for kind,ce in pending:by[kind].append(ce)
                if set(by)!=set(reg['course_loss_mass']):raise ValueError('Actual optimizer update lacks a registered source stratum')
                self.journal.append('train',step,loss=values['train/objective'],ce=values['train/ce'],reward=None,
                    lr=values['train/lr'],lr_position='post_scheduler_next_update',
                    head_lr_derived=values['train/lr']*c['head_learning_rate']/c['learning_rate'],
                    grad_norm=values['train/grad_norm_pre_clip'],records_seen=values['train/records_seen'],
                    source_ce={k:sum(v)/len(v) for k,v in by.items()},source_counts={k:len(v) for k,v in by.items()},
                    raw_upstream_metrics=values)
                pending.clear()
            else:
                point=read(out/'validation-curve.json')[-1]
                if point['step']!=step:raise ValueError('Validation event step differs')
                self.journal.append('validation',step,loss=point['weighted_target_ce'],phase_loss=point['phase_weighted_ce'],
                    stratum_ce=point['stratum_ce'],stratum_accuracy=point['stratum_accuracy'],stratum_counts=point['stratum_counts'],
                    riichi_type_counts=point['riichi_type_counts'],operation_type_counts=point['operation_type_counts'],
                    validation_data_sha256=read(reg['training_config']['data_receipt'])['files']['validation']['sha256'])
                contract_queue.add(step)
                if step==0 or step%c['game_evaluation_every_steps']==0 or step==last:queue.add(step)
                flush()
    trainer.batch_loss=batch;trainer.training_requests=admitted
    trainer.start_wandb=lambda a,d:Tracker(MetricJournal(metrics,c['run_id'],c['stage'],c['stage_offset']),c['run_id'])
    base_lora.question_loss=loss;base_lora.validation_profile=profile
    try:
        with curriculum_parent_loader(initialization):base_lora.train(reg['training_config'])
        flush()
        actual=read(out/'training-run/training_metrics.json')
        if actual['optimizer_steps']!=last or actual['records_seen']!=reg['train_records']*c['epochs']:
            raise ValueError('Calling course did not finish the registered full exposure')
        if queue!=finished or len(epoch_rng)!=1 or epoch_rng[0].epochs!=c['epochs']:raise ValueError('Missing epoch/game evidence')
        if c.get('riichi_input_schema') and set(contracts)!={0,last,*range(c['validation_every_steps'],last+1,c['validation_every_steps'])}:
            raise ValueError('A context checkpoint contract is missing')
        verify();export_csv(metrics)
        for name in ('plot_curriculum.py','plot_curriculum_fixed_grid.py'):
            subprocess.run([sys.executable,str(ROOT/'tools'/name),'--metrics-dir',str(metrics)],check=True,timeout=180)
        write(out/'completion.json',dict(status='stage2_fit_and_curves_complete',optimizer_steps=last,
            actual_records_seen=actual['records_seen'],curve_steps=sorted(finished),new_optimizer=True,
            parent_identity=c['parent_identity'],automatic_stage3_started=False,automatic_release=False))
    finally:
        trainer.random=originals['random'];trainer.batch_loss=originals['batch'];trainer.start_wandb=originals['tracker']
        trainer.training_requests=originals['requests'];base_lora.question_loss=originals['loss'];base_lora.validation_profile=originals['profile']


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--registered',type=Path,required=True);run(p.parse_args().registered.resolve())
