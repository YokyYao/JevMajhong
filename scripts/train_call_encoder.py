"""Train the call-only LoRA using upstream training and exact-action protection.

The upstream source files remain unchanged. A local evaluation callback adds
admissibility to checkpoint selection without changing reported NLL values.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from call_policy import digest
from encoder_training_guard import ExactRecallGuard,exact_metrics
from head_rl import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());out=ROOT/cfg['out'];data=ROOT/cfg['data']
    preparation=json.loads((data/'provenance.json').read_text())
    if preparation['config']!=cfg or not preparation['complete'] or preparation['human_test_opened']:
        raise ValueError('Prepared inputs differ')
    for split,h in preparation['files'].items():
        if digest(data/(split+'.jsonl'))!=h:raise ValueError('Prepared dataset changed')
    for name,h in preparation['initialization_files'].items():
        if digest(ROOT/cfg['initialization']/name)!=h:raise ValueError('Warm initialization changed')
    for name,h in preparation['reference_files'].items():
        if digest(ROOT/cfg['reference']/name)!=h:raise ValueError('Incumbent changed')
    if (out/'protocol.json').exists():raise ValueError('Training has already started; inspect before resuming')
    code=['scripts/train_call_encoder.py','scripts/prepare_call_encoder.py','encoder_training_guard.py',
          'encoder_call_policy.py','call_policy.py','semantic.py','vendor/JevAny/jevany/train.py',
          'vendor/JevAny/jevany/model.py','vendor/JevAny/jevany/checkpoint.py']
    protocol={'config':cfg,'prepared_provenance_sha256':digest(data/'provenance.json'),
        'trainer_config_sha256':digest(ROOT/cfg['trainer_config']),
        'code_sha256':{n:digest(ROOT/n) for n in code},'reference_files':preparation['reference_files'],
        'initialization_files':preparation['initialization_files'],
        'method':'independent call encoder LoRA and full pointer head; ordinaryB never trained',
        'selection':'actual validation NLL with exact chi/pon TP >=half warm-start and nonzero in every view',
        'primary_success_metric':'new-wall full-East terminal net points, not supervised accuracy',
        'test_opened':False,'deployed':False}
    write(out/'protocol.json',protocol)
    records=[json.loads(l) for l in (data/'validation.jsonl').read_text().splitlines()]
    from jevany import train as trainer
    original_evaluate,original_update=trainer.evaluate_during_training,trainer.early_stop_update
    guard=ExactRecallGuard(cfg['views'],cfg['min_exact_recall_fraction']);curve=[];started=time.monotonic()
    def evaluate(*pos,**kw):
        summary=original_evaluate(*pos,**kw)
        step=kw.get('step',pos[5] if len(pos)>5 else None)
        train_dir=Path(kw.get('out_dir',pos[4] if len(pos)>4 else ''))
        rows=json.loads((train_dir/'training_eval'/f'step-{step:06d}'/'development'/'rows.json').read_text())
        metrics=exact_metrics(records,rows);admission=guard.assess(metrics)
        item={'step':step,'nll':summary['clean']['nll'],'views':metrics,'guard':admission,
              'seconds':time.monotonic()-started}
        curve.append(item);write(out/f'validation-{step:06d}.json',item);write(out/'validation-curve.json',curve)
        status={'status':'call_encoder_validation','step':step,'max_steps':cfg['max_steps'],
                'nll':item['nll'],'admissible':admission['admissible'],'seconds':item['seconds']}
        write(out/'status.json',status);print(json.dumps(status),flush=True)
        return summary
    def update(value,best,stale,maximize=True,min_delta=0.):
        return guard.update(original_update,value,best,stale,maximize=maximize,min_delta=min_delta)
    trainer.evaluate_during_training=evaluate;trainer.early_stop_update=update
    write(out/'status.json',{'status':'loading_call_encoder_for_training','step':0,'max_steps':cfg['max_steps']})
    try:
        base=Path(os.environ.get('JEV_BASE_MODEL_DIR',Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
        train_dir=trainer.main(['--config',str(ROOT/cfg['trainer_config']),'--base_load_path',str(base)])
        selection=json.loads((train_dir/'selection.json').read_text());metrics=json.loads((train_dir/'training_metrics.json').read_text())
        if metrics['global_effective_batch']!=16:raise ValueError('Unexpected global batch')
        if metrics['rejected_records'] or metrics['admitted_records']!=preparation['records']['train']:
            raise ValueError('Training token admission lost prepared records')
        for name,h in protocol['code_sha256'].items():
            if digest(ROOT/name)!=h:raise ValueError('Training source changed')
        for name,h in preparation['reference_files'].items():
            if digest(ROOT/cfg['reference']/name)!=h:raise ValueError('OrdinaryB changed')
        selected=selection.get('selected_checkpoint');step=selection['step']
        status='encoder_refit_complete' if step>0 else 'encoder_refit_not_improved'
        result={'status':status,'step':metrics['optimizer_steps'],'selected_step':step,
            'checkpoint':str((train_dir/selected).resolve().relative_to(ROOT)) if selected else None,
            'validation':next(r for r in curve if r['step']==step),
            'incumbent_files_unchanged':True,'human_test_opened':False,'deployed':False,
            'strength_improvement_proven':False,'next':'native call/discard parity and fresh full-East development required'}
        write(out/'result.json',result);write(out/'status.json',result);print(json.dumps(result),flush=True)
    except Exception as error:
        write(out/'training-failure.json',{'type':type(error).__name__,'error':str(error),'ordinaryB_source_preserved':True})
        raise
    finally:
        trainer.evaluate_during_training=original_evaluate;trainer.early_stop_update=original_update


if __name__=='__main__':main()
