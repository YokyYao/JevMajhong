"""Native curriculum continuing only the verified clean-parent Mahjong model."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from encoder_training_guard import ExactRecallGuard, exact_metrics
from head_rl import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    data=ROOT/cfg['data'];out=ROOT/cfg['out']
    if (out/'protocol.json').exists() or (out/'invalidation.json').exists():
        raise ValueError('Refuse overwriting a started or invalidated experiment; use a new run')
    try:
        prepared=json.loads((data/'provenance.json').read_text())
        if not prepared['complete'] or prepared['config']!=cfg or prepared['model_test_opened']:
            raise ValueError('Prepared data incomplete or different')
        for split in ('train','validation'):
            if digest(data/(split+'.jsonl'))!=prepared['files'][split]:
                raise ValueError('Training/validation changed')
        for name,key in [('replay_call_repair.py','repair_sha256'),('semantic.py','semantic_sha256'),('clean_policy_data.py','policy_data_sha256'),('replay_dora_repair.py','dora_repair_sha256')]:
            if digest(ROOT/name)!=prepared[key]:raise ValueError('Data-producing code changed')
        trainer_config=tomllib.loads((ROOT/cfg['trainer_config']).read_text())
        if trainer_config.get('init_from')!=cfg['warm_start'] or trainer_config.get('resume'):
            raise ValueError('Only the registered clean-parent warm start is permitted')
        parent_cfg=json.loads((ROOT/cfg['clean_parent_config']).read_text())
        parent=ROOT/parent_cfg['out'];warm=ROOT/cfg['warm_start']
        if digest(parent/'protocol.json')!=prepared['parent_protocol_sha256'] or digest(parent/'result.json')!=prepared['parent_result_sha256']:
            raise ValueError('Clean parent identity changed')
        parent_protocol=json.loads((parent/'protocol.json').read_text())
        for name,h in parent_protocol['code_sha256'].items():
            if digest(ROOT/name)!=h:raise ValueError('Parent source changed')
        for name,h in prepared['warm_start_files'].items():
            if digest(warm/name)!=h:raise ValueError('Warm checkpoint changed')
        completed_checkpoint_files(warm,200)
        parent_initial=json.loads((parent/'initialization-audit.json').read_text())
        if digest(parent/'initialization-audit.json')!=prepared['parent_initialization_sha256'] or not parent_initial['fresh_lora_b_zero'] or parent_initial['warm_start_requested']:
            raise ValueError('Parent was not initialized from clean Qwen')
        if trainer_config['max_steps']!=cfg['max_steps'] or trainer_config['eval_every_steps']!=cfg['validation_every']:
            raise ValueError('Stage limits differ')
        reference_files=json.loads((ROOT/cfg['reference_protocol']).read_text())['reference_files']
        for name,h in reference_files.items():
            if digest(ROOT/cfg['reference']/name)!=h:raise ValueError('FrozenB changed')
        code=['scripts/train_clean_native_curriculum.py','scripts/prepare_clean_native_curriculum.py','clean_policy_data.py','checkpoint_identity.py',
              'semantic.py','replay_call_repair.py','replay_dora_repair.py','encoder_training_guard.py',
              'vendor/JevAny/jevany/train.py','vendor/JevAny/jevany/model.py','vendor/JevAny/jevany/checkpoint.py']
        protocol={'config':cfg,'code_sha256':{n:digest(ROOT/n) for n in code},
            'trainer_config_sha256':digest(ROOT/cfg['trainer_config']),
            'prepared_provenance_sha256':digest(data/'provenance.json'),'reference_files':reference_files,
            'selection':'minimum actual phase-stratified validation NLL with nonzero exact chi and pon TP in every view',
            'ordinary_reference':'frozen originalB; clean candidate learns discard, riichi, chi/pon/pass and post-call discard',
            'initialization':'continued verified clean-parent LoRA/head; no oldB or invalidated-v1 weights',
            'warm_start_files':prepared['warm_start_files'],'clean_parent_protocol_sha256':prepared['parent_protocol_sha256'],
            'test_opened':False,'deployed':False,'strength_improvement_proven':False}
        write(out/'protocol.json',protocol)
        records=[json.loads(l) for l in (data/'validation.jsonl').read_text().splitlines()]
        from jevany import train as trainer
        import torch
        original_model=trainer.DecisionModel
        original_evaluate=trainer.evaluate_during_training
        original_update=trainer.early_stop_update
        guard=ExactRecallGuard(cfg['views'],0.0)
        curve=[];start=time.monotonic()

        def verify_warm(model):
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            adapter=get_peft_model_state_dict(model.lm)
            expected=load_file(str(warm/'adapter_model.safetensors'),device='cpu')
            if set(adapter)!=set(expected) or any(not torch.equal(adapter[k].detach().cpu(),expected[k]) for k in expected):
                raise ValueError('Loaded adapter differs from registered clean checkpoint')
            expected_head=torch.load(warm/'head.pt',map_location='cpu',weights_only=True)['head']
            actual_head=model.head.state_dict()
            if set(expected_head)!=set(actual_head) or any(not torch.equal(actual_head[k].detach().cpu(),v) for k,v in expected_head.items()):
                raise ValueError('Loaded pointer differs from registered clean checkpoint')
            write(out/'initialization-audit.json',{'clean_parent_fresh_origin':True,'warm_start_exact_adapter_and_head':True,
                'adapter_tensor_count':len(expected),'no_oldB_weights_loaded':True,
                'trainable_parameters':sum(v.numel() for v in model.trainable_parameters())})

        def evaluate(*pos,**kw):
            current_step=kw.get('step',pos[5] if len(pos)>5 else None)
            if current_step==0:verify_warm(kw.get('model',pos[0] if pos else None))
            summary=original_evaluate(*pos,**kw)
            step=kw.get('step',pos[5] if len(pos)>5 else None)
            directory=Path(kw.get('out_dir',pos[4] if len(pos)>4 else ''))
            rows=json.loads((directory/'training_eval'/f'step-{step:06d}'/'development'/'rows.json').read_text())
            metrics=exact_metrics(records,rows);admission=guard.assess(metrics)
            by_phase={}
            for phase in ('closed_discard','open_discard','reaction'):
                subset=[r for r in records if r['_meta']['phase']==phase]
                ids={r['_meta']['id'] for r in subset}
                by_phase[phase]=exact_metrics(subset,[r for r in rows if r['id'] in ids])
            item={'step':step,'nll':summary['clean']['nll'],'views':metrics,
                'by_phase':by_phase,'guard':admission,'seconds':time.monotonic()-start}
            curve.append(item);write(out/f'validation-{step:06d}.json',item);write(out/'validation-curve.json',curve)
            status={'status':'clean_policy_validation','step':step,'max_steps':cfg['max_steps'],
                'nll':item['nll'],'admissible':admission['admissible'],'seconds':item['seconds']}
            write(out/'status.json',status);print(json.dumps(status),flush=True)
            return summary

        def update(value,best,stale,maximize=True,min_delta=0.):
            return guard.update(original_update,value,best,stale,maximize=maximize,min_delta=min_delta)

        trainer.evaluate_during_training=evaluate;trainer.early_stop_update=update
        write(out/'status.json',{'status':'loading_verified_clean_parent_curriculum','step':0,'max_steps':cfg['max_steps']})
        try:
            base=Path(os.environ.get('JEV_BASE_MODEL_DIR',Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
            train_dir=trainer.main(['--config',str(ROOT/cfg['trainer_config']),'--base_load_path',str(base)])
        finally:
            trainer.DecisionModel=original_model;trainer.evaluate_during_training=original_evaluate;trainer.early_stop_update=original_update
        selection=json.loads((train_dir/'selection.json').read_text())
        metrics=json.loads((train_dir/'training_metrics.json').read_text())
        if metrics['global_effective_batch']!=16 or metrics['rejected_records'] or metrics['admitted_records']!=prepared['records']['train']:
            raise ValueError('Token admission or effective batch differs')
        for name,h in protocol['code_sha256'].items():
            if digest(ROOT/name)!=h:raise ValueError('Active training source changed')
        for name,h in reference_files.items():
            if digest(ROOT/cfg['reference']/name)!=h:raise ValueError('FrozenB changed')
        step=selection['step'];selected=selection['selected_checkpoint']
        result={'status':'clean_policy_candidate' if step>0 else 'clean_policy_no_admissible_improvement',
            'optimizer_steps':metrics['optimizer_steps'],'selected_step':step,
            'checkpoint':str((train_dir/selected).resolve().relative_to(ROOT)) if selected else None,
            'validation':next(r for r in curve if r['step']==step),'frozenB_unchanged':True,
            'human_test_opened':False,'deployed':False,'strength_improvement_proven':False,
            'next':'native inference parity then fresh three-opponent-suite full-East net points'}
        write(out/'result.json',result);write(out/'status.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='validation'}),flush=True)
    except Exception as exc:
        out.mkdir(parents=True,exist_ok=True)
        write(out/'training-failure.json',{'type':type(exc).__name__,'error':str(exc),'deployed':False})
        raise


if __name__=='__main__':main()
