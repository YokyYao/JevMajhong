"""Verify saved full-source AdamW decisions through normal model serving."""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def audit():
    from tools.train_endgame_fourth_adamw import validate_config, validate_assets, progress_state
    from tools.train_endgame_fourth_chase import verify, read_caches
    from tools.train_endgame_constrained_head import head_predictions
    from tools.audit_endgame_parameter_attribution import control_comparison
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.reference import inference_request
    from jevmahjong.endgame.constrained_head import source_loss_values
    from jevmahjong.endgame.fourth_chase import fourth_chase_action_keys
    from jevmahjong.endgame.fourth_chase_training import chase_validation,chase_guards,chase_training_metrics
    out=Path('runs/endgame-fourth-adamw-v1')
    if (out/'postfit-summary-private.json').exists():raise ValueError('Existing audit; do not silently restart')
    jobs=json.loads((out/'jobs-private.json').read_text())
    if json.loads((out/f"fit-job-{jobs['fit']}-exit.json").read_text())['exit_code']!=0:raise ValueError('Fit failed')
    config=json.loads((ROOT/'configs/training/endgame-fourth-adamw-v1.json').read_text());validate_config(config)
    protocol=json.loads((out/'protocol-private.json').read_text());verify(protocol)
    ready,parent,train,validation,fourth=validate_assets(config)
    result=json.loads((out/'fit-result-private.json').read_text())
    if (result['status']!='fourth_full_source_adamw_fit_complete'
            or json.loads((out/'training-run/selection.json').read_text())['step']!=result['selected_step']):
        raise ValueError('Inconsistent fit/selection state')
    source=Path(config['prepared_data_run'])
    reference=json.loads((source/'reference-targets-private.json').read_text())['rows'];state=None
    concise=[]
    for point in result['curve']:
        step=point['step'];pred=json.loads((out/'training-run/predictions'/f'step-{step:06d}-private.json').read_text())
        metrics=chase_validation(validation,pred,reference)
        flags=chase_guards(step,metrics,ready['initial_validation'],ready['source_config'],exposure_met=step>0)
        state=progress_state(step,metrics,flags,state,config)
        if metrics!=point['metrics'] or flags!=point['guards'] or state!=point['progress']:
            raise ValueError('Saved probabilities do not reproduce metrics, guards and stopping decisions')
        concise.append(dict(step=step,top2=metrics['fourth']['top2'],soft_top2=metrics['fourth']['soft_top2'],
            fourth=metrics['fourth']['fourth'],strict_train_hits=point['training149']['counts']['strict_target_hits'],
            old_expert_ce=metrics['legacy']['original_expert_ce'],
            third=metrics['legacy']['third_pursuit']['candidate'],hold=metrics['legacy']['utility_by_stratum']['hold'],
            failed_common_guards=[k for k,v in flags.items() if k.endswith('_passed') and not v
                                  and k not in ('common_guards_passed','all_guards_passed')]))
    if state['selected_step']!=result['selected_step']:raise ValueError('Wrong selected checkpoint')
    if [u['step'] for u in result['updates']]!=list(range(1,result['last_step']+1)):
        raise ValueError('Missing actual updates')
    if any(u['positive_records']!=826 or u['per_source_monotonic_veto'] is not False for u in result['updates']):
        raise ValueError('Training did not apply the registered full-source update recipe')
    import torch
    from safetensors.torch import load_file
    parent_adapter=load_file(str(parent/'adapter_model.safetensors'),device='cpu')
    parent_head=torch.load(parent/'head.pt',map_location='cpu',weights_only=True)['head']
    initial=torch.cat([v.reshape(-1) for v in parent_head.values()]).double()
    for step in {0,result['last_step'],result['selected_step']}:
        path=out/'training-run/checkpoints'/f'step-{step:06d}'
        adapter=load_file(str(path/'adapter_model.safetensors'),device='cpu')
        if set(adapter)!=set(parent_adapter) or any(not torch.equal(v,parent_adapter[k]) for k,v in adapter.items()):
            raise ValueError('Frozen LoRA changed')
        meta=torch.load(path/'head.pt',map_location='cpu',weights_only=True)
        if (meta['optimizer_step']!=step or meta['policy_version']!='endgame-fourth-chase-v1'
                or set(meta['head'])!=set(parent_head)
                or any(not torch.equal(v,parent_head[k]) for k,v in meta['head'].items())!=(step>0)):
            raise ValueError('Saved head/metadata update mismatch')
        if step==result['last_step']:
            final=torch.cat([meta['head'][k].reshape(-1) for k in parent_head]).double()
            delta=float(torch.linalg.vector_norm(final-initial)/torch.linalg.vector_norm(initial))
            if abs(delta-result['updates'][-1]['relative_head_l2_from_parent'])>1e-9:
                raise ValueError('Saved parameter distance differs from actual update log')
    sys.path.insert(0,ready['source_config']['upstream_source'])
    from jevmahjong.policy.routing import load_model
    from jevany.api import validate_distribution
    train_cache,_=read_caches(config,ready['parent_identity'],train,validation)
    features=train_cache['features']
    for v in features.values():v['decide']=v['decide'].to('cuda');v['options']=v['options'].to('cuda')
    reports=[]
    for step in sorted({result['last_step']}|({result['selected_step']} if result['selected_step']>0 else set())):
        checkpoint=out/'training-run/checkpoints'/f'step-{step:06d}';before=bundle_identity(checkpoint)
        client=load_model(checkpoint);client.runtime.model.eval().requires_grad_(False)
        if client.runtime.inference_options.prefix_cache_size!=0:raise ValueError('Inference prefix cache must be disabled')
        actual={}
        for row in validation+fourth:
            rid=row['_meta']['id'];keys=list(row['questions']['action']['criteria'])
            answer=client.system_one(**inference_request(row))
            values,_=validate_distribution(answer['answers']['action']['probabilities'],keys)
            actual[rid]=dict(zip(keys,values))
        expected=json.loads((out/'training-run/predictions'/f'step-{step:06d}-private.json').read_text())
        expected.update(head_predictions(client.runtime.model.head,fourth,features))
        control=control_comparison(validation+fourth,actual,expected,1e-7)
        flips=sum(max(fourth_chase_action_keys(row),key=actual[row['_meta']['id']].get)
                  !=max(fourth_chase_action_keys(row),key=expected[row['_meta']['id']].get) for row in validation+fourth)
        if not control['passed'] or flips:raise ValueError('Actual serving differs from cached decisions')
        logged=next(p for p in result['curve'] if p['step']==step)
        metrics=chase_validation(validation,{r['_meta']['id']:actual[r['_meta']['id']] for r in validation},reference)
        learning=chase_training_metrics(fourth,{r['_meta']['id']:actual[r['_meta']['id']] for r in fourth},reference)
        if metrics!=logged['metrics'] or learning!=logged['training149']:raise ValueError('Actual serving changed learning metrics')
        if step==result['last_step']:
            losses=source_loss_values(client.runtime.model.head,train,features)
            if any(abs(losses[k]-v)>1e-7 for k,v in result['final_source_ce'].items()):raise ValueError('Final losses differ')
        if bundle_identity(checkpoint)!=before:raise ValueError('Audit mutated model files')
        reports.append(dict(step=step,queries=len(validation)+len(fourth),control=control,new_scope_choice_flips=flips))
        del client;gc.collect();torch.cuda.empty_cache()
    validate_assets(config);verify(protocol)
    summary=dict(status='fourth_adamw_saved_weight_and_service_audit_complete',last_step=result['last_step'],
        selected_step=result['selected_step'],stop_reason=result['stop_reason'],curve=concise,
        saved_lora_identical_to_formal_v8=True,saved_head_distance_from_parent=delta,service_audits=reports,
        initial_source_ce=result['initial_source_ce'],final_source_ce=result['final_source_ce'],
        source_and_parent_unchanged=True,full_positive_passes=len(result['updates']),
        eligible_for_development=result['selected_step']>0,new_simulations=0,independent_confirmation=False,
        deployed=False,raw_publication=False)
    write(out/'postfit-summary-private.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Saved checkpoint auditing requires a scheduled GPU allocation')
    audit()
