"""Postfit normal-serving audit; never promote an in-memory-only LoRA gain."""
import gc
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def audit():
    from tools.train_endgame_fourth_chase import verify
    from tools.train_endgame_fourth_adamw import validate_assets
    from tools.train_endgame_fourth_lora_balanced import validate_config
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.reference import inference_request
    from jevmahjong.endgame.fourth_chase import fourth_chase_action_keys
    from jevmahjong.endgame.fourth_chase_training import chase_validation,chase_guards,chase_training_metrics
    out=Path('runs/endgame-fourth-lora-balanced-v1');jobs=json.loads((out/'jobs-private.json').read_text())
    if (out/'postfit-summary-private.json').exists():raise ValueError('Inspect existing audit; do not restart')
    if json.loads((out/f"fit-job-{jobs['fit']}-exit.json").read_text())['exit_code']!=0:raise ValueError('Fit did not complete successfully')
    registration=json.loads((out/'registered-code-private.json').read_text());verify(registration)
    config=registration['configuration'];validate_config(config)
    ready,parent,train,validation,fourth=validate_assets(config)
    result=json.loads((out/'fit-result-private.json').read_text())
    if result['status']!='fourth_balanced_joint_lora_fit_complete':raise ValueError('Incomplete fit')
    anchor=json.loads((out/'selection-anchor-private.json').read_text())
    if anchor['fourth']['top2']<ready['initial_validation']['fourth']['top2']:raise ValueError('Formal zero anchor was lowered')
    reference=json.loads((Path(config['prepared_data_run'])/'reference-targets-private.json').read_text())['rows']
    for point in result['curve']:
        p=json.loads((out/'training-run/predictions'/f"step-{point['step']:06d}-private.json").read_text())
        metrics=chase_validation(validation,p,reference)
        guards=chase_guards(point['step'],metrics,anchor,ready['source_config'],exposure_met=point['unique_positive_exposure']==826)
        if metrics!=point['metrics'] or guards!=point['guards']:raise ValueError('Logged selection metrics do not reproduce')
    selected=result['selected_step'];last=result['last_step']
    if json.loads((out/'training-run/selection.json').read_text())['step']!=selected:raise ValueError('Selection differs')
    if selected and not next(p for p in result['curve'] if p['step']==selected)['guards']['eligible_nonzero']:
        raise ValueError('In-memory selection was ineligible')
    sys.path.insert(0,ready['source_config']['upstream_source'])
    import torch
    from safetensors.torch import load_file
    from jevany.api import validate_distribution
    from jevmahjong.policy.routing import load_model
    parent_adapter=load_file(str(parent/'adapter_model.safetensors'),device='cpu')
    parent_head=torch.load(parent/'head.pt',map_location='cpu',weights_only=True)['head']
    reports=[]
    for step in sorted({last}|({selected} if selected else set())):
        checkpoint=out/'training-run/checkpoints'/f'step-{step:06d}';before=bundle_identity(checkpoint)
        adapter=load_file(str(checkpoint/'adapter_model.safetensors'),device='cpu')
        meta=torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=True)
        if set(adapter)!=set(parent_adapter) or set(meta['head'])!=set(parent_head):raise ValueError('Saved architecture changed')
        if (not any(not torch.equal(v,parent_adapter[k]) for k,v in adapter.items())
                or not any(not torch.equal(v,parent_head[k]) for k,v in meta['head'].items())
                or meta['optimizer_step']!=step or meta['policy_version']!='endgame-fourth-chase-v1'):
            raise ValueError('Actual LoRA/head update or metadata missing')
        client=load_model(checkpoint);client.runtime.model.eval().requires_grad_(False)
        if client.runtime.inference_options.prefix_cache_size!=0:raise ValueError('Normal inference cache must be disabled')
        actual={}
        for row in validation+fourth:
            keys=list(row['questions']['action']['criteria']);answer=client.system_one(**inference_request(row))
            values,_=validate_distribution(answer['answers']['action']['probabilities'],keys)
            actual[row['_meta']['id']]=dict(zip(keys,values))
        metrics=chase_validation(validation,{r['_meta']['id']:actual[r['_meta']['id']] for r in validation},reference)
        point=next(p for p in result['curve'] if p['step']==step)
        flags=chase_guards(step,metrics,anchor,ready['source_config'],exposure_met=point['unique_positive_exposure']==826)
        training=chase_training_metrics(fourth,{r['_meta']['id']:actual[r['_meta']['id']] for r in fourth},reference)
        old=json.loads((out/'training-run/predictions'/f'step-{step:06d}-private.json').read_text())
        difference=max(abs(actual[r['_meta']['id']][k]-old[r['_meta']['id']][k])
                       for r in validation for k in r['questions']['action']['criteria'])
        flips=sum(max(fourth_chase_action_keys(r),key=actual[r['_meta']['id']].get)
                  !=max(fourth_chase_action_keys(r),key=old[r['_meta']['id']].get) for r in validation)
        # BF16 unmerged training and normal loaded/merged service may differ.
        # Report that difference; qualification uses ACTUAL service metrics.
        reports.append(dict(step=step,normal_service_queries=335,metrics=metrics,guards=flags,training149=training,
            probability_max_difference_vs_unmerged=difference,choice_flips_vs_unmerged=flips,
            eligible_selected=step==selected and selected>0 and flags['eligible_nonzero']))
        write(out/f'normal-service-step-{step:06d}-private.json',actual)
        if bundle_identity(checkpoint)!=before:raise ValueError('Normal serving changed model files')
        del client;gc.collect();torch.cuda.empty_cache()
    verify(registration);validate_assets(config)
    summary=dict(status='fourth_lora_normal_service_audit_complete',last_step=last,selected_step=selected,
        audits=reports,eligible_for_consumed_development=any(r['eligible_selected'] for r in reports),
        actual_existing_lora_and_head_updates_verified=True,new_simulations=0,
        independent_confirmation=False,deployed=False,raw_publication=False)
    write(out/'postfit-summary-private.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Checkpoint auditing requires a scheduler GPU allocation')
    audit()
