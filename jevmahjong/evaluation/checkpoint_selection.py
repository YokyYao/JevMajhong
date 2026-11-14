"""Explicit pre-game declaration of one completed final checkpoint.

The training run's CE selection is preserved. This diagnostic selector is
available for CE-selected zero, or an exact final budget registered in the
training configuration. No full match may already have tested the fit.
Game promotion still uses the unchanged own-win/risk/confirmation protocol.
"""
import datetime
import json
import os
from pathlib import Path
import torch

from jevmahjong.core.io import digest
from jevmahjong.training.base_lora import validation_profile
from .unified_contract import checked_candidate,bundle,read,verify_files

METHOD='registered_final_training_checkpoint'
SCHEMA='final-trained-checkpoint-probe-v1'


def same_adapter_architecture(before,after):
    """PEFT serializes its unordered target-module set in process-specific order."""
    def canonical(path):
        cfg=read(path);modules=cfg.get('target_modules')
        if isinstance(modules,list):
            if any(not isinstance(name,str) for name in modules) or len(set(modules))!=len(modules):
                raise ValueError('Invalid adapter target module set')
            cfg['target_modules']=sorted(modules)
        return cfg
    return canonical(before)==canonical(after)


def _final_candidate(run,base):
    root=Path(run).resolve();fit=read(root/'result.json')
    args=base['training_protocol']['trainer_args'];trainer=Path(args['out']).resolve()
    metrics_file=trainer/'training_metrics.json';metrics=read(metrics_file)
    step=fit['optimizer_steps']
    cfg=base['training_config'];rule=cfg.get('evaluation_candidate_rule');predeclared=rule=='final_registered_budget_v1'
    if rule is not None and not predeclared:raise ValueError('Unknown predeclared evaluation rule')
    if predeclared and (isinstance(cfg.get('registered_final_step'),bool)
            or not isinstance(cfg.get('registered_final_step'),int) or cfg['registered_final_step']!=step
            or args.get('max_steps')!=step or args.get('early_stop_patience',0)!=0):
        raise ValueError('Completed final step differs from the registered fixed training budget')
    if (not predeclared and base['selected_step']!=0) or not isinstance(step,int) or step<=0 or metrics['optimizer_steps']!=step:
        raise ValueError('Only an untested fit with CE-selected zero and a completed final step may be declared')
    curve=read(root/'validation-curve.json')
    if not curve or curve[-1]['step']!=step or max(r['step'] for r in curve)!=step:
        raise ValueError('Final step differs from the completed validation trace')
    final=trainer/'checkpoints'/f'step-{step:06d}'
    model=bundle(final);initial=trainer/'checkpoints/step-000000' if predeclared else Path(base['model']['directory'])
    initial_model=bundle(initial)
    if not same_adapter_architecture(final/'adapter_config.json',initial/'adapter_config.json'):
        raise ValueError('Final checkpoint changed adapter architecture')
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        if model['files'][name]!=initial_model['files'][name]:
            raise ValueError('Final checkpoint changed adapter architecture or tokenization')
    from safetensors.torch import load_file
    old=load_file(str(initial/'adapter_model.safetensors'),device='cpu')
    new=load_file(str(final/'adapter_model.safetensors'),device='cpu')
    if set(old)!=set(new) or any(old[k].shape!=new[k].shape for k in old):
        raise ValueError('Final adapter tensor coverage differs')
    if not any(not torch.equal(old[k],new[k]) for k in old):
        raise ValueError('Final adapter is unchanged')
    before=torch.load(initial/'head.pt',map_location='cpu',weights_only=True)
    after=torch.load(final/'head.pt',map_location='cpu',weights_only=True)
    fields=('base','base_revision','lora','head_dim','head_residual_dim','head_type',
        'option_isolation','special_embeddings','multimodal','backbone_adapter','branch_mode','decision_mode')
    if any(before.get(k)!=after.get(k) for k in fields) or after.get('head_residual_dim',0)!=0:
        raise ValueError('Final pointer architecture differs')
    a,b=before['head'],after['head']
    if set(a)!=set(b) or any(a[k].shape!=b[k].shape for k in a) or not any(not torch.equal(a[k],b[k]) for k in a):
        raise ValueError('Final pointer is missing, changed shape or unchanged')
    rows_file=trainer/'training_eval'/f'step-{step:06d}'/'development/rows.json'
    rows=read(rows_file);actual=validation_profile(base['validation_records'],rows,base['training_config']['validation_phase_weights'])
    recorded=[r for r in curve if r['step']==step]
    if len(recorded)!=1 or any(recorded[0][k]!=v for k,v in actual.items()):
        raise ValueError('Final validation prediction profile differs')
    files=dict(base['receipt']);files.update({str(final/name):value for name,value in model['files'].items()})
    if predeclared:files.update({str(initial/name):value for name,value in initial_model['files'].items()})
    files.update({str(metrics_file):digest(metrics_file),str(rows_file):digest(rows_file)})
    declaration=dict(schema=SCHEMA,method=METHOD,run=str(root),step=step,
        training_validation_selected_step=base['selected_step'],model=model,inputs=files,
        validation_profile=actual,selection_code_sha256=digest(Path(__file__)))
    return declaration,rows


def declare_final_candidate(run,target,runtime_roots):
    """Write once, before any game using this fit; never alter training outputs."""
    target=Path(target)
    if target.exists():raise ValueError('Final candidate declaration already exists')
    if not runtime_roots:raise ValueError('Actual runtime roots are required')
    base=checked_candidate(run);declaration,_=_final_candidate(run,base);checked=0
    for directory in runtime_roots:
        directory=Path(directory).resolve()
        if not directory.is_dir():raise ValueError('Runtime root missing')
        for path in directory.rglob('protocol.json'):
            checked+=1;cfg=read(path).get('config',{});candidate=cfg.get('candidate_run')
            if not isinstance(candidate,str):continue
            candidate=Path(candidate)
            if not candidate.is_absolute():candidate=directory.parent/candidate
            games=path.parent/'games-private.jsonl'
            if candidate.resolve()==Path(run).resolve() and games.exists() and games.stat().st_size:
                raise ValueError('This fit already has full-match evidence; no retrospective checkpoint selection')
    declaration.update(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        runtime_protocols_checked=checked,no_prior_full_matches_for_fit=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream:
        json.dump(declaration,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return declaration


def candidate_for_evaluation(cfg):
    base=checked_candidate(cfg['candidate_run']);spec=cfg.get('candidate_selection')
    if spec is None:return base
    if set(spec)!={'method','manifest','sha256'} or spec['method']!=METHOD:
        raise ValueError('Explicit registered final-checkpoint method required')
    manifest=Path(spec['manifest'])
    if digest(manifest)!=spec['sha256']:raise ValueError('Candidate declaration changed')
    declared=read(manifest);actual,rows=_final_candidate(cfg['candidate_run'],base)
    if any(declared.get(k)!=v for k,v in actual.items()) or declared.get('no_prior_full_matches_for_fit') is not True:
        raise ValueError('Declared final candidate changed')
    verify_files(declared['inputs'])
    files=dict(declared['inputs']);files[str(manifest.resolve())]=digest(manifest)
    return dict(base,model=declared['model'],selected_step=declared['step'],
        selected_validation=dict(step=declared['step'],**declared['validation_profile']),
        selected_validation_rows=rows,receipt=files,eligible_for_development=True,
        selection_method=METHOD,training_validation_selected_step=declared['training_validation_selected_step'])
