"""Bind a complete fresh-LoRA checkpoint rather than a frozen-encoder head."""
import hashlib
import json
import math
from pathlib import Path
from jevmahjong.core.io import digest

MODEL_FILES=('adapter_model.safetensors','adapter_config.json','head.pt',
    'tokenizer.json','tokenizer_config.json','chat_template.jinja')
FORMAT_FILES=('data/base_lora_pool.py','policy/call_features.py','policy/call_helpers.py',
    'policy/routing.py','core/state.py','training/base_lora.py','training/initialization.py')


def read(path):return json.loads(Path(path).read_text())


def bundle(directory):
    root=Path(directory).resolve();files={name:digest(root/name) for name in MODEL_FILES}
    identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(directory=str(root),files=files,identity_sha256=identity)


def verify_files(files):
    for name,checksum in files.items():
        if digest(Path(name))!=checksum:raise ValueError('Bound input changed: '+name)


def checked_candidate(run):
    """Recompute selection arithmetic and verify the selected complete bundle."""
    import torch
    from jevmahjong.training.base_lora import package_digest,validation_profile
    root=Path(run).resolve();fit=read(root/'result.json');protocol=read(root/'protocol.json');cfg=protocol['config']
    warm=cfg.get('initialization','fresh')=='continued'
    status='continued_base_LoRA_training_complete' if warm else 'fresh_base_LoRA_training_complete'
    if (fit['status']!=status or fit['model_id']!=cfg['model_id']
            or not fit['actual_update_proof'] or fit['old_adapters_loaded']!=warm or fit['external_residual_networks']):
        raise ValueError('Completed explicitly initialized unified LoRA required')
    verify_files(protocol['input_sha256']);verify_files(protocol['code_sha256'])
    if package_digest(Path(cfg['upstream_source'])/'jevany')!=cfg['upstream_source_sha256']:
        raise ValueError('Training upstream package changed')
    package=Path(__file__).resolve().parents[1]
    old={name.split('/jevmahjong/',1)[1]:h for name,h in protocol['code_sha256'].items()}
    for name in FORMAT_FILES:
        if digest(package/name)!=old.get(name):raise ValueError('Training/serving format or loss differs: '+name)
    initialization=read(root/'initialization-audit.json');admission=read(root/'token-admission.json');proof=read(root/'actual-update-proof.json')
    from jevmahjong.training.initialization import bind_parent,check_initialization
    parent=bind_parent(cfg,protocol['trainer_args']) if warm else None
    if warm and (read(root/'parent-initialization-private.json')!=parent or protocol.get('parent_initialization')!=parent):
        raise ValueError('Recorded parent initialization differs')
    check_initialization(cfg,fit,initialization,proof,parent)
    if admission['rejected_records'] or admission['admitted_records']!=admission['requested_records']:
        raise ValueError('Token admission unverified')
    trainer=Path(protocol['trainer_args']['out']).resolve();selection=read(trainer/'selection.json')
    if warm:
        from safetensors.torch import load_file
        from jevmahjong.training.initialization import exact_loaded_tensors
        initial=trainer/'checkpoints/step-000000'
        exact_loaded_tensors(load_file(str(initial/'adapter_model.safetensors'),device='cpu'),
            torch.load(initial/'head.pt',map_location='cpu',weights_only=True)['head'],parent['model']['directory'])
    selected=(trainer/selection['selected_checkpoint']).resolve()
    if not selected.is_relative_to(trainer/'checkpoints') or str(selected)!=fit['selected_checkpoint']:
        raise ValueError('Selected complete checkpoint route differs')
    if selection['step']!=fit['selected_step']:raise ValueError('Selected step differs')
    records=[json.loads(line) for line in Path(protocol['trainer_args']['eval_data']).read_text().splitlines() if line.strip()]
    curve=read(root/'validation-curve.json')
    if curve!=fit['validation_curve'] or not curve or curve[0]['step']!=0:
        raise ValueError('Complete frozen validation curve required')
    recomputed=[]
    for item in curve:
        rows=read(trainer/'training_eval'/f"step-{item['step']:06d}"/'development/rows.json')
        actual=validation_profile(records,rows,cfg['validation_phase_weights'])
        if any(actual[k]!=item[k] for k in actual):raise ValueError('Recorded validation target metric differs')
        recomputed.append(dict(step=item['step'],**actual))
    best=min(recomputed,key=lambda v:v['weighted_target_ce'])
    if best['step']!=selection['step'] or abs(best['weighted_target_ce']-selection['value'])>1e-12:
        raise ValueError('Selected checkpoint is not the registered validation minimum')
    model=bundle(selected);meta=torch.load(selected/'head.pt',map_location='cpu',weights_only=True);adapter=read(selected/'adapter_config.json')
    if (meta['base']!=cfg['base'] or meta.get('base_revision')!=cfg['base_revision']
            or meta['lora']!=protocol['trainer_args']['lora'] or adapter['r']!=meta['lora']
            or adapter['base_model_name_or_path']!=cfg['base'] or meta.get('head_residual_dim',0)!=0):
        raise ValueError('Selected new adapter/head base identity differs')
    files={str(root/name):digest(root/name) for name in ('result.json','protocol.json','initialization-audit.json','token-admission.json','actual-update-proof.json','validation-curve.json')}
    files[str(trainer/'selection.json')]=digest(trainer/'selection.json')
    if warm:files[str(root/'parent-initialization-private.json')]=digest(root/'parent-initialization-private.json')
    files.update({str(selected/name):h for name,h in model['files'].items()})
    return dict(model_id=fit['model_id'],initialization_mode='continued' if warm else 'fresh',model=model,training_config=cfg,training_protocol=protocol,
        validation_records=records,selected_validation=best,selected_step=best['step'],receipt=files,
        selected_validation_rows=read(trainer/'training_eval'/f"step-{best['step']:06d}"/'development/rows.json'),
        eligible_for_development=best['step']>0 and best['weighted_target_ce']<recomputed[0]['weighted_target_ce'])


def assert_same_candidate(protocol,candidate):
    if protocol['candidate_bundle']!=candidate['model'] or protocol['fit_receipt']!=candidate['receipt']:
        raise ValueError('Evaluation used a different unified adapter or head')
    verify_files(protocol['fit_receipt'])


def check_recipes(preflight,development,confirmation):
    """Check candidate/reference/objective links before any model is selected."""
    shared=('candidate_run','candidate_model_id','candidate_selection','reference','confirmation_objective','max_dealin_rate_increase','fast_gamma')
    if (not isinstance(development.get('candidate_model_id'),str) or not development['candidate_model_id']
            or development.get('confirmation_objective')!='hand_win'
            or not math.isfinite(development['max_dealin_rate_increase'])
            or not 0<=development['max_dealin_rate_increase']<=1
            or not 0<development['fast_gamma']<1):raise ValueError('Explicit unified candidate and valid win/risk objective required')
    if (preflight.get('preflight_only') is not True or development.get('preflight_only',False)
            or confirmation.get('preflight_only',False) or preflight['stage']!='development'
            or development['stage']!='development' or confirmation['stage']!='independent_confirmation'
            or any(p.get(k)!=development.get(k) for p in (preflight,confirmation) for k in shared)):
        raise ValueError('Unified evaluation identities or stages differ')
    if (confirmation.get('development_out')!=development['out']
            or any(c.get('required_native_preflight')!=str(Path(preflight['out'])/'native-preflight.json') for c in (development,confirmation))
            or len({p['out'] for p in (preflight,development,confirmation)})!=3):
        raise ValueError('Unified preflight/development/confirmation route differs')
    ranges=[]
    for c in (development,confirmation):
        if c.get('reference_cache_out') or not isinstance(c['seeds'],int) or c['seeds']<2:
            raise ValueError('Fresh independent multi-wall reference games required')
        ranges.append((c['seed_start'],c['seed_start']+c['seeds']))
    if max(ranges[0][0],ranges[1][0])<min(ranges[0][1],ranges[1][1]):
        raise ValueError('Development/confirmation wall overlap')
    return dict(status='passed_unified_recipe_links',ranges=ranges,actual_weights_verified=False)
