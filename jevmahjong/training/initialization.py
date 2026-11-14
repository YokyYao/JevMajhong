"""Bind continued LoRA training to a complete, independently checked parent."""
import json
from pathlib import Path
import subprocess
import sys
import torch

from jevmahjong.core.io import digest


def verified_parent(cfg):
    """An older run is checked with its own immutable verifier, never this one."""
    from .base_lora import package_digest
    spec = cfg['parent_initialization']; source = Path(spec['source']).resolve()
    if package_digest(source/'jevmahjong') != spec['source_sha256']:
        raise ValueError('Parent verifier source changed')
    if Path(spec['run']).resolve() == Path(cfg['out']).resolve():
        raise ValueError('Parent cannot be the new output')
    program = ('import sys,json;sys.path.insert(0,sys.argv[1]);'
        'from jevmahjong.evaluation.unified_contract import checked_candidate;'
        'v=checked_candidate(sys.argv[2]);'
        'print(json.dumps({k:v[k] for k in ("model","receipt","training_protocol","training_config","selected_step")}))')
    process = subprocess.run([sys.executable,'-c',program,str(source),spec['run']],
        text=True,capture_output=True,timeout=60)
    if process.returncode:
        raise ValueError('Pinned parent verification failed: '+process.stderr[-1500:])
    parent = json.loads(process.stdout)
    if (parent['model']['identity_sha256'] != spec['identity_sha256']
            or parent['selected_step'] != spec['selected_step']
            or parent['training_config'].get('initialization','fresh') != 'fresh'
            or parent['training_config']['base'] != cfg['base']
            or parent['training_config']['base_revision'] != cfg['base_revision']):
        raise ValueError('Expected complete fresh-base parent identity differs')
    return parent


def check_groups(parent_train,train,validation):
    def groups(file):
        with Path(file).open() as stream:
            values = {json.loads(line)['_meta']['group_id'] for line in stream if line.strip()}
        if not values:raise ValueError('Nonempty training and validation groups required')
        return values
    ancestor,current,held = groups(parent_train),groups(train),groups(validation)
    if held & (ancestor | current):
        raise ValueError('Validation overlaps parent or current training groups')
    return dict(parent_train_groups=len(ancestor),train_groups=len(current),
        validation_groups=len(held),parent_validation_overlap=0,train_validation_overlap=0)


def bind_parent(cfg,args):
    if args.get('resume'):raise ValueError('Continuation uses new optimizer; resume is forbidden')
    parent = verified_parent(cfg)
    directory = Path(parent['model']['directory']).resolve()
    if Path(args.get('init_from','')).resolve() != directory:
        raise ValueError('init_from must be the verified complete parent checkpoint')
    protocol = parent['training_protocol']; files = dict(parent['receipt'])
    for receipt in (protocol['input_sha256'],protocol['code_sha256']):
        for name,checksum in receipt.items():
            if name in files and files[name] != checksum:raise ValueError('Conflicting parent file identity')
            files[name] = checksum
    group_proof = check_groups(protocol['trainer_args']['data'],args['data'],args['eval_data'])
    return dict(model=parent['model'],selected_step=parent['selected_step'],files=files,
        group_proof=group_proof,parent_training_file=protocol['trainer_args']['data'])


def exact_loaded_tensors(adapters,head,directory):
    """Check every tensor after upstream warm_start, not only key counts."""
    from safetensors.torch import load_file
    directory = Path(directory)
    expected = load_file(str(directory/'adapter_model.safetensors'),device='cpu')
    expected_head = torch.load(directory/'head.pt',map_location='cpu',weights_only=True)['head']
    for actual,saved in ((adapters,expected),(head,expected_head)):
        if set(actual) != set(saved) or any(not torch.equal(actual[k].detach().cpu(),saved[k]) for k in saved):
            raise ValueError('Loaded parent adapter/head tensors differ')
    return dict(parent_adapter_exact=True,parent_head_exact=True,adapter_tensors=len(expected),head_tensors=len(expected_head))


def check_initialization(cfg,fit,audit,proof,parent_binding=None):
    """Shared by training receipts and serving admission; no fitness shortcuts."""
    warm = cfg.get('initialization','fresh') == 'continued'
    if cfg.get('initialization','fresh') not in ('fresh','continued'):
        raise ValueError('Unknown LoRA initialization mode')
    expected_status = 'continued_base_LoRA_training_complete' if warm else 'fresh_base_LoRA_training_complete'
    common = (fit['status']==expected_status and fit['old_adapters_loaded']==warm
        and fit.get('pretrained_base_frozen') is True
        and not fit['external_residual_networks'] and not audit['external_residual_networks']
        and audit['old_adapters_loaded']==warm and not audit['pretrained_base_requires_grad']
        and proof['head_parameters_changed'])
    if warm:
        spec = cfg['parent_initialization']
        valid = (parent_binding is not None and audit.get('initialization')=='continued'
            and audit.get('parent_adapter_exact') is True and audit.get('parent_head_exact') is True
            and audit.get('parent_identity_sha256')==spec['identity_sha256']
            and parent_binding['model']['identity_sha256']==spec['identity_sha256']
            and audit.get('parent_group_proof')==parent_binding['group_proof']
            and audit.get('new_optimizer') is True
            and proof.get('continued_LoRA_parameters_changed') is True)
    else:
        valid = (audit.get('fresh_lora_B_zero') is True and audit.get('fresh_lora_A_nonzero') is True
            and proof.get('fresh_LoRA_B_now_nonzero') is True)
    if not (common and valid):raise ValueError('Initialization or actual LoRA/head updates unverified')
