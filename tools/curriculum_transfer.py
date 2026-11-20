"""Bridge an explicitly sealed curriculum final into pinned warm-start checks.

This does not admit a partial/unqualified parent, resume its optimizer, select
by minimum CE, or change the frozen research implementation on disk.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tomllib

ROOT=Path(__file__).resolve().parents[1]
METHOD='curriculum_registered_final_with_quality_v1'
RESEARCH_METHOD='curriculum_registered_final_research_v2'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical_protocol(protocol,root=ROOT):
    """Add checked absolute aliases; preserve every original bound file entry."""
    result=copy.deepcopy(protocol)
    for section in ('input_sha256','code_sha256'):
        original=dict(result[section]);resolved={}
        for name,expected in original.items():
            path=Path(name);path=(path if path.is_absolute() else root/path).resolve()
            if sha(path)!=expected:raise ValueError('Parent protocol input changed: '+name)
            key=str(path)
            if key in resolved and resolved[key]!=expected:raise ValueError('Conflicting canonical parent file')
            if key in original and original[key]!=expected:raise ValueError('Conflicting parent path alias')
            resolved[key]=expected
        result[section].update(resolved)
    for key in ('data','eval_data'):
        path=Path(result['trainer_args'][key]);path=(path if path.is_absolute() else root/path).resolve()
        expected=result['input_sha256'].get(str(path))
        if expected is None or sha(path)!=expected:raise ValueError('Unbound ancestral dataset: '+key)
        result['trainer_args'][key]=str(path)
    return result


def verify_curriculum_parent(cfg):
    from tools.curriculum_parent import require_stage1_candidate,require_child_input_contract
    spec=cfg['parent_initialization']
    required={'method','manifest','sha256','identity_sha256','selected_step','source','source_sha256','run','child_data_ready'}
    if set(spec)!=required or spec['method'] not in (METHOD,RESEARCH_METHOD) or cfg.get('initialization')!='continued':
        raise ValueError('Explicit sealed curriculum continuation specification required')
    seal_path=Path(spec['manifest']).resolve()
    if sha(seal_path)!=spec['sha256']:raise ValueError('Parent seal changed')
    seal=read(seal_path)
    if (seal.get('status')!='curriculum_stage1_final_parent_sealed' or seal.get('method')!=spec['method']
        or seal.get('selection_from_minimum_CE') is not False or seal.get('optimizer_resumed') is not False
        or not seal.get('stage_quality',{}).get('passed')
        or seal['selected_step']!=spec['selected_step']
        or seal['model']['identity_sha256']!=spec['identity_sha256']
        or seal['base']!=cfg['base'] or seal['base_revision']!=cfg['base_revision']):
        raise ValueError('Parent is not the qualified registered final curriculum checkpoint')
    bound=seal['bound_files']
    for path,expected in bound.items():
        if sha(path)!=expected:raise ValueError('Sealed parent input changed: '+path)
    reg_path=Path(seal['parent_registration']).resolve();reg=read(reg_path)
    if (bound.get(str(reg_path))!=sha(reg_path) or Path(reg['source']).resolve()!=Path(spec['source']).resolve()
        or reg['source_sha256']!=spec['source_sha256'] or reg['registered_final_step']!=seal['selected_step']
        or Path(spec['run']).resolve()!=(ROOT/reg['curriculum']['out']).resolve()
        or Path(spec['run']).resolve()==Path(cfg['out']).resolve()):
        raise ValueError('Sealed parent source, run or registration differs')
    evidence=Path(seal['parent_evidence']).resolve();fit=Path(spec['run']).resolve()
    validation=fit/'metrics/validation_metrics.jsonl'
    for p in (evidence,fit/'completion.json',validation,fit/'protocol.json'):
        if bound.get(str(p))!=sha(p):raise ValueError('Required audited parent artifact is unbound')
    metrics=[json.loads(s) for s in validation.read_text().splitlines() if s]
    if spec['method']==RESEARCH_METHOD:
        from tools.curriculum_research_parent import load_bound_policy,require_research_candidate
        policy_file=Path(seal['research_parent_policy_binding']).resolve()
        if bound.get(str(policy_file))!=sha(policy_file):raise ValueError('Research-policy binding is not sealed')
        policy,policy_files=load_bound_policy(policy_file,reg_path)
        if any(bound.get(p)!=h for p,h in policy_files.items()):raise ValueError('Research-policy evidence is unbound')
        grade=require_research_candidate(reg,read(fit/'completion.json'),read(evidence),metrics[-1],spec['identity_sha256'],policy)
        if grade!=seal['stage_quality'] or seal['original_stage1_advancement']!=read(evidence)['stage1_advancement']:
            raise ValueError('Research grade or preserved original diagnostic differs')
    else:require_stage1_candidate(reg,read(fit/'completion.json'),read(evidence),metrics[-1],spec['identity_sha256'])
    child_path=Path(spec['child_data_ready']).resolve()
    if str(child_path)!=seal.get('child_data_ready') or bound.get(str(child_path))!=sha(child_path):
        raise ValueError('Continuation child differs from the sealed replay dataset')
    child=read(child_path);require_child_input_contract(reg,child)
    if cfg.get('riichi_input_schema')!=reg['curriculum'].get('riichi_input_schema'):
        raise ValueError('Child model-serving input schema differs')
    args=tomllib.loads(Path(cfg['trainer_config']).read_text())
    if args.get('resume') or Path(args.get('init_from','')).resolve()!=Path(seal['model']['directory']).resolve():
        raise ValueError('Use the sealed adapter/head with a new optimizer, never resume')
    for split,key in (('train','data'),('validation','eval_data')):
        expected=(ROOT/child['files'][split]['path']).resolve()
        if Path(args[key]).resolve()!=expected or bound.get(str(expected))!=sha(expected):
            raise ValueError('Actual child trainer data differs from sealed inputs')
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if package_digest(Path(spec['source'])/'jevmahjong')!=spec['source_sha256']:
        raise ValueError('Pinned parent verifier source changed')
    model=bundle(seal['model']['directory'])
    if model!=seal['model']:raise ValueError('Sealed parent model changed')
    if reg['curriculum'].get('riichi_input_schema'):
        from tools.curriculum_input_contract import verify_checkpoint
        contract=verify_checkpoint(model['directory'],reg,bundle)
        if contract['policy_identity_sha256']!=read(evidence)['final_policy_identity_sha256']:
            raise ValueError('Sealed parent public input contract differs')
    original_protocol=read(fit/'protocol.json');protocol=canonical_protocol(original_protocol)
    if protocol['config']!=reg['training_config'] or protocol['trainer_args']['data']!=seal['parent_training_file']:
        raise ValueError('Actual parent training protocol differs from registration')
    receipt=dict(bound);receipt[str(seal_path)]=spec['sha256']
    return dict(model=model,receipt=receipt,training_protocol=protocol,training_config=reg['training_config'],
        selected_step=seal['selected_step'])


@contextmanager
def curriculum_parent_loader(initialization_module):
    """Keep pinned bind_parent, group/lineage and exact tensor-load checks."""
    original=initialization_module.verified_parent
    initialization_module.verified_parent=verify_curriculum_parent
    try:yield
    finally:initialization_module.verified_parent=original
