"""Early-stop current3A and continue the best available historical checkpoint."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_yaku_best3a_v1 as inherited
from tools import curriculum_yaku_canonical_v2 as canonical
from tools.curriculum_transfer import read,sha,canonical_protocol
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_parent import rows,require_replay
from tools.curriculum_progression_parent_v1 import extend
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,fields,terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

RUN=ROOT/'runs/curriculum-yaku-early-v2'
ENTRY=ROOT/'cluster/curriculum-yaku-early-v2.slurm'
CONFIG=ROOT/'configs/training/curriculum-yaku-early-v2.json'
PROSPECTIVE=ROOT/'runs/curriculum-early3B-candidates-v2'
CATALOGUE=PROSPECTIVE/'catalogue-private.json'
PREPARED=PROSPECTIVE/'entry-binding-v2-private.json'
QUEUE=PROSPECTIVE/'queue-v2-private.json'
METHOD='user_authorized_early_stop_best_historical_common16_direct3B_v2'
STATUS='actual_best_available_historical3A_checkpoint_selected_for_direct3B'
PUBLIC=('tools/curriculum_yaku_early_v2.py','cluster/curriculum-yaku-early-v2.slurm',
    'configs/training/curriculum-yaku-early-v2.json','docs/experiments/plans/curriculum-early3A-direct3B-20261125.md')


def verify(path):
    value=read(path)
    if any(sha(name)!=expected for name,expected in value['files'].items()):raise ValueError('Continuation input changed')
    return value


def best(candidates):
    return max(candidates,key=lambda row:(row['common16']['top2_rate'],row['common16']['net_points'],-row['step'],row['id']))


def verify_parent(cfg):
    spec=cfg['parent_initialization'];path=Path(spec['manifest']);sealed=read(path);candidate=sealed['selected_candidate']
    if (cfg.get('initialization')!='continued' or spec['method']!=METHOD or sealed['method']!=METHOD
            or sealed['status']!=STATUS or sha(path)!=spec['sha256']
            or spec['selected_step']!=candidate['step'] or sealed['selected_step']!=candidate['step']
            or spec['identity_sha256']!=sealed['model']['identity_sha256']
            or spec['policy_identity_sha256']!=sealed['policy_identity']
            or sealed['user_authorized_early_stop_direct3B'] is not True
            or sealed['stage3A_grade_blocks_3B'] is not False
            or best(read(CATALOGUE)['candidates'])['id']!=candidate['id']):
        raise ValueError('Exact actual early-stop selected parent and explicit direct3B authorization required')
    if any(sha(name)!=expected for name,expected in sealed['bound_files'].items()):raise ValueError('Actual sealed parent source changed')
    parent=read(sealed['parent_registration']);sys.path[:0]=[parent['source'],parent['training_config']['upstream_source']]
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest
    import tomllib
    args=tomllib.loads(Path(cfg['trainer_config']).read_text());data=read(sealed['child_data_ready'])
    if (bundle(sealed['model']['directory'])!=sealed['model'] or cfg['loss_method']!=inherited.old_register.METHOD
            or cfg['base']!=sealed['base'] or cfg['base_revision']!=sealed['base_revision']
            or spec['source']!=parent['source'] or spec['source_sha256']!=parent['source_sha256']
            or package_digest(Path(parent['source'])/'jevmahjong')!=parent['source_sha256']
            or args.get('resume') or Path(args['init_from']).resolve()!=Path(sealed['model']['directory']).resolve()
            or cfg['out']==parent['curriculum']['out']):raise ValueError('Exact saved adapter/head and fresh child optimizer required')
    for split,key in (('train','data'),('validation','eval_data')):
        if Path(args[key]).resolve()!=(ROOT/data['files'][split]['path']).resolve():raise ValueError('Prepared3B data changed')
    protocol=canonical_protocol(read(Path(candidate['fit'])/'protocol.json'))
    receipt=dict(sealed['bound_files']);receipt[str(path)]=sha(path)
    return dict(model=sealed['model'],receipt=receipt,training_protocol=protocol,
        training_config=parent['training_config'],selected_step=candidate['step'])


@contextmanager
def parent_loader(initialization):
    before=initialization.verified_parent;initialization.verified_parent=verify_parent
    try:yield
    finally:initialization.verified_parent=before


def check_job(role):
    resource();verify(PREPARED);queue=read(QUEUE)
    if str(queue[role+'_job'])!=os.environ['SLURM_JOB_ID']:raise ValueError('Exact early3B phase job required')
    return queue


def prepare():
    check_job('prepare');catalogue=read(CATALOGUE);candidate=best(catalogue['candidates'])
    if candidate['id']!=catalogue['selected']:raise ValueError('Historical comparison choice changed')
    parent=read(candidate['parent_registration']);pc=parent['curriculum'];fit=Path(candidate['fit']);step=candidate['step']
    files=dict(parent['inputs']);files.update(verify(PREPARED)['files'])
    if any(sha(name)!=value for name,value in files.items()):raise ValueError('Selected parent inputs changed')
    sys.path[:0]=[parent['source'],parent['training_config']['upstream_source']]
    from jevmahjong.evaluation.unified_contract import bundle
    import tomllib
    model=bundle(fit/'training-run/checkpoints'/f'step-{step:06d}')
    if model['identity_sha256']!=candidate['model_identity']:raise ValueError('Actual saved historical weights differ')
    RUN.mkdir()
    baseline=inherited.parent64
    make_cache=FunctionType(baseline.__code__,dict(baseline.__globals__,RUN=RUN),
        baseline.__name__,baseline.__defaults__,baseline.__closure__)
    cache=make_cache(parent,step,model,candidate['contract'])
    c=read(CONFIG);data_path=ROOT/c['data_ready'];data=read(data_path)
    args=tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    for split,key in (('train','data'),('validation','eval_data')):
        path=ROOT/data['files'][split]['path']
        if sha(path)!=data['files'][split]['sha256']:raise ValueError('Prepared3B data changed')
        require_replay(rows(args[key]),rows(path))
    for path in (data_path.parent/'completion-binding-private.json',
        (ROOT/c['objective_preflight']).parent/'completion-binding-private.json',
        (ROOT/c['validation_adapter_preflight']).parent/'completion-binding-private.json',
        (ROOT/c['epoch_recipe']).parent/'completion-binding-private.json'):extend(files,path)
    paths=[Path(candidate['parent_registration']),fit/'protocol.json',fit/'policy-contracts-private.json',
        fit/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json',CATALOGUE,PREPARED,QUEUE]
    paths += [Path(model['directory'])/name for name in model['files']]
    paths += [cache/name for name in ('protocol.json','summary.json','matches-private.jsonl')]
    for path in paths:files[str(path)]=sha(path)
    sealed=dict(status=STATUS,method=METHOD,selected_step=step,selected_candidate=candidate,model=model,
        policy=candidate['contract']['policy'],policy_identity=candidate['contract']['policy_identity'],
        parent_registration=candidate['parent_registration'],parent_evidence=str(Path(candidate['run'])/'final-evidence-private.json'),
        parent_evaluation_cache=str(cache),original_guarded_evaluation_cache=pc['original_guarded_evaluation_cache'],
        original_guarded_identity=pc['parent_identity'],child_data_ready=str(data_path),base=pc['base'],base_revision=pc['base_revision'],
        bound_files=files,stage3A_grade_blocks_3B=False,user_authorized_early_stop_direct3B=True,
        raw_grade_overridden=False,optimizer_resumed=False,selection_from_minimum_CE=False)
    write(RUN/'parent-seal-private.json',sealed)
    inherited.function(inherited.old_register.register,step,PARENT_METHOD=METHOD,STATUS=STATUS,verify_parent=verify_parent,
        PUBLIC_FILES=(*inherited.old_register.PUBLIC_FILES,*PUBLIC))(CONFIG,RUN/'parent-seal-private.json')
    registered=RUN/'fit-plan/registered.json';reg=read(registered);launch=dict(reg['inputs']);launch[str(registered)]=sha(registered)
    write(RUN/'launch-binding-private.json',dict(files=launch,registered_sha256=sha(registered),
        actual_registration_before_first_update=True,user_authorized_early_stop_direct3B=True))
    print(dict(status='early_stopped3A_best_historical_parent_registered_for3B',selected=candidate['id'],optimizer_steps=2500),flush=True)


def save_terminal(role,queue):
    job=str(queue[role+'_job']);actual=scheduler(job);receipt=read(ROOT/'logs'/f'curriculum-yaku-early-v2-exit-{job}.json')
    if actual.returncode or receipt.get('phase')!=role:raise ValueError('Actual successful role terminal required')
    scheduler_exit(receipt,actual.stdout,ENTRY)
    write(RUN/(role+'-terminal-scheduler-private.json'),dict(raw=actual.stdout,exit=receipt))


def fit():
    queue=check_job('fit');save_terminal('prepare',queue);candidate=best(read(CATALOGUE)['candidates'])
    inherited.function(inherited.old_train.train,candidate['step'],verify_parent=verify_parent,parent_loader=parent_loader)(RUN/'fit-plan/registered.json')


def audit():
    queue=check_job('audit');save_terminal('fit',queue);candidate=best(read(CATALOGUE)['candidates'])
    inherited.function(inherited.old_audit.audit,candidate['step'],verify_parent=verify_parent)(RUN/'fit-plan/registered.json',RUN/'final-evidence-private.json')


def capture():
    queue=check_job('capture');out=(RUN if RUN.exists() else PROSPECTIVE)/'terminal-v2';out.mkdir();proofs={}
    for role in ('prepare','fit','audit'):
        job=str(queue[role+'_job']);actual=scheduler(job);raw=actual.stdout
        if actual.returncode:
            path=RUN/(role+'-terminal-scheduler-private.json')
            if not path.exists():raise ValueError('Actual role terminal unavailable')
            raw=read(path)['raw']
        record=terminal_record(raw,job,ENTRY);path=ROOT/'logs'/f'curriculum-yaku-early-v2-exit-{job}.json'
        receipt=read(path) if path.exists() else None
        if record['successful']:scheduler_exit(receipt,raw,ENTRY)
        record['exit_receipt']=receipt;proofs[role]=record
    write(out/'terminal-evidence-private.json',dict(status='early_direct3B_actual_terminals_preserved',scheduler_proofs=proofs,
        automatic_release=False,final_confirmation_opened=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,choices=('prepare','fit','audit','capture'))
    args=parser.parse_args();globals()[args.phase]()
