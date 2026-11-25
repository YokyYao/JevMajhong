"""User-authorized best trained 3A checkpoint -> direct existing 3B course."""
import argparse
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import CodeType,FunctionType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_ri_balanced_replay_v2 as parent_runtime
from tools import curriculum_yaku_canonical_v2 as previous
from tools import register_curriculum_yaku_balanced_v1 as old_register
from tools import train_curriculum_yaku_balanced_v1 as old_train
from tools import audit_curriculum_yaku_balanced_v1 as old_audit
from tools.curriculum_transfer import read,sha,canonical_protocol
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_parent import rows,require_replay
from tools.curriculum_progression_parent_v1 import extend
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,fields,terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

RUN=ROOT/'runs/curriculum-yaku-best3a-v1'
PROSPECTIVE=ROOT/'runs/curriculum-yaku-best3a-entry-v1'
PREPARED=PROSPECTIVE/'completion-binding-private.json'
QUEUE=PROSPECTIVE/'queue-private.json'
CONFIG=ROOT/'configs/training/curriculum-yaku-best3a-v1.json'
ENTRY=ROOT/'cluster/curriculum-yaku-best3a-v1.slurm'
METHOD='user_authorized_best_common16_top2_then_net_trained3A_to_3B_v1'
STATUS='actual_completed3A_best_checkpoint_selected_for_direct3B'
STEPS=(618,1236,1854,2472)
PUBLIC=('tools/curriculum_yaku_best3a_v1.py','tests/core/test_curriculum_yaku_best3a_v1.py',
    'configs/training/curriculum-yaku-best3a-v1.json','cluster/curriculum-yaku-best3a-v1.slurm',
    'docs/experiments/plans/curriculum-best3A-direct3B-20261125.md')


def verify(path):
    value=read(path)
    if any(sha(name)!=expected for name,expected in value['files'].items()):raise ValueError('Bound continuation source changed')
    return value


def choose(values):
    if set(values)!=set(STEPS):raise ValueError('All four trained checkpoints on one common grid required')
    for point in values.values():
        if (not all(math.isfinite(point[key]) for key in ('top2_rate','net_points'))
                or not 0<=point['top2_rate']<=1 or point['matches']!=16):
            raise ValueError('Finite same16 development summaries required')
    return max(STEPS,key=lambda step:(values[step]['top2_rate'],values[step]['net_points'],-step))


def verify_parent(cfg):
    spec=cfg['parent_initialization'];path=Path(spec['manifest']);sealed=read(path)
    if (spec['method']!=METHOD or sealed['method']!=METHOD or sealed['status']!=STATUS
            or sha(path)!=spec['sha256'] or sealed['selected_step'] not in STEPS
            or spec['selected_step']!=sealed['selected_step']
            or sealed['stage3A_grade_blocks_3B'] is not False
            or sealed['user_authorized_best_checkpoint_direct3B'] is not True
            or spec['identity_sha256']!=sealed['model']['identity_sha256']
            or spec['policy_identity_sha256']!=sealed['policy_identity']
            or cfg['initialization']!='continued' or cfg['loss_method']!=old_register.METHOD
            or cfg['base']!=sealed['base'] or cfg['base_revision']!=sealed['base_revision']):
        raise ValueError('Exact explicitly selected parent identity and direct3B authorization required')
    if choose({int(key):value for key,value in sealed['selection_values'].items()})!=sealed['selected_step']:
        raise ValueError('Actual selected checkpoint differs from the declared common-grid rule')
    if any(sha(name)!=expected for name,expected in sealed['bound_files'].items()):raise ValueError('Selected source changed')
    parent=read(sealed['parent_registration']);parent_runtime.require_registration(parent)
    sys.path[:0]=[parent['source'],parent['training_config']['upstream_source']]
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest
    import tomllib
    args=tomllib.loads(Path(cfg['trainer_config']).read_text());data=read(sealed['child_data_ready'])
    if (bundle(sealed['model']['directory'])!=sealed['model']
            or package_digest(Path(parent['source'])/'jevmahjong')!=parent['source_sha256']
            or spec['source_sha256']!=parent['source_sha256'] or spec['source']!=parent['source']
            or args.get('resume') or Path(args['init_from']).resolve()!=Path(sealed['model']['directory']).resolve()
            or cfg['out']==parent['curriculum']['out']):
        raise ValueError('Actual selected adapter/head and a fresh independent child optimizer required')
    for split,key in (('train','data'),('validation','eval_data')):
        if Path(args[key]).resolve()!=(ROOT/data['files'][split]['path']).resolve():raise ValueError('Child dataset changed')
    protocol=canonical_protocol(read(ROOT/parent['curriculum']['out']/'protocol.json'))
    receipt=dict(sealed['bound_files']);receipt[str(path)]=sha(path)
    return dict(model=sealed['model'],receipt=receipt,training_protocol=protocol,
        training_config=parent['training_config'],selected_step=sealed['selected_step'])


@contextmanager
def parent_loader(initialization):
    before=initialization.verified_parent;initialization.verified_parent=verify_parent
    try:yield
    finally:initialization.verified_parent=before


def selected_code(code,step,registration=False):
    constants=[]
    for value in code.co_consts:
        if isinstance(value,CodeType):value=selected_code(value,step,registration)
        elif isinstance(value,str):value=value.replace('step-002472/',f'step-{step:06d}/')
        elif registration and type(value) is int and value==2472:value=step
        constants.append(value)
    return code.replace(co_consts=tuple(constants))


def function(original,step,**overrides):
    return FunctionType(selected_code(original.__code__,step,original is old_register.register),
        dict(original.__globals__,**overrides),original.__name__,original.__defaults__,original.__closure__)


def parent64(reg,step,model,contract):
    """Keep existing same16 events, add only the other48 development games."""
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    c=reg['curriculum'];fit=ROOT/c['out'];source=fit/'evaluations'/f'step-{step:06d}'
    games=rows(source/'matches-private.jsonl')
    expected={(seed,profile,seat) for seed in range(c['curve_seed_start'],c['curve_seed_start']+8)
        for profile in c['opponent_profiles'] for seat in range(4)}
    if len(games) not in (16,64) or len({(r['seed'],r['profile'],r['seat']) for r in games})!=len(games):
        raise ValueError('Existing selected development games incomplete')
    out=RUN/'selected-parent64';out.mkdir()
    with (out/'matches-private.jsonl').open('x') as stream:
        for row in games:stream.write(json.dumps(row)+'\n')
        if len(games)==16:
            from jevmahjong.policy.routing import load_model
            from jevmahjong.policy.unified import UnifiedPolicyClient
            from tools.curriculum_complete_legal_class_policy_v1 import CompleteLegalClassPolicyAgent
            from tools.evaluate_curriculum import play
            agent=CompleteLegalClassPolicyAgent(UnifiedPolicyClient(load_model(model['directory'])))
            for seed in range(c['curve_seed_start']+2,c['curve_seed_start']+8):
                for profile in c['opponent_profiles']:
                    for seat in range(4):
                        row=play(agent,seed,seat,profile);games.append(row)
                        stream.write(json.dumps(row)+'\n');stream.flush()
    if len(games)!=64 or {(r['seed'],r['profile'],r['seat']) for r in games}!=expected:
        raise ValueError('Selected parent64 must use exactly the existing development grid')
    protocol=dict(read(source/'protocol.json'),checkpoint=model,policy=contract['policy'],
        policy_identity=contract['policy_identity'],groups=8,
        seeds=list(range(c['curve_seed_start'],c['curve_seed_start']+8)),
        existing_same16_events_preserved=True,extra_games_are_development_not_confirmation=True)
    write(out/'protocol.json',protocol)
    summary=aggregate_complete_matches(games);summary.update(checkpoint_sha256=model['identity_sha256'],
        policy_identity=contract['policy_identity'],curve_grid_subset=aggregate_complete_matches(games[:16]))
    write(out/'summary.json',summary)
    return out


def select_and_seal():
    parent=read(parent_runtime.REGISTERED);parent_runtime.require_registration(parent)
    fit=ROOT/parent['curriculum']['out'];done=read(fit/'completion.json')
    if done['optimizer_steps']!=2472 or done['actual_records_seen']!=180456:
        raise ValueError('Wait for the complete current course; never select from an unfinished run')
    sys.path[:0]=[parent['source'],parent['training_config']['upstream_source']]
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.evaluation.unified_contract import bundle
    import tomllib
    c=read(CONFIG);values={};files=dict(parent['inputs']);files.update(verify(PREPARED)['files'])
    for step in STEPS:
        folder=fit/'evaluations'/f'step-{step:06d}'
        actual=rows(folder/'matches-private.jsonl')
        common=[row for row in actual if row['seed']<parent['curriculum']['curve_seed_start']+2]
        expected={(seed,profile,seat) for seed in range(parent['curriculum']['curve_seed_start'],parent['curriculum']['curve_seed_start']+2)
            for profile in parent['curriculum']['opponent_profiles'] for seat in range(4)}
        if len(common)!=16 or {(r['seed'],r['profile'],r['seat']) for r in common}!=expected or any(not r['completed'] or not r['ledger_passed'] for r in common):
            raise ValueError('All candidates must share the same completed16 development cells')
        values[step]=aggregate_complete_matches(common)
        for name in ('protocol.json','summary.json','matches-private.jsonl'):files[str(folder/name)]=sha(folder/name)
    step=choose(values);checkpoint=fit/'training-run/checkpoints'/f'step-{step:06d}';model=bundle(checkpoint)
    contract=read(fit/'policy-contracts-private.json')[str(step)]
    if contract['model_identity']!=model['identity_sha256']:raise ValueError('Saved selected parameters differ')
    RUN.mkdir()
    write(RUN/'selection-private.json',dict(method=METHOD,selected_step=step,values=values,
        stage3A_grade_blocks_3B=False,selection_is_development_not_independent_proof=True))
    cache=parent64(parent,step,model,contract)
    data_path=ROOT/c['data_ready'];data=read(data_path)
    args=tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    for split,key in (('train','data'),('validation','eval_data')):
        path=ROOT/data['files'][split]['path']
        if sha(path)!=data['files'][split]['sha256']:raise ValueError('Existing3B data changed')
        require_replay(rows(args[key]),rows(path))
    for path in (data_path.parent/'completion-binding-private.json',
        (ROOT/c['objective_preflight']).parent/'completion-binding-private.json',
        (ROOT/c['validation_adapter_preflight']).parent/'completion-binding-private.json',
        (ROOT/c['epoch_recipe']).parent/'completion-binding-private.json'):extend(files,path)
    for path in (parent_runtime.REGISTERED,fit/'completion.json',fit/'protocol.json',fit/'policy-contracts-private.json',
        fit/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json',RUN/'selection-private.json',
        PROSPECTIVE/'parent-fit-terminal-private.json',PREPARED,QUEUE):files[str(path)]=sha(path)
    for path in [*[checkpoint/name for name in model['files']],*[cache/name for name in ('protocol.json','summary.json','matches-private.jsonl')]]:
        files[str(path)]=sha(path)
    sealed=dict(status=STATUS,method=METHOD,selected_step=step,model=model,policy=contract['policy'],
        policy_identity=contract['policy_identity'],parent_registration=str(parent_runtime.REGISTERED),
        parent_evidence=str(parent_runtime.RUN/'final-evidence-private.json'),
        parent_evaluation_cache=str(cache),original_guarded_evaluation_cache=parent['curriculum']['original_guarded_evaluation_cache'],
        original_guarded_identity=parent['curriculum']['parent_identity'],child_data_ready=str(data_path),
        base=parent['curriculum']['base'],base_revision=parent['curriculum']['base_revision'],bound_files=files,
        selection_values={str(k):v for k,v in values.items()},stage3A_grade_blocks_3B=False,
        user_authorized_best_checkpoint_direct3B=True,raw_grade_overridden=False,optimizer_resumed=False,
        selection_from_minimum_CE=False,child_fit_started=False)
    write(RUN/'parent-seal-private.json',sealed)
    return step


def check_job(role):
    resource();verify(PREPARED);queue=read(QUEUE)
    if str(queue[role+'_job'])!=os.environ['SLURM_JOB_ID']:raise ValueError('Exact own direct3B job required')
    return queue


def prepare():
    queue=check_job('prepare');job=str(queue['parent_fit_job']);actual=scheduler(job)
    receipt=read(ROOT/'logs'/f'curriculum-ri-balanced-replay-v2-exit-{job}.json')
    if actual.returncode or receipt.get('phase')!='fit':raise ValueError('Actual successful current FIT terminal required')
    scheduler_exit(receipt,actual.stdout,parent_runtime.ENTRY)
    # This archive is an entry artifact, not a student output.
    if RUN.exists():raise ValueError('Never overwrite a selected continuation')
    archive=PROSPECTIVE/'parent-fit-terminal-private.json';write(archive,dict(raw=actual.stdout,exit=receipt))
    step=select_and_seal()
    function(old_register.register,step,PARENT_METHOD=METHOD,STATUS=STATUS,verify_parent=verify_parent,
        PUBLIC_FILES=(*old_register.PUBLIC_FILES,*PUBLIC))(CONFIG,RUN/'parent-seal-private.json')
    registered=RUN/'fit-plan/registered.json';reg=read(registered);files=dict(reg['inputs'])
    files[str(registered)]=sha(registered)
    write(RUN/'launch-binding-private.json',dict(files=files,registered_sha256=sha(registered),
        actual_registration_before_first_update=True,user_authorized_direct3B=True,selected_parent_step=step))


def fit():
    queue=check_job('fit');save_terminal('prepare',queue)
    step=read(RUN/'selection-private.json')['selected_step']
    function(old_train.train,step,verify_parent=verify_parent,parent_loader=parent_loader)(RUN/'fit-plan/registered.json')


def save_terminal(role,queue):
    job=str(queue[role+'_job']);actual=scheduler(job);receipt=read(ROOT/'logs'/f'curriculum-yaku-best3a-v1-exit-{job}.json')
    if actual.returncode or receipt.get('phase')!=role:raise ValueError('Actual completed role required')
    scheduler_exit(receipt,actual.stdout,ENTRY)
    write(RUN/(role+'-terminal-scheduler-private.json'),dict(raw=actual.stdout,exit=receipt))


def audit():
    queue=check_job('audit');save_terminal('fit',queue)
    step=read(RUN/'selection-private.json')['selected_step']
    function(old_audit.audit,step,verify_parent=verify_parent)(RUN/'fit-plan/registered.json',RUN/'final-evidence-private.json')


def capture():
    queue=check_job('capture');out=(RUN if RUN.exists() else PROSPECTIVE)/'terminal-v1';out.mkdir();proofs={}
    for role in ('prepare','fit','audit'):
        job=str(queue[role+'_job']);actual=scheduler(job);raw=actual.stdout
        if actual.returncode:
            path=RUN/(role+'-terminal-scheduler-private.json')
            if not path.exists():raise ValueError('Actual role terminal unavailable')
            raw=read(path)['raw']
        record=terminal_record(raw,job,ENTRY)
        path=ROOT/'logs'/f'curriculum-yaku-best3a-v1-exit-{job}.json';receipt=read(path) if path.exists() else None
        if record['successful']:scheduler_exit(receipt,raw,ENTRY)
        record['exit_receipt']=receipt;proofs[role]=record
    write(out/'terminal-evidence-private.json',dict(status='direct3B_actual_terminals_preserved',scheduler_proofs=proofs,
        automatic_release=False,final_confirmation_opened=False))


def bind():
    allocation=resource()
    if PREPARED.exists():raise ValueError('Never overwrite direct3B preparation')
    original=read(previous.CONFIG);candidate=read(CONFIG)
    allowed=dict(original,run_id='jev-curriculum-yaku-best3a-v1',stage='stage3B-yaku-best3a-v1',
        plan_root=str(RUN.relative_to(ROOT)/'fit-plan'),out=str(RUN.relative_to(ROOT)/'stage3B-fit'),
        parent_selection=METHOD,admission_mode='user_authorized_direct3B_no_stage3A_grade_veto')
    if candidate!=allowed:raise ValueError('Change only parent selection/admission and output; preserve prepared3B recipe')
    files={}
    extend(files,previous.PREFLIGHT/'completion-binding-private.json')
    for name in PUBLIC:files[str(ROOT/name)]=sha(ROOT/name)
    tests=subprocess.run([sys.executable,'-m','unittest','tests.core.test_curriculum_yaku_best3a_v1','-v'],
        capture_output=True,text=True,timeout=120)
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    subprocess.run(['bash','-n',str(ENTRY)],check=True,timeout=15)
    PROSPECTIVE.mkdir(parents=True);(PROSPECTIVE/'tests.log').write_text(tests.stdout+tests.stderr)
    files[str(PROSPECTIVE/'tests.log')]=sha(PROSPECTIVE/'tests.log')
    write(PREPARED,dict(files=files,allocation=allocation,student_updates=0,
        user_authorized_best_checkpoint_direct3B=True,stage3A_grade_blocks_3B=False))


def enqueue():
    resource();verify(PREPARED)
    if QUEUE.exists():raise ValueError('No duplicate direct3B jobs')
    parent=read(parent_runtime.RUN/'fit-queue-private.json');job=str(parent['fit_job']);actual=scheduler(job)
    if actual.returncode or fields(actual.stdout).get('JobState')!='RUNNING' or fields(actual.stdout).get('Command')!=str(parent_runtime.ENTRY):
        raise ValueError('Exact current running training required')
    old_jobs=[('324602',str(previous.ENTRY)),('324603',str(previous.ENTRY)),
        ('324914',str(ROOT/'cluster/curriculum-ri-anchor-activation-v3.slurm'))]
    old={}
    for old_job,entry in old_jobs:
        observed=scheduler(old_job);value=fields(observed.stdout)
        if observed.returncode or value.get('JobState')!='PENDING' or value.get('Command')!=entry:
            raise ValueError('Only the verified own pending superseded routes may be cancelled')
        old[old_job]=observed.stdout
    jobs={};observed={}
    def submit(role,dependency,limit,cpus=4,memory='48G'):
        result=subprocess.run(['sbatch','--parsable','--time='+limit,'--cpus-per-task='+str(cpus),'--mem='+memory,
            '--dependency='+dependency,'--kill-on-invalid-dep=yes','--export=ALL,JEV_BEST3A_PHASE='+role,str(ENTRY)],
            cwd=ROOT,capture_output=True,text=True,timeout=30)
        value=result.stdout.strip().split(';')[0]
        if result.returncode or not value.isdigit():raise RuntimeError(result.stderr or result.stdout)
        jobs[role]=value;return value
    try:
        preparation=submit('prepare','afterok:'+job,'00:45:00')
        fit_job=submit('fit','afterok:'+preparation,'24:00:00')
        audit_job=submit('audit','afterok:'+fit_job,'02:00:00')
        submit('capture','afterany:'+preparation+':'+fit_job+':'+audit_job,'00:05:00',1,'6G')
        for role,new_job in jobs.items():
            current=scheduler(new_job);value=fields(current.stdout)
            if current.returncode or value.get('JobState')!='PENDING' or value.get('Command')!=str(ENTRY):
                raise ValueError('Actual direct3B graph differs')
            edges={'prepare':['afterok:'+job],'fit':['afterok:'+preparation],
                'audit':['afterok:'+fit_job],
                'capture':['afterany:'+prior for prior in (preparation,fit_job,audit_job)]}[role]
            if any(edge+'(' not in value.get('Dependency','') for edge in edges):
                raise ValueError('Actual direct3B dependency edge differs')
            observed[role]=current.stdout
        write(QUEUE,dict(**{role+'_job':value for role,value in jobs.items()},parent_fit_job=job,
            observed=observed,superseded_pending_jobs=old,prepared_sha256=sha(PREPARED),
            user_authorized_direct3B=True,stage3A_grade_blocks_3B=False))
        for old_job,_ in old_jobs:
            current=scheduler(old_job)
            if fields(current.stdout).get('JobState')=='CANCELLED':continue
            subprocess.run(['scancel',old_job],check=True,capture_output=True,text=True,timeout=20)
        write(PROSPECTIVE/'superseded-routes-private.json',dict(before=old,
            after={old_job:scheduler(old_job).stdout for old_job,_ in old_jobs},current_main_untouched=True))
    except BaseException:
        for new_job in jobs.values():
            observed=scheduler(new_job)
            if fields(observed.stdout).get('Command')==str(ENTRY):subprocess.run(['scancel',new_job],capture_output=True,text=True,timeout=20)
        raise
    print(dict(status='best_trained_checkpoint_direct3B_queued_after_current_fit',jobs=jobs),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,
        choices=('bind','enqueue','prepare','fit','audit','capture'))
    args=parser.parse_args();globals()[args.phase]()
