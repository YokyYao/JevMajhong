"""Finite, source-bound placement lifecycle after genuine parent/data completion."""
import argparse
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools import curriculum_placement_balanced_course_v1 as course
from tools import integrate_curriculum_placement_balanced_v1 as integration
from tools import prepare_curriculum_placement_launch_v1 as original
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields, terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

CONTROL = ROOT/'runs/curriculum-placement-balanced-launch-v1'
RUN = ROOT/Path(course.IDENTITY['out']).parent
ENTRY = ROOT/'cluster/curriculum-placement-balanced-course-v1.slurm'
BINDING = CONTROL/'entry-preflight-v1/completion-binding-private.json'
QUEUE = CONTROL/'queue-private.json'
PARENT_TERMINAL_QUEUE = CONTROL/'parent-terminal-queue-private.json'
LAUNCH_CONFIG = RUN/'launch-config-private.json'
DATA = ROOT/'runs/curriculum-placement-recovered-exposures-v1/data-v1/data-ready-private.json'
ORIGINAL_DATA = ROOT/'runs/curriculum-targeted-win-resume-v3/placement-data-v5/data-v1/data-ready-private.json'
SEAL = RUN/'qualified-yaku-parent-v1/parent-seal-private.json'
PREFIX = 'curriculum-placement-balanced-course-exit-'
ROLES = ('register', 'integration', 'prepare', 'fit', 'audit')
LIMITS = dict(register='00:35:00', integration='00:40:00', prepare='00:20:00',
              fit='1-00:00:00', audit='02:00:00', capture='00:05:00', **{'parent-terminal':'00:05:00'})
PUBLIC = (
    'tools/launch_curriculum_placement_balanced_v1.py',
    'cluster/curriculum-placement-balanced-course-v1.slurm',
    'tests/core/test_curriculum_placement_balanced_launch_v1.py',
    'docs/experiments/plans/curriculum-placement-balanced-launch-20261125.md',
    'configs/training/curriculum-placement-assessment-v1.json',
)


def prepare_tree():
    tree = ast.parse(inspect.getsource(original.prepare))
    replacements = {
        'curriculum-placement-preflight-exit-': PREFIX,
        'cluster/curriculum-placement-preflight-v1.slurm': str(ENTRY.relative_to(ROOT)),
        'placement_actual_upstream_integration_and_launch_bound_v1':
            'placement_retained_objective_actual_upstream_integration_and_launch_bound_v1',
    }
    counts = dict.fromkeys((*replacements, 'job', 'summarizer'), 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value,str) and node.value in replacements:
            old=node.value;node.value=replacements[old];counts[old]+=1
        elif isinstance(node,ast.ImportFrom) and node.module=='tools.curriculum_placement_metrics_v1':
            node.module='tools.curriculum_placement_parent_metrics_v2';counts['summarizer']+=1
    class Job(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value == 319427:
                counts['job']+=1
                return ast.copy_location(ast.Name(id='ACTUAL_INTEGRATION_JOB',ctx=ast.Load()),node)
            return node
    tree=Job().visit(tree)
    if counts != dict.fromkeys(counts,1):
        raise ValueError('Review original preparation: exact job/path/summarizer substitutions changed')
    return ast.fix_missing_locations(tree)


def require_integration(proof,reg,job):
    original.require_integration(proof,reg,job)
    if (proof.get('loss_method')!=integration.loss.METHOD
            or proof.get('placement_parent_lineage_method')!=course.parent.METHOD
            or proof.get('global_student_stage_offset')!=5968):
        raise ValueError('Actual retained loss and explicit balanced-yaku integration required')


def prepare_function(job):
    if not str(job).isdigit() or int(job)<=0:raise ValueError('Actual integration job ID required')
    def writer(path,value):
        value=copy.deepcopy(value)
        if Path(path).name=='launch-binding-private.json':
            value['files'][str(BINDING.resolve())]=sha(BINDING)
            value['files'][str(QUEUE.resolve())]=sha(QUEUE)
            value['placement_parent_lineage_method']=course.parent.METHOD
        original.write(path,value)
    scope=dict(vars(original), ACTUAL_INTEGRATION_JOB=int(job),
        require_registration=course.require_registration,verify_parent=course.parent.verify_parent,
        require_integration=require_integration,write=writer,
        PUBLIC_FILES=(*original.PUBLIC_FILES,*course.parent.PUBLIC,*course.PUBLIC,*integration.PUBLIC,*PUBLIC))
    exec(compile(prepare_tree(),str(Path(original.__file__)),'exec'),scope)
    return scope['prepare']


def verify_entry():
    resource();b=read(BINDING)
    if any(sha(path)!=digest for path,digest in b['files'].items()):
        raise ValueError('Bound lifecycle software changed')
    return b


def bind():
    allocation=resource()
    if BINDING.exists():raise ValueError('Never overwrite a lifecycle binding')
    out=BINDING.parent;out.mkdir(parents=True,exist_ok=False)
    snapshots=out/'inputs';snapshots.mkdir()
    for name in PUBLIC:shutil.copyfile(ROOT/name,snapshots/Path(name).name)
    groups=[
        (ROOT/'runs/curriculum-placement-balanced-parent-v1/preflight-v1/completion-binding-private.json',
         (*course.parent.PUBLIC,*course.parent.producer.PUBLIC),'tests.core.test_curriculum_placement_balanced_parent_v1'),
        (course.BINDING,course.registration_function().__globals__['PUBLIC_FILES'],
         'tests.core.test_curriculum_placement_balanced_course_v1'),
        (integration.BINDING,integration.integration_function().__globals__['PUBLIC_FILES'],
         'tests.core.test_curriculum_placement_balanced_integration_v1'),
    ]
    files={}
    # A fresh checkout can reproduce all prerequisite software bindings here.
    # Existing immutable bindings are verified and reused without retesting.
    for path,public,test_module in groups:
        if path.exists():
            saved=read(path)
            if any(sha(p)!=h for p,h in saved['files'].items()):raise ValueError('Prerequisite entry changed')
            files.update(saved['files']);files[str(path.resolve())]=sha(path);continue
        target=path.parent;target.mkdir(parents=True,exist_ok=False)
        result=subprocess.run([os.sys.executable,'-m','unittest',test_module],cwd=ROOT,
            capture_output=True,text=True,timeout=60)
        receipt=target/'lifecycle-unit-tests-private.json'
        write(receipt,dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise RuntimeError(result.stderr)
        bound={str((ROOT/name).resolve()):sha(ROOT/name) for name in public}
        bound[str(receipt.resolve())]=sha(receipt)
        # Retain all already verified lower-level bindings in a fresh entry.
        bound.update(files);write(path,dict(files=bound))
        files.update(bound);files[str(path.resolve())]=sha(path)
    result=subprocess.run([os.sys.executable,'-m','unittest',
        'tests.core.test_curriculum_placement_balanced_launch_v1'],cwd=ROOT,
        capture_output=True,text=True,timeout=60)
    receipt=out/'unit-tests-private.json'
    write(receipt,dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError(result.stderr)
    subprocess.run(['bash','-n',str(ENTRY)],check=True,timeout=10)
    for path in [*[ROOT/name for name in (*PUBLIC,*course.registration_function().__globals__['PUBLIC_FILES'],
                    *integration.integration_function().__globals__['PUBLIC_FILES'])],
                 Path(original.__file__),Path(course.parent.retained.__file__),
                 Path(course.parent.producer.__file__),receipt]:
        files[str(path.resolve())]=sha(path)
    write(BINDING,dict(files=files,allocation=allocation,student_updates=0,
        actual_registration_or_fit_started=False,full_data_and_parent_still_required=True))
    print(json.dumps(dict(status='finite_placement_lifecycle_bound_not_submitted',student_updates=0)),flush=True)


def receipt(role,job):
    value=read(ROOT/'logs'/f'{PREFIX}{job}.json')
    if str(value['job_id'])!=str(job) or value.get('phase')!=role:
        raise ValueError('Actual own phase receipt required')
    return value


def archive_success(role,q):
    job=str(q[role+'_job']);value=receipt(role,job)
    return original.terminal(job,value,ENTRY,RUN/f'{role}-terminal-scheduler-private.json')


def verify_queue(role):
    verify_entry();q=read(QUEUE);job=os.environ.get('SLURM_JOB_ID','')
    actual=scheduler(job);v=fields(actual.stdout)
    if (str(q[role+'_job'])!=job or actual.returncode or v.get('Command')!=str(ENTRY)
            or not v.get('UserId','').endswith(f'({os.getuid()})')
            or v.get('JobState')!='RUNNING' or v.get('TimeLimit')!=LIMITS[role]):
        raise ValueError('Exact actual bound lifecycle phase and finite resources required')
    if any(sha(p)!=h for p,h in q['files'].items()):raise ValueError('Pre-release source/data admission changed')
    return q


def preserve_parent_terminal():
    verify_entry();q=read(PARENT_TERMINAL_QUEUE)
    if str(q['job'])!=os.environ.get('SLURM_JOB_ID'):raise ValueError('Exact parent terminal job required')
    if q['binding_sha256']!=sha(BINDING) or q['parent_queue_sha256']!=sha(course.parent.QUEUE):
        raise ValueError('Bound parent terminal/producer queue changed')
    parent_queue=read(course.parent.QUEUE);job=str(parent_queue['capture_job'])
    path=ROOT/'logs'/f'curriculum-yaku-balanced-exit-{job}.json';value=read(path)
    actual=scheduler(job)
    if actual.returncode:raise ValueError('Genuine parent capture terminal expired; never reconstruct it')
    if value.get('phase')!='capture':raise ValueError('Original parent capture exit required')
    scheduler_exit(value,actual.stdout,course.parent.ENTRY)
    destination=course.parent.PARENT.parent.parent/'capture-terminal-for-placement-private.json'
    if destination.exists():raise ValueError('Never overwrite a parent terminal archive')
    write(destination,dict(raw=actual.stdout,exit=value))
    write(CONTROL/'parent-terminal-preserved-private.json',dict(status='actual_yaku_capture_terminal_preserved',
        capture_job=job,model_loaded=False,student_updates=0,placement_fit_started=False))


def enqueue_parent_terminal():
    verify_entry()
    if PARENT_TERMINAL_QUEUE.exists():raise ValueError('No duplicate parent-terminal job')
    q=read(course.parent.QUEUE);fit=scheduler(q['fit_job']);cap=scheduler(q['capture_job'])
    if (fit.returncode or fields(fit.stdout).get('JobState')!='RUNNING'
            or fields(fit.stdout).get('Command')!=str(course.parent.ENTRY)
            or cap.returncode or fields(cap.stdout).get('JobState')!='PENDING'
            or fields(cap.stdout).get('Command')!=str(course.parent.ENTRY)):
        raise ValueError('Bind during the actual yaku fit with its exact pending capture')
    job=None;released=False
    try:
        result=subprocess.run(['sbatch','--hold','--parsable','--dependency=afterany:'+str(q['capture_job']),
            '--kill-on-invalid-dep=yes','--export=ALL,JEV_PLACEMENT_BALANCED_PHASE=parent-terminal',str(ENTRY)],
            cwd=ROOT,capture_output=True,text=True,check=True,timeout=30)
        job=result.stdout.strip().split(';')[0];held=scheduler(job);v=fields(held.stdout)
        if (not job.isdigit() or held.returncode or v.get('JobState')!='PENDING'
                or v.get('Reason')!='JobHeldUser' or v.get('Command')!=str(ENTRY)
                or v.get('TimeLimit')!=LIMITS['parent-terminal']
                or f'afterany:{q["capture_job"]}(' not in v.get('Dependency','')):
            raise ValueError('Actual held finite parent-terminal dependency required')
        write(PARENT_TERMINAL_QUEUE,dict(job=job,actual_fit=fit.stdout,actual_capture=cap.stdout,
            actual_held_terminal_job=held.stdout,binding_sha256=sha(BINDING),
            parent_queue_sha256=sha(course.parent.QUEUE)))
        subprocess.run(['scontrol','release',job],check=True,capture_output=True,text=True,timeout=20)
        released=True
    finally:
        if job and job.isdigit() and not released:subprocess.run(['scancel',job],capture_output=True,timeout=20)


def submit():
    verify_entry()
    if QUEUE.exists() or integration.REGISTERED.exists() or (ROOT/course.IDENTITY['out']).exists():
        raise ValueError('Never duplicate or silently restart a placement fit')
    _,_,_,parent_files,_=course.parent.parent_artifacts(course.parent.PARENT)
    data=read(DATA);original_data=read(ORIGINAL_DATA);course.parent.producer.require_data(data,original_data)
    for split in ('train','validation'):
        path=ROOT/data['files'][split]['path']
        if sha(path)!=data['files'][split]['sha256']:raise ValueError('Actual prepared placement file changed')
    files=dict(parent_files)
    for path in (BINDING,DATA,ORIGINAL_DATA):files[str(path.resolve())]=sha(path)
    RUN.mkdir(parents=True,exist_ok=True)
    jobs={};released=False
    try:
        for role in (*ROLES,'capture'):
            dependencies=([('afterany',jobs[r]) for r in ROLES] if role=='capture' else
                          [] if role=='register' else [('afterok',jobs[ROLES[ROLES.index(role)-1]])])
            args=['sbatch','--hold','--parsable','--time='+LIMITS[role],
                '--cpus-per-task='+('1' if role=='capture' else '4'),
                '--mem='+('4G' if role=='capture' else '48G'),'--kill-on-invalid-dep=yes',
                '--export=ALL,JEV_PLACEMENT_BALANCED_PHASE='+role]
            if dependencies:args.append('--dependency='+','.join(k+':'+j for k,j in dependencies))
            result=subprocess.run([*args,str(ENTRY)],cwd=ROOT,capture_output=True,text=True,check=True,timeout=30)
            job=result.stdout.strip().split(';')[0]
            if not job.isdigit():raise ValueError('Actual numeric phase job ID required')
            jobs[role]=job
        observations={role:scheduler(job) for role,job in jobs.items()}
        for role,result in observations.items():
            v=fields(result.stdout)
            if (result.returncode or v.get('JobId')!=jobs[role] or v.get('JobState')!='PENDING'
                    or not v.get('UserId','').endswith(f'({os.getuid()})')
                    or v.get('Reason')!='JobHeldUser' or v.get('AllocTRES') not in ('','(null)')
                    or v.get('Command')!=str(ENTRY) or v.get('TimeLimit')!=LIMITS[role]):
                raise ValueError('All finite phase jobs must be genuinely held before release')
            expected=([('afterany',jobs[r]) for r in ROLES] if role=='capture' else
                      [] if role=='register' else [('afterok',jobs[ROLES[ROLES.index(role)-1]])])
            if any(f'{kind}:{job}(' not in v.get('Dependency','') for kind,job in expected):
                raise ValueError('Actual phase dependency differs')
        config=dict(status='planned_full_placement_launch_after_actual_integration',
            registered=str(integration.REGISTERED.relative_to(ROOT)),
            integration_out=str(integration.OUT.relative_to(ROOT)),integration_job=int(jobs['integration']),
            criteria_config='configs/training/curriculum-placement-assessment-v1.json',
            automatic_release=False,open_independent_confirmation=False)
        write(LAUNCH_CONFIG,config);files[str(LAUNCH_CONFIG.resolve())]=sha(LAUNCH_CONFIG)
        write(QUEUE,dict(status='complete_placement_jobs_bound_while_held',
            **{role+'_job':job for role,job in jobs.items()},files=files,
            actual_observations={role:r.stdout for role,r in observations.items()},automatic_release=False))
        for role in reversed((*ROLES,'capture')):
            subprocess.run(['scontrol','release',jobs[role]],capture_output=True,text=True,check=True,timeout=20)
        released=True
        print(json.dumps(dict(status='qualified_full_placement_chain_submitted',jobs=jobs)),flush=True)
    finally:
        if not released:
            for job in jobs.values():subprocess.run(['scancel',job],capture_output=True,timeout=20)


def run(role):
    q=verify_queue(role)
    if role=='register':
        course.parent.seal(course.parent.PARENT,DATA,ORIGINAL_DATA,SEAL.parent)
        course.register(course.CONFIG,SEAL)
    elif role=='integration':
        archive_success('register',q);integration.integrate()
    elif role=='prepare':
        archive_success('integration',q);prepare_function(q['integration_job'])(LAUNCH_CONFIG)
    elif role=='fit':
        archive_success('prepare',q);course.train(integration.REGISTERED)
    elif role=='audit':
        archive_success('fit',q);course.audit(integration.REGISTERED,RUN/'final-evidence-private.json')
        evidence=read(RUN/'final-evidence-private.json')
        write(RUN/'audit-execution-private.json',dict(status='actual_full_placement_evidence_independently_audited',
            actual_fit_job=q['fit_job'],original_launch_binding_unchanged=True,
            development_grade_passed=evidence['development_assessment']['passed'],
            strategy_grade_overridden=False,independent_v8_confirmation_performed=False,automatic_release=False))
    else:
        out=RUN/'placement-balanced-terminal-v1';out.mkdir(exist_ok=False)
        proofs={}
        for prior in ROLES:
            job=q[prior+'_job'];path=RUN/f'{prior}-terminal-scheduler-private.json'
            value=receipt(prior,job) if (ROOT/'logs'/f'{PREFIX}{job}.json').exists() else None
            actual=scheduler(job)
            write(out/f'{prior}-observation-private.json',dict(returncode=actual.returncode,raw=actual.stdout,stderr=actual.stderr))
            if actual.returncode:
                saved=read(path)
                if saved['receipt']!=value:raise ValueError('Original archived phase receipt changed')
                raw=saved['raw']
            else:raw=actual.stdout
            proof=terminal_record(raw,job,ENTRY)
            if proof['successful']:
                if value is None or value['phase']!=prior:raise ValueError('Actual successful phase receipt required')
                scheduler_exit(value,raw,ENTRY)
            proofs[prior]=dict(**proof,exit_receipt=value)
        evidence=RUN/'final-evidence-private.json'
        write(out/'terminal-evidence-private.json',dict(status='actual_balanced_placement_terminals_preserved_not_confirmed',
            scheduler_proofs=proofs,original_development_grade=(read(evidence)['development_assessment'] if evidence.exists() else None),
            strategy_grade_overridden=False,independent_confirmation_started=False,automatic_release=False))
        paths=[BINDING,QUEUE,*out.glob('*-private.json')]
        write(out/'completion-binding-private.json',dict(files={str(p.resolve()):sha(p) for p in paths}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('bind','submit','enqueue-parent-terminal','parent-terminal',*ROLES,'capture'),required=True)
    args=parser.parse_args()
    if args.phase=='bind':bind()
    elif args.phase=='submit':submit()
    elif args.phase=='enqueue-parent-terminal':enqueue_parent_terminal()
    elif args.phase=='parent-terminal':preserve_parent_terminal()
    else:run(args.phase)
