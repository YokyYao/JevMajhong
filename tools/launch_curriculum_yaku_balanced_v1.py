"""Frozen replay2472-to-yaku2500 entry, with a finite qualified-only bridge."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields, terminal_record
from tools.curriculum_yaku_balanced_parent_v1 import require_qualified_parent, seal, extend, gpu_only
from tools.curriculum_yaku_parent_loss_v1 import METHOD

RUN = ROOT/'runs/curriculum-yaku-balanced-parent-v1'
PARENT_RUN = ROOT/'runs/curriculum-ri-balanced-replay-fit-v1'
CONFIG = ROOT/'configs/training/curriculum-yaku-balanced-fit-v1.json'
ENTRY = ROOT/'cluster/curriculum-yaku-balanced-v1.slurm'
BINDING = PARENT_RUN/'balanced-yaku-entry-binding-v1-private.json'
QUEUE = PARENT_RUN/'balanced-yaku-queue-v1-private.json'
PREFLIGHT = ROOT/'runs/curriculum-balanced-yaku-course-v1/preflight-v1'
BRIDGE_QUEUE = PARENT_RUN/'balanced-yaku-bridge-queue-v1-private.json'
PUBLIC = (
    'tools/curriculum_yaku_balanced_parent_v1.py', 'tools/curriculum_yaku_parent_runtime_v1.py',
    'tools/register_curriculum_yaku_balanced_v1.py', 'tools/train_curriculum_yaku_balanced_v1.py',
    'tools/evaluate_curriculum_yaku_parent_v1.py', 'tools/audit_curriculum_yaku_balanced_v1.py',
    'tools/launch_curriculum_yaku_balanced_v1.py',
    'tests/core/test_curriculum_yaku_balanced_parent_v1.py',
    'tests/core/test_curriculum_yaku_balanced_course_v1.py',
    'configs/training/curriculum-yaku-balanced-fit-v1.json', 'cluster/curriculum-yaku-balanced-v1.slurm',
    'docs/experiments/plans/curriculum-yaku-balanced-course-20261125.md',
)


def require_same_recipe(candidate, original):
    expected = dict(original, run_id='jev-curriculum-yaku-balanced-parent-v1',
        stage='stage3B-yaku-balanced-parent-v1', stage_offset=3468,
        plan_root='runs/curriculum-yaku-balanced-parent-v1/fit-plan',
        out='runs/curriculum-yaku-balanced-parent-v1/stage3B-fit',
        parent_registration='runs/curriculum-ri-balanced-replay-fit-v1/fit-plan/registered.json',
        parent_evidence='runs/curriculum-ri-balanced-replay-fit-v1/final-evidence-private.json',
        parent_jobs_from='runs/curriculum-ri-balanced-replay-fit-v1/fit-queue-private.json',
        parent_selection='actual_registered_final2472_only')
    if candidate != expected:
        raise ValueError('Keep actual3B data/masses/budget/rules; change only explicit parent/objective/output interfaces')


def qualified_parent(c):
    queue = read(ROOT/c['parent_jobs_from'])
    c = dict(c, parent_fit_job=int(queue['fit_job']), parent_audit_job=int(queue['audit_job']),
             parent_terminal_capture_job=int(queue['capture_job']))
    parent = read(ROOT/c['parent_registration']); fit = ROOT/parent['curriculum']['out']
    require_qualified_parent(parent, read(fit/'completion.json'), read(ROOT/c['parent_evidence']),
        read(PARENT_RUN/'audit-execution-private.json'), c)
    return c


def verify_binding():
    b = read(BINDING)
    if (b['status'] != 'balanced_yaku_entry_bound_prospectively_not_parent_or_fit_admitted'
            or not b['parent_not_admitted'] or b['strategy_grade_overridden']
            or b['student_updates'] != 0
            or any(sha(p) != h for p, h in b['files'].items())):
        raise ValueError('Before-completion retained3B entry binding changed')
    return b


def bind():
    allocation = resource(); c = read(CONFIG)
    require_same_recipe(c, read(ROOT/'configs/training/curriculum-yaku-parent-fit-v1.json'))
    if BINDING.exists(): raise ValueError('Never overwrite an entry binding')
    from tools.curriculum_ri_balanced_replay_runtime_v1 import require_registration
    require_registration(read(ROOT/c['parent_registration']))
    pf = read(PREFLIGHT/'summary-private.json'); prior = read(PREFLIGHT/'completion-binding-private.json')
    if (pf['status'] != 'balanced_yaku_full_entry_preflight_passed_not_parent_or_fit_admitted'
            or any(sha(p) != h for p, h in prior['files'].items())):
        raise ValueError('Frozen prospective parent registration and complete entry preflight required')
    files = {str((ROOT/p).resolve()): sha(ROOT/p) for p in PUBLIC}
    files.update(prior['files'])
    for p in (CONFIG, PREFLIGHT/'summary-private.json', PREFLIGHT/'completion-binding-private.json',
              ROOT/c['parent_registration']): files[str(p.resolve())] = sha(p)
    write(BINDING, dict(status='balanced_yaku_entry_bound_prospectively_not_parent_or_fit_admitted',
        files=files,
        allocation=allocation, parent_not_admitted=True, strategy_grade_overridden=False,
        student_updates=0, child_jobs_submitted=False))
    print(json.dumps(dict(status='balanced3B_entry_bound_not_submitted', student_updates=0)), flush=True)


def archive_parent_capture(c):
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    job=c['parent_terminal_capture_job']
    receipt=read(ROOT/'logs'/f'curriculum-ri-balanced-replay-exit-{job}.json')
    if receipt.get('phase')!='capture':raise ValueError('Actual parent capture phase required')
    path=PARENT_RUN/'capture-terminal-for-yaku-private.json'
    if path.exists():
        proof=read(path)
        if proof['exit']!=receipt:raise ValueError('Archived parent capture exit differs')
        raw=proof['raw']
    else:
        result=scheduler(job)
        if result.returncode:raise ValueError('Actual parent capture terminal expired; never synthesize evidence')
        raw=result.stdout
        scheduler_exit(receipt,raw,ROOT/'cluster/curriculum-ri-balanced-replay-v1.slurm')
        write(path,dict(raw=raw,exit=receipt))
    scheduler_exit(receipt,raw,ROOT/'cluster/curriculum-ri-balanced-replay-v1.slurm')
    return path


def submit(after_job=None):
    resource(); verify_binding(); c = qualified_parent(read(CONFIG))
    if RUN.exists() or QUEUE.exists(): raise ValueError('Never submit duplicate child registration/fit')
    cap = read(PARENT_RUN/'terminal-v1/terminal-evidence-private.json')
    if any(not p['successful'] for p in cap['scheduler_proofs'].values()):
        raise ValueError('Actual parent fit and audit must have successful captured terminals')
    archive_parent_capture(c)
    jobs = []
    def enqueue(*args):
        r = subprocess.run(['sbatch', '--parsable', '--hold', *args, str(ENTRY)],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        value = r.stdout.strip().split(';')[0]
        if r.returncode or not value.isdigit(): raise ValueError(r.stderr or r.stdout)
        jobs.append(value); return value
    try:
        preparation = ['--time=00:35:00', '--export=ALL,JEV_YAKU_PARENT_PHASE=prepare']
        if after_job is not None: preparation += ['--dependency=afterok:'+str(after_job),'--kill-on-invalid-dep=yes']
        prepare_job = enqueue(*preparation)
        fit_job = enqueue('--dependency=afterok:'+prepare_job, '--kill-on-invalid-dep=yes',
            '--export=ALL,JEV_YAKU_PARENT_PHASE=fit')
        audit_job = enqueue('--time=02:00:00', '--dependency=afterok:'+fit_job, '--kill-on-invalid-dep=yes',
            '--export=ALL,JEV_YAKU_PARENT_PHASE=audit,JEV_YAKU_PARENT_FIT_JOB='+fit_job)
        capture_job = enqueue('--time=00:05:00', '--cpus-per-task=1', '--mem=4G',
            '--dependency=afterany:'+':'.join((prepare_job, fit_job, audit_job)), '--kill-on-invalid-dep=yes',
            '--export=ALL,JEV_YAKU_PARENT_PHASE=capture')
        observed = {role: scheduler(job) for role, job in
            [('prepare', prepare_job), ('fit', fit_job), ('audit', audit_job), ('capture', capture_job)]}
        if any(r.returncode or fields(r.stdout).get('Reason') != 'JobHeldUser'
               or fields(r.stdout).get('Command') != str(ENTRY) for r in observed.values()):
            raise ValueError('Actual held child job identities differ')
        dependencies = {'fit': [('afterok', prepare_job)], 'audit': [('afterok', fit_job)],
            'capture': [('afterany', job) for job in (prepare_job, fit_job, audit_job)]}
        for role, required in dependencies.items():
            dependency = fields(observed[role].stdout).get('Dependency', '')
            if any(f'{kind}:{job}(' not in dependency for kind, job in required):
                raise ValueError('Actual child dependency differs: '+role)
        if after_job is not None and f'afterok:{after_job}(' not in fields(observed['prepare'].stdout).get('Dependency',''):
            raise ValueError('Child preparation must await successful finite bridge exit')
        write(QUEUE, dict(status='actual_qualified_parent_yaku_jobs_bound_while_held',
            prepare_job=prepare_job, fit_job=fit_job, audit_job=audit_job, capture_job=capture_job,
            binding_sha256=sha(BINDING), observed_jobs={k: v.stdout for k, v in observed.items()},
            original_parent_grade_preserved=True, actual_qualified_parent_request=c))
        for job in (capture_job, audit_job, fit_job, prepare_job):
            subprocess.run(['scontrol', 'release', job], check=True, capture_output=True, text=True, timeout=20)
        print(json.dumps(dict(status='qualified_retained_parent3B_jobs_submitted', **{k: v for k, v in
            read(QUEUE).items() if k.endswith('_job')})), flush=True)
    except BaseException as error:
        cleanup = [dict(job=job, returncode=subprocess.run(['scancel', job], capture_output=True,
            text=True, timeout=20).returncode) for job in jobs]
        path = PARENT_RUN/'balanced-yaku-submit-error-v1-private.json'
        if not path.exists(): write(path, dict(error=repr(error), cleanup=cleanup))
        raise


def enqueue_parent_bridge():
    allocation=resource();verify_binding();c=read(CONFIG)
    if BRIDGE_QUEUE.exists() or QUEUE.exists():raise ValueError('No duplicate balanced-yaku bridge or child fit')
    queue=read(ROOT/c['parent_jobs_from']);fit=scheduler(queue['fit_job']);cap=scheduler(queue['capture_job'])
    fv,cv=fields(fit.stdout),fields(cap.stdout)
    if (fit.returncode or cap.returncode or fv.get('JobState')!='RUNNING'
            or fv.get('Command')!=str(ROOT/'cluster/curriculum-ri-balanced-replay-v1.slurm')
            or cv.get('JobState')!='PENDING' or cv.get('Command')!=fv.get('Command')
            or any(f'afterany:{queue[r+"_job"]}(' not in cv.get('Dependency','') for r in ('fit','audit'))):
        raise ValueError('Bind while actual replay2472 parent runs with its exact pending capture')
    job=None
    try:
        result=subprocess.run(['sbatch','--parsable','--hold','--time=00:05:00','--cpus-per-task=1','--mem=4G',
            '--job-name=jev-balanced-yaku-bridge','--dependency=afterany:'+str(queue['capture_job']),
            '--kill-on-invalid-dep=yes','--export=ALL,JEV_YAKU_PARENT_PHASE=bridge',str(ENTRY)],
            cwd=ROOT,capture_output=True,text=True,timeout=30)
        job=result.stdout.strip().split(';')[0]
        if result.returncode or not job.isdigit():raise ValueError(result.stderr or result.stdout)
        observed=scheduler(job);v=fields(observed.stdout)
        if (observed.returncode or v.get('JobState')!='PENDING' or v.get('Reason')!='JobHeldUser'
                or v.get('Command')!=str(ENTRY) or v.get('TimeLimit')!='00:05:00'
                or f'afterany:{queue["capture_job"]}(' not in v.get('Dependency','')):
            raise ValueError('Actual held finite qualified-only bridge contract differs')
        write(BRIDGE_QUEUE,dict(status='finite_balanced_yaku_bridge_bound_during_real_parent_fit',
            bridge_job=job,parent_jobs=queue,binding_sha256=sha(BINDING),allocation=allocation,
            observed_fit=fit.stdout,observed_capture=cap.stdout,observed_held_bridge=observed.stdout))
        subprocess.run(['scontrol','release',job],check=True,capture_output=True,text=True,timeout=20)
        print(json.dumps({'status':'finite_balanced_yaku_bridge_released','bridge_job':job}),flush=True)
    except BaseException:
        if job and job.isdigit():subprocess.run(['scancel',job],capture_output=True,text=True,timeout=20)
        raise


def bridge():
    allocation=resource();verify_binding();queue=read(BRIDGE_QUEUE)
    if str(queue['bridge_job'])!=os.environ['SLURM_JOB_ID'] or queue['binding_sha256']!=sha(BINDING):
        raise ValueError('Exact bound finite bridge required')
    captured=read(PARENT_RUN/'terminal-v1/terminal-evidence-private.json')
    grade=captured.get('strategy_grade')
    if grade and grade.get('passed') is False:
        write(PARENT_RUN/'balanced-yaku-bridge-decision-private.json',dict(
            status='balanced_yaku_not_started_parent_failed_original_strategy_grade',
            original_grade=grade,allocation=allocation,student_updates=0))
        print(json.dumps({'status':'balanced_yaku_not_started_parent_failed_original_strategy_grade'}),flush=True)
        return
    submit(after_job=queue['bridge_job'])  # Requires genuine qualified final2472; no grade override.
    write(PARENT_RUN/'balanced-yaku-bridge-decision-private.json',dict(
        status='qualified_balanced_yaku_chain_submitted_after_bridge_success',allocation=allocation))


def prepare():
    gpu_only(); b = verify_binding(); c = qualified_parent(read(CONFIG)); queue = read(QUEUE)
    if str(queue['prepare_job']) != os.environ['SLURM_JOB_ID'] or queue['binding_sha256'] != sha(BINDING):
        raise ValueError('Actual bound preparation job required')
    if RUN.exists(): raise ValueError('Never overwrite a parent, registration or fit')
    RUN.mkdir(); request = RUN/'parent-request-private.json'; write(request, c)
    seal(request, RUN/'parent')
    from tools.register_curriculum_yaku_balanced_v1 import register
    register(request, RUN/'parent/parent-seal-private.json')
    registered = RUN/'fit-plan/registered.json'; reg = read(registered)
    files = {**b['files'], **reg['inputs']}
    for p in (BINDING, QUEUE, registered): files[str(p)] = sha(p)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Registered continuation source changed')
    write(RUN/'launch-binding-private.json', dict(files=files, registered_sha256=sha(registered),
        candidate_step=2500, actual_registration_before_first_update=True, optimizer_updates=0))
    write(RUN/'prepare-execution-private.json', dict(status='actual_retained_yaku_parent_sealed_and2500_registered',
        registered_sha256=sha(registered), optimizer_updates=0, original_3A_grade_preserved=True))
    print(json.dumps(dict(status='qualified_retained_yaku2500_registered', student_updates=0)), flush=True)


def train():
    gpu_only(); verify_binding()
    queue = read(QUEUE)
    if str(queue['fit_job']) != os.environ['SLURM_JOB_ID']:
        raise ValueError('Actual bound child FIT job required')
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    prepare_job = queue['prepare_job']
    receipt = read(ROOT/'logs'/f'curriculum-yaku-balanced-exit-{prepare_job}.json')
    if receipt['phase'] != 'prepare': raise ValueError('Actual PREPARE exit required')
    archive = RUN/'prepare-terminal-scheduler-private.json'
    if archive.exists():
        saved = read(archive)
        if saved['exit'] != receipt: raise ValueError('Archived preparation receipt differs')
        raw = saved['raw']
    else:
        actual = scheduler(prepare_job)
        if actual.returncode: raise ValueError('Actual preparation terminal unavailable before fit')
        raw = actual.stdout
        scheduler_exit(receipt, raw, ENTRY)
        write(archive, dict(raw=raw, exit=receipt))
    scheduler_exit(receipt, raw, ENTRY)
    from tools.train_curriculum_yaku_balanced_v1 import train as fit
    fit(RUN/'fit-plan/registered.json')


def audit(fit_job):
    gpu_only(); verify_binding()
    queue = read(QUEUE)
    if str(queue['audit_job']) != os.environ['SLURM_JOB_ID'] or str(queue['fit_job']) != str(fit_job):
        raise ValueError('Actual bound child audit/FIT jobs required')
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    receipt = read(ROOT/'logs'/f'curriculum-yaku-balanced-exit-{fit_job}.json')
    if receipt['phase'] != 'fit': raise ValueError('Actual child FIT phase receipt required')
    actual = scheduler(fit_job)
    if actual.returncode: raise ValueError('Actual child fit terminal unavailable')
    scheduler_exit(receipt, actual.stdout, ENTRY)
    write(RUN/'fit-terminal-scheduler-private.json', dict(raw=actual.stdout, exit=receipt))
    launch = read(RUN/'launch-binding-private.json')
    if any(sha(p) != h for p, h in launch['files'].items()): raise ValueError('Original child binding changed')
    from tools.audit_curriculum_yaku_balanced_v1 import audit as independent_audit
    independent_audit(RUN/'fit-plan/registered.json', RUN/'final-evidence-private.json')
    if any(sha(p) != h for p, h in launch['files'].items()): raise ValueError('Audit changed a bound child input')
    write(RUN/'audit-execution-private.json', dict(status='actual_yaku2500_full_evidence_independently_audited',
        actual_fit_job=fit_job, original_launch_binding_unchanged=True, loss_method=METHOD, automatic_release=False))


def capture():
    allocation = resource(); verify_binding(); queue = read(QUEUE)
    if str(queue['capture_job']) != os.environ['SLURM_JOB_ID']:
        raise ValueError('Actual bound child terminal-preservation job required')
    out = (RUN if RUN.exists() else PARENT_RUN)/'yaku-balanced-terminal-v1'; out.mkdir(exist_ok=False)
    proofs = {}
    for role in ('prepare', 'fit', 'audit'):
        job = queue[role+'_job']; actual = scheduler(job)
        write(out/(role+'-observation-private.json'), dict(stdout=actual.stdout, stderr=actual.stderr,
            returncode=actual.returncode))
        raw = actual.stdout; path = ROOT/'logs'/f'curriculum-yaku-balanced-exit-{job}.json'
        receipt = read(path) if path.exists() else None
        archive_path = RUN/(role+'-terminal-scheduler-private.json')
        if actual.returncode and role in ('prepare', 'fit') and archive_path.exists():
            archive = read(archive_path)
            if archive['exit'] != receipt: raise ValueError('Archived child fit exit differs')
            raw = archive['raw']
        proof = terminal_record(raw, job, ENTRY)
        if proof['successful']:
            from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
            if receipt is None or receipt['phase'] != role: raise ValueError('Successful child phase receipt required')
            scheduler_exit(receipt, raw, ENTRY)
        proof['exit_receipt'] = receipt; proofs[role] = proof
    evidence = RUN/'final-evidence-private.json'
    write(out/'terminal-evidence-private.json', dict(status='actual_retained_yaku_terminals_preserved_not_stage4_admitted',
        scheduler_proofs=proofs, original_strategy_grade=(read(evidence)['stage3B_advancement'] if evidence.exists() else None),
        strategy_grade_overridden=False, stage4_parent_sealed=False, student_updates=0, allocation=allocation))
    paths = [BINDING, QUEUE, *out.glob('*-private.json')]
    write(out/'completion-binding-private.json', dict(files={str(p): sha(p) for p in paths}))
    print(json.dumps(dict(status='actual_retained_yaku_terminals_preserved')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('bind', 'submit', 'enqueue-parent-bridge', 'bridge', 'prepare', 'fit', 'audit', 'capture'), required=True)
    p.add_argument('--fit-job', type=int)
    a = p.parse_args()
    if a.phase == 'bind': bind()
    elif a.phase == 'submit': submit()
    elif a.phase == 'enqueue-parent-bridge': enqueue_parent_bridge()
    elif a.phase == 'bridge': bridge()
    elif a.phase == 'prepare': prepare()
    elif a.phase == 'fit': train()
    elif a.phase == 'capture': capture()
    else:
        if not a.fit_job: p.error('--fit-job required for audit')
        audit(a.fit_job)
