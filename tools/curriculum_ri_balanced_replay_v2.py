"""Isolated retry of V1 with pinned-loader canonical requests.

Reuse V1's numerical train/evaluate/audit functions and strict admission.
Only request construction, explicit run identity and output/entry paths differ.
The failed V1 registration, sources, logs and terminals remain immutable.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import CodeType, FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import curriculum_ri_balanced_replay_runtime_v1 as original_runtime
from tools import register_curriculum_ri_balanced_replay_v1 as original_registration
from tools import train_curriculum_ri_balanced_replay_v1 as original_training
from tools import evaluate_curriculum_ri_balanced_replay_v1 as original_evaluation
from tools import audit_curriculum_ri_balanced_replay_v1 as original_auditing
from tools.curriculum_ri_canonical_replay_v2 import canonical_replay, METHOD
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields, terminal_record
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

RUN = ROOT/'runs/curriculum-ri-balanced-replay-fit-v2'
REGISTERED = RUN/'fit-plan/registered.json'
ENTRY = ROOT/'cluster/curriculum-ri-balanced-replay-v2.slurm'
PROOF = ROOT/'runs/curriculum-ri-canonical-replay-v2/preflight-v1/completion-binding-private.json'
FAILED = original_runtime.RUN/'terminal-v1/completion-binding-private.json'
QUEUE = RUN/'fit-queue-private.json'
PUBLIC = (
    'tools/curriculum_ri_canonical_replay_v2.py',
    'tests/core/test_curriculum_ri_canonical_replay_v2.py',
    'tools/curriculum_ri_balanced_replay_v2.py',
    'tools/evaluate_curriculum_ri_balanced_replay_v2.py',
    'tests/core/test_curriculum_ri_balanced_replay_v2.py',
    'cluster/curriculum-ri-balanced-replay-v2.slurm',
    'docs/experiments/plans/curriculum-ri-canonical-retry-20261125.md',
)


def scoped(function, **overrides):
    return FunctionType(function.__code__, dict(function.__globals__, **overrides),
        function.__name__, function.__defaults__, function.__closure__)


def curriculum_view(old):
    result = scoped(original_runtime.curriculum_view, RUN=RUN)(old)
    result.update(run_id='jev-curriculum-ri-balanced-replay-v2', stage='stage3-ri-balanced-replay-v2')
    return result


trainer_view = scoped(original_runtime.trainer_view, RUN=RUN)
check_matched = scoped(original_runtime.check_matched, RUN=RUN,
    curriculum_view=curriculum_view, trainer_view=trainer_view)
base_require_registration = scoped(original_runtime.require_registration, RUN=RUN, check_matched=check_matched)


def verify_binding(path):
    binding = read(path)
    if any(sha(p) != h for p, h in binding['files'].items()):
        raise ValueError('Frozen prerequisite/source changed: '+str(path))
    return binding


def require_prerequisites():
    verify_binding(PROOF)
    proof = read(PROOF.parent/'summary-private.json')
    if (proof['status'] != 'full87595_loader_metadata_cause_reproduced_and_canonical_equality_passed'
            or proof['unique_rows'] != 87595 or proof['physical_slots'] != 90228
            or not proof['v1_failure_reproduced'] or not proof['all_nonmetadata_fields_exact']
            or not proof['every_existing_metadata_field_exact'] or proof['student_updates'] != 0
            or proof['actual_token_admission_claimed'] is not False):
        raise ValueError('Actual full-data canonical cause proof required')
    verify_binding(FAILED)
    terminal = read(FAILED.parent/'terminal-evidence-private.json')
    failed_queue = read(original_runtime.RUN/'fit-queue-private.json')
    fit = terminal['scheduler_proofs']['fit']
    if (terminal['status'] != 'balanced_replay_actual_terminals_preserved_not_promoted'
            or fit['state'] != 'FAILED' or fit['successful'] is not False
            or fit['job_id'] != str(failed_queue['fit_job'])
            or fit['exit_receipt']['phase'] != 'fit' or fit['exit_receipt']['exit_code'] != 1):
        raise ValueError('Actual failed V1 attempt must be preserved, never treated as a candidate')


def require_registration(reg):
    base_require_registration(reg)
    require_prerequisites()
    for path in (PROOF, FAILED, *[ROOT/name for name in PUBLIC]):
        if reg['inputs'].get(str(path)) != sha(path):
            raise ValueError('Explicit canonical retry input not bound: '+str(path))


def require_launch(registered):
    require_prerequisites()
    scoped(original_runtime.require_launch, RUN=RUN)(registered)


def register():
    require_prerequisites()
    # Same V1 registration recipe; add isolated identity and actual failure proof.
    scoped(original_registration.register, RUN=RUN,
        PUBLIC=(*original_registration.PUBLIC, *PUBLIC, str(PROOF.relative_to(ROOT)), str(FAILED.relative_to(ROOT))),
        curriculum_view=curriculum_view, trainer_view=trainer_view,
        require_registration=require_registration)()
    scoped(original_registration.authorize, RUN=RUN,
        require_registration=require_registration, require_launch=require_launch)()


def train_function(registered):
    reg = read(registered)
    def replay_factory(raw, plans):
        import tomllib
        from jevany import train as trainer
        args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
        return canonical_replay(raw, plans, args['data'], trainer.load_records)
    def expansion(replay, rows, report):
        result = original_runtime.expand_after_unique_admission(replay, rows, report)
        write(ROOT/reg['curriculum']['out']/'canonical-request-admission-private.json',
            dict(replay.canonicalization_report, actual_unique_token_admission=report,
                strict_full_canonical_object_equality_passed=True))
        return result
    function = original_training.train
    old = 'tools/evaluate_curriculum_ri_balanced_replay_v1.py'
    changed = 0
    def replace_entry(code):
        nonlocal changed
        constants = []
        for value in code.co_consts:
            if isinstance(value, CodeType):
                value = replace_entry(value)
            elif value == old:
                value = 'tools/evaluate_curriculum_ri_balanced_replay_v2.py'
                changed += 1
            constants.append(value)
        return code.replace(co_consts=tuple(constants))
    code = replace_entry(function.__code__)
    if changed != 1:
        raise ValueError('Exactly one declared evaluator entry path substitution required')
    scope = dict(function.__globals__, require_registration=require_registration,
        require_launch=require_launch, EpochReplay=replay_factory,
        expand_after_unique_admission=expansion)
    return FunctionType(code, scope, function.__name__, function.__defaults__, function.__closure__)


def evaluate(registered, checkpoint, step):
    scoped(original_evaluation.evaluate, require_registration=require_registration)(registered, checkpoint, step)


def train(registered):
    require_registration(read(registered)); require_launch(registered)
    train_function(registered)(registered)


def audit(registered, out):
    scoped(original_auditing.audit, require_registration=require_registration,
        require_launch=require_launch)(registered, out)
    evidence = read(ROOT/read(registered)['curriculum']['out']/'canonical-request-admission-private.json')
    report = evidence['actual_unique_token_admission']
    if (evidence['unique_rows'] != 87595 or not evidence['strict_full_canonical_object_equality_passed']
            or report['requested_records'] != 87595 or report['admitted_records'] != 87595
            or report['rejected_records'] != 0 or report['rejected'] != []):
        raise ValueError('Full actual real token admission and unchanged canonical requests required')


def submit(arguments):
    result = subprocess.run(['sbatch', '--parsable', *arguments, str(ENTRY)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    job = result.stdout.strip().split(';')[0]
    if result.returncode or not job.isdigit():
        raise RuntimeError(result.stderr or result.stdout)
    return job


def enqueue():
    allocation = resource(); require_registration(read(REGISTERED)); require_launch(REGISTERED)
    if QUEUE.exists():
        raise ValueError('No duplicate retry submission')
    jobs = []
    try:
        fit = submit(['--hold', '--export=ALL,JEV_RI_REPLAY_V2_PHASE=fit']); jobs.append(fit)
        check = submit(['--time=02:00:00', '--job-name=jev-ri-canonical-audit',
            '--dependency=afterok:'+fit, '--kill-on-invalid-dep=yes',
            '--export=ALL,JEV_RI_REPLAY_V2_PHASE=audit']); jobs.append(check)
        capture = submit(['--time=00:05:00', '--cpus-per-task=1', '--mem=4G',
            '--job-name=jev-ri-canonical-terminal', '--dependency=afterany:'+fit+':'+check,
            '--kill-on-invalid-dep=yes', '--export=ALL,JEV_RI_REPLAY_V2_PHASE=capture']); jobs.append(capture)
        actual = {name: scheduler(job) for name, job in [('fit', fit), ('audit', check), ('capture', capture)]}
        if any(r.returncode for r in actual.values()):
            raise ValueError('Cannot verify newly created scheduler jobs')
        fv, av, cv = (fields(actual[name].stdout) for name in ('fit', 'audit', 'capture'))
        if (fv.get('Reason') != 'JobHeldUser' or fv.get('JobState') != 'PENDING'
                or any(v.get('Command') != str(ENTRY) for v in (fv, av, cv))
                or f'afterok:{fit}(' not in av.get('Dependency', '')
                or any(f'afterany:{j}(' not in cv.get('Dependency', '') for j in (fit, check))):
            raise ValueError('Held finite fit/audit/terminal dependency contract differs')
        bound = read(RUN/'launch-binding-private.json')['files']
        frozen = dict(bound)
        for p in (RUN/'launch-gate-private.json', PROOF, FAILED): frozen[str(p)] = sha(p)
        write(QUEUE, dict(status='canonical_retry_jobs_bound_while_held', fit_job=fit,
            audit_job=check, capture_job=capture, files=frozen, allocation=allocation,
            canonical_request_method=METHOD, failed_v1_weights_reused=False,
            actual_observations={k: r.stdout for k, r in actual.items()}))
        subprocess.run(['scontrol', 'release', fit], check=True, capture_output=True, text=True, timeout=20)
        print(json.dumps(dict(status='canonical_retry_fit_released', fit_job=fit,
            audit_job=check, capture_job=capture)), flush=True)
    except BaseException as error:
        cleanup = []
        for job in jobs:
            r = subprocess.run(['scancel', job], capture_output=True, text=True, timeout=20)
            cleanup.append(dict(job=job, returncode=r.returncode, stderr=r.stderr))
        write(RUN/'enqueue-failure-private.json', dict(error=repr(error), cleanup=cleanup))
        raise


def bound_job(role):
    resource(); queue = verify_binding(QUEUE)
    if str(queue[role+'_job']) != os.environ['SLURM_JOB_ID']:
        raise ValueError('Wrong explicitly bound job')
    return queue


def fit_job():
    bound_job('fit'); train(REGISTERED)


def audit_job():
    queue = bound_job('audit'); job = queue['fit_job']; actual = scheduler(job)
    receipt = read(ROOT/'logs'/f'curriculum-ri-balanced-replay-v2-exit-{job}.json')
    if actual.returncode or receipt['phase'] != 'fit':
        raise ValueError('Actual completed fit required')
    scheduler_exit(receipt, actual.stdout, ENTRY)
    write(RUN/'fit-terminal-scheduler-private.json', dict(raw=actual.stdout, exit=receipt))
    audit(REGISTERED, RUN/'final-evidence-private.json')
    write(RUN/'audit-execution-private.json', dict(
        status='actual_balanced_replay2472_full_evidence_independently_audited',
        actual_fit_job=job, canonical_request_method=METHOD,
        automatic_release=False, automatic_stage3B=False))


def capture_job():
    queue = bound_job('capture'); out = RUN/'terminal-v1'; out.mkdir(exist_ok=False)
    results = {}
    for role in ('fit', 'audit'):
        job = queue[role+'_job']; actual = scheduler(job); raw = actual.stdout
        if actual.returncode:
            if role != 'fit': raise ValueError('Actual audit terminal expired before capture')
            raw = read(RUN/'fit-terminal-scheduler-private.json')['raw']
        record = terminal_record(raw, job, ENTRY)
        path = ROOT/'logs'/f'curriculum-ri-balanced-replay-v2-exit-{job}.json'
        receipt = read(path) if path.exists() else None
        if record['successful']:
            if receipt is None or receipt.get('phase') != role:
                raise ValueError('Exact successful phase exit required')
            scheduler_exit(receipt, raw, ENTRY)
        record['exit_receipt'] = receipt; results[role] = record
        write(out/(role+'-terminal-private.json'), record)
    final = RUN/'final-evidence-private.json'
    write(out/'terminal-evidence-private.json', dict(
        status='balanced_replay_actual_terminals_preserved_not_promoted', scheduler_proofs=results,
        strategy_grade=read(final)['stage3_advancement'] if final.exists() else None,
        canonical_request_method=METHOD, automatic_release=False, automatic_stage3B=False))
    paths = [QUEUE, *out.glob('*-private.json')]
    write(out/'completion-binding-private.json', dict(files={str(p): sha(p) for p in paths}))
    print(json.dumps(dict(status='canonical_retry_actual_terminals_preserved_not_promoted')), flush=True)


def prepare():
    resource()
    result = subprocess.run([sys.executable, '-m', 'unittest',
        'tests.core.test_curriculum_ri_canonical_replay_v2',
        'tests.core.test_curriculum_ri_balanced_replay_v2'], capture_output=True, text=True, timeout=90)
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    print(result.stdout+result.stderr, flush=True)
    subprocess.run(['bash', '-n', str(ENTRY)], check=True, timeout=15)
    register(); enqueue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'fit', 'audit', 'capture'), required=True)
    args = parser.parse_args()
    dict(prepare=prepare, fit=fit_job, audit=audit_job, capture=capture_job)[args.phase]()
