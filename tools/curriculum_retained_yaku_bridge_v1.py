"""Finite after-terminal continuation: preserve exits, admit only real success.

This is a scheduler job, never a poller. The original qualified-parent gate
and already-bound3B launcher remain the sole admission/submission mechanism.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import curriculum_yaku_parent_launch_v1 as child
from tools.curriculum_yaku_retained_parent_v1 import require_qualified_parent
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields, terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

RUN = child.PARENT_RUN/'automatic-yaku-bridge-v1'
BINDING = RUN/'binding-private.json'
QUEUE = RUN/'queue-private.json'
ENTRY = ROOT/'cluster/curriculum-retained-yaku-bridge-v1.slurm'
CAPTURE_ENTRY = ROOT/'cluster/curriculum-parent-retention-terminal-v1.slurm'
PUBLIC = ('tools/curriculum_retained_yaku_bridge_v1.py',
    'tests/core/test_curriculum_retained_yaku_bridge_v1.py',
    'cluster/curriculum-retained-yaku-bridge-v1.slurm',
    'docs/experiments/plans/curriculum-retained-yaku-bridge-20261124.md')


def require_parent(parent, done, evidence, execution, cap, config):
    require_qualified_parent(parent, done, evidence, execution, config)
    if (cap['status'] != 'parent_retention_actual_terminals_preserved_not_child_admitted'
            or cap['original_strategy_grade'] != evidence['stage3_advancement']
            or cap['strategy_grade_overridden'] or cap['child_parent_sealed'] or cap['student_updates'] != 0
            or set(cap['scheduler_proofs']) != {'fit', 'audit'}
            or any(not p['successful'] for p in cap['scheduler_proofs'].values())):
        raise ValueError('Actual original qualification and successful preserved phases required')


def capture_record(raw, receipt, job):
    proof = terminal_record(raw, job, CAPTURE_ENTRY)
    if not proof['successful']: raise ValueError('Successful actual terminal-capture job required')
    scheduler_exit(receipt, raw, CAPTURE_ENTRY)
    return dict(raw=raw, exit=receipt)


def child_submission_needed(queue, directory, expected_binding):
    if queue.exists():
        actual = read(queue)
        if (actual['status'] != 'actual_qualified_parent_yaku_jobs_bound_while_held'
                or actual['binding_sha256'] != expected_binding
                or actual['original_parent_grade_preserved'] is not True):
            raise ValueError('An existing child queue must match the actual frozen launcher')
        return False
    if directory.exists(): raise ValueError('Existing child artifacts without their queue require investigation')
    return True


def prepare_after_bridge(command, job):
    """All newly created compute jobs remain blocked until this bridge succeeds.

    Other dependency chains and cancellation commands pass through unchanged.
    This protects against a timeout even between sbatch and receipt creation.
    """
    if not str(job).isdigit(): raise ValueError('Actual numeric bridge job required')
    if command and command[0] == 'sbatch' and '--export=ALL,JEV_YAKU_PARENT_PHASE=prepare' in command:
        if any(v.startswith('--dependency') for v in command):
            raise ValueError('Original preparation unexpectedly already has a dependency')
        return [command[0], '--dependency=afterok:'+str(job), '--kill-on-invalid-dep=yes', *command[1:]]
    return command


def require_held(raw, capture_job):
    v = fields(raw); tres = dict(t.split('=', 1) for t in v.get('ReqTRES', '').split(',') if '=' in t)
    if (v.get('JobState') != 'PENDING' or v.get('Reason') != 'JobHeldUser'
            or v.get('Command') != str(ENTRY) or f'afterany:{capture_job}(' not in v.get('Dependency', '')
            or v.get('TimeLimit') != '00:05:00' or v.get('NumCPUs') != '1'
            or tres.get('gres/gpu') != '1' or tres.get('mem') != '4G'):
        raise ValueError('Actual held5-minute GPU bridge with the exact terminal dependency required')


def verify_binding():
    binding = read(BINDING)
    if (binding['status'] != 'finite_yaku_bridge_bound_before_parent_completion'
            or binding['strategy_grade_overridden'] or binding['student_updates'] != 0
            or any(sha(p) != h for p, h in binding['files'].items())):
        raise ValueError('Before-completion source and original admission binding changed')
    return binding


def bind():
    allocation = resource(); child.verify_binding(); c = read(child.CONFIG)
    parent_queue = ROOT/c['parent_jobs_from']; queue = read(parent_queue)
    fit = scheduler(queue['fit_job']); cap = scheduler(queue['capture_job'])
    fv, cv = fields(fit.stdout), fields(cap.stdout)
    if (fit.returncode or cap.returncode or fv.get('JobState') != 'RUNNING'
            or fv.get('JobId') != str(queue['fit_job'])
            or fv.get('Command') != str(ROOT/'cluster/curriculum-progression-parent-retention-v1.slurm')
            or cv.get('JobId') != str(queue['capture_job'])
            or cv.get('JobState') != 'PENDING' or cv.get('Command') != str(CAPTURE_ENTRY)
            or any(f'afterany:{queue[role+"_job"]}(' not in cv.get('Dependency', '') for role in ('fit', 'audit'))):
        raise ValueError('Bind while the actual parent is live and its terminal job is pending')
    if BINDING.exists(): raise ValueError('Never overwrite a bridge binding')
    test_log = ROOT/'logs/curriculum-retained-yaku-bridge-tests-v2.log'; log = test_log.read_text()
    if 'Ran 6 tests' not in log or not log.endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Six admission/capture/duplicate/held-dependency tests must pass first')
    subprocess.run(['bash', '-n', str(ENTRY)], check=True, timeout=10)
    files = {str(p.resolve()): sha(p) for p in (parent_queue, child.BINDING, child.CONFIG, test_log,
        ROOT/'tools/curriculum_yaku_parent_launch_v1.py', *[ROOT/p for p in PUBLIC])}
    RUN.mkdir(parents=True, exist_ok=True)
    write(BINDING, dict(status='finite_yaku_bridge_bound_before_parent_completion', files=files,
        fit_job=str(queue['fit_job']), audit_job=str(queue['audit_job']), capture_job=str(queue['capture_job']),
        actual_parent_fit_observed=fit.stdout, actual_capture_dependency_observed=cap.stdout,
        original_child_binding_sha256=sha(child.BINDING), allocation=allocation,
        strategy_grade_overridden=False, student_updates=0, no_polling=True))
    print(json.dumps(dict(status='finite_yaku_bridge_bound_not_submitted', student_updates=0)), flush=True)


def submit():
    resource(); binding = verify_binding()
    if QUEUE.exists(): raise ValueError('Never submit a duplicate bridge')
    result = subprocess.run(['sbatch', '--parsable', '--hold', '--time=00:05:00', '--cpus-per-task=1',
        '--mem=4G', '--gres=gpu:nvidia-rtx-5000-ada-generation:1', '--nice=0',
        '--dependency=afterany:'+binding['capture_job'], '--kill-on-invalid-dep=yes', str(ENTRY)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    job = result.stdout.strip().split(';')[0]
    if result.returncode or not job.isdigit(): raise ValueError(result.stderr or result.stdout)
    try:
        actual = scheduler(job)
        if actual.returncode: raise ValueError('Actual held bridge could not be observed')
        require_held(actual.stdout, binding['capture_job'])
        write(QUEUE, dict(status='finite_yaku_bridge_actual_held_job_bound', job_id=job,
            binding_sha256=sha(BINDING), capture_job=binding['capture_job'], observed_held_job=actual.stdout,
            child_not_admitted_by_submission=True, student_updates=0))
        subprocess.run(['scontrol', 'release', job], check=True, capture_output=True, text=True, timeout=20)
        print(json.dumps(dict(status='finite_yaku_bridge_submitted', job_id=job,
            after_actual_terminal_capture=binding['capture_job'], child_updates=0)), flush=True)
    except BaseException:
        subprocess.run(['scancel', job], capture_output=True, text=True, timeout=20)
        raise


def preserve_capture(job):
    archive = child.PARENT_RUN/'capture-terminal-for-yaku-private.json'
    receipt = read(ROOT/'logs'/f'curriculum-parent-terminal-exit-{job}.json')
    if archive.exists():
        saved = read(archive)
        if saved['exit'] != receipt: raise ValueError('Previously saved capture receipt differs')
        capture_record(saved['raw'], receipt, job)
        return archive
    actual = scheduler(job); prior = child.PARENT_RUN/'parent-retention-terminal-v1/capture-job-observed-completed-private.txt'
    if actual.returncode and not prior.exists():
        raise ValueError('Real capture terminal missing; never invent an expired record')
    raw = actual.stdout if not actual.returncode else prior.read_text()
    write(archive, capture_record(raw, receipt, job))
    return archive


def execute():
    allocation = resource(); binding = verify_binding(); queue = read(QUEUE)
    if queue['job_id'] != os.environ['SLURM_JOB_ID'] or queue['binding_sha256'] != sha(BINDING):
        raise ValueError('Only the actually bound finite bridge job may execute')
    out = RUN/'execution-v1'; out.mkdir(exist_ok=False)
    c = read(child.CONFIG); parent_queue = read(ROOT/c['parent_jobs_from'])
    c.update(parent_fit_job=int(parent_queue['fit_job']), parent_audit_job=int(parent_queue['audit_job']),
        parent_terminal_capture_job=int(parent_queue['capture_job']))
    try:
        archive = preserve_capture(binding['capture_job'])
        parent = read(ROOT/c['parent_registration']); fit = ROOT/parent['curriculum']['out']
        evidence = read(ROOT/c['parent_evidence'])
        cap_path = child.PARENT_RUN/'parent-retention-terminal-v1/terminal-evidence-private.json'
        if any(sha(p) != h for p, h in read(cap_path.parent/'completion-binding-private.json')['files'].items()):
            raise ValueError('Actual completed parent terminal artifacts changed')
        require_parent(parent, read(fit/'completion.json'), evidence,
            read(child.PARENT_RUN/'audit-execution-private.json'),
            read(cap_path), c)
        child.verify_binding()
        if child_submission_needed(child.QUEUE, child.RUN, sha(child.BINDING)):
            original_process_api = child.subprocess
            child.subprocess = SimpleNamespace(run=lambda command, **kwargs:
                original_process_api.run(prepare_after_bridge(command, queue['job_id']), **kwargs))
            try: child.submit()
            finally: child.subprocess = original_process_api
            observed = read(child.QUEUE)['observed_jobs']['prepare']
            if (f'afterok:{queue["job_id"]}(' not in fields(observed).get('Dependency', '')
                    or fields(observed).get('KillOInInvalidDependent') != 'Yes'):
                raise ValueError('The actual child preparation must depend on successful bridge completion')
            action = 'qualified_parent_child_chain_submitted'
        else: action = 'existing_child_chain_preserved_not_restarted'
        write(out/'summary-private.json', dict(status=action, original_parent_grade=evidence['stage3_advancement'],
            actual_child_queue=str(child.QUEUE), actual_capture_archive=str(archive), allocation=allocation,
            strategy_grade_overridden=False, bridge_student_updates=0, no_polling=True))
        print(json.dumps(dict(status=action, child_queue=str(child.QUEUE))), flush=True)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        write(out/'not-admitted-private.json', dict(status='child_not_admitted_missing_or_invalid_original_evidence',
            reason=repr(error), strategy_grade_overridden=False, child_submission_not_retried=True,
            bridge_student_updates=0, allocation=allocation, no_polling=True))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--phase', choices=['bind', 'submit', 'execute'], required=True)
    globals()[p.parse_args().phase]()
