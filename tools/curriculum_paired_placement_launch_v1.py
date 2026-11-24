"""Continue only qualified paired-parent final3B through the unchanged Course4."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields, terminal_record
from tools.curriculum_progression_parent_v1 import extend
from tools.curriculum_placement_parent_v1 import require_final
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

A = ROOT/'runs/curriculum-progression-riichi-paired-v1'
B = ROOT/'runs/curriculum-yaku-paired-v1'
RUN = ROOT/'runs/curriculum-placement-paired-v1'
CONFIG = ROOT/'configs/training/curriculum-placement-paired-v1.json'
ENTRY = ROOT/'cluster/curriculum-placement-paired-v1.slurm'
B_ENTRY = ROOT/'cluster/curriculum-yaku-paired-v1.slurm'
BINDING = A/'placement-continuation-launch-binding-private.json'
QUEUE = A/'placement-continuation-queue-private.json'
PUBLIC = (
    'tools/curriculum_paired_placement_launch_v1.py',
    'tools/curriculum_placement_paired_parent_v1.py',
    'tools/prepare_curriculum_placement_paired_launch_v1.py',
    'configs/training/curriculum-placement-paired-v1.json',
    'cluster/curriculum-placement-paired-v1.slurm',
    'tests/core/test_curriculum_paired_placement_launch_v1.py',
    'docs/experiments/plans/curriculum-paired-placement-continuation-20261124.md',
)


def require_same_recipe(candidate, original):
    expected = dict(original)
    expected.update(run_id='jev-curriculum-placement-paired-parent-v1',
        stage='stage4-placement-paired-parent-v1',
        plan_root='runs/curriculum-placement-paired-v1/fit-plan',
        out='runs/curriculum-placement-paired-v1/stage4-fit')
    if candidate != expected:
        raise ValueError('Only explicit run/output identities may change; retain the original Course4 recipe')


def bind():
    allocation = resource()
    require_same_recipe(read(CONFIG), read(ROOT/'configs/training/curriculum-placement-fit-v1.json'))
    jobs = read(A/'yaku-continuation-queue-private.json')
    fit = scheduler(jobs['actual_parent_fit_job']); actual = fields(fit.stdout)
    last = None
    with (A/'stage3-fit/metrics/train_metrics.jsonl').open() as stream:
        for line in stream:
            if line.strip(): last = json.loads(line)['local_step']
    if (fit.returncode or actual.get('JobState') != 'RUNNING'
            or actual.get('Command') != str(ROOT/'cluster/curriculum-progression-paired-v1.slurm')
            or last is None or not 0 < last < 2400 or RUN.exists() or B.exists()):
        raise ValueError('Bind before current3A completion and conditional3B preparation')
    observed = {}
    for role in ('fit', 'audit'):
        job = jobs[role+'_job']; result = scheduler(job); v = fields(result.stdout)
        if (result.returncode or v.get('JobId') != str(job)
                or v.get('JobState') != 'PENDING' or v.get('Command') != str(B_ENTRY)):
            raise ValueError('The actual unchanged conditional3B chain must still be pending')
        observed[role] = result.stdout
    paths = [ROOT/name for name in PUBLIC]
    paths += [ROOT/'configs/training/curriculum-placement-fit-v1.json',
        ROOT/'configs/training/curriculum-placement-assessment-v1.json',
        ROOT/'tools/curriculum_placement_parent_v1.py',
        ROOT/'tools/prepare_curriculum_placement_launch_v1.py',
        A/'yaku-continuation-queue-private.json',
        A/'yaku-terminal-capture-binding-v1-private.json',
        A/'yaku-terminal-capture-v1-queue-private.json']
    write(BINDING, dict(status='unchanged_placement_entry_bound_before_paired3A_completion',
        observed_parent_step=last, actual_parent_scheduler=fit.stdout,
        pending_yaku_scheduler=observed, files={str(p.resolve()): sha(p) for p in paths},
        binding_allocation=allocation, student_updates=0, parent_not_admitted=True,
        strategy_grade_overridden=False))
    print(json.dumps(dict(status='conditional_placement_entry_bound_not_admitted',
        observed_parent_step=last, student_updates=0)), flush=True)


def verify_binding():
    b = read(BINDING)
    if (b['status'] != 'unchanged_placement_entry_bound_before_paired3A_completion'
            or not b['parent_not_admitted'] or b['strategy_grade_overridden']
            or not 0 < b['observed_parent_step'] < 2400
            or any(sha(p) != h for p, h in b['files'].items())):
        raise ValueError('Original before-completion continuation source binding changed')
    return b


def require_queued_job(role):
    q = read(QUEUE); job = str(q[role+'_job'])
    actual = scheduler(job); v = fields(actual.stdout)
    if (os.environ.get('SLURM_JOB_ID') != job or actual.returncode
            or v.get('JobId') != job or v.get('JobState') != 'RUNNING'
            or v.get('Command') != str(ENTRY)):
        raise ValueError('Exact actual queued role on the original GPU entry required')
    return q


def normalized_parent_terminal(parent, done, evidence, execution, captured, binding, registered):
    if (binding['status'] != 'conditional_yaku_terminal_capture_bound_before_fit_start'
            or binding['automatic_fit'] or binding['strategy_grade_overridden']
            or captured['status'] != 'conditional_yaku_actual_terminals_preserved_not_placement_admitted'
            or not captured['both_scheduler_jobs_successful']
            or captured['strategy_grade_overridden'] or captured['child_parent_sealed']
            or captured['original_final_evidence'] != evidence
            or set(captured['scheduler_proofs']) != {'fit', 'audit'}):
        raise ValueError('Original successful real3B terminals and unchanged final evidence required')
    proofs = {}
    for role in ('fit', 'audit'):
        p = captured['scheduler_proofs'][role]; job = str(binding[role+'_job'])
        receipt = p['exit_receipt']
        allowed = {'actual_terminal_scontrol'}
        if role == 'fit': allowed.add('original_independent_audit_fit_archive')
        if (p['job_id'] != job or not p['successful'] or p['source'] not in allowed
                or receipt is None or str(receipt['job_id']) != job or receipt['phase'] != role):
            raise ValueError('Wrong original terminal role, source, receipt or job')
        scheduler_exit(receipt, p['raw'], B_ENTRY)
        source = ('original_independent_audit_scontrol_archive'
            if p['source'] == 'original_independent_audit_fit_archive' else p['source'])
        proofs[role] = dict(source=source, raw=p['raw'], receipt=receipt)
    result = dict(status='actual_final3B_fit_and_audit_terminal_evidence_preserved_not_child_admitted',
        selected_step=2500, parent_registration=str(registered.resolve()),
        final_identity=evidence['final_identity'], final_policy_identity=evidence['final_policy_identity'],
        original_strategy_grade=evidence['stage3B_advancement'], scheduler_proofs=proofs,
        parent_fit_job=int(binding['fit_job']), parent_audit_job=int(binding['audit_job']),
        strategy_grade_overridden=False, child_parent_sealed=False, student_updates=0,
        automatic_release=False, raw_private=True)
    # The original qualification contract is reused without changing any grade.
    require_final(parent, done, evidence, execution, result)
    return result


def prepare():
    gpu_only(); b = verify_binding(); q = require_queued_job('prepare')
    require_same_recipe(read(CONFIG), read(ROOT/'configs/training/curriculum-placement-fit-v1.json'))
    registered = B/'fit-plan/registered.json'; parent = read(registered)
    fit = ROOT/parent['curriculum']['out']; evidence = read(B/'final-evidence-private.json')
    capture_dir = A/'yaku-terminal-capture-v1'
    capture_binding_path = A/'yaku-terminal-capture-binding-v1-private.json'
    captured_binding = read(capture_binding_path)
    files = dict(b['files'])
    extend(files, capture_dir/'completion-binding-private.json')
    for path, expected in captured_binding['files'].items():
        if path in files and files[path] != expected: raise ValueError('Conflicting frozen source')
        files[path] = expected
    for path in (BINDING, QUEUE): files[str(path.resolve())] = sha(path)
    if any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('Frozen capture or continuation source changed')
    terminal = normalized_parent_terminal(parent, read(fit/'completion.json'), evidence,
        read(B/'audit-execution-private.json'), read(capture_dir/'terminal-evidence-private.json'),
        captured_binding, registered)
    if (int(q['parent_audit_job']) != terminal['parent_audit_job']
            or int(q['parent_capture_job']) != int(read(A/'yaku-terminal-capture-v1-queue-private.json')['capture_job'])):
        raise ValueError('Queued prerequisites belong to another actual parent')
    if RUN.exists(): raise ValueError('Never overwrite a parent, registration or fit')
    RUN.mkdir(); out = RUN/'parent-terminal-v1'; out.mkdir()
    for path in (registered, B/'final-evidence-private.json', B/'audit-execution-private.json',
                 fit/'completion.json'):
        files[str(path.resolve())] = sha(path)
    write(out/'terminal-evidence-private.json', terminal)
    files[str((out/'terminal-evidence-private.json').resolve())] = sha(out/'terminal-evidence-private.json')
    write(out/'completion-binding-private.json', dict(files=files))
    request = RUN/'parent-request-private.json'
    write(request, dict(parent_selection='actual_registered_final2500_only',
        admission_mode='original_stage3B_grade_pass_required', parent_registration=str(registered),
        parent_terminal=str(out/'terminal-evidence-private.json'),
        parent_fit_job=terminal['parent_fit_job'], parent_audit_job=terminal['parent_audit_job'],
        child_data_ready='runs/curriculum-placement-exposures-v1/data-v1/data-ready-private.json',
        original_child_data_ready='runs/curriculum-placement-data-v3/data-v1/data-ready-private.json',
        out='runs/curriculum-placement-paired-v1/parent', automatic_fit=False))
    from tools.curriculum_placement_paired_parent_v1 import seal
    seal(request)
    from tools import register_curriculum_placement_v1 as registration
    original = registration.PUBLIC_FILES
    registration.PUBLIC_FILES = tuple(dict.fromkeys((*original, *PUBLIC)))
    try: registration.register(CONFIG, RUN/'parent/parent-seal-private.json')
    finally: registration.PUBLIC_FILES = original
    write(RUN/'launch-request-private.json', dict(
        status='planned_full_placement_launch_after_actual_integration',
        registered=str(RUN/'fit-plan/registered.json'), integration_out=str(RUN/'upstream-integration-v1'),
        integration_job=int(q['integration_job']),
        criteria_config='configs/training/curriculum-placement-assessment-v1.json',
        automatic_release=False, open_independent_confirmation=False))
    print(json.dumps(dict(status='qualified_paired3B_complete_placement_data_registered_not_fit_admitted',
        student_updates=0, integration_pending=True)), flush=True)


def integration():
    gpu_only(); verify_binding(); require_queued_job('integration')
    from tools.preflight_curriculum_placement_v1 import preflight
    preflight(RUN/'fit-plan/registered.json', RUN/'upstream-integration-v1')


def train():
    gpu_only(); verify_binding(); require_queued_job('fit')
    from tools.prepare_curriculum_placement_paired_launch_v1 import prepare as admit
    admit(RUN/'launch-request-private.json')
    from tools.train_curriculum_placement_course_v1 import train as original_train
    original_train(RUN/'fit-plan/registered.json')


def audit():
    gpu_only(); verify_binding(); q = require_queued_job('audit')
    registered = RUN/'fit-plan/registered.json'; launch = read(RUN/'launch-binding-private.json')
    if launch['registered_sha256'] != sha(registered) or any(sha(p) != h for p, h in launch['files'].items()):
        raise ValueError('Original immutable fit launch differs')
    job = q['fit_job']; receipt = read(ROOT/'logs'/f'curriculum-placement-paired-exit-{job}.json')
    if receipt['phase'] != 'fit': raise ValueError('Actual successful full FIT receipt required')
    from tools.prepare_curriculum_placement_launch_v1 import terminal
    terminal(job, receipt, ENTRY, RUN/'fit-terminal-scheduler-private.json')
    from tools.audit_curriculum_placement_course_v1 import audit as original_audit
    path = RUN/'final-evidence-private.json'; original_audit(registered, path)
    if any(sha(p) != h for p, h in launch['files'].items()): raise ValueError('Audit changed frozen source')
    write(RUN/'audit-execution-private.json', dict(
        status='actual_full_placement_evidence_independently_audited', actual_fit_job=int(job),
        original_launch_binding_unchanged=True, development_grade_passed=read(path)['development_assessment']['passed'],
        strategy_grade_overridden=False, independent_v8_confirmation_performed=False, automatic_release=False))


def capture():
    allocation = resource(); verify_binding(); q = require_queued_job('capture')
    out = A/'placement-terminal-capture-v1'; out.mkdir(exist_ok=False)
    proofs = {}; successful = True
    for role in ('fit', 'audit'):
        job = str(q[role+'_job']); actual = scheduler(job)
        write(out/(role+'-scheduler-observation-private.json'), dict(
            stdout=actual.stdout, stderr=actual.stderr, returncode=actual.returncode))
        path = ROOT/'logs'/f'curriculum-placement-paired-exit-{job}.json'
        receipt = read(path) if path.exists() else None
        raw = actual.stdout; source = 'actual_terminal_scontrol'
        archive = RUN/'fit-terminal-scheduler-private.json'
        if actual.returncode and role == 'fit' and archive.exists():
            saved = read(archive)
            if saved['receipt'] != receipt or saved['source'] != 'actual_terminal_scontrol':
                raise ValueError('Exact original fit archive and receipt required')
            raw = saved['raw']; source = 'original_independent_audit_fit_archive'
        proof = terminal_record(raw, job, ENTRY)
        if proof['successful']:
            if receipt is None or receipt.get('phase') != role: raise ValueError('Actual phase receipt required')
            scheduler_exit(receipt, raw, ENTRY)
        proof.update(source=source, exit_receipt=receipt); proofs[role] = proof
        successful = successful and proof['successful']
        write(out/(role+'-terminal-private.json'), proof)
    evidence = RUN/'final-evidence-private.json'
    write(out/'terminal-evidence-private.json', dict(
        status='actual_paired_placement_terminals_preserved_not_independent_confirmation_admitted',
        scheduler_proofs=proofs, both_scheduler_jobs_successful=successful,
        original_final_evidence=read(evidence) if evidence.exists() else None,
        strategy_grade_overridden=False, independent_confirmation_registered=False,
        this_task_optimizer_updates=0, automatic_release=False, capture_allocation=allocation))
    paths = [BINDING, QUEUE, *out.glob('*-private.json')]
    write(out/'completion-binding-private.json', dict(files={str(p.resolve()): sha(p) for p in paths}))
    print(json.dumps(dict(status='actual_placement_terminals_preserved',
        both_jobs_successful=successful, confirmation_admitted=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['bind', 'prepare', 'integration', 'fit', 'audit', 'capture'], required=True)
    phase = parser.parse_args().phase
    {'bind': bind, 'prepare': prepare, 'integration': integration,
     'fit': train, 'audit': audit, 'capture': capture}[phase]()
