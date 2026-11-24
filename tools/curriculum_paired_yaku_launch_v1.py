"""Continue a qualified paired 3A final through the unchanged fixed 3B recipe."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_yaku_paired_parent_v1 import require_qualified_parent, seal, extend
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields

RUN = ROOT/'runs/curriculum-yaku-paired-v1'
CONFIG = ROOT/'configs/training/curriculum-yaku-paired-v1.json'
PARENT_RUN = ROOT/'runs/curriculum-progression-riichi-paired-v1'
BINDING = PARENT_RUN/'yaku-continuation-launch-binding-private.json'
PUBLIC = ('tools/curriculum_paired_yaku_launch_v1.py',
    'tools/curriculum_yaku_paired_parent_v1.py',
    'configs/training/curriculum-yaku-paired-v1.json',
    'cluster/curriculum-yaku-paired-v1.slurm',
    'tests/core/test_curriculum_paired_yaku_launch_v1.py',
    'docs/experiments/plans/curriculum-paired-yaku-continuation-20261124.md')


def require_same_recipe(candidate, original):
    expected = dict(original)
    del expected['parent_fit_job']; del expected['parent_audit_job']
    expected.update(run_id='jev-curriculum-yaku-paired-parent-v1',
        stage='stage3B-yaku-paired-parent-v1',
        plan_root='runs/curriculum-yaku-paired-v1/fit-plan',
        out='runs/curriculum-yaku-paired-v1/stage3B-fit',
        parent_registration='runs/curriculum-progression-riichi-paired-v1/fit-plan/registered.json',
        parent_evidence='runs/curriculum-progression-riichi-paired-v1/final-evidence-private.json',
        parent_jobs_from='runs/curriculum-progression-riichi-paired-v1/terminal-capture-v2-queue-private.json',
        parent_terminal_capture_binding='runs/curriculum-progression-riichi-paired-v1/terminal-capture-binding-v2-private.json')
    if candidate != expected:
        raise ValueError('Only explicit parent/runtime/output identities may change; keep the original 3B recipe')


def fixed_inputs():
    paths = [ROOT/name for name in PUBLIC]
    paths += [ROOT/'configs/training/curriculum-yaku-fit-v2.json',
        PARENT_RUN/'fit-plan/registered.json',
        PARENT_RUN/'terminal-capture-binding-v2-private.json']
    return paths


def bind():
    allocation = resource(); c = read(CONFIG)
    require_same_recipe(c, read(ROOT/'configs/training/curriculum-yaku-fit-v2.json'))
    capture = read(ROOT/c['parent_terminal_capture_binding'])
    actual = scheduler(capture['fit_job']); v = fields(actual.stdout)
    step = None
    with (PARENT_RUN/'stage3-fit/metrics/train_metrics.jsonl').open() as stream:
        for line in stream:
            if line.strip(): step = json.loads(line)['local_step']
    if (BINDING.exists() or actual.returncode or v.get('JobState') != 'RUNNING'
            or v.get('JobId') != str(capture['fit_job'])
            or v.get('Command') != str(ROOT/'cluster/curriculum-progression-paired-v1.slurm')
            or step is None or not 0 < step < 2400):
        raise ValueError('Bind the immutable continuation entry before the actual paired fit completes')
    write(BINDING, dict(status='unchanged_yaku_recipe_entry_bound_during_paired3A_fit',
        observed_parent_step=step, observed_parent_scheduler=actual.stdout,
        binding_allocation=allocation, files={str(p.resolve()): sha(p) for p in fixed_inputs()},
        student_updates=0, parent_not_admitted=True, strategy_grade_overridden=False))
    print(json.dumps(dict(status='conditional_yaku_entry_bound_not_admitted', observed_parent_step=step)), flush=True)


def verify_binding():
    b = read(BINDING)
    if (b['status'] != 'unchanged_yaku_recipe_entry_bound_during_paired3A_fit'
            or not b['parent_not_admitted'] or b['strategy_grade_overridden']
            or not 0 < b['observed_parent_step'] < 2400
            or any(sha(p) != h for p, h in b['files'].items())):
        raise ValueError('Before-completion continuation entry binding changed')
    return b


def prepare():
    gpu_only(); b = verify_binding(); c = read(CONFIG)
    require_same_recipe(c, read(ROOT/'configs/training/curriculum-yaku-fit-v2.json'))
    queued = read(ROOT/c['parent_jobs_from'])
    c.update(parent_fit_job=int(queued['fit_job']), parent_audit_job=int(queued['audit_job']),
        parent_terminal_capture_job=int(queued['capture_job']))
    parent = read(ROOT/c['parent_registration'])
    fit = ROOT/parent['curriculum']['out']
    require_qualified_parent(parent, read(fit/'completion.json'), read(ROOT/c['parent_evidence']),
        read(PARENT_RUN/'audit-execution-private.json'), c)
    old_run = ROOT/'runs/curriculum-yaku-course-v1'
    preflight = old_run/'launch-preflight-v1/summary-private.json'
    io = old_run/'launch-io-fix-v2/summary-private.json'
    if (read(preflight)['status'] != 'yaku_launch_full_synthetic_validation_integration_passed_not_parent_or_fit_admitted'
            or read(io)['status'] != 'actual_yaku_conditional_config_directory_and_exclusive_write_checked'):
        raise ValueError('Reuse the actual unchanged full validation and exclusive I/O integrations')
    files = dict(b['files']); files[str(BINDING)] = sha(BINDING)
    extend(files, preflight.parent/'completion-binding-private.json')
    extend(files, io.parent/'completion-binding-private.json')
    for path in (preflight, io, ROOT/c['parent_jobs_from']): files[str(path.resolve())] = sha(path)
    if RUN.exists(): raise ValueError('Never overwrite a continuation parent, registration or fit')
    RUN.mkdir()
    request = RUN/'parent-request-private.json'; write(request, c)
    seal(request, RUN/'parent')
    from tools import register_curriculum_yaku_course_v3 as registration
    original = registration.PUBLIC_FILES
    registration.PUBLIC_FILES = tuple(dict.fromkeys((*original, *PUBLIC)))
    try:
        registration.register(request, RUN/'parent/parent-seal-private.json')
    finally:
        registration.PUBLIC_FILES = original
    registered = RUN/'fit-plan/registered.json'; reg = read(registered)
    files.update(reg['inputs']); files[str(registered)] = sha(registered)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Continuation launch input changed')
    write(RUN/'launch-binding-private.json', dict(files=files, registered_sha256=sha(registered),
        candidate_step=2500, actual_registration_before_first_update=True, optimizer_updates=0,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname()))
    write(RUN/'prepare-execution-private.json', dict(status='actual_yaku_parent_sealed_and2500_recipe_registered',
        registered_sha256=sha(registered), bound_inputs=len(files), optimizer_updates=0, fit_started=False,
        original_3A_grade_preserved=True, automatic_release=False))
    print(json.dumps(dict(status='qualified_paired_parent_yaku2500_registered', student_updates=0)), flush=True)


def train():
    gpu_only(); verify_binding()
    from tools.train_curriculum_yaku_course_v3 import train as original_train
    original_train(RUN/'fit-plan/registered.json')


def audit(fit_job):
    gpu_only(); verify_binding()
    receipt_path = ROOT/'logs'/f'curriculum-yaku-paired-exit-{fit_job}.json'
    receipt = read(receipt_path)
    if receipt['phase'] != 'fit': raise ValueError('Actual yaku FIT phase exit required')
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    archive = RUN/'fit-terminal-scheduler-private.json'
    if archive.exists():
        captured = read(archive)
        if captured['exit'] != receipt: raise ValueError('Actual yaku fit receipt changed')
        raw = captured['raw']
    else:
        observed = scheduler(fit_job)
        if observed.returncode: raise ValueError('Actual yaku fit terminal missing')
        raw = observed.stdout
        scheduler_exit(receipt, raw, ROOT/'cluster/curriculum-yaku-paired-v1.slurm')
        write(archive, dict(raw=raw, exit=receipt))
    scheduler_exit(receipt, raw, ROOT/'cluster/curriculum-yaku-paired-v1.slurm')
    launch = read(RUN/'launch-binding-private.json')
    if any(sha(p) != h for p, h in launch['files'].items()): raise ValueError('Bound yaku launch input changed')
    from tools.audit_curriculum_yaku_course_v1 import audit as original_audit
    original_audit(RUN/'fit-plan/registered.json', RUN/'final-evidence-private.json')
    if any(sha(p) != h for p, h in launch['files'].items()): raise ValueError('Audit changed bound yaku source')
    write(RUN/'audit-execution-private.json', dict(status='actual_yaku2500_full_evidence_independently_audited',
        actual_fit_job=fit_job, original_launch_binding_unchanged=True, automatic_release=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['bind', 'prepare', 'fit', 'audit'], required=True)
    p.add_argument('--fit-job', type=int)
    a = p.parse_args()
    if a.phase == 'bind': bind()
    elif a.phase == 'prepare': prepare()
    elif a.phase == 'fit': train()
    else:
        if not a.fit_job: p.error('--fit-job required for audit')
        audit(a.fit_job)
