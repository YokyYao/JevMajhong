"""Finite matched calls13-loss control; preserve both failed 3A controls."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_progression_parent_loss_v1 import METHOD
from tools.curriculum_progression_parent_runtime_v1 import require_registration
from tools.curriculum_progression_paired_v1 import require_failed_RI_control
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

CONTROL = ROOT/'runs/curriculum-progression-riichi-paired-v1'
RUN = ROOT/'runs/curriculum-progression-parent-retention-v1'
ENTRY = ROOT/'cluster/curriculum-progression-parent-retention-v1.slurm'
PREFLIGHT = CONTROL/'parent-retention-launch-preflight-v1'
PUBLIC = (
    'tools/curriculum_progression_parent_loss_v1.py',
    'tools/curriculum_progression_parent_runtime_v1.py',
    'tools/curriculum_progression_parent_fit_v1.py',
    'tools/curriculum_progression_parent_control_v1.py',
    'tools/preflight_curriculum_progression_parent_v1.py',
    'tools/train_curriculum_progression_parent_v1.py',
    'tools/evaluate_curriculum_progression_parent_v1.py',
    'tools/audit_curriculum_progression_parent_v1.py',
    'tools/launch_curriculum_progression_parent_v1.py',
    'tests/core/test_curriculum_progression_parent_loss_v1.py',
    'tests/core/test_curriculum_progression_parent_control_v1.py',
    'cluster/curriculum-progression-parent-retention-v1.slurm',
    'cluster/curriculum-parent-retention-terminal-v1.slurm',
    'docs/experiments/plans/curriculum-3A-parent-loss-retention-20261124.md',
    'docs/experiments/plans/curriculum-3A-parent-loss-launch-20261124.md',
)


def matched_recipe(reg, control, args, previous):
    old = control['curriculum']
    expected = dict(old, run_id='jev-curriculum-progression-parent-retention-v1',
        stage='stage3-parent-loss-retained-v1',
        out=str(RUN.relative_to(ROOT)/'stage3-fit'),
        plan_root=str(RUN.relative_to(ROOT)/'fit-plan'), loss_method=METHOD)
    if reg['curriculum'] != expected or args != dict(previous, out=str(RUN/'stage3-fit/training-run')):
        raise ValueError('Only output identity and the replay objective may change')
    for key in ('source', 'source_sha256', 'parent_model', 'course_loss_weights',
                'course_loss_mass', 'expected_record_exposures', 'train_records',
                'validation_records', 'registered_final_step', 'expected_orders'):
        if reg[key] != control[key]:
            raise ValueError('Matched parent/data/budget/order differs: '+key)
    cfg = dict(control['training_config'], model_id=expected['run_id'], out=expected['out'],
        trainer_config=str(RUN/'fit-plan/trainer.toml'),
        trainer_config_sha256=reg['training_config']['trainer_config_sha256'], loss_method=METHOD)
    if reg['training_config'] != cfg or reg.get('both_failed_3A_parameter_updates_excluded') is not True:
        raise ValueError('Initializer/configuration differs from the isolated control')


def verify(reg):
    require_registration(reg)
    control = read(reg['parent_loss_control_registration'])
    from tools.curriculum_progression_paired_v1 import verify_retry
    verify_retry(control)
    matched_recipe(reg, control,
        tomllib.loads(Path(reg['training_config']['trainer_config']).read_text()),
        tomllib.loads(Path(control['training_config']['trainer_config']).read_text()))
    for path, expected in reg['inputs'].items():
        if sha(path) != expected:
            raise ValueError('Frozen parent-retention source/input changed: '+path)


def prepare():
    gpu_only()
    control_path = CONTROL/'fit-plan/registered.json'
    old = read(control_path)
    from tools.curriculum_progression_paired_v1 import verify_retry
    from tools.curriculum_progression_parent_v1 import verify_parent
    verify_retry(old)
    fit = ROOT/old['curriculum']['out']
    paths = [control_path, fit/'completion.json', CONTROL/'final-evidence-private.json',
        CONTROL/'audit-execution-private.json', CONTROL/'fit-terminal-scheduler-private.json',
        fit/'metrics/validation_metrics.jsonl', CONTROL/'launch-binding-private.json',
        PREFLIGHT/'summary-private.json', PREFLIGHT/'completion-binding-private.json']
    queue_path = CONTROL/'parent-retention-queue-v1-private.json'
    queue = read(queue_path)
    import os
    if str(queue['fit_job']) != os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Preparation must run in the explicitly bound fit job')
    for p, h in queue['files'].items():
        if sha(p) != h: raise ValueError('Held submission source binding changed: '+p)
    paths.append(queue_path)
    done, evidence, execution, terminal = (read(p) for p in paths[1:5])
    last = [json.loads(line) for line in paths[5].open()][-1]
    trigger = require_failed_RI_control(done, evidence, execution, last, old['curriculum']['development_targets'])
    scheduler_exit(terminal['exit'], terminal['raw'], ROOT/'cluster/curriculum-progression-paired-v1.slurm')
    if str(execution['actual_fit_job']) != str(terminal['exit']['job_id']):
        raise ValueError('Actual independent control audit/fit receipt differs')
    pf = read(PREFLIGHT/'summary-private.json'); binding = read(PREFLIGHT/'completion-binding-private.json')
    if (pf['status'] != 'full_validation_parent_retention_launch_preflight_passed'
            or pf['validation_records'] != 23688 or pf['train_records'] != 87595
            or pf['student_updates'] != 0 or pf['unit_test_returncode'] != 0
            or pf['method'] != METHOD):
        raise ValueError('Full actual TRAIN/VAL launch preflight required')
    for p, expected in binding['files'].items():
        if sha(p) != expected:
            raise ValueError('Preflight-bound implementation changed: '+p)
    sys.path[:0] = [old['source'], old['training_config']['upstream_source']]
    verify_parent(old['training_config'])
    if RUN.exists():
        raise ValueError('Never overwrite a registration or restart a fit silently')
    reg = copy.deepcopy(old)
    c = reg['curriculum']; config = reg['training_config']
    c.update(run_id='jev-curriculum-progression-parent-retention-v1',
        stage='stage3-parent-loss-retained-v1', out=str(RUN.relative_to(ROOT)/'stage3-fit'),
        plan_root=str(RUN.relative_to(ROOT)/'fit-plan'), loss_method=METHOD)
    plan = RUN/'fit-plan'; plan.mkdir(parents=True)
    args = tomllib.loads(Path(old['training_config']['trainer_config']).read_text())
    args['out'] = str(RUN/'stage3-fit/training-run')
    trainer = plan/'trainer.toml'
    trainer.write_text(''.join(f'{key} = {json.dumps(value)}\n' for key, value in args.items()))
    config.update(model_id=c['run_id'], out=c['out'], trainer_config=str(trainer),
        trainer_config_sha256=sha(trainer), loss_method=METHOD)
    config_path = plan/'training-config.json'; write(config_path, config)
    files = dict(old['inputs'])
    for p in paths+[trainer, config_path]+[ROOT/p for p in PUBLIC]:
        files[str(p.resolve())] = sha(p)
    files.update(binding['files'])
    # These claims described the previous order-only experiment, not this one.
    for key in ('same_parent_data_objective_budget_only_epoch_order_changed',
                'paired_control_registration', 'paired_control_failed_RI_grade'):
        reg.pop(key, None)
    reg.update(inputs=files, parent_loss_control_registration=str(control_path),
        parent_loss_control_failed_grade=trigger,
        both_failed_3A_parameter_updates_excluded=True,
        only_replay_objective_changed_from_paired_control=True)
    verify(reg)
    registered = plan/'registered.json'; write(registered, reg)
    write(RUN/'launch-binding-private.json', dict(files={**files, str(registered): sha(registered)},
        registered_sha256=sha(registered), selected_initializer='qualified_calls388',
        both_failed_3A_parameter_updates_excluded=True, new_optimizer_updates=0))
    print(json.dumps(dict(status='parent_retention_registered_not_started', student_updates=0)), flush=True)


def train(registered):
    gpu_only(); verify(read(registered))
    from tools.train_curriculum_progression_parent_v1 import train as fit
    fit(registered)


def audit(registered, fit_job):
    gpu_only(); reg = read(registered); verify(reg)
    receipt = read(ROOT/'logs'/f'curriculum-parent-retention-exit-{fit_job}.json')
    if receipt['phase'] != 'fit':
        raise ValueError('Actual completed FIT phase required')
    raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(fit_job)], text=True, timeout=20)
    scheduler_exit(receipt, raw, ENTRY)
    write(RUN/'fit-terminal-scheduler-private.json', dict(raw=raw, exit=receipt))
    b = read(RUN/'launch-binding-private.json')
    if any(sha(p) != h for p, h in b['files'].items()):
        raise ValueError('Actual launch binding changed')
    from tools.audit_curriculum_progression_parent_v1 import audit as check
    check(registered, RUN/'final-evidence-private.json')
    verify(reg)
    write(RUN/'audit-execution-private.json', dict(
        status='actual_progression2400_full_evidence_independently_audited',
        actual_fit_job=fit_job, original_launch_binding_unchanged=True,
        loss_method=METHOD, automatic_release=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('prepare', 'fit', 'audit'), required=True)
    p.add_argument('--registered', type=Path, default=RUN/'fit-plan/registered.json')
    p.add_argument('--fit-job', type=int)
    a = p.parse_args()
    if a.phase == 'prepare': prepare()
    elif a.phase == 'fit': train(a.registered.resolve())
    else:
        if a.fit_job is None: p.error('--fit-job required for audit')
        audit(a.registered.resolve(), a.fit_job)
