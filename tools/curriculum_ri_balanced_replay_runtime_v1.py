"""A separate prospective fit contract, without changing the active trial."""
import copy
from pathlib import Path
import tomllib

from tools.curriculum_transfer import read, sha
from tools.curriculum_progression_parent_runtime_v1 import policy_contract, prediction_parity, require_step0
from tools.curriculum_ri_inverse_repeat_loss_v1 import METHOD
from tools.curriculum_progression_parent_loss_v1 import MASS, METHOD as VALIDATION_METHOD

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT/'runs/curriculum-progression-parent-retention-v1'
RUN = ROOT/'runs/curriculum-ri-balanced-replay-fit-v1'
POINTS = [0, 618, 1236, 1854, 2472]


def expand_after_unique_admission(replay, rows, report):
    pool, expanded = replay.admit(rows, report)
    unique_report = dict(report, physical_exposure_slots=len(pool),
        physical_expansion_after_unique_token_admission=True)
    return pool, unique_report, expanded


def curriculum_view(old):
    return dict(old, run_id='jev-curriculum-ri-balanced-replay-v1',
        stage='stage3-ri-balanced-replay-v1', out=str(RUN.relative_to(ROOT)/'stage3-fit'),
        plan_root=str(RUN.relative_to(ROOT)/'fit-plan'), loss_method=METHOD,
        expected_record_exposures=180456, validation_and_game_steps=POINTS,
        checkpoint_every_steps=103)


def trainer_view(old):
    return dict(old, out=str(RUN/'stage3-fit/training-run'), max_steps=2472,
        checkpoint_every_steps=103, eval_every_steps=618)


def check_matched(reg, old, args, old_args):
    if (reg['status'] != 'balanced_replay2472_registered_not_started'
            or reg['registered_final_step'] != 2472 or reg['train_records'] != 87595
            or reg['validation_records'] != 23688 or reg['physical_train_slots_per_epoch'] != 90228
            or reg['expected_record_exposures'] != 180456 or reg['course_loss_mass'] != MASS
            or reg['curriculum'] != curriculum_view(old['curriculum'])
            or args != trainer_view(old_args)
            or reg['validation_loss_method'] != VALIDATION_METHOD
            or reg['standby_only_after_control_failed_strategy'] is not True):
        raise ValueError('Only registered replay mapping, physical budget and aligned cadence may change')
    cfg = dict(old['training_config'], model_id=reg['curriculum']['run_id'],
        out=reg['curriculum']['out'], trainer_config=str(RUN/'fit-plan/trainer.toml'),
        trainer_config_sha256=reg['training_config']['trainer_config_sha256'],
        registered_final_step=2472, loss_method=METHOD)
    if reg['training_config'] != cfg:
        raise ValueError('Same exact initializer, frozen base, canonical validation and optimizer recipe required')
    for key in ('source', 'source_sha256', 'parent_model', 'course_loss_weights'):
        if reg[key] != old[key]: raise ValueError('Changed matched parent/source: '+key)


def require_registration(reg):
    old = read(reg['sampling_control_registration'])
    from tools.curriculum_progression_parent_control_v1 import verify
    verify(old)
    check_matched(reg, old,
        tomllib.loads(Path(reg['training_config']['trainer_config']).read_text()),
        tomllib.loads(Path(old['training_config']['trainer_config']).read_text()))
    if reg['sampling_control_registration'] != str(CONTROL/'fit-plan/registered.json'):
        raise ValueError('Exact current control required')
    expected = [str(ROOT/'runs/curriculum-ri-every-update-replay-v1'/f'epoch-{e:02d}-private.json') for e in (1, 2)]
    if reg['replay_epoch_plans'] != expected: raise ValueError('Exact already prepared TRAIN-only replay plans required')
    for p, h in reg['inputs'].items():
        if sha(p) != h: raise ValueError('Frozen balanced-replay input changed: '+p)


def failed_control(done, evidence, execution, captured):
    """An infrastructure failure or interim decline cannot trigger a new fit."""
    proofs = captured['scheduler_proofs']
    if (done['status'] != 'progression2400_fit_and_real_curves_complete'
            or done['optimizer_steps'] != 2400 or done['actual_records_seen'] != 175190
            or evidence['status'] != 'progression2400_final_evidence_audit_passed'
            or evidence['optimizer_steps'] != 2400 or evidence['saved_final_reloaded_menus'] != 23688
            or execution['status'] != 'actual_progression2400_full_evidence_independently_audited'
            or captured['status'] != 'parent_retention_actual_terminals_preserved_not_child_admitted'
            or any(proofs[r]['successful'] is not True or proofs[r]['state'] != 'COMPLETED'
                or proofs[r]['exit_code'] != '0:0' for r in ('fit', 'audit'))
            or str(execution['actual_fit_job']) != proofs['fit']['job_id']
            or evidence['stage3_advancement']['passed'] is not False
            or captured['original_strategy_grade'] != evidence['stage3_advancement']
            or captured['strategy_grade_overridden'] is not False):
        raise ValueError('Actual completed, successfully audited control strategy failure required')
    return copy.deepcopy(evidence['stage3_advancement'])


def require_launch(registered):
    gate = read(RUN/'launch-gate-private.json')
    if gate['registered_sha256'] != sha(registered): raise ValueError('Launch registration differs')
    for p, h in gate['files'].items():
        if sha(p) != h: raise ValueError('Actual prior terminal/audit gate changed: '+p)
    paths = gate['evidence_paths']
    grade = failed_control(*(read(paths[k]) for k in ('completion', 'audit', 'execution', 'capture')))
    if grade != gate['unaltered_failed_control_grade']:
        raise ValueError('Prior strategy grade differs')
