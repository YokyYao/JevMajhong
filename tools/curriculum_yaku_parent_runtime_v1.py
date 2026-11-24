"""Explicit registration contract for retained-parent3B continuation."""
from tools.curriculum_yaku_parent_loss_v1 import METHOD, MASS
from tools.curriculum_complete_legal_class_policy_v1 import METHOD as ROUTE
from tools.curriculum_yaku_runtime_v1 import policy_contract, prediction_parity, require_step0


def require_registration(reg):
    c = reg['curriculum']
    if (reg['status'] != 'yaku2500_registered_not_started' or reg['registered_final_step'] != 2500
            or reg['train_records'] != 91247 or reg['validation_records'] != 24516
            or c['epochs'] != 2 or c['accumulation'] != 73 or c['seed'] != 2026112305
            or c['loss_method'] != METHOD or reg['training_config']['loss_method'] != METHOD
            or reg['course_loss_mass'] != MASS or c['selection_method'] != ROUTE
            or c['expected_record_exposures'] != 182494
            or c['validation_and_game_steps'] != [0, 625, 1250, 1875, 2500]
            or c['optimizer_resume'] is not False or c['new_optimizer'] is not True
            or reg.get('retained_parent_replay_objective_explicit') is not True):
        raise ValueError('Exact retained-parent yaku2500 registration required')
