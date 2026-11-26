"""Registration for the isolated calls13-loss retention control."""
from tools.curriculum_progression_parent_loss_v1 import METHOD, MASS
from tools.curriculum_complete_legal_class_policy_v1 import METHOD as ROUTE
from tools.curriculum_progression_runtime_v1 import policy_contract, prediction_parity, require_step0


def require_registration(reg):
    c = reg['curriculum']
    if (reg['status'] != 'progression2400_registered_not_started'
            or reg['registered_final_step'] != 2400
            or reg['train_records'] != 87595 or reg['validation_records'] != 23688
            or c['epochs'] != 2 or c['accumulation'] != 73 or c['seed'] != 2026112304
            or c['loss_method'] != METHOD or reg['training_config']['loss_method'] != METHOD
            or reg['course_loss_mass'] != MASS or c['selection_method'] != ROUTE
            or c['expected_record_exposures'] != 175190
            or c['validation_and_game_steps'] != [0, 600, 1200, 1800, 2400]
            or c['optimizer_resume'] is not False or c['new_optimizer'] is not True
            or reg.get('both_failed_3A_parameter_updates_excluded') is not True):
        raise ValueError('Exact parent-loss-retained progression registration required')
