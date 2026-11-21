"""Explicit experimental transfer of a failed calls final, never a passing grade.

Reuse the frozen structural sealer/verifier under a scoped distinct method.
The original quality grade stays false in the seal, registration and audit.
Only the net-gain component may fail; the new experiment is declared after
inspecting that failure and before any recovery update.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics

from tools import curriculum_calls_parent as strict
from tools.curriculum_parent import ROOT, ParentNotReady, read, sha

METHOD = 'curriculum_calls_experimental_recovery_v1'
POLICY = ROOT/'configs/training/curriculum-calls-recovery-policy-v1.json'
ORIGINAL_QUALIFY = strict.qualify


def policy():
    value = read(POLICY)
    if (value['method'] != METHOD or value['original_calls_quality_must_remain_failed'] is not True
        or value['created_after_inspecting_calls_final_results'] is not True
        or value['prospective_for_future_recovery_updates_only'] is not True
        or value['research_admission_is_not_qualified_advancement'] is not True
        or value['automatic_release'] is not False):
        raise ParentNotReady('Explicit failed-parent experimental policy required')
    return value


def recovery_grade(reg, completion, audit, curve, summary):
    p = policy()
    c = reg['curriculum']
    first, last = curve[0], curve[-1]
    original = dict(plain_agreement_gain=last['stratum_accuracy']['plain']-first['stratum_accuracy']['plain'],
        riichi_balanced_gain=strict.balanced(last['riichi_type_counts'])-strict.balanced(first['riichi_type_counts']),
        five_call_type_balanced_gain=statistics.mean(strict.balanced(last['operation_type_counts'][op])
            -strict.balanced(first['operation_type_counts'][op]) for op in ('chi','pon','ankan','daiminkan','kakan')),
        passed=False, development_only_not_strength_proof=True)
    limits = c['development_targets']
    if (reg['registered_final_step'] != p['calls_final_step'] or summary.get('matches') != p['calls_final_matches']
        or summary.get('checkpoint_sha256') != p['calls_parent_identity']
        or original != audit.get('stage2_advancement')
        or original['plain_agreement_gain'] < -limits['max_plain_agreement_drop_from_step0']
        or original['riichi_balanced_gain'] < -limits['max_riichi_balanced_accuracy_drop_from_step0']
        or original['five_call_type_balanced_gain'] < limits['min_five_call_type_balanced_accuracy_gain_from_step0']
        or audit['comparison_to_parent']['dealin']['gain'] > limits['max_final64_dealin_increase_over_parent']
        or not math.isfinite(audit['comparison_to_parent']['net_points']['gain'])
        or audit['comparison_to_parent']['net_points']['gain'] >= limits['min_final64_net_gain_over_parent']):
        raise ParentNotReady('Recovery applies only to the recorded net-only failed final')
    # The original strict implementation must reach its final quality failure,
    # after every original completion/reload/exposure/schedule check passed.
    try:
        ORIGINAL_QUALIFY(reg,completion,audit,curve,summary)
    except ParentNotReady as error:
        if str(error) != 'The original calls-course quality targets did not pass':
            raise
    else:
        raise ParentNotReady('A passing original final does not belong to this failed-parent experiment')
    return original


@contextmanager
def recovery_contract():
    original_method, original_qualify = strict.METHOD, strict.qualify
    strict.METHOD, strict.qualify = METHOD, recovery_grade
    try:
        yield
    finally:
        strict.METHOD, strict.qualify = original_method, original_qualify


def seal_recovery(registered,evidence,fit_exit,audit_exit,observations,child_data_ready,out):
    p = policy()
    if registered.resolve() != (ROOT/p['calls_registration']).resolve() or evidence.resolve() != (ROOT/p['calls_evidence']).resolve():
        raise ParentNotReady('Another parent/evidence is outside this experiment')
    if out.exists(): raise ParentNotReady('Never overwrite an experimental parent seal')
    with recovery_contract():
        strict.seal(registered,evidence,fit_exit,audit_exit,observations,child_data_ready,out)
    sealed = read(out)
    assert sealed['stage_quality']['passed'] is False and sealed['method'] == METHOD
    sealed['research_admission'] = dict(admitted_for_experiment=True,
        experiment_run_id=p['experiment_run_id'], policy=str(POLICY), policy_sha256=sha(POLICY),
        original_quality_unchanged_failed=True, qualified_advancement=False,
        declared_after_parent_results_before_child_updates=True)
    for path in (POLICY,Path(__file__)):
        sealed['bound_files'][str(path.resolve())] = sha(path)
    out.write_text(json.dumps(sealed,indent=2)+'\n')
    return sealed


def verify_recovery_parent(cfg):
    p = policy()
    sealed = read(Path(cfg['parent_initialization']['manifest']))
    admission = sealed.get('research_admission',{})
    if (cfg['model_id'] != p['experiment_run_id'] or admission.get('experiment_run_id') != p['experiment_run_id']
        or admission.get('admitted_for_experiment') is not True or admission.get('qualified_advancement') is not False
        or admission.get('original_quality_unchanged_failed') is not True or sealed['stage_quality']['passed'] is not False
        or admission.get('policy_sha256') != sha(POLICY)
        or sealed['bound_files'].get(str(POLICY.resolve())) != sha(POLICY)
        or sealed['bound_files'].get(str(Path(__file__).resolve())) != sha(Path(__file__))):
        raise ParentNotReady('Wrong or changed experimental recovery admission')
    with recovery_contract():
        return strict.verify_calls_parent(cfg)


@contextmanager
def recovery_parent_loader(initialization_module):
    original = initialization_module.verified_parent
    initialization_module.verified_parent = verify_recovery_parent
    try:
        yield
    finally:
        initialization_module.verified_parent = original
