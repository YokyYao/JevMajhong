"""Scoped recovery hooks around the unchanged numerical attack pipeline."""
from contextlib import contextmanager
import math

from tools.curriculum_calls_recovery_parent import policy, recovery_parent_loader, verify_recovery_parent


@contextmanager
def hooks(train=None, audit=None):
    old_train = train.calls_parent_loader if train else None
    old_audit = audit.verify_calls_parent if audit else None
    if train: train.calls_parent_loader = recovery_parent_loader
    if audit: audit.verify_calls_parent = verify_recovery_parent
    try:
        yield
    finally:
        if train: train.calls_parent_loader = old_train
        if audit: audit.verify_calls_parent = old_audit


def assessment(evidence, pre_calls_comparison, limits):
    values=dict(net=pre_calls_comparison['net_points']['gain'],top2=pre_calls_comparison['top2']['gain'],
        win=evidence['comparison_to_parent']['hand_win']['gain'],dealin=evidence['comparison_to_parent']['dealin']['gain'])
    if not all(type(v) in (int,float) and math.isfinite(v) for v in values.values()):
        raise ValueError('Finite actual recovery game comparisons required')
    if evidence['original_stage2_advancement']['passed'] is not False:
        raise ValueError('The original failed calls grade must remain visible')
    passed=(evidence['stage3_advancement']['passed']
        and values['net']>=limits['min_final64_net_gain_over_pre_calls_stage1']
        and values['top2']>=limits['min_final64_top2_gain_over_pre_calls_stage1']
        and values['win']>=limits['min_hand_win_gain_over_calls_parent']
        and values['dealin']<=limits['max_dealin_gain_over_calls_parent'])
    return dict(passed=passed, original_calls_quality_passed=False,
        requires_original_attack_targets_and_additional_recovery_targets=True,
        comparison_values=values, original_gate_not_relabelled=True,
        development_only_not_v8_parity=True, automatic_release=False)
