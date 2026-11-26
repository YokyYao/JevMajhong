"""Prospective final acceptance arithmetic; no outcomes or candidate admission.

The actual runner/auditor must independently verify identities, untouched
sources and native ledgers before these arithmetic conditions support a claim.
"""
import math
from tools.curriculum_confirmation_stats_v1 import GROUPS,PRIMARY,PROFILES,SEATS

CRITERIA=dict(method='final32_joint_primary_lower_bounds_and_secondary_disclosure_v1',
    seed_start=2026300000,seed_groups=32,seats=[0,1,2,3],opponent_profiles=['efficiency','calling'],
    matches_per_policy=256,primary_metrics=['top2_rate','net_points'],
    primary_regression_margins=dict(top2_rate=0.,net_points=0.),
    nominal_joint_primary_lower_confidence=.95,
    primary_bound_method='two97.5_percent_one_sided_t_lower_bounds_Bonferroni_alpha0.05',
    require_positive_primary_cluster_variance=True,
    secondary_metrics=['hand_win_rate','first_rate','fourth_rate','dealin_hand_rate'],
    secondary_interpretation='report_every_point_direction_and_interval_supported_regression_qualify_claim_scope',
    no_conditional_fourth_escape_veto=True,no_secondary_point_estimate_veto=True,
    sole_registered_final_candidate_before_outcomes=True,no_intermediate_results_for_selection=True,
    no_adaptive_extra_games_or_rescue_seed_replacement=True,
    require_native_ledgers_and_source_provenance_audit=True,
    automatic_goal_completion_or_release=False,raw_matches_private=True)


def require_criteria(actual):
    if actual!=CRITERIA:raise ValueError('Exact final acceptance must be frozen before outcomes')


def assess_statistics(result,criteria):
    require_criteria(criteria)
    if (result['method']!='paired_logical_seed_mean_t_df31_v1'
        or result['independent_seed_groups']!=GROUPS or result['matches_per_policy']!=256
        or result['matches_per_seed']!=8 or result['primary_metrics']!=list(PRIMARY)
        or result['primary_lower_bounds_nominal_joint_confidence']!=.95
        or result['joint_bound_rule']!=criteria['primary_bound_method']):
        raise ValueError('Original32-seed paired statistics and joint-primary bound required')
    metrics=result['metrics'];directions=dict(hand_win_rate=1,first_rate=1,fourth_rate=-1,dealin_hand_rate=-1)
    if set(metrics)!=set(PRIMARY)|set(directions):raise ValueError('Every primary and secondary outcome required')
    for name,stat in metrics.items():
        low,high=stat['ci95'];mean=stat['gain'];std=stat['cluster_std']
        if (stat['independent_seed_groups']!=32 or stat['df']!=31
            or any(not math.isfinite(v) for v in (low,high,mean,std,stat['candidate'],stat['reference']))
            or std<0 or not low<=mean<=high or abs((stat['candidate']-stat['reference'])-mean)>1e-9):
            raise ValueError('Complete finite consistent whole-seed estimates required')
    primary_ok=all(metrics[name]['ci95'][0]>=0 and metrics[name]['cluster_std']>0
        and metrics[name]['all_observed_differences_zero'] is False for name in PRIMARY)
    secondary={}
    for name,direction in directions.items():
        stat=metrics[name];low,high=stat['ci95'];gain=stat['gain']
        regression=direction*gain<0
        supported=stat['cluster_std']>0 and (high<0 if direction>0 else low>0)
        secondary[name]=dict(gain=gain,ci95=stat['ci95'],observed_regression=regression,
            interval_supports_regression=supported,positive_estimated_seed_variance=stat['cluster_std']>0,
            interval_is_descriptive_not_joint_secondary_claim=True)
    return dict(primary_arithmetic_conditions_met=primary_ok,
        primary_metrics=list(PRIMARY),secondary_tradeoffs=secondary,
        unqualified_all_metrics_noninferiority_claim_allowed=False,
        matched_native_ledger_and_full_provenance_audit_still_required=True,
        zero_variance_primary_cannot_supply_t_inference=any(metrics[name]['cluster_std']==0 for name in PRIMARY),
        no_conditional_fourth_escape_veto=True,task_completion_or_release_authorized=False)


def require_final_parent(reg,completion,evidence,execution):
    """Admit only the actual original-qualified final full-placement candidate."""
    c=reg['curriculum'];final=reg['registered_final_step']
    if (completion['status']!='placement_full_budget_fit_and_real_curves_complete'
        or completion['optimizer_steps']!=final or completion['actual_records_seen']!=reg['expected_record_exposures']
        or completion['independent_confirmation_opened'] or completion['automatic_release']
        or evidence['status']!='placement_full_budget_native_and_validation_evidence_audited'
        or evidence['optimizer_steps']!=final or evidence['actual_record_exposures']!=reg['expected_record_exposures']
        or evidence['actual_optimizer_observations']!=final or not evidence['source_and_launch_hashes_unchanged']
        or evidence['saved_final_reloaded_menus']!=reg['validation_records'] or not evidence['all376_optimizer_states_match']
        or evidence['development_assessment']['passed'] is not True or evidence['automatic_release']
        or not evidence['candidate_from_registered_final_budget'] or not evidence['CE_minimum_is_diagnostic_only']
        or execution['status']!='actual_full_placement_evidence_independently_audited'
        or execution['development_grade_passed'] is not True or execution['strategy_grade_overridden']
        or not execution['original_launch_binding_unchanged'] or execution['independent_v8_confirmation_performed']
        or execution['automatic_release'] or not c['final_candidate_only']):
        raise ValueError('Actual original-qualified audited final placement budget required before confirmation')
    return dict(selected_step=final,identity=evidence['final_identity'],policy_identity=evidence['final_policy_identity'])
