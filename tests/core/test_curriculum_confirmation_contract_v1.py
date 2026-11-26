import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.curriculum_confirmation_stats_v1 import compare
from tools.curriculum_confirmation_contract_v1 import CRITERIA,assess_statistics,require_final_parent
from tools.curriculum_confirmation_native_v1 import FrozenReferenceEventAdapter,audit_matches
from tools.curriculum_confirmation_provenance_v1 import scan_jsonl


def rows():
    result=[]
    for index in range(32):
        for profile in ('efficiency','calling'):
            for seat in range(4):
                scores=[40000,30000,20000,10000]
                result.append(dict(seed=910000000+index,seat=seat,profile=profile,completed=True,ledger_passed=True,
                    final_scores=scores,rank=seat+1,net_points=scores[seat]-25000,hands=4,wins=1,dealin_hands=0,
                    choices=dict(chi=0,pon=0,ankan=0,daiminkan=0,kakan=0,reach=1),
                    mjai_log=[dict(type='hora',actor=seat),dict(type='reach',actor=seat)]))
    return result


class ConfirmationContractTests(unittest.TestCase):
    def test_partial_or_numerically_failed_curriculum_cannot_admit_a_final_candidate(self):
        reg=dict(curriculum=dict(final_candidate_only=True),registered_final_step=2500,
            expected_record_exposures=182494,validation_records=24516)
        completion=dict(status='placement_full_budget_fit_and_real_curves_complete',optimizer_steps=2500,
            actual_records_seen=182494,independent_confirmation_opened=False,automatic_release=False)
        evidence=dict(status='placement_full_budget_native_and_validation_evidence_audited',optimizer_steps=2500,
            actual_record_exposures=182494,actual_optimizer_observations=2500,source_and_launch_hashes_unchanged=True,
            saved_final_reloaded_menus=24516,all376_optimizer_states_match=True,development_assessment=dict(passed=True),
            automatic_release=False,candidate_from_registered_final_budget=True,CE_minimum_is_diagnostic_only=True,
            final_identity='candidate',final_policy_identity='policy')
        execution=dict(status='actual_full_placement_evidence_independently_audited',development_grade_passed=True,
            strategy_grade_overridden=False,original_launch_binding_unchanged=True,independent_v8_confirmation_performed=False,
            automatic_release=False)
        self.assertEqual(require_final_parent(reg,completion,evidence,execution)['selected_step'],2500)
        for field,value in (('optimizer_steps',128),('saved_final_reloaded_menus',10),
                            ('candidate_from_registered_final_budget',False)):
            bad=copy.deepcopy(evidence);bad[field]=value
            with self.assertRaises(ValueError):require_final_parent(reg,completion,bad,execution)
        bad=copy.deepcopy(evidence);bad['development_assessment']['passed']=False
        with self.assertRaises(ValueError):require_final_parent(reg,completion,bad,execution)

    def test_primary_arithmetic_pass_keeps_secondary_regression_and_unverified_scope_visible(self):
        base=rows();candidate=copy.deepcopy(base)
        for row in candidate:
            seat=row['seat'];improve=21000 if seat==2 and row['seed']%2==0 else 1000+row['seed']%2*1000
            row['final_scores'][seat]+=improve;row['final_scores'][(seat+1)%4]-=improve
            row['net_points']+=improve
            row['rank']=sorted(range(4),key=lambda p:(-row['final_scores'][p],p)).index(seat)+1
            row['dealin_hands']=1+row['seed']%2
        arithmetic=compare(candidate,base,910000000);result=assess_statistics(arithmetic,CRITERIA)
        self.assertTrue(result['primary_arithmetic_conditions_met'])
        self.assertTrue(result['secondary_tradeoffs']['dealin_hand_rate']['interval_supports_regression'])
        self.assertTrue(result['matched_native_ledger_and_full_provenance_audit_still_required'])
        self.assertFalse(result['unqualified_all_metrics_noninferiority_claim_allowed'])
        self.assertFalse(result['task_completion_or_release_authorized'])
        bad=copy.deepcopy(arithmetic);bad['metrics']['top2_rate']['ci95'][0]=-.01
        self.assertFalse(assess_statistics(bad,CRITERIA)['primary_arithmetic_conditions_met'])

    def test_zero_variance_or_changed_statistics_cannot_be_promoted_to_proof(self):
        data=rows();result=compare(data,data,910000000)
        self.assertFalse(assess_statistics(result,CRITERIA)['primary_arithmetic_conditions_met'])
        bad=copy.deepcopy(result);bad['independent_seed_groups']=256
        with self.assertRaises(ValueError):assess_statistics(bad,CRITERIA)
        with self.assertRaises(ValueError):assess_statistics(result,dict(CRITERIA,no_conditional_fourth_escape_veto=False))

    def test_all_native_records_are_checked_and_financial_errors_are_not_ignored(self):
        data=rows();calls=[]
        def ledger(log,scores):calls.append(log);return dict(hands=4)
        proof=audit_matches(data,910000000,ledger,lambda log,seat:0)
        self.assertEqual(len(calls),256);self.assertEqual(proof['complete_matches'],256)
        # Mock dependencies exercise delegation only, never a real ledger proof.
        bad=copy.deepcopy(data);bad[-1]['choices']['daiminkan']=1
        with self.assertRaises(ValueError):audit_matches(bad,910000000,ledger,lambda log,seat:0)
        def failing(log,scores):raise RuntimeError('native settlement failed')
        with self.assertRaises(RuntimeError):audit_matches(data,910000000,failing,lambda log,seat:0)

    def test_original_reference_act_receives_only_the_unchanged_observation_once(self):
        calls=[];action=object();observation=object()
        class Original:
            def act(self,obs):calls.append(obs);return action
        adapter=FrozenReferenceEventAdapter(Original())
        self.assertIs(adapter.act(observation,current_public_event=dict(type='tsumo')),action)
        self.assertEqual(calls,[observation])

    def test_streaming_seed_inventory_detects_nested_or_stringified_seed_without_claiming_global_proof(self):
        with TemporaryDirectory() as directory:
            a=Path(directory)/'a.jsonl';b=Path(directory)/'b.jsonl'
            a.write_text('{"seed":2026200000}\n{"group":"family:2026300001:seat2"}\n')
            b.write_text('{"meta":{"seed":2026300031}}\n{"larger_id":120263000001}\n')
            result=scan_jsonl([a,b,a])
            self.assertEqual(result['total_files'],2);self.assertEqual(result['reserved_literal_occurrences'],2)
            self.assertTrue(result['all_inventoried_files_stable'])
            self.assertTrue(result['checked_files_only_not_global_unused_grid_proof'])
            self.assertFalse(result['confirmation_admitted'])
            with self.assertRaises(ValueError):scan_jsonl([])
            with self.assertRaises(ValueError):scan_jsonl([Path(directory)/'absent.jsonl'])
