import copy
import unittest

from tools.curriculum_parent import ParentNotReady,require_stage1_candidate
from tools.curriculum_research_parent import POLICY,read,require_research_candidate


class ResearchParentTests(unittest.TestCase):
    def setUp(self):
        self.policy=read(POLICY)
        self.reg=dict(registered_final_step=584,train_records=9292,curriculum=dict(
            run_id='jev-curriculum-stable-v2-stage1',epochs=4,batch=1,accumulation=64,
            candidate_rule='registered_final_full_four_epochs_584_updates',curve_seed_start=2026200000,
            opponent_profiles=['efficiency','calling'],stage_end_seed_groups=8))
        self.done=dict(status='stage1_fit_and_curves_complete',optimizer_steps=584,actual_records_seen=37168)
        self.audit=dict(status='stage1_final_evidence_audit_passed',optimizer_steps=584,
            source_and_data_hashes_unchanged=True,heldout_groups_excluded_from_all_stages=True,
            stage1_advancement=dict(passed=False),saved_reloads=[dict(step=584,identity='final')],
            final_summary=dict(matches=64,seed_groups=8,top2_rate=.625,net_points=2500.,dealin_hand_rate=.14),
            final_comparison_to_v8={k:dict(seed_groups=8,gain=0.,ci95=[-1.,1.])
                for k in ('top2','net_points','hand_win','dealin','fourth')})
        self.validation=dict(local_step=584,riichi_type_counts=dict(tp=30,fn=7,tn=25,fp=45),
            stratum_counts=dict(riichi_yes=37,riichi_no=70),stratum_accuracy=dict(plain=.51),stratum_ce=dict(plain=1.34))

    def grade(self,audit=None,validation=None,reg=None,policy=None):
        return require_research_candidate(reg or self.reg,self.done,audit or self.audit,
            validation or self.validation,'final',policy or self.policy)

    def test_explicit_research_admission_preserves_failed_original_grade(self):
        with self.assertRaises(ParentNotReady):
            require_stage1_candidate(self.reg,self.done,self.audit,self.validation,'final')
        result=self.grade()
        self.assertTrue(result['passed']);self.assertFalse(result['original_stage1_advancement']['passed'])
        self.assertTrue(result['admission_is_not_stage1_mastery_or_v8_strength_proof'])
        self.assertFalse(self.audit['stage1_advancement']['passed'])

    def test_short_grid_or_worse_retention_cannot_pass(self):
        for key,value in [('matches',16),('seed_groups',2),('net_points',-100.),('top2_rate',.5),
                          ('dealin_hand_rate',.2),('dealin_hand_rate',float('nan'))]:
            audit=copy.deepcopy(self.audit);audit['final_summary'][key]=value
            with self.subTest(metric=key,value=value),self.assertRaises(ParentNotReady):self.grade(audit=audit)
        audit=copy.deepcopy(self.audit);audit['final_comparison_to_v8'].pop('net_points')
        with self.assertRaises(ParentNotReady):self.grade(audit=audit)

    def test_collapse_basic_loss_and_population_change_fail(self):
        for counts in (dict(tp=0,fn=37,tn=70,fp=0),dict(tp=10,fn=27,tn=10,fp=60),dict(tp=30,fn=6,tn=25,fp=45)):
            value=copy.deepcopy(self.validation);value['riichi_type_counts']=counts
            with self.assertRaises(ParentNotReady):self.grade(validation=value)
        for key,value in [('stratum_accuracy',.4),('stratum_ce',2.),('stratum_ce',float('nan'))]:
            validation=copy.deepcopy(self.validation);validation[key]['plain']=value
            with self.assertRaises(ParentNotReady):self.grade(validation=validation)

    def test_context_route_needs_full_serving_audit(self):
        reg=copy.deepcopy(self.reg);reg['curriculum']['riichi_input_schema']='public-riichi-context-v1'
        with self.assertRaises(ParentNotReady):self.grade(reg=reg)
        audit=dict(self.audit,input_schema='public-riichi-context-v1',all_saved_checkpoint_contracts_verified=True,
            final_raw_public_serving_menus=107,final_raw_public_max_probability_error=0.,final_policy_identity_sha256='policy')
        self.assertTrue(self.grade(reg=reg,audit=audit)['passed'])

    def test_no_intermediate_or_post_result_policy_replacement(self):
        validation=dict(self.validation,local_step=438)
        with self.assertRaises(ParentNotReady):self.grade(validation=validation)
        policy=copy.deepcopy(self.policy);policy['complete_match_development']['min_net_points']=-100000.
        with self.assertRaises(ParentNotReady):self.grade(policy=policy)
        reg=copy.deepcopy(self.reg);reg['curriculum']['run_id']='unregistered_variant'
        with self.assertRaises(ParentNotReady):self.grade(reg=reg)
