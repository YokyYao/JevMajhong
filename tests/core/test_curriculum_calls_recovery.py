import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from tools import curriculum_calls_parent as strict
from tools.curriculum_calls_recovery_parent import recovery_grade,recovery_contract,policy
from tools.curriculum_parent import ParentNotReady
from tools.attack_recovery_runtime import hooks,assessment


class RecoveryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.reg=dict(registered_final_step=324,train_records=20237,curriculum=dict(
            stage='stage2-calls-v1',candidate_rule='registered_final_full_two_epochs_324_updates',epochs=2,batch=1,
            accumulation=125,validation_every_steps=81,development_targets=dict(max_plain_agreement_drop_from_step0=.02,
            max_riichi_balanced_accuracy_drop_from_step0=.05,min_five_call_type_balanced_accuracy_gain_from_step0=.03,
            min_final64_net_gain_over_parent=0.,max_final64_dealin_increase_over_parent=.02)))
        self.done=dict(status='stage2_fit_and_curves_complete',optimizer_steps=324,actual_records_seen=40474)
        self.curve=[dict(step=i,stratum_accuracy={'plain':.5},riichi_type_counts=dict(tp=1,fn=3,tn=3,fp=1),
            operation_type_counts={op:dict(tp=1 if i==0 else 2,fn=3 if i==0 else 2,tn=3,fp=1)
                for op in ('chi','pon','ankan','daiminkan','kakan')}) for i in (0,81,162,243,324)]
        self.summary=dict(matches=64,seed_groups=8,net_points=100.,checkpoint_sha256=policy()['calls_parent_identity'])
        self.audit=dict(status='stage2_final_evidence_audit_passed',optimizer_steps=324,real_records_seen=40474,
            source_and_data_hashes_unchanged=True,heldout_groups_excluded=True,exact_train_and_validation_replay=True,
            all376_optimizer_states_match=True,saved_final_reloaded_menus=4721,saved_final_max_probability_error=1e-7,
            checked_complete_games=224,comparison_to_parent=dict(net_points={'gain':-10.},dealin={'gain':0.}),
            final_summary=self.summary.copy(),stage2_advancement=dict(plain_agreement_gain=0.,riichi_balanced_gain=0.,
            five_call_type_balanced_gain=.125,passed=False,development_only_not_strength_proof=True))

    def test_research_returns_original_false_and_does_not_make_strict_parent_pass(self):
        self.assertIs(recovery_grade(self.reg,self.done,self.audit,self.curve,self.summary)['passed'],False)
        with self.assertRaises(ParentNotReady):strict.qualify(self.reg,self.done,self.audit,self.curve,self.summary)

    def test_integrity_failure_and_other_quality_failure_never_bypass(self):
        broken=copy.deepcopy(self.audit);broken['saved_final_reloaded_menus']=1
        with self.assertRaises(ParentNotReady):recovery_grade(self.reg,self.done,broken,self.curve,self.summary)
        broken=copy.deepcopy(self.curve);broken[-1]['stratum_accuracy']['plain']=.1
        with self.assertRaises(ParentNotReady):recovery_grade(self.reg,self.done,self.audit,broken,self.summary)
        broken=copy.deepcopy(self.audit);broken['comparison_to_parent']['dealin']['gain']=.03
        with self.assertRaises(ParentNotReady):recovery_grade(self.reg,self.done,broken,self.curve,self.summary)

    def test_changed_parent_or_forged_old_grade_rejected(self):
        with self.assertRaises(ParentNotReady):
            recovery_grade(self.reg,self.done,self.audit,self.curve,dict(self.summary,checkpoint_sha256='other'))
        broken=copy.deepcopy(self.audit);broken['stage2_advancement']['passed']=True
        with self.assertRaises(ParentNotReady):recovery_grade(self.reg,self.done,broken,self.curve,self.summary)

    def test_scoped_method_and_runtime_hooks_restore_on_failure(self):
        before=(strict.METHOD,strict.qualify)
        with self.assertRaises(RuntimeError):
            with recovery_contract():raise RuntimeError('fixture')
        self.assertEqual((strict.METHOD,strict.qualify),before)
        train=SimpleNamespace(calls_parent_loader=object());audit=SimpleNamespace(verify_calls_parent=object())
        before=(train.calls_parent_loader,audit.verify_calls_parent)
        with self.assertRaises(RuntimeError):
            with hooks(train=train,audit=audit):raise RuntimeError('fixture')
        self.assertEqual((train.calls_parent_loader,audit.verify_calls_parent),before)

    def test_recovery_must_restore_pre_calls_baseline_not_just_improve_weaker_parent(self):
        limits=policy()['recovery_targets']
        evidence=dict(original_stage2_advancement={'passed':False},stage3_advancement={'passed':True},
            comparison_to_parent=dict(hand_win={'gain':.01},dealin={'gain':-.01}))
        comparison=dict(net_points={'gain':-1.},top2={'gain':0.})
        self.assertFalse(assessment(evidence,comparison,limits)['passed'])
        comparison['net_points']['gain']=1.
        self.assertTrue(assessment(evidence,comparison,limits)['passed'])
        self.assertIs(assessment(evidence,comparison,limits)['original_calls_quality_passed'],False)

    def test_numerical_recipe_is_the_prepared_recipe_with_new_provenance(self):
        root=Path(__file__).resolve().parents[2]
        original=json.loads((root/'configs/training/curriculum-stage3-attack-v1.json').read_text())
        recovery=json.loads((root/'configs/training/curriculum-stage3-attack-recovery-v2.json').read_text())
        recovery.pop('recovery_policy')
        for key in ('run_id','plan_root','out'):
            original.pop(key);recovery.pop(key)
        self.assertEqual(original,recovery)


if __name__=='__main__':unittest.main()
