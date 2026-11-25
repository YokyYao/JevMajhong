import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import curriculum_yaku_ri_anchor_parent_v1 as parent


class AugmentedYakuParentTests(unittest.TestCase):
    def fixture(self):
        reg=dict(validation_loss_method=parent.PARENT_VALIDATION_LOSS)
        done=dict(status='ri_anchor2596_fit_and_real_curves_complete',optimizer_steps=2596,
            actual_records_seen=189508,unique_train_rows=92570,loss_method=parent.PARENT_LOSS,
            curve_steps=list(parent.parent_runtime.POINTS),automatic_stage4_started=False,automatic_release=False)
        evidence=dict(status='ri_anchor2596_final_evidence_audit_passed',optimizer_steps=2596,
            actual_records_seen=189508,stage3_advancement={'passed':True},basic_retention={'passed':True},
            retention_quality_cohort='original23688',saved_final_reloaded_menus=24998,
            checked_complete_games=320,all376_optimizer_states_match=True,source_and_data_hashes_unchanged=True,
            source_split_replay_and_exposure_exact=True,final_reserved_grid_unused=True,
            automatic_stage4_started=False,automatic_release=False)
        execution=dict(status='actual_ri_anchor2596_full_evidence_independently_audited',
            automatic_stage3B=False,automatic_release=False,actual_fit_job='101')
        return [reg,done,evidence,execution,dict(fit_job='101')]

    def require(self, values):
        with patch.object(parent.parent_runtime,'require_registration') as check:
            parent.require_qualified_parent(*values)
            check.assert_called_once_with(values[0])

    def test_full_original_qualification_and_registration_required(self):
        self.require(self.fixture())
        with patch.object(parent.parent_runtime,'require_registration',side_effect=ValueError('changed source')):
            with self.assertRaisesRegex(ValueError,'changed source'):
                parent.require_qualified_parent(*self.fixture())

    def test_no_partial_old_or_minimum_loss_parent(self):
        for index,key,value in [(1,'optimizer_steps',1298),(1,'actual_records_seen',180456),
                (1,'unique_train_rows',87595),(1,'loss_method',parent.PARENT_VALIDATION_LOSS),
                (1,'curve_steps',[0,2596]),(0,'validation_loss_method',parent.PARENT_LOSS),
                (2,'optimizer_steps',2472),(2,'actual_records_seen',180456)]:
            bad=self.fixture();bad[index][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.require(bad)

    def test_failed_retention_or_strategy_and_combined_cohort_refused(self):
        for key in ('stage3_advancement','basic_retention'):
            bad=self.fixture();bad[2][key]['passed']=False
            with self.subTest(key=key),self.assertRaises(ValueError):self.require(bad)
        bad=self.fixture();bad[2]['retention_quality_cohort']='combined24998'
        with self.assertRaises(ValueError):self.require(bad)

    def test_full_reload_games_and_immutable_inputs_required(self):
        for key,value in [('saved_final_reloaded_menus',23688),('checked_complete_games',16),
                ('all376_optimizer_states_match',False),('source_and_data_hashes_unchanged',False),
                ('source_split_replay_and_exposure_exact',False),('final_reserved_grid_unused',False)]:
            bad=self.fixture();bad[2][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.require(bad)

    def test_actual_execution_identity_and_no_automatic_promotion(self):
        for index,key,value in [(3,'actual_fit_job','other'),(3,'automatic_stage3B',True),
                (3,'automatic_release',True),(1,'automatic_release',True),
                (2,'automatic_stage4_started',True)]:
            bad=self.fixture();bad[index][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.require(bad)

    def test_parent_loader_restores_original_after_exception(self):
        previous=object();initialization=SimpleNamespace(verified_parent=previous)
        with self.assertRaisesRegex(RuntimeError,'abort'):
            with parent.parent_loader(initialization):
                self.assertIs(initialization.verified_parent,parent.verify_parent)
                raise RuntimeError('abort')
        self.assertIs(initialization.verified_parent,previous)


if __name__=='__main__':unittest.main()
