import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from tools import curriculum_yaku_balanced_parent_v1 as parent


class BalancedParentTests(unittest.TestCase):
    def fixture(self):
        reg=dict(validation_loss_method=parent.CANONICAL_PARENT_LOSS)
        done=dict(status='balanced_replay2472_fit_and_real_curves_complete',optimizer_steps=2472,
            actual_records_seen=180456,loss_method=parent.PARENT_LOSS)
        evidence=dict(status='balanced_replay2472_final_evidence_audit_passed',optimizer_steps=2472,
            actual_records_seen=180456,stage3_advancement={'passed':True},basic_retention={'passed':True},
            saved_final_reloaded_menus=23688,checked_complete_games=320,all376_optimizer_states_match=True,
            source_and_data_hashes_unchanged=True)
        execution=dict(status='actual_balanced_replay2472_full_evidence_independently_audited',
            automatic_release=False,automatic_stage3B=False,actual_fit_job=123)
        return [reg,done,evidence,execution,{'parent_fit_job':123}]

    def require(self,args):
        # These are unit fixtures, never actual parent-admission evidence. The
        # real caller independently verifies the complete immutable registration.
        with patch.object(parent,'require_parent_registration') as check:
            parent.require_qualified_parent(*args)
            check.assert_called_once_with(args[0])

    def test_full_original_qualification_contract(self):
        self.require(self.fixture())
        for target in ('stage3_advancement','basic_retention'):
            args=self.fixture();args[2][target]['passed']=False
            with self.assertRaises(ValueError):self.require(args)

    def test_no_old_final_step_or_physical_budget_relabeling(self):
        for index,key,value in [(1,'optimizer_steps',2400),(1,'actual_records_seen',175190),
                (2,'optimizer_steps',2400),(2,'actual_records_seen',175190),
                (1,'loss_method',parent.CANONICAL_PARENT_LOSS),
                (0,'validation_loss_method',parent.PARENT_LOSS)]:
            args=self.fixture();args[index][key]=value
            with self.assertRaises(ValueError):self.require(args)

    def test_partial_reload_games_optimizer_or_changed_sources_refused(self):
        for key,value in [('saved_final_reloaded_menus',100),('checked_complete_games',16),
                ('all376_optimizer_states_match',False),('source_and_data_hashes_unchanged',False)]:
            args=self.fixture();args[2][key]=value
            with self.assertRaises(ValueError):self.require(args)

    def test_wrong_execution_and_automatic_admission_refused(self):
        for key,value in [('actual_fit_job',999),('automatic_release',True),('automatic_stage3B',True),
                ('status','actual_progression2400_full_evidence_independently_audited')]:
            args=self.fixture();args[3][key]=value
            with self.assertRaises(ValueError):self.require(args)

    def test_registration_verification_failure_is_propagated(self):
        with patch.object(parent,'require_parent_registration',side_effect=ValueError('changed inputs')):
            with self.assertRaisesRegex(ValueError,'changed inputs'):
                parent.require_qualified_parent(*self.fixture())

    def test_parent_loader_restores_verifier_on_failure(self):
        old=object();initialization=SimpleNamespace(verified_parent=old)
        with self.assertRaisesRegex(RuntimeError,'abort'):
            with parent.parent_loader(initialization):
                self.assertIs(initialization.verified_parent,parent.verify_parent)
                raise RuntimeError('abort')
        self.assertIs(initialization.verified_parent,old)

    def test_actual_scheduler_resource_and_finite_time_required(self):
        with patch.object(parent,'resource',return_value='TimeLimit=14:30:00') as query:
            self.assertEqual(parent.gpu_only(),'TimeLimit=14:30:00')
            query.assert_called_once_with()
        for raw in ('TimeLimit=UNLIMITED','TimeLimit=00:00:00','TimeLimit=Unknown',''):
            with patch.object(parent,'resource',return_value=raw):
                with self.assertRaises(RuntimeError):parent.gpu_only()
        with patch.object(parent,'resource',side_effect=ValueError('unallocated login')):
            with self.assertRaises(ValueError):parent.gpu_only()


if __name__=='__main__':unittest.main()
