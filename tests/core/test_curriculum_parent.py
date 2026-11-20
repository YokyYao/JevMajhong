import copy
import unittest

from tools.curriculum_parent import ParentNotReady, require_stage1_candidate


class CurriculumParentTests(unittest.TestCase):
    def setUp(self):
        self.reg = dict(registered_final_step=6, train_records=5,
            curriculum=dict(epochs=2, batch=1, accumulation=2, candidate_rule='registered_final_full_test'))
        self.done = dict(status='stage1_fit_and_curves_complete', optimizer_steps=6, actual_records_seen=10)
        self.audit = dict(status='stage1_final_evidence_audit_passed', optimizer_steps=6,
            source_and_data_hashes_unchanged=True, heldout_groups_excluded_from_all_stages=True,
            stage1_advancement=dict(passed=True), saved_reloads=[dict(step=6, identity='final')])
        self.validation = dict(local_step=6, riichi_type_counts=dict(tp=30, fn=7, tn=56, fp=14),
            stratum_counts=dict(riichi_yes=37, riichi_no=70),
            stratum_accuracy=dict(plain=.55), stratum_ce=dict(plain=1.2))

    def test_registered_final_candidate_can_pass_gate(self):
        require_stage1_candidate(self.reg, self.done, self.audit, self.validation, 'final')

    def test_passed_flag_cannot_hide_failed_actual_recall(self):
        bad = copy.deepcopy(self.validation)
        bad['riichi_type_counts'].update(tp=3, fn=34)
        with self.assertRaises(ParentNotReady):
            require_stage1_candidate(self.reg, self.done, self.audit, bad, 'final')

    def test_minimum_ce_checkpoint_cannot_replace_registered_final(self):
        with self.assertRaises(ParentNotReady):
            require_stage1_candidate(self.reg, self.done, self.audit, self.validation, 'minimum-ce')

    def test_partial_fit_is_not_a_parent(self):
        bad = dict(self.done, optimizer_steps=4)
        with self.assertRaises(ParentNotReady):
            require_stage1_candidate(self.reg, bad, self.audit, self.validation, 'final')

    def test_evidence_pass_without_quality_gate_is_rejected(self):
        bad = dict(self.audit, stage1_advancement=None)
        with self.assertRaises(ParentNotReady):
            require_stage1_candidate(self.reg, self.done, bad, self.validation, 'final')
