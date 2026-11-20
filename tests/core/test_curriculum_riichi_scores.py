import unittest

from tools.audit_curriculum_riichi_scores import ranking_summary


class RiichiRankingTests(unittest.TestCase):
    def test_separable_scores_and_ties_have_expected_discrimination(self):
        labels = [True, False, True, False]
        separated = ranking_summary(labels, [3., -2., 1., -1.])
        self.assertEqual(separated['auc'], 1.)
        self.assertTrue(separated['oracle_both60'])
        tied = ranking_summary(labels, [.5]*4)
        self.assertEqual(tied['auc'], .5)
        self.assertEqual(tied['oracle_best_balanced'], .5)
        self.assertFalse(tied['oracle_both60'])

    def test_audit_does_not_return_a_validation_selected_threshold(self):
        result = ranking_summary([True, False, True, False], [4., 1., 2., 3.])
        self.assertEqual(set(result), {'auc', 'oracle_best_balanced', 'oracle_both60'})
