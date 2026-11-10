import unittest
from scripts.audit_call_validation import action_metrics


class ExactCallMetricsTests(unittest.TestCase):
    def test_wrong_chi_combination_is_not_an_exact_hit(self):
        report = action_metrics([
            {'gold': 'chi', 'pred': 'chi', 'exact': True, 'target_probability': .7},
            {'gold': 'chi', 'pred': 'chi', 'exact': False, 'target_probability': .1},
            {'gold': 'pon', 'pred': 'none', 'exact': False, 'target_probability': .1}])
        chi = report['by_target']['chi']
        self.assertEqual(chi['precision'], 1)
        self.assertEqual(chi['recall'], 1)
        self.assertEqual(chi['exact_tp'], 1)
        self.assertEqual(chi['exact_precision'], .5)
        self.assertEqual(chi['exact_recall'], .5)
        self.assertEqual(report['exact_call_recall'], 1 / 3)
        self.assertEqual(report['by_target']['pon']['exact_recall'], 0)

    def test_inconsistent_exact_type_is_rejected(self):
        with self.assertRaises(ValueError):
            action_metrics([{'gold': 'chi', 'pred': 'none', 'exact': True, 'target_probability': .1}])


if __name__ == '__main__': unittest.main()
