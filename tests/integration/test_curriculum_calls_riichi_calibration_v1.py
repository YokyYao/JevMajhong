import unittest

from tools.audit_curriculum_calls_riichi_calibration_v1 import diagnostic


class CalibrationNumbers(unittest.TestCase):
    def test_hand_calculated_tie_auc_brier_confusion_and_bins(self):
        result = diagnostic([(1, .5), (1, .9), (0, .1), (0, .5)])
        self.assertEqual(result['roc_auc'], .875)
        self.assertAlmostEqual(result['brier'], .13)
        self.assertEqual((result['fixed_0_5_tp'], result['fixed_0_5_fn'],
                          result['fixed_0_5_tn'], result['fixed_0_5_fp']), (2, 0, 1, 1))
        self.assertEqual(result['fixed_0_5_balanced_accuracy'], .75)
        self.assertAlmostEqual(result['equal_width_10bin_ece'], .05)

    def test_one_class_has_no_discrimination_statistic(self):
        result = diagnostic([(0, 0.), (0, .1)])
        self.assertIsNone(result['roc_auc'])
        self.assertIsNone(result['fixed_0_5_balanced_accuracy'])
        self.assertAlmostEqual(result['brier'], .005)


if __name__ == '__main__': unittest.main()
