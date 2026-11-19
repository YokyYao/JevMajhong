import unittest
from unittest.mock import Mock
from jevmahjong.endgame.exposure import TrainingExposure, exposure_early_stop_update


def row(rid, source):
    return {'_meta': {'id': rid}, 'questions': {'action': {'src': source}}}


class ActualExposureTests(unittest.TestCase):
    def test_repeated_batches_cannot_replace_unseen_critical_positions(self):
        ledger = TrainingExposure([row('first', 'win'), row('second', 'win'), row('old', 'replay')], {'win': 1})
        ledger.observe(['first', 'old', 'first'])
        self.assertEqual(ledger.snapshot()['unique_records_seen'], 2)
        self.assertFalse(ledger.snapshot()['minimum_exposure_met'])
        ledger.observe(['second'])
        self.assertTrue(ledger.snapshot()['minimum_exposure_met'])
        with self.assertRaises(ValueError):
            ledger.observe(['foreign'])

    def test_invalid_source_cannot_silently_disable_coverage(self):
        with self.assertRaises(ValueError):
            TrainingExposure([row('x', 'win')], {'typo': 1})
        with self.assertRaises(ValueError):
            TrainingExposure([row('x', 'win'), row('x', 'win')], {'win': 1})

    def test_safe_incomplete_exposure_does_not_select_or_exhaust_patience(self):
        original = Mock()
        result = exposure_early_stop_update(original, 1e6, -.5, 1, step=8,
            exposure_met=False, other_guards_passed=True, maximize=False)
        self.assertEqual(result, (-.5, 0, False))
        original.assert_not_called()

    def test_guard_failures_still_trigger_stopping_before_coverage(self):
        original = Mock(return_value=(-.5, 2, False))
        result = exposure_early_stop_update(original, 1e6, -.5, 1, step=16,
            exposure_met=False, other_guards_passed=False, maximize=False)
        self.assertEqual(result, (-.5, 2, False))
        original.assert_called_once_with(1e6, -.5, 1, maximize=False)

    def test_baseline_and_fully_exposed_candidates_use_normal_selection(self):
        for step, seen in ((0, False), (48, True)):
            original = Mock(return_value=(-.6, 0, True))
            result = exposure_early_stop_update(original, -.6, -.5, 0, step=step,
                exposure_met=seen, other_guards_passed=True, maximize=False)
            self.assertEqual(result, (-.6, 0, True))
            original.assert_called_once()


if __name__ == '__main__':
    unittest.main()
