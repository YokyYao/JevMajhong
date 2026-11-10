import copy
import unittest
from scripts.refit_call_query import eligible_report


class ExactSelectionTests(unittest.TestCase):
    def setUp(self):
        self.warm = {'views': {v: {'by_target': {k: {'recall': .8, 'exact_recall': .4}
            for k in ('chi', 'pon')}} for v in ('native', 'screen_confirmed', 'screen_unknown')}}

    def test_correct_type_without_correct_combination_cannot_pass_exact_guard(self):
        r = copy.deepcopy(self.warm)
        r['views']['native']['by_target']['chi']['exact_recall'] = 0
        self.assertTrue(eligible_report(r, self.warm, .5, exact=False))
        self.assertFalse(eligible_report(r, self.warm, .5, exact=True))

    def test_missing_screen_call_ability_is_not_hidden_by_native_success(self):
        r = copy.deepcopy(self.warm)
        r['views']['screen_unknown']['by_target']['chi']['exact_recall'] = .1
        self.assertTrue(eligible_report(r, self.warm, .5, exact=False))
        self.assertFalse(eligible_report(r, self.warm, .5, exact=True))

    def test_retaining_half_exact_recall_in_every_view_is_eligible(self):
        r = copy.deepcopy(self.warm)
        for view in r['views'].values():
            for target in view['by_target'].values(): target['exact_recall'] = .2
        self.assertTrue(eligible_report(r, self.warm, .5, exact=True))


if __name__ == '__main__': unittest.main()
