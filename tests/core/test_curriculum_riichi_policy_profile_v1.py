import copy
import math
import unittest

from tools.curriculum_basic_repair import MASS
from tools.curriculum_riichi_policy_profile_v1 import profile


class RiichiPolicyProfileTests(unittest.TestCase):
    def fixture(self):
        rows = []; predictions = []
        for source in MASS:
            category = source.split(':')[1]
            criteria = {'d1': {'type': 'dahai'}, 'd2': {'type': 'dahai'}}
            if category != 'plain': criteria['r'] = {'type': 'reach'}
            probabilities = [.8, .2] if category == 'plain' else [.2, .2, .6] if category == 'riichi_yes' else [.35, .25, .4]
            rows.append(dict(_meta=dict(id=source), questions=dict(action=dict(
                src='curriculum1:'+source, label='r' if category == 'riichi_yes' else 'd1', criteria=criteria))))
            predictions.append(dict(id=source, keys=list(criteria), p=probabilities))
        return rows, predictions

    def ce(self, question, keys, probabilities):
        return -math.log(probabilities[keys.index(question['label'])])

    def test_selected_grade_changes_only_choices_not_raw_probabilities_or_ce(self):
        rows, predictions = self.fixture(); before = copy.deepcopy((rows, predictions))
        result = profile(rows, predictions, {'weighted_target_ce': 1.}, self.ce)
        self.assertFalse(result['raw_source_quality']['all_source_quality_passed'])
        self.assertTrue(result['selected_source_quality']['all_source_quality_passed'])
        self.assertEqual(result['scope_counts'], {'non_RI': 2, 'RI': 4})
        self.assertEqual(result['changed_choices'], {'non_RI': 0, 'RI': 2})
        for field in ('weighted_target_ce', 'source_stratum_ce', 'stratum_ce', 'stratum_counts'):
            self.assertEqual(result['raw_profile'][field], result['selected_profile'][field])
        self.assertEqual((rows, predictions), before)

    def test_duplicate_missing_or_invalid_probability_records_are_rejected(self):
        rows, predictions = self.fixture()
        for invalid in (predictions[:-1], predictions+[predictions[0]]):
            with self.assertRaises(ValueError): profile(rows, invalid, {'weighted_target_ce': 1.}, self.ce)
        broken = copy.deepcopy(predictions); broken[0]['p'][0] = float('nan')
        with self.assertRaises(ValueError): profile(rows, broken, {'weighted_target_ce': 1.}, self.ce)

    def test_source_class_cannot_substitute_for_actual_menu_and_label(self):
        rows, predictions = self.fixture()
        broken = copy.deepcopy(rows)
        i = next(i for i, row in enumerate(rows) if row['_meta']['id'] == 'legacy:riichi_yes')
        broken[i]['questions']['action']['label'] = 'd1'
        with self.assertRaises(ValueError): profile(broken, predictions, {'weighted_target_ce': 1.}, self.ce)
        broken = copy.deepcopy(rows); broken[i]['questions']['action']['criteria']['r']['type'] = 'ankan'
        with self.assertRaises(ValueError): profile(broken, predictions, {'weighted_target_ce': 1.}, self.ce)


if __name__ == '__main__': unittest.main()
