import copy
import unittest

from tools import curriculum_placement_parent_metrics_v2 as new
from tools import curriculum_placement_metrics_v1 as old
from tools import curriculum_yaku_parent_loss_v1 as replay
from tests.core.test_curriculum_placement_metrics_v1 import fixture, materialize


class PlacementParentMetricTests(unittest.TestCase):
    def test_all32_terminal_diagnostics_and_serving_selection_are_unchanged(self):
        records, predictions = fixture([.004, .006, .99]); original = copy.deepcopy(records)
        before = old.new_profile(records, predictions, materialize)
        after = new.new_profile(records, predictions, materialize)
        for key, value in before.items():
            if key != 'source_loss_components': self.assertEqual(value, after[key])
        for source in new.NEW_SOURCES:
            for key in ('objective', 'standard_target_ce', 'conditional_discard_ce', 'masked_MC_ce'):
                self.assertEqual(before['source_loss_components'][source][key], after['source_loss_components'][source][key])
            self.assertEqual(after['source_loss_components'][source]['terminal_soft_objective'],
                before['source_loss_components'][source]['objective'])
        self.assertEqual(records, original)

    def test_full_soft_RI_profile_preserves_original_objective(self):
        records, predictions = fixture([.45, .3, .25])
        for row in records:
            q = row['questions']['action']; q['target'] = {'a': .6, 'b': .1, 'c': .3}; q['label'] = 'a'
            q['criteria'] = {'a': {'type': 'reach'}, 'b': {'type': 'dahai'}, 'c': {'type': 'dahai'}}
        before = old.new_profile(records, predictions, materialize)
        after = new.new_profile(records, predictions, materialize)
        for source in new.NEW_SOURCES:
            a = before['source_loss_components'][source]; b = after['source_loss_components'][source]
            self.assertEqual(a['objective'], b['objective'])
            self.assertAlmostEqual(b['objective']-b['standard_target_ce'], b['signed_objective_minus_standard_ce'])
            self.assertEqual(before['selected_target_mass_by_source'][source], after['selected_target_mass_by_source'][source])

    def test_complete57_profile_aggregates_signed_and_terminal_components(self):
        records, predictions = fixture([.4, .6, 0.]); fresh = new.new_profile(records, predictions, materialize)
        components = {s: dict.fromkeys(replay.COMPONENTS, 0.) for s in replay.MASS}
        for values in components.values():
            values.update(objective=2., standard_target_ce=3., signed_objective_minus_standard_ce=-1.)
        inherited = dict(source_loss_components=components, stratum_counts=dict.fromkeys(replay.MASS, 1))
        result = new.combine({}, inherited, fresh, 57)
        self.assertEqual(len(result['source_loss_components']), 57)
        self.assertAlmostEqual(result['weighted_class_objective']-result['weighted_target_ce'], result['signed_objective_adjustment'])
        self.assertAlmostEqual(result['terminal_soft_objective'], .2*next(iter(fresh['source_loss_components'].values()))['objective'])
        self.assertNotIn('extra_RI_class_loss', result)
        self.assertTrue(result['validation_loss_is_not_policy_strength'])

    def test_missing_sources_still_fail_instead_of_renormalising(self):
        records, predictions = fixture([.4, .6, 0.])
        with self.assertRaises(ValueError): new.new_profile(records[:-1], predictions[:-1], materialize)
        fresh = new.new_profile(records, predictions, materialize)
        with self.assertRaises(ValueError): new.combine({}, dict(source_loss_components={}, stratum_counts={}), fresh, 32)


    def test_actual_upstream_rounding_is_supported_without_mutating_teacher_labels(self):
        records, predictions = fixture([.2, .3, .5])
        for row in records:
            q = row['questions']['action']; q['target'] = {'a': .2, 'b': .3, 'c': .5+1e-15}; q['label'] = 'c'
        original = copy.deepcopy(records)
        def upstream_normalized(row):
            encoded = materialize(row); q = encoded['questions'][0]
            q['target'] = [v/sum(q['target']) for v in q['target']]
            return encoded
        with self.assertRaises(ValueError): old.new_profile(records, predictions, upstream_normalized)
        actual = new.new_profile(records, predictions, upstream_normalized)
        self.assertEqual(records, original)
        self.assertEqual(len(actual['source_loss_components']), 32)
        self.assertEqual(actual['records'], 32)


if __name__ == '__main__': unittest.main()

