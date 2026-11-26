import copy
import math
import unittest

from tools.curriculum_placement_metrics_v1 import new_profile, selected_key, summarize
from tools.curriculum_placement_exposures_v1 import REPEAT_FIELD
from tools.curriculum_placement_mix_v1 import NEW_SOURCES, REPLAY_MASS, masses
from tools.curriculum_placement_objective_v1 import COMPONENTS


def fixture(probabilities):
    # Synthetic legal-menu-shaped records only; these are not native states.
    rows = [dict(_meta=dict(id=f'v{i}', split='validation', group_id=f'g{i}'),
        questions=dict(action=dict(src=s, label='b', target={'a': .4, 'b': .6, 'c': 0.},
            criteria={'a': {'type': 'none'}, 'b': {'type': 'pon'}, 'c': {'type': 'chi'}})))
        for i, s in enumerate(NEW_SOURCES)]
    predictions = [dict(id=r['_meta']['id'], keys=['a', 'b', 'c'], p=probabilities[:]) for r in rows]
    return rows, predictions


def materialize(row):
    q = row['questions']['action']; keys = list(q['criteria'])
    return dict(questions=[dict(qid='action', src=q['src'], keys=keys,
        label=keys.index(q['label']), target=[q['target'][k] for k in keys])])


class PlacementMetricTests(unittest.TestCase):
    def test_low_masked_loss_does_not_hide_untested_full_menu_selection(self):
        rows, predictions = fixture([.4, .6, 0.]); before = copy.deepcopy(rows)
        a = new_profile(rows, predictions, materialize)
        other = [dict(p, p=[.004, .006, .99]) for p in predictions]
        b = new_profile(rows, other, materialize)
        for source in NEW_SOURCES:
            self.assertAlmostEqual(a['source_loss_components'][source]['objective'],
                                   b['source_loss_components'][source]['objective'])
            self.assertEqual(a['selected_unlabelled_rate_by_source'][source], 0)
            self.assertEqual(b['selected_unlabelled_rate_by_source'][source], 1)
            self.assertEqual(b['selected_target_mass_by_source'][source], 0)
        self.assertEqual(rows, before)
        self.assertTrue(b['target_mass_is_not_predicted_win_or_top2_probability'])

    def test_serving_win_menu_and_full_RI_class_have_distinct_real_selection_rules(self):
        q = dict(criteria={'w': {'type': 'hora'}, 'r': {'type': 'reach'}, 'd': {'type': 'dahai'}})
        self.assertEqual(selected_key(q, ['w', 'r', 'd'], [.35, .4, .25]), 'r')
        self.assertEqual(selected_key(q, ['w', 'r', 'd'], [.4, .4, .2]), 'w')
        q = dict(criteria={'r': {'type': 'reach'}, 'a': {'type': 'dahai'}, 'b': {'type': 'dahai'}})
        self.assertEqual(selected_key(q, ['r', 'a', 'b'], [.45, .3, .25]), 'a')

    def test_sparse_update_summary_weights_actual_example_counts(self):
        weights = dict.fromkeys(masses(), 1.)
        new = NEW_SOURCES[0]; old = next(iter(REPLAY_MASS)); weights[new] = 2.; weights[old] = 3.
        pending = [dict(source=s, **dict.fromkeys(COMPONENTS, value))
                   for s, value in ((new, 2.), (new, 4.), (old, 10.))]
        result = summarize(pending, weights)
        self.assertEqual(result['loss'], 14.)
        self.assertEqual(result['source_counts'], {new: 2, old: 1})
        weights[new] = math.inf
        with self.assertRaises(ValueError): summarize(pending, weights)

    def test_missing_duplicate_train_or_reordered_validation_is_rejected(self):
        rows, predictions = fixture([.4, .6, 0.])
        with self.assertRaises(ValueError): new_profile(rows[:-1], predictions[:-1], materialize)
        with self.assertRaises(ValueError): new_profile(rows+[rows[0]], predictions+[predictions[0]], materialize)
        bad = copy.deepcopy(rows); bad[0]['_meta']['split'] = 'train'
        with self.assertRaises(ValueError): new_profile(bad, predictions, materialize)
        bad = copy.deepcopy(rows); bad[0]['_meta'][REPEAT_FIELD] = dict(original_id='another')
        with self.assertRaises(ValueError): new_profile(bad, predictions, materialize)
        bad = copy.deepcopy(predictions); bad[0]['keys'] = ['b', 'a', 'c']
        with self.assertRaises(ValueError): new_profile(rows, bad, materialize)
