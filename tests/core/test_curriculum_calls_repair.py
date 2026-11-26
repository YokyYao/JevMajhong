import copy
import math
import unittest
from unittest.mock import patch

from tools import curriculum_calls_repair as repair
from tools.curriculum_parent import ParentNotReady
from tools.curriculum_transfer import METHOD, RESEARCH_METHOD


class CallsRepairTests(unittest.TestCase):
    def fixture(self):
        rows = []; predictions = []
        for category in repair.MASS:
            phase = category.split(':')[-1]
            if phase in ('riichi_yes', 'riichi_no'):
                criteria = {'d': {'type': 'dahai'}, 'r': {'type': 'reach'}}
                label = 'r' if phase == 'riichi_yes' else 'd'
            elif phase == 'kan':
                criteria = {'p': {'type': 'none'}, 'k': {'type': 'daiminkan'}}
                label = 'p' if ':teacher:' in category else 'k'
            elif phase == 'call':
                criteria = {'p': {'type': 'none'}, 'c': {'type': 'chi'}}; label = 'c'
            else:
                criteria = {'d': {'type': 'dahai'}, 'x': {'type': 'dahai'}}; label = 'd'
            rows.append({'_meta': {'id': category, 'group_id': category}, 'state': {'hand': ['1m']},
                'questions': {'action': {'src': 'curriculum2:'+category, 'criteria': criteria,
                    'type': 'choice', 'instructions': 'Choose', 'label': label}}})
            predictions.append({'id': category, 'keys': list(criteria),
                'p': [.9 if k == label else .1 for k in criteria]})
        return rows, predictions

    def ce(self, q, keys, p):
        return -math.log(p[keys.index(q['label'])])

    def test_full_profile_retains_sources_and_zero_positive_teacher_kan(self):
        rows, predictions = self.fixture()
        result = repair.profile(rows, predictions, repair.MASS, {'weighted_target_ce': 123}, self.ce)
        self.assertAlmostEqual(result['weighted_target_ce'], -math.log(.9))
        self.assertEqual(len(result['stratum_counts']), 12)
        self.assertEqual(result['basic_replay']['stratum_counts'], {'plain': 2, 'riichi_yes': 2, 'riichi_no': 2})
        self.assertTrue(result['basic_replay_source_quality']['all_source_quality_passed'])
        self.assertEqual(result['riichi_by_source']['teacher'], {'tp': 1, 'tn': 1})
        self.assertEqual(result['operation_type_counts_by_source']['advanced:teacher']['daiminkan'],
            {'tp': 0, 'fn': 0, 'tn': 1, 'fp': 0})
        self.assertEqual(result['operation_type_counts']['daiminkan'], {'tp': 1, 'fn': 0, 'tn': 1, 'fp': 0})

    def test_duplicate_missing_menu_and_invalid_probability_cannot_pass(self):
        for change in ('duplicate', 'missing', 'menu', 'nan', 'mass'):
            rows, predictions = self.fixture(); mass = dict(repair.MASS)
            if change == 'duplicate': predictions[-1] = copy.deepcopy(predictions[0])
            elif change == 'missing': predictions.pop()
            elif change == 'menu': predictions[0]['keys'][0] = 'wrong'
            elif change == 'nan': predictions[0]['p'][0] = float('nan')
            else: mass[next(iter(mass))] += .01
            with self.assertRaises(ValueError):
                repair.profile(rows, predictions, mass, {'weighted_target_ce': 0}, self.ce)

    def test_warm_start_checks_every_basic_input_and_prediction(self):
        rows, predictions = self.fixture(); parent = repair.basic_projection(rows)
        parent_predictions = [p for p in predictions if p['id'].startswith('basic:')]
        result = repair.require_parent_prediction_parity(parent, parent_predictions, rows, predictions)
        self.assertEqual(result['checked_basic_menus'], 6); self.assertEqual(result['max_probability_error'], 0)
        for change in ('state', 'label', 'probability', 'candidate_order', 'extra_basic'):
            child = copy.deepcopy(rows); scores = copy.deepcopy(predictions)
            if change == 'state': child[0]['state']['hand'] = ['2m']
            elif change == 'label': child[0]['questions']['action']['label'] = 'x'
            elif change == 'probability': scores[0]['p'] = [.8, .2]
            elif change == 'candidate_order': scores[0]['keys'].reverse(); scores[0]['p'].reverse()
            else:
                child.append(copy.deepcopy(child[0])); child[-1]['_meta']['id'] = 'extra'
                scores.append(dict(scores[0], id='extra'))
            with self.assertRaises(ValueError):
                repair.require_parent_prediction_parity(parent, parent_predictions, child, scores)
        with self.assertRaises(ValueError):
            repair.require_parent_prediction_parity(parent, parent_predictions, rows, predictions, tolerance=.1)

    def test_aggregate_seal_and_research_override_do_not_admit_parent(self):
        cfg = {'parent_initialization': {'manifest': '/tmp/example-seal', 'sha256': 'digest', 'method': METHOD}}
        qualified = {'method': METHOD, 'source_specific_parent_quality_checked': True,
            'source_specific_sealer': 'tools/seal_curriculum_basic_repair.py'}
        for change in ('aggregate_only', 'research', 'wrong_sealer'):
            seal = copy.deepcopy(qualified); config = copy.deepcopy(cfg)
            if change == 'aggregate_only': seal.pop('source_specific_parent_quality_checked')
            elif change == 'research': config['parent_initialization']['method'] = seal['method'] = RESEARCH_METHOD
            else: seal['source_specific_sealer'] = 'other.py'
            with patch.object(repair, 'sha', return_value='digest'), patch.object(repair, 'read', return_value=seal), \
                patch.object(repair, 'verify_curriculum_parent') as original:
                with self.assertRaises(ParentNotReady): repair.verify_source_qualified_parent(config)
                original.assert_not_called()

    def test_loader_restores_original_on_error(self):
        class Initialization:
            verified_parent = object()
        previous = Initialization.verified_parent
        with self.assertRaises(RuntimeError):
            with repair.source_qualified_parent_loader(Initialization):
                self.assertIs(Initialization.verified_parent, repair.verify_source_qualified_parent)
                raise RuntimeError('Fixture failure')
        self.assertIs(Initialization.verified_parent, previous)


if __name__ == '__main__':
    unittest.main()
