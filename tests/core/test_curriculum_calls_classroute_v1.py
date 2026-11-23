import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

from tools import curriculum_calls_classroute_v1 as course
from tools.curriculum_calls_riichi_retention_v1 import annotate as old_annotate, parts as old_parts
from tools.curriculum_calls_classroute_parent_v1 import require_policy_grade
from tools.curriculum_parent import ParentNotReady
from tests.core import test_curriculum_calls_repair as fixtures


class CallsClassRouteTests(unittest.TestCase):
    def fixture(self):
        rows, predictions = fixtures.CallsRepairTests().fixture()
        item = copy.deepcopy(rows[0]); item['_meta']['id'] = course.POST
        item['questions']['action'].update(src='curriculum2:'+course.POST,
            criteria={'d': {'type': 'dahai'}, 'x': {'type': 'dahai'}}, label='d')
        rows.append(item); predictions.append(dict(id=course.POST, keys=['d', 'x'], p=[.8, .2]))
        return rows, predictions

    def ce(self, q, keys, p): return -math.log(max(p[keys.index(q['label'])], 1e-30))

    def encoded(self, row):
        q = row['questions']['action']; keys = list(q['criteria'])
        return dict(questions=[dict(qid='action', src=q['src'], keys=keys, label=keys.index(q['label']))])

    def test_loss_values_and_gradients_preserve_old12_and_add_only_post_CE(self):
        rows, _ = self.fixture(); original = copy.deepcopy(rows)
        for row in rows:
            encoded = course.annotate(self.encoded(row), row); q = encoded['questions'][0]
            z = torch.linspace(-.7, .8, len(q['keys']), dtype=torch.float64, requires_grad=True)
            y = z.detach().clone().requires_grad_(); actual = course.parts(z, q, 'cpu')
            if q['src'] == 'curriculum2:'+course.POST:
                expected = torch.nn.functional.cross_entropy(y[None], torch.tensor([q['label']]))
                self.assertEqual(float(actual['advanced_action_ce']), 0.)
                self.assertEqual(float(actual['post_reach_action_ce']), float(actual['objective']))
            else:
                old = old_annotate(self.encoded(row), row)['questions'][0]
                expected = old_parts(y, old, 'cpu')['objective']
                self.assertEqual(float(actual['post_reach_action_ce']), 0.)
            self.assertTrue(torch.equal(actual['objective'], expected))
            self.assertTrue(torch.equal(torch.autograd.grad(actual['objective'], z)[0], torch.autograd.grad(expected, y)[0]))
        self.assertEqual(rows, original)

    def test_real_observation_summary_matches_cached_objective_not_raw_CE_alias(self):
        rows, predictions = self.fixture(); pending = []; weights = {k: 1. for k in course.MASS}
        for row, prediction in zip(rows, predictions):
            q = course.annotate(self.encoded(row), row)['questions'][0]
            z = torch.linspace(-.5, .8, len(q['keys']), dtype=torch.float64, requires_grad=True)
            prediction['p'] = z.detach().softmax(0).tolist()
            course.weighted_loss(z, q, 'cpu', weights, pending)
        actual = course.profile(rows, predictions, course.MASS, {'weighted_target_ce': 123}, self.ce)
        summary = course.summarize(pending, weights)
        for source, parts in actual['source_loss_components'].items():
            for name, value in parts.items(): self.assertAlmostEqual(value, summary['source_loss_components'][source][name], places=12)
        with self.assertRaises(ValueError): course.summarize(pending[:-1], weights)
        with self.assertRaises(ValueError): course.profile(rows, predictions[:-1], course.MASS, {'weighted_target_ce': 1}, self.ce)

    def test_selected_RI_and_raw_metrics_remain_separate(self):
        rows, predictions = self.fixture()
        for row, p in zip(rows, predictions):
            if row['questions']['action']['src'].endswith('riichi_no'):
                row['questions']['action']['criteria']['x'] = {'type': 'dahai'}
                p.update(keys=['d', 'r', 'x'], p=[.35, .4, .25])
        before = copy.deepcopy((rows, predictions))
        actual = course.profile(rows, predictions, course.MASS, {'weighted_target_ce': 123}, self.ce)
        self.assertFalse(actual['raw_basic_source_quality']['all_source_quality_passed'])
        self.assertTrue(actual['basic_replay_source_quality']['all_source_quality_passed'])
        self.assertEqual(actual['raw_basic_replay']['stratum_ce'], actual['basic_replay']['stratum_ce'])
        self.assertEqual((rows, predictions), before)

    def test_recipe_freezes_two_complete_actual_epochs(self):
        c = json.loads((Path(__file__).resolve().parents[2]/'configs/training/curriculum-calls-classroute-v1.json').read_text())
        c['loss_mass'] = course.MASS
        counts = dict(zip(course.MASS, [8826, 3106, 194, 272, 1075, 812, 5590, 3569, 1786, 9352, 13399, 729, 312]))
        proposed = course.layout(counts)
        data = dict(status=course.READY, categories={'train': counts}, loss_mass=course.MASS,
            example_weights=proposed['example_weights'], count_only_accumulation=253,
            files={'validation': {'records': 12479}})
        self.assertEqual(course.require_recipe(c, data)['final_step'], 388)
        for name, value in [('selection_method', 'raw'), ('epochs', 1), ('accumulation', 254),
                            ('candidate_rule', 'minimum_loss'), ('automatic_release', True)]:
            altered = dict(c); altered[name] = value
            with self.assertRaises(ValueError): course.require_recipe(altered, data)

    def test_native_evidence_cannot_be_substituted_for_source_grade_or_other_route(self):
        raw = {'all_source_quality_passed': False}; selected = {'all_source_quality_passed': True}
        native = dict(status='fixed_RI_route_native64_independent_cached_audit_passed', raw_parameters_unchanged=True,
            source_grade_not_overridden=True, baseline16_exact=True, final_reserved_grid_unused=True,
            checked_complete_match_ledgers=192, parent_admitted=False, new_optimizer_updates=0,
            policy={'method': course.ROUTE})
        require_policy_grade(raw, selected, native)
        for field, value in [('source_grade_not_overridden', False), ('new_optimizer_updates', 1), ('parent_admitted', True)]:
            with self.assertRaises(ParentNotReady): require_policy_grade(raw, selected, dict(native, **{field: value}))
        with self.assertRaises(ParentNotReady): require_policy_grade(raw, raw, native)

    def test_full3266_warmstart_checks_choice_boundary_even_inside_probability_tolerance(self):
        source, predictions = self.fixture(); base = next(r for r in source if r['questions']['action']['src'].endswith('riichi_yes'))
        parent = []; child = []; before = []; after = []
        for i in range(3266):
            old = copy.deepcopy(base); old['_meta']['id'] = str(i)
            old['questions']['action']['src'] = old['questions']['action']['src'].replace('curriculum2:basic:', 'curriculum1:')
            parent.append(old); current = copy.deepcopy(old); current['questions']['action']['src'] = base['questions']['action']['src']; child.append(current)
            p = dict(id=str(i), keys=['d', 'r'], p=[.4999999, .5000001]); before.append(p); after.append(copy.deepcopy(p))
        self.assertEqual(course.warmstart_parity(parent, before, child, after)['checked_basic_menus'], 3266)
        after[-1]['p'] = [.5000001, .4999999]
        with self.assertRaises(ValueError): course.warmstart_parity(parent, before, child, after)


if __name__ == '__main__': unittest.main()
