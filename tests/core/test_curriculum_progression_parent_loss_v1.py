import copy
import unittest

import torch

from tools import curriculum_calls_classroute_v1 as parent
from tools import curriculum_progression_objective_v1 as progression
from tools import curriculum_progression_parent_loss_v1 as retained


def fixture(source, kinds, label, target=None):
    keys = ['k'+str(i) for i in range(len(kinds))]
    raw = dict(src=source, label=keys[label], criteria={k: dict(type=kind) for k, kind in zip(keys, kinds)})
    q = dict(qid='action', src=source, keys=keys, label=label, target=target)
    if target is not None:
        raw['target'] = dict(zip(keys, target))
    return {'questions': {'action': raw}}, {'questions': [q]}


class ParentLossRetentionTests(unittest.TestCase):
    def compare_values_and_gradients(self, request, encoded, annotate_old, parts_old):
        old = annotate_old(copy.deepcopy(encoded), request)['questions'][0]
        new = retained.annotate(copy.deepcopy(encoded), request)['questions'][0]
        a = torch.linspace(-.7, .9, len(new['keys']), dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        expected = parts_old(a, old, 'cpu')['objective']
        actual = retained.parts(b, new, 'cpu')['objective']
        self.assertTrue(torch.equal(expected, actual))
        expected.backward(); actual.backward()
        self.assertTrue(torch.equal(a.grad, b.grad))

    def test_all_thirteen_parent_sources_keep_exact_objective_and_gradient(self):
        self.assertEqual(len(parent.MASS), 13)
        for source in parent.MASS:
            if source.endswith('riichi_yes'):
                kinds, label = ['dahai', 'reach', 'dahai', 'dahai'], 1
            elif source.endswith('riichi_no'):
                kinds, label = ['dahai', 'reach', 'dahai', 'dahai'], 2
            elif source.endswith(':call') or source.endswith(':kan'):
                kinds, label = ['none', 'pon', 'daiminkan'], 0
            else:
                kinds, label = ['dahai', 'dahai', 'dahai'], 1
            with self.subTest(source=source):
                request, encoded = fixture('curriculum2:'+source, kinds, label)
                self.compare_values_and_gradients(request, encoded, parent.annotate, parent.parts)

    def test_all_six_nonparent_sources_keep_progression_objective_and_gradient(self):
        sources = [s for s in retained.MASS if not s.startswith('curriculum2:')]
        self.assertEqual(len(sources), 6)
        for source in sources:
            target = [.3, 0., .7] if ':mc:' in source else None
            request, encoded = fixture(source, ['dahai', 'dahai', 'dahai'], 2, target)
            with self.subTest(source=source):
                self.compare_values_and_gradients(request, encoded, progression.annotate, progression.parts)

    def test_positive_RI_signed_adjustment_is_not_misreported_as_nonnegative_extra_loss(self):
        request, encoded = fixture('curriculum2:basic:teacher:riichi_yes', ['reach', 'dahai', 'dahai'], 0)
        question = retained.annotate(encoded, request)['questions'][0]
        result = retained.parts(torch.tensor([0., 0., 0.], dtype=torch.float64), question, 'cpu')
        self.assertLess(float(result['signed_objective_minus_standard_ce']), 0)
        self.assertGreater(float(result['parent_RI_max_ce']), 0)
        self.assertAlmostEqual(float(result['objective']), float(result['parent_RI_max_ce']))

    def test_actual_weighted_observations_reconstruct_the_optimized_objective(self):
        observer = []; weights = {s: 1.5 for s in retained.MASS}; losses = []
        for source, kinds, label in (
                ('curriculum2:basic:legacy:riichi_yes', ['reach', 'dahai', 'dahai'], 0),
                ('curriculum2:basic:teacher:riichi_no', ['reach', 'dahai', 'dahai'], 1)):
            request, encoded = fixture(source, kinds, label)
            q = retained.annotate(encoded, request)['questions'][0]
            logits = torch.tensor([.3, -.1, .8], dtype=torch.float64, requires_grad=True)
            losses.append(retained.weighted_loss(logits, q, 'cpu', weights, observer))
        summary = retained.summarize(observer, weights)
        self.assertAlmostEqual(summary['loss'], float(torch.stack(losses).mean().detach()))
        self.assertAlmostEqual(summary['loss']-summary['ce'], summary['signed_objective_adjustment'])
        self.assertEqual(sum(summary['source_counts'].values()), 2)

    def test_repeat_annotation_and_unregistered_weights_are_rejected(self):
        request, encoded = fixture('curriculum2:basic:legacy:plain', ['dahai', 'dahai'], 0)
        retained.annotate(encoded, request)
        with self.assertRaises(ValueError):
            retained.annotate(encoded, request)
        with self.assertRaises(ValueError):
            retained.weighted_loss(torch.tensor([.1, .2]), encoded['questions'][0], 'cpu', {})

    def test_input_request_labels_menus_and_source_masses_remain_unchanged(self):
        request, encoded = fixture('curriculum2:basic:legacy:riichi_no', ['dahai', 'reach', 'dahai'], 2)
        original = copy.deepcopy(request)
        retained.annotate(encoded, request)
        self.assertEqual(request, original)
        self.assertIs(retained.MASS, progression.MASS)


if __name__ == '__main__':
    unittest.main()
