import copy
import unittest
import torch

from tools import curriculum_placement_parent_loss_v2 as new
from tools import curriculum_placement_parent_loss_v1 as old
from tests.core.test_curriculum_progression_parent_loss_v1 import fixture


def normalized_fixture(source, kinds, label, target):
    request, encoded = fixture(source, kinds, label, target)
    if target is not None: encoded['questions'][0]['target'] = [v/sum(target) for v in target]
    return request, encoded


class PlacementNormalizedAlignmentTests(unittest.TestCase):
    def test_rounded_soft_labels_match_upstream_without_changing_raw_request(self):
        target = [.1]*9+[.1+1e-15]
        self.assertNotEqual(sum(target), 1.)  # Python3.12 compensates summation of ten equal0.1 values.
        request, encoded = normalized_fixture(new.NEW_SOURCES[0], ['none']+['dahai']*9, 9, target)
        before = copy.deepcopy(encoded); raw = copy.deepcopy(request)
        with self.assertRaises(ValueError): old.annotate(copy.deepcopy(encoded), request)
        annotated = new.annotate(encoded, request); new.materialization_unchanged(before, annotated)
        self.assertEqual(request, raw)

    def test_full_soft_RI_objective_and_gradient_match_the_original_formula(self):
        request, encoded = normalized_fixture(new.NEW_SOURCES[0], ['reach']+['dahai']*9, 9, [.1]*9+[.1+1e-15])
        reference = copy.deepcopy(request)
        reference['questions']['action']['target'] = dict(zip(encoded['questions'][0]['keys'], encoded['questions'][0]['target']))
        a = old.annotate(copy.deepcopy(encoded), reference)['questions'][0]
        b = new.annotate(copy.deepcopy(encoded), request)['questions'][0]
        x = torch.linspace(-1., 1., 10, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        left = old.parts(x, a, 'cpu'); right = new.parts(y, b, 'cpu')
        for key in new.COMPONENTS: self.assertTrue(torch.equal(left[key], right[key]))
        left['objective'].backward(); right['objective'].backward(); self.assertTrue(torch.equal(x.grad, y.grad))

    def test_untested_options_still_have_zero_gradient_and_partial_RI_is_rejected(self):
        target = [.1]*8+[.2, 0.]
        request, encoded = normalized_fixture(new.NEW_SOURCES[0], ['none']+['dahai']*9, 8, target)
        q = new.annotate(encoded, request)['questions'][0]
        z = torch.tensor([0.]*9+[90.], dtype=torch.float64, requires_grad=True)
        new.parts(z, q, 'cpu')['objective'].backward(); self.assertEqual(float(z.grad[-1]), 0.)
        request, encoded = normalized_fixture(new.NEW_SOURCES[0], ['reach']+['dahai']*9, 8, target)
        with self.assertRaises(ValueError): new.annotate(encoded, request)

    def test_all25_inherited_objectives_and_gradients_remain_exact(self):
        for source in new.replay.MASS:
            kinds = ['reach', 'dahai', 'dahai'] if source.endswith(('riichi_yes', 'riichi_no')) else ['dahai']*3
            label = 0 if source.endswith('riichi_yes') else 1
            target = [.4, .6, 0.] if ':mc:' in source else None
            request, encoded = normalized_fixture(source, kinds, label, target)
            a = old.annotate(copy.deepcopy(encoded), request)['questions'][0]
            b = new.annotate(copy.deepcopy(encoded), request)['questions'][0]
            x = torch.linspace(-.8, .9, 3, dtype=torch.float64, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            left = old.parts(x, a, 'cpu'); right = new.parts(y, b, 'cpu')
            with self.subTest(source=source):
                for key in new.COMPONENTS: self.assertTrue(torch.equal(left[key], right[key]))
                left['objective'].backward(); right['objective'].backward(); self.assertTrue(torch.equal(x.grad, y.grad))

    def test_wrong_materialized_target_nonunit_raw_and_reannotation_are_rejected(self):
        source = new.NEW_SOURCES[0]
        request, encoded = normalized_fixture(source, ['none', 'pon'], 1, [.4, .6])
        encoded['questions'][0]['target'] = [.3, .7]
        with self.assertRaises(ValueError): new.annotate(encoded, request)
        request, encoded = normalized_fixture(source, ['none', 'pon'], 1, [.4, .7])
        with self.assertRaises(ValueError): new.annotate(encoded, request)
        request, encoded = normalized_fixture(source, ['none', 'pon'], 1, [.4, .6])
        q = new.annotate(encoded, request)['questions'][0]
        with self.assertRaises(ValueError): new.annotate(encoded, request)
        observer = []; weights = dict.fromkeys(new.masses(), 1.)
        new.weighted_loss(torch.zeros(2, requires_grad=True), q, 'cpu', weights, observer)
        self.assertEqual(new.summarize(observer, weights)['loss_method'], new.METHOD)


if __name__ == '__main__': unittest.main()
