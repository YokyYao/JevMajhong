import copy
import unittest
import torch

from tools import curriculum_placement_parent_loss_v1 as new
from tools import curriculum_yaku_parent_loss_v1 as replay
from tools import curriculum_placement_objective_v1 as terminal
from tests.core.test_curriculum_progression_parent_loss_v1 import fixture


class PlacementParentLossTests(unittest.TestCase):
    def compare(self, source, kinds, label, target=None):
        request, encoded = fixture(source, kinds, label, target)
        raw = copy.deepcopy(request); original = copy.deepcopy(encoded)
        old = replay if source in replay.MASS else terminal
        a = old.annotate(copy.deepcopy(encoded), request)['questions'][0]
        annotated = new.annotate(copy.deepcopy(encoded), request); new.materialization_unchanged(original, annotated)
        b = annotated['questions'][0]
        x = torch.linspace(-.8, .9, len(kinds), dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        before = old.parts(x, a, 'cpu'); after = new.parts(y, b, 'cpu')
        for key in ('objective', 'standard_target_ce'): self.assertTrue(torch.equal(before[key], after[key]))
        if source in replay.MASS:
            for key in replay.COMPONENTS: self.assertTrue(torch.equal(before[key], after[key]))
            self.assertEqual(float(after['terminal_soft_objective'].detach()), 0.)
        else: self.assertTrue(torch.equal(after['terminal_soft_objective'], after['objective']))
        before['objective'].backward(); after['objective'].backward()
        self.assertTrue(torch.equal(x.grad, y.grad)); self.assertEqual(request, raw)

    def test_all25_replay_sources_preserve_values_and_gradients(self):
        for source in replay.MASS:
            if source.endswith('riichi_yes'): kinds, label = ['reach', 'dahai', 'dahai'], 0
            elif source.endswith('riichi_no'): kinds, label = ['reach', 'dahai', 'dahai'], 1
            elif ':call' in source or ':kan' in source: kinds, label = ['none', 'pon', 'daiminkan'], 0
            else: kinds, label = ['dahai', 'dahai', 'dahai'], 1
            target = [.4, .6, 0.] if ':mc:' in source else None
            with self.subTest(source=source): self.compare(source, kinds, label, target)

    def test_all32_new_sources_preserve_soft_and_full_RI_semantics(self):
        for source in new.NEW_SOURCES:
            for kinds, target in ((['none', 'pon', 'chi'], [.4, .6, 0.]),
                                  (['reach', 'dahai', 'dahai'], [.6, .1, .3]),
                                  (['hora', 'none', 'ankan'], [.4, .2, .4])):
                with self.subTest(source=source, kinds=kinds):
                    self.compare(source, kinds, target.index(max(target)), target)

    def test_untested_action_zero_gradient_and_partial_RI_rejection(self):
        source = new.NEW_SOURCES[0]
        request, encoded = fixture(source, ['none', 'pon', 'chi'], 1, [.4, .6, 0.])
        q = new.annotate(encoded, request)['questions'][0]
        z = torch.tensor([0., 1., 90.], dtype=torch.float64, requires_grad=True)
        new.parts(z, q, 'cpu')['objective'].backward(); self.assertEqual(float(z.grad[2]), 0.)
        request, encoded = fixture(source, ['reach', 'dahai', 'dahai'], 1, [.4, .6, 0.])
        with self.assertRaises(ValueError): new.annotate(encoded, request)

    def test_sparse_observer_signed_and_terminal_accounting_and_caps(self):
        weights = {s: 1.2 for s in new.masses()}; observed = []; losses = []
        for source, target in (('curriculum2:basic:legacy:riichi_yes', None), (new.NEW_SOURCES[0], [.6, .1, .3])):
            request, encoded = fixture(source, ['reach', 'dahai', 'dahai'], 0, target)
            q = new.annotate(encoded, request)['questions'][0]
            losses.append(new.weighted_loss(torch.zeros(3, dtype=torch.float64, requires_grad=True), q, 'cpu', weights, observed))
        result = new.summarize(observed, weights)
        self.assertAlmostEqual(result['loss'], float(torch.stack(losses).mean().detach()))
        self.assertAlmostEqual(result['loss']-result['ce'], result['signed_objective_adjustment'])
        self.assertNotIn('extra_RI_class_loss', result); self.assertEqual(len(result['source_counts']), 2)
        invalid = dict(weights); invalid[new.NEW_SOURCES[0]] = 17.
        with self.assertRaises(ValueError): new.summarize(observed, invalid)
        corrupted = copy.deepcopy(observed); corrupted[-1]['terminal_soft_objective'] += 1
        with self.assertRaises(ValueError): new.summarize(corrupted, weights)

    def test_unknown_source_and_reannotation_rejected_and_mass_unchanged(self):
        self.assertEqual(len(new.masses()), 57)
        self.assertAlmostEqual(float(sum(new.masses()[s] for s in replay.MASS)), .8)
        self.assertAlmostEqual(float(sum(new.masses()[s] for s in new.NEW_SOURCES)), .2)
        request, encoded = fixture('unknown', ['dahai', 'dahai'], 0)
        with self.assertRaises(ValueError): new.annotate(encoded, request)
        request, encoded = fixture(new.NEW_SOURCES[0], ['none', 'pon'], 0, [.6, .4])
        new.annotate(encoded, request)
        with self.assertRaises(ValueError): new.annotate(encoded, request)


if __name__ == '__main__': unittest.main()
