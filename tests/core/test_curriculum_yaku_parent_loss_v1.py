import copy
import unittest
import torch

from tools import curriculum_yaku_parent_loss_v1 as new
from tools import curriculum_yaku_objective_v1 as previous
from tools import curriculum_progression_parent_loss_v1 as parent
from tests.core.test_curriculum_progression_parent_loss_v1 import fixture


class YakuParentLossTests(unittest.TestCase):
    def compare(self, source, kinds, label, target=None):
        request, encoded = fixture(source, kinds, label, target)
        raw = copy.deepcopy(request); original = copy.deepcopy(encoded)
        expected_module = previous if source in new.NEW_SOURCES else parent
        a = expected_module.annotate(copy.deepcopy(encoded), request)['questions'][0]
        annotated = new.annotate(copy.deepcopy(encoded), request)
        new.materialization_unchanged(original, annotated)
        b = annotated['questions'][0]
        x = torch.linspace(-.9, .8, len(kinds), dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        expected = expected_module.parts(x, a, 'cpu')
        actual = new.parts(y, b, 'cpu')
        for field in ('objective', 'standard_target_ce'):
            self.assertTrue(torch.equal(expected[field], actual[field]))
        expected['objective'].backward(); actual['objective'].backward()
        self.assertTrue(torch.equal(x.grad, y.grad))
        self.assertEqual(request, raw)

    def test_all_nineteen_replay_sources_keep_exact_parent_values_and_gradients(self):
        for source in parent.MASS:
            if source.endswith('riichi_yes'): kinds, label = ['reach', 'dahai', 'dahai'], 0
            elif source.endswith('riichi_no'): kinds, label = ['reach', 'dahai', 'dahai'], 1
            elif ':call' in source or ':kan' in source: kinds, label = ['none', 'pon', 'daiminkan'], 0
            else: kinds, label = ['dahai', 'dahai', 'dahai'], 1
            target = [.4, .6, 0.] if ':mc:' in source else None
            with self.subTest(source=source): self.compare(source, kinds, label, target)

    def test_six_new_route_losses_keep_existing_RI_and_ordinary_semantics(self):
        for source in new.NEW_SOURCES:
            for kinds, label in ((['dahai', 'dahai', 'dahai'], 1),
                                  (['reach', 'dahai', 'dahai'], 0),
                                  (['reach', 'dahai', 'dahai'], 2)):
                with self.subTest(source=source, label=label): self.compare(source, kinds, label)

    def test_masses_weights_and_epoch_budget_are_unchanged(self):
        self.assertIs(new.MASS, previous.MASS)
        self.assertEqual(set(new.MASS), set(parent.MASS)|set(new.NEW_SOURCES))
        self.assertAlmostEqual(sum(new.MASS[s] for s in parent.MASS), .8)
        self.assertEqual(new.RECORDS, 91247); self.assertEqual(new.FINAL_STEP, 2500)
        self.assertIs(new.weights, previous.weights); self.assertIs(new.epoch_order, previous.epoch_order)

    def test_signed_optimizer_accounting_and_pristine_annotation(self):
        weights = {s: 1.2 for s in new.MASS}; observed = []; losses = []
        for source in ('curriculum2:basic:legacy:riichi_yes', new.NEW_SOURCES[0]):
            request, encoded = fixture(source, ['reach', 'dahai', 'dahai'], 0)
            question = new.annotate(encoded, request)['questions'][0]
            losses.append(new.weighted_loss(torch.zeros(3, dtype=torch.float64, requires_grad=True),
                question, 'cpu', weights, observed))
            with self.assertRaises(ValueError): new.annotate(encoded, request)
        result = new.summarize(observed, weights)
        self.assertAlmostEqual(result['loss'], float(torch.stack(losses).mean().detach()))
        self.assertAlmostEqual(result['loss']-result['ce'], result['signed_objective_adjustment'])
        self.assertNotIn('extra_RI_class_loss', result)


if __name__ == '__main__': unittest.main()
