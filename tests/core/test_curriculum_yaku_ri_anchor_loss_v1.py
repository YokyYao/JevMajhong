import copy
import unittest
import torch

from tools import curriculum_progression_parent_loss_v1 as retained
from tools import curriculum_ri_parent_anchor_v1 as anchor
from tools import curriculum_ri_parent_anchor_weighted_v1 as old_weighted
from tools import curriculum_ri_inverse_repeat_loss_v1 as inverse
from tools import curriculum_yaku_parent_loss_v1 as yaku
from tools import curriculum_yaku_ri_anchor_loss_v1 as objective
from tools import curriculum_yaku_ri_anchor_epoch_v1 as epoch
from tests.core.test_curriculum_ri_parent_anchor_v1 import fixture


def population():
    rows = []; encoded = {}; predictions = []
    for index, source in enumerate(yaku.MASS):
        kinds = ['dahai', 'reach', 'dahai'] if source in anchor.RI_SOURCES else ['dahai'] * 3
        label = 1 if source.endswith(':riichi_yes') else 2
        target = [.3, 0., .7] if ':mc:' in source else None
        row, rec = fixture(source, kinds, label, str(index), target)
        row['state'] = dict(public_test_index=index)
        rows.append(row); encoded[str(index)] = rec
        if source in anchor.RI_SOURCES:
            predictions.append(dict(id=str(index), keys=['k0', 'k1', 'k2'], p=[.15, .6, .25]))
    return rows, encoded, predictions


class YakuAnchorLossTests(unittest.TestCase):
    def annotated(self):
        rows, raw, predictions = population(); targets = objective.make_targets(rows, predictions)
        result = []
        for row in rows:
            encoded = yaku.annotate(copy.deepcopy(raw[row['_meta']['id']]), row)
            objective.attach_weight(encoded, 1.25, row['_meta']['id'])
            epoch.attach_targets(encoded, row, targets)
            result.append(encoded['questions'][0])
        return rows, raw, targets, result

    def test_every_retained19_value_and_gradient_match_exact_previous_anchor_objective(self):
        rows, raw, targets, current = self.annotated()
        for row, question in zip(rows, current):
            if question['src'] not in retained.MASS:
                continue
            old = anchor.annotate(copy.deepcopy(raw[row['_meta']['id']]), row, targets)
            old_weighted.attach(old, 1.25, row['_meta']['id'])
            a = torch.tensor([-.3, .8, .2], dtype=torch.float64, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            actual = objective.weighted_loss(a, question, 'cpu')
            expected = old_weighted.weighted_loss(b, old['questions'][0], 'cpu')
            self.assertTrue(torch.equal(actual, expected))
            actual.backward(); expected.backward()
            self.assertTrue(torch.equal(a.grad, b.grad))

    def test_six_new_routes_keep_exact_previous_route_values_and_gradients_without_KL(self):
        _, _, _, questions = self.annotated()
        for question in questions:
            if question['src'] not in yaku.NEW_SOURCES:
                continue
            a = torch.tensor([-.3, .8, .2], dtype=torch.float64, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            actual = objective.weighted_loss(a, question, 'cpu')
            expected = yaku.parts(b, question, 'cpu')['objective'] * 1.25
            self.assertTrue(torch.equal(actual, expected))
            actual.backward(); expected.backward()
            self.assertTrue(torch.equal(a.grad, b.grad))
            self.assertEqual(float(objective.parts(a, question, 'cpu')['parent_RI_KL'].detach()), 0.)

    def test_observer_reconstructs_base_anchor_and_route_contributions(self):
        _, _, _, questions = self.annotated(); pending = []; losses = []
        for question in questions:
            logits = torch.tensor([-.3, .8, .2], dtype=torch.float64, requires_grad=True)
            losses.append(objective.weighted_loss(logits, question, 'cpu', pending))
        summary = objective.summarize(pending)
        self.assertEqual(set(summary['source_counts']), set(yaku.MASS))
        self.assertAlmostEqual(summary['loss'], float(torch.stack(losses).mean().detach()), places=12)
        self.assertAlmostEqual(summary['loss'], summary['base25_objective_without_anchor']
            + summary['parent_RI_anchor_adjustment'], places=12)
        self.assertAlmostEqual(summary['parent_RI_anchor_adjustment'], 4 * summary['parent_RI_KL'], places=12)

    def test_legacy_weight_schema_missing_targets_changed_input_and_invalid_accounting_rejected(self):
        rows, raw, targets, questions = self.annotated()
        question = copy.deepcopy(questions[0]); question[inverse.MARKER] = inverse.METHOD
        with self.assertRaises(ValueError):
            objective.weighted_loss(torch.zeros(3), question, 'cpu')
        _, _, predictions = population()
        with self.assertRaises(ValueError):
            objective.make_targets(rows, predictions[:-1])
        row = next(row for row in rows if row['questions']['action']['src'] in anchor.RI_SOURCES)
        rec = yaku.annotate(copy.deepcopy(raw[row['_meta']['id']]), row)
        objective.attach_weight(rec, 1., row['_meta']['id'])
        changed = copy.deepcopy(row); changed['state']['public_test_index'] += 1
        with self.assertRaises(ValueError):
            epoch.attach_targets(rec, changed, targets)
        pending = []
        objective.weighted_loss(torch.zeros(3, requires_grad=True), questions[0], 'cpu', pending)
        pending[0]['objective'] += 1
        with self.assertRaises(ValueError):
            objective.summarize(pending)


if __name__ == '__main__':
    unittest.main()
