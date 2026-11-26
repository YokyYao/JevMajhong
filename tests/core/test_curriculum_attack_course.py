import copy
import unittest

import torch

from tools.curriculum_attack_losses import canonical_question, loss, mixture


def expert(phase, kind='dahai', offered_reach=False):
    criteria = {'a0': {'type': kind}}
    if offered_reach and kind != 'reach':
        criteria['a1'] = {'type': 'reach'}
    return dict(_meta=dict(label_type='expert_replay', phase=phase),
        questions=dict(action=dict(criteria=criteria, label='a0', src='expert_replay')))


class AttackLossTests(unittest.TestCase):
    def test_new_tags_keep_masked_unknown_actions_at_zero_gradient(self):
        values = []
        for unknown in (-1000., 1000.):
            logits = torch.tensor([.3, -.2, unknown], requires_grad=True)
            q = dict(src='curriculum3:mc:kan_window', target=[.6, .4, 0.])
            value = loss(logits, q, 'cpu', {'mc:kan_window': 2.})
            value.backward()
            self.assertEqual(logits.grad[-1].item(), 0.)
            self.assertGreater(logits.grad[:2].abs().sum().item(), 0.)
            values.append((value.item(), logits.grad))
        self.assertEqual(values[0][0], values[1][0])
        self.assertTrue(torch.equal(values[0][1], values[1][1]))

    def test_source_loss_contract_cannot_silently_become_hard_or_unmasked(self):
        for q in (dict(src='paired_future_masked', target=[.5, .5]),
                  dict(src='curriculum3:mc:call_window', label=0),
                  dict(src='curriculum3:expert:call', target=[.5, .5]),
                  dict(src='curriculum3:unknown', label=0)):
            with self.assertRaises(ValueError):
                canonical_question(q)

    def test_mixture_preserves_replay_mass_and_natural_riichi_ratio(self):
        rows = [expert('closed_discard') for _ in range(40)]
        rows += [expert('closed_discard', 'reach') for _ in range(2)]
        rows += [expert('closed_discard', offered_reach=True) for _ in range(3)]
        for phase, n in (('open_discard', 20), ('call_window', 10), ('kan_window', 7)):
            rows += [expert(phase) for _ in range(n)]
        for phase in ('closed_discard', 'open_discard', 'call_window', 'kan_window'):
            rows.append(dict(_meta=dict(label_type='paired_future_masked', phase=phase),
                questions=dict(action=dict(target={'a0': .5, 'a1': .5}))))
        result = mixture(rows)
        self.assertAlmostEqual(sum(v for k, v in result['loss_mass'].items() if k.startswith('expert:')), .75)
        self.assertAlmostEqual(sum(v for k, v in result['loss_mass'].items() if k.startswith('mc:')), .25)
        self.assertAlmostEqual(result['loss_mass']['expert:plain'], .75*.25)
        self.assertAlmostEqual(result['example_weights']['expert:riichi_yes'], result['example_weights']['expert:riichi_no'])
        self.assertAlmostEqual(sum(result['counts'][k]*w for k, w in result['example_weights'].items())/len(rows), 1.)
        missing = [r for r in rows if not (r['_meta']['label_type'] == 'paired_future_masked' and r['_meta']['phase'] == 'kan_window')]
        with self.assertRaises(ValueError):
            mixture(missing)
        singleton = copy.deepcopy(rows)
        singleton[-1]['questions']['action']['target'] = {'a0': 1., 'a1': 0.}
        with self.assertRaises(ValueError):
            mixture(singleton)


if __name__ == '__main__':
    unittest.main()
