import copy
import math
from types import SimpleNamespace
import unittest
import torch
from sampling_loss import SamplingLossHooks
from call_sampling_prior import CLASSES, option_class


def loss(z, q, dev):
    return torch.nn.functional.cross_entropy(z[None], torch.tensor([q['label']]))


class SamplingLossTests(unittest.TestCase):
    def test_correct_raw_population_logits_stationary_in_sampled_likelihood(self):
        module = SimpleNamespace(encode_batch=lambda *a: [], question_loss=loss)
        z = torch.tensor([math.log(.1), math.log(.9)], dtype=torch.float64, requires_grad=True)
        offset = [0., -math.log(9.)]
        with SamplingLossHooks(module, {}) as hooks:
            objective = sum(module.question_loss(z, {'label': y, '_sampling_log_inclusion': offset}, 'cpu') for y in (0, 1)) / 2
            objective.backward()
            self.assertTrue(torch.allclose(z.grad, torch.zeros_like(z), atol=1e-12))
            self.assertEqual(hooks.adjusted_losses, 2)
        self.assertIs(module.question_loss, loss)

    def test_equal_inclusion_matches_original_ce_and_keeps_raw_logits(self):
        module = SimpleNamespace(encode_batch=lambda *a: [], question_loss=loss)
        z = torch.tensor([.2, -.4], requires_grad=True);before = z.detach().clone()
        with SamplingLossHooks(module, {}):
            q = {'label': 1, '_sampling_log_inclusion': [-2., -2.]}
            self.assertAlmostEqual(float(module.question_loss(z, q, 'cpu').detach()), float(loss(z, {'label': 1}, 'cpu').detach()), places=6)
            self.assertTrue(torch.equal(z.detach(), before))
            self.assertTrue(torch.equal(module.question_loss(z, {'label': 1}, 'cpu'), loss(z, {'label': 1}, 'cpu')))

    def test_restores_hooks_after_error_and_rejects_bad_offsets(self):
        encode = lambda *a: []
        module = SimpleNamespace(encode_batch=encode, question_loss=loss)
        with self.assertRaises(ValueError):
            with SamplingLossHooks(module, {}):
                module.question_loss(torch.zeros(2), {'label': 0, '_sampling_log_inclusion': [float('nan'), 0.]}, 'cpu')
        self.assertIs(module.encode_batch, encode)
        self.assertIs(module.question_loss, loss)

    def test_real_sdk_shuffle_alignment_without_input_or_tokens_mutation(self):
        from jevany import train as trainer
        record = {'state': {'hand': ['1m']}, '_meta': {'id': 'probe', 'phase': 'reaction'},
                  'questions': {'action': {'type': 'choice', 'src': 'probe', 'label': 'chi',
                    'criteria': {'chi': {'type': 'chi', 'call_features': {'loses_closed_status': True}},
                                 'pass': {'type': 'none'}}}}}
        original = copy.deepcopy(record)
        args = SimpleNamespace(seed=7, p_none=0., p_none_distract=0., p_distract=0.,
                               p_none_pair=0., max_state=4096, max_branch=8192, max_packed=8192)
        model = SimpleNamespace(encode=lambda *a, **kw: {'ids': [11, 22]})
        weights = {k: 2. for k in CLASSES};weights['none'] = 9.
        with SamplingLossHooks(trainer, weights) as hooks:
            for epoch in range(4):
                variants = trainer.encode_batch(model, None, args, [record], epoch)
                q = variants[0].rec['questions'][0]
                self.assertEqual(q['keys'][q['label']], 'chi')
                expected = [-math.log(weights[option_class(record['questions']['action']['criteria'][k])]) for k in q['keys']]
                self.assertEqual(q['_sampling_log_inclusion'], expected)
                self.assertEqual(variants[0].enc['ids'], [11, 22])
            self.assertEqual(hooks.reaction_records, 4)
        self.assertEqual(record, original)

    def test_rejects_augmentation_and_misaligned_metadata(self):
        module = SimpleNamespace(encode_batch=lambda *a: [], question_loss=loss)
        with SamplingLossHooks(module, {}) as hooks:
            with self.assertRaises(ValueError):
                module.encode_batch(None, None, SimpleNamespace(p_none=1.), [], 0)
            with self.assertRaises(ValueError):hooks.annotate([{}], [])


if __name__ == '__main__':
    unittest.main()
