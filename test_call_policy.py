import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from collections import Counter

import torch
from call_policy import CallRouter, is_call_request, request_hash, screenshot_view, metrics


def example():
    return {'state': {'hand': '5z 5z 1m', 'scores': [25000]*4, 'last_discard': 124,
                      'screen_observation': {'metadata_confirmed': False}},
        'questions': {'action': {'type': 'choice', 'instructions': 'Choose', 'label': 'a0',
            'criteria': {'a0': {'type': 'pon', 'pai': 'P', 'consumed': ['P', 'P']}, 'a1': {'type': 'none'}}}},
        '_meta': {'reaction_relative_seat': 2}}


class CallPolicyTests(unittest.TestCase):
    def test_missing_information_views_preserve_candidates_without_fabricating_id(self):
        native = example(); before = copy.deepcopy(native)
        screen = screenshot_view(native, unknown=True)
        self.assertIsNone(screen['state']['scores'])
        self.assertIsNone(screen['state']['last_discard'])
        self.assertEqual(screen['state']['screen_observation']['reaction']['tile'], '5z')
        self.assertEqual(screen['questions'], native['questions'])
        self.assertEqual(native, before)
        altered = copy.deepcopy(native); altered['questions']['action']['label'] = 'a1'
        self.assertEqual(request_hash(native), request_hash(altered))

    def test_router_restores_original_and_never_changes_discard_or_kan_path(self):
        original, trained = object(), object()
        model = SimpleNamespace(head=original)
        observed = []
        def infer(**request):
            observed.append((model.head, request))
            raise RuntimeError('Simulated inference failure')
        client = SimpleNamespace(runtime=SimpleNamespace(model=model), system_one=infer)
        router = CallRouter(client, trained)
        request = example(); request.pop('_meta'); request['questions']['action'].pop('label')
        before = copy.deepcopy(request)
        with self.assertRaisesRegex(RuntimeError, 'Simulated'): router.system_one(**request)
        self.assertIs(observed[-1][0], trained)
        self.assertIsNone(observed[-1][1]['state']['scores'])
        self.assertIs(model.head, original)
        self.assertEqual(request, before)
        for kind in ('dahai', 'daiminkan', 'hora', 'reach'):
            other = copy.deepcopy(request)
            other['questions']['action']['criteria']['a0']['type'] = kind
            self.assertFalse(is_call_request(other))
            with self.assertRaises(RuntimeError): router.system_one(**other)
            self.assertIs(observed[-1][0], original)
            self.assertIs(model.head, original)

    def test_padded_batch_matches_native_head_and_query_gradient_preserves_key(self):
        from jevany.model import PointerHead
        from scripts.train_call_repair import logits_batch
        torch.manual_seed(12)
        head = PointerHead(4, 4).requires_grad_(False)
        head.q.requires_grad_(True)
        rows = [{'decide': torch.randn(4), 'options': torch.randn(n, 4)} for n in [2, 4]]
        logits = logits_batch(head, rows, 'cpu')
        self.assertTrue(torch.isneginf(logits[0, 2:]).all())
        for i, row in enumerate(rows):
            torch.testing.assert_close(logits[i, :len(row['options'])], head(row['decide'], row['options']))
        before = copy.deepcopy(head.state_dict())
        optimizer = torch.optim.SGD(head.q.parameters(), lr=.1)
        target = torch.tensor([0, 1])
        initial = float(torch.nn.functional.cross_entropy(logits, target).detach())
        for _ in range(20):
            optimizer.zero_grad(); loss = torch.nn.functional.cross_entropy(logits_batch(head, rows, 'cpu'), target)
            loss.backward(); optimizer.step()
        self.assertLess(float(loss.detach()), initial)
        self.assertTrue(torch.equal(head.k.weight, before['k.weight']))
        self.assertFalse(torch.equal(head.q.weight, before['q.weight']))

    def test_pass_only_classifier_cannot_hide_zero_call_recall(self):
        report = metrics([{'gold': kind, 'pred': 'none', 'exact': kind == 'none', 'target_probability': .5}
                          for kind in ('chi', 'pon', 'none')])
        self.assertEqual(report['by_target']['chi']['recall'], 0)
        self.assertEqual(report['by_target']['pon']['recall'], 0)
        self.assertLess(report['macro_f1'], .2)

    def test_real_repaired_dataset_is_group_isolated_and_consumes_only_owned_tiles(self):
        from desktop.jev_sidecar.tiles import mjai
        data = Path(json.loads(Path('call-repair-20261109.json').read_text())['data'])
        if not (data / 'provenance.json').is_file(): self.skipTest('Private training corpus absent')
        groups, hashes = {}, set()
        for split in ('train', 'validation', 'test'):
            groups[split] = set()
            for line in (data / f'{split}.jsonl').read_text().splitlines():
                row = json.loads(line); h = request_hash(row)
                self.assertEqual(h, row['_meta']['text_sha256']); self.assertNotIn(h, hashes); hashes.add(h)
                groups[split].add(row['_meta']['group_id'])
                owned = Counter(mjai(tile) for tile in row['state']['hand'].split())
                for option in row['questions']['action']['criteria'].values():
                    if option['type'] == 'none': continue
                    self.assertEqual(len(option['consumed']), 2)
                    self.assertFalse(Counter(option['consumed']) - owned)
                    if option['type'] == 'chi': self.assertEqual(row['_meta']['reaction_relative_seat'], 3)
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])


if __name__ == '__main__': unittest.main()
