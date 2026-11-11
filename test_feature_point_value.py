"""Native pointer replay, frozen ordinary routing and score utility round trips."""
from types import SimpleNamespace
import unittest

import torch
from desktop.tests.test_stage2 import reaction
from feature_point_value import FeaturePointClient, point_features
from guide_call_features import NAMES, VERSION
from scripts.fit_feature_points import paired_features
from jevany.model import PointerHead


class Ordinary:
    def __init__(self): self.requests = []
    def system_one(self, **request): self.requests.append(request); return {'ordinary': True}


class Teacher:
    def __init__(self, head):
        self.runtime = SimpleNamespace(model=SimpleNamespace(head=head))
        self.decide, self.options = torch.randn(8), torch.randn(2, 8)
        self.corrupt = False
    def system_one(self, **request):
        p = self.runtime.model.head(self.decide, self.options).softmax(-1)
        if self.corrupt: p = torch.tensor([.999, .001])
        return {'answers': {'action': {'probabilities': dict(zip(request['questions']['action']['criteria'], p.tolist()))}}}


def build():
    torch.manual_seed(29); head = PointerHead(8, dp=4)
    point = {'strategy_feature_version': VERSION, 'strategy_feature_names': list(NAMES),
        'scope': ['chi', 'pon', 'none'], 'target_scale': 1000., 'coefficient': torch.randn(4 + len(NAMES))}
    teacher, ordinary = Teacher(head), Ordinary()
    return FeaturePointClient(ordinary, teacher, point), teacher, ordinary


def native_request():
    request = reaction(); request['state'].pop('screen_observation', None)
    return request


class FeaturePointValueTest(unittest.TestCase):
    def test_native_replay_and_fitter_features_agree_exactly(self):
        client, teacher, ordinary = build(); request = native_request()
        try:
            output = client.system_one(**request)
            captured = client.last
            features = point_features(client.head, teacher.decide, teacher.options, request)
            self.assertTrue(torch.equal(features, captured['features']))
            for action in range(2):
                row = dict(decide=teacher.decide, options=teacher.options,
                    strategy_features=features[:, 4:].float(), action=action,
                    teacher_action=captured['teacher_index'])
                self.assertTrue(torch.equal(paired_features(client.head, row).double(), captured['relative_features'][action]))
            self.assertEqual(float(captured['values'][captured['teacher_index']]), 0.)
            keys = list(output['answers']['action']['probabilities'])
            self.assertEqual(max(range(2), key=lambda i: output['answers']['action']['probabilities'][keys[i]]), captured['chosen'])
            self.assertFalse(ordinary.requests)
        finally: client.close()

    def test_ordinary_payload_is_exact_and_does_not_capture_teacher(self):
        client, teacher, ordinary = build(); request = native_request()
        request['questions']['action']['criteria'] = {'a0': {'type': 'dahai', 'pai': '1m'}}
        try:
            self.assertEqual(client.system_one(**request), {'ordinary': True})
            self.assertEqual(ordinary.requests, [request]); self.assertEqual(client.checks, 0)
        finally: client.close()

    def test_corrupt_native_teacher_distribution_is_rejected(self):
        client, teacher, ordinary = build(); teacher.corrupt = True
        try:
            with self.assertRaises(ValueError): client.system_one(**native_request())
        finally: client.close()

    def test_zero_advantage_keeps_teacher_for_either_action_order(self):
        seen = set()
        for reverse in (False, True):
            client, teacher, ordinary = build()
            if reverse: teacher.options = teacher.options.flip(0)
            client.coefficient.zero_()
            try:
                client.system_one(**native_request())
                self.assertEqual(client.last['chosen'], client.last['teacher_index'])
                self.assertTrue(torch.equal(client.last['values'], torch.zeros(2, dtype=torch.float64)))
                seen.add(client.last['teacher_index'])
            finally: client.close()
        self.assertEqual(seen, {0, 1})


if __name__ == '__main__': unittest.main()
