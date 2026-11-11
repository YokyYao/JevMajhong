"""Verify that the full-policy arm really has no ordinary/rare oldB fallback."""
import copy
import unittest
from desktop.tests.test_stage2 import reaction
from encoder_call_policy import EncoderCallRouter
from stage2_ablation import FullStage2Client


class Stub:
    def __init__(self, name): self.name = name; self.requests = []
    def system_one(self, **request):
        self.requests.append(request)
        return {'source': self.name}


class Stage2AblationTest(unittest.TestCase):
    def test_ordinary_rare_and_call_all_reach_stage2_in_full_arm(self):
        candidate = Stub('stage2'); full = FullStage2Client(candidate)
        for kind in ('dahai', 'reach', 'ankan', 'kakan', 'daiminkan'):
            payload = reaction()
            payload['questions']['action']['criteria'] = {'a0': {'type': kind}}
            before = copy.deepcopy(payload)
            self.assertEqual(full.system_one(**payload)['source'], 'stage2')
            self.assertEqual(payload, before)
        self.assertEqual(full.system_one(**reaction())['source'], 'stage2')
        self.assertEqual(len(candidate.requests), 6)

    def test_hybrid_and_full_share_calls_but_only_hybrid_routes_ordinary_to_oldB(self):
        old, new = Stub('oldB'), Stub('stage2')
        full = FullStage2Client(new); hybrid = EncoderCallRouter(old, full)
        self.assertEqual(hybrid.system_one(**reaction())['source'], 'stage2')
        payload = reaction(); payload['questions']['action']['criteria'] = {'a0': {'type': 'dahai', 'pai': '1m'}}
        self.assertEqual(hybrid.system_one(**payload)['source'], 'oldB')
        self.assertEqual(full.system_one(**payload)['source'], 'stage2')
        self.assertEqual(old.requests, [payload])


if __name__ == '__main__': unittest.main()
