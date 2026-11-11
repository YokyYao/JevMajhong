"""Portable call-policy dispatch/feature mechanisms; no model download or UI required."""
import copy
import unittest

from desktop.jev_sidecar.engine import Board, request, reaction_candidates
from desktop.jev_sidecar.tiles import parse
from desktop.jev_sidecar.call_policy import CallPolicyClient, is_call_request
from desktop.jev_sidecar.call_features import enrich_request


class Stub:
    def __init__(self): self.requests = []
    def system_one(self, **payload):
        self.requests.append(payload)
        return {'answers': {'action': {'probabilities': {'a0': .5, 'a1': .5}}}}


def reaction():
    board = Board(parse('123456m345p67s11z'), rivers=[[], [], [], []],
        metadata_confirmed=True, reaction={'tile': '5s', 'seat': 3, 'offered': ['chi']})
    payload = request(board, [])
    payload['questions']['action']['criteria'] = {f'a{i}': v['action'] for i, v in enumerate(reaction_candidates(board))}
    return payload


class CallPolicyTests(unittest.TestCase):
    def test_call_routes_to_new_adapter_and_features_do_not_mutate_input(self):
        old, new = Stub(), Stub(); router = CallPolicyClient(old, new)
        payload = reaction(); before = copy.deepcopy(payload)
        router.system_one(**payload)
        self.assertEqual(payload, before)
        self.assertFalse(old.requests); self.assertEqual(len(new.requests), 1)
        self.assertTrue(all('call_features' in c for c in new.requests[0]['questions']['action']['criteria'].values()))

    def test_ordinary_and_rare_requests_keep_original_payload(self):
        old, new = Stub(), Stub(); router = CallPolicyClient(old, new)
        for kind in ('dahai', 'reach', 'daiminkan', 'hora'):
            payload = reaction()
            payload['questions']['action']['criteria']['a0']['type'] = kind
            self.assertFalse(is_call_request(payload))
            router.system_one(**payload)
            self.assertEqual(old.requests[-1], payload)
        self.assertFalse(new.requests)

    def test_unconfirmed_metadata_mask_preserves_confirmed_counters(self):
        old, new = Stub(), Stub(); router = CallPolicyClient(old, new)
        payload = reaction(); payload['state']['honba'] = 3; payload['state']['sticks'] = 2
        payload['state']['screen_observation'] = {'metadata_confirmed': False,
            'confirmed_metadata': ['honba', 'sticks']}
        router.system_one(**payload)
        state = new.requests[0]['state']
        self.assertIsNone(state['scores']); self.assertIsNone(state['riichi'])
        self.assertEqual((state['honba'], state['sticks']), (3, 2))

    def test_unknown_threat_river_is_not_treated_as_known_safe(self):
        payload = reaction(); payload['state']['riichi'] = [1]; payload['state']['rivers'][1] = None
        enriched = enrich_request(payload)
        for criterion in enriched['questions']['action']['criteria'].values():
            self.assertIsNone(criterion['call_features']['safe_discard_at_best_shanten'])


if __name__ == '__main__': unittest.main()
