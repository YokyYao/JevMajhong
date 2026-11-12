"""Real-game regressions: public state must reach inference and invalidate advice."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from desktop.jev_sidecar.call_policy import CallPolicyClient
from desktop.jev_sidecar.dataset import DecisionRecorder
from desktop.jev_sidecar.engine import Adviser, Board, decision_identity, reaction_candidates, candidates
from desktop.jev_sidecar.tiles import parse


class Capture:
    def __init__(self): self.payloads = []
    def system_one(self, **payload):
        self.payloads.append(copy.deepcopy(payload))
        keys = list(payload['questions']['action']['criteria'])
        return {'answers': {'action': {'probabilities': {k: 1 / len(keys) for k in keys}}}}


def call_board():
    return Board(parse('123456m345p67s11z'), rivers=[[], ['9m'], [], []],
        riichi=[1], confirmed_metadata=['riichi'],
        reaction={'tile': '5s', 'seat': 3, 'offered': ['chi']})


class PlayInputTests(unittest.TestCase):
    def test_positive_riichi_survives_router_without_confirming_other_metadata(self):
        ordinary, calling = Capture(), Capture()
        adviser = Adviser(); adviser.model = CallPolicyClient(ordinary, calling)
        board = call_board(); original = copy.deepcopy(board)
        result = adviser.recommend(board, use_model=True)
        state = calling.payloads[0]['state']
        self.assertEqual(state['riichi'], [1])
        self.assertIsNone(state['scores']); self.assertIsNone(state['dealer'])
        self.assertTrue(all(c['call_features']['riichi_threat_count'] == 1
                            for c in calling.payloads[0]['questions']['action']['criteria'].values()))
        trace = result['model_input']
        self.assertEqual(trace['route'], 'calling')
        self.assertEqual(trace['model_request'], calling.payloads[0])
        self.assertEqual(trace['source_request']['state']['riichi'], [1])
        self.assertEqual(board, original); self.assertFalse(ordinary.payloads)
        # The trace must remain the submitted input even if a client mutates it later.
        calling.payloads[0]['state']['riichi'].clear()
        self.assertEqual(trace['model_request']['state']['riichi'], [1])

    def test_public_state_changes_invalidate_advice(self):
        board = call_board(); key = decision_identity(board, 1)
        self.assertEqual(key, decision_identity(copy.deepcopy(board), 1))
        for field, value in [('rivers', [[], ['9m', '8p'], [], []]),
                             ('indicators', ['1z']), ('scores', [30000, 20000, 25000, 25000]),
                             ('dealer', 2), ('round_wind', 1), ('riichi', [1, 2]),
                             ('confirmed_metadata', []), ('metadata_confirmed', True),
                             ('melds', [[], [], [{'type': 1, 'tiles': '5z 5z 5z'}], []])]:
            changed = copy.deepcopy(board); setattr(changed, field, value)
            with self.subTest(field=field):
                self.assertNotEqual(key, decision_identity(changed, 1))
        self.assertNotEqual(key, decision_identity(board, 0))

    def test_own_riichi_locks_calls_but_allows_ron(self):
        board = call_board(); board.riichi = [0, 1]
        self.assertEqual([v['action']['type'] for v in reaction_candidates(board)], ['none'])
        result = Adviser().recommend(board, use_model=True)
        self.assertEqual(result['top3'][0]['action']['type'], 'none')
        board.reaction['offered'].append('hora')
        result = Adviser().recommend(board, use_model=True)
        self.assertEqual(result['top3'][0]['action']['type'], 'hora')
        # A drawn red five must not permit replacing the original non-red five.
        locked = Board(parse('123m123455p123s10p'), riichi=[0], drawn='0p')
        self.assertEqual({v['tile'] for v in candidates(locked)}, {'0p'})

    def test_recorder_retains_changed_inference_on_same_image(self):
        adviser = Adviser(); adviser.model = CallPolicyClient(Capture(), Capture())
        board = call_board()
        first = adviser.recommend(board, use_model=True)
        board.riichi = [1, 2]
        second = adviser.recommend(board, use_model=True)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = DecisionRecorder(tmp)
            recorder.prediction('same-image', first, .1)
            recorder.prediction('same-image', first, .1)
            recorder.prediction('same-image', second, .1)
            rows = [json.loads(s) for s in (Path(tmp) / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['result']['model_input']['model_request']['state']['riichi'], [1, 2])
        self.assertNotIn('label', rows[1]['result']['model_input']['model_request']['questions']['action'])


if __name__ == '__main__': unittest.main()
