import copy
import importlib.util
import unittest

from desktop.jev_sidecar.engine import Adviser, candidates, reaction_candidates, request
from desktop.jev_sidecar.selftest import model_cases
from desktop.jev_sidecar.unified_policy import UnifiedPolicyClient, serving_request, identity


def payload(board):
    values = reaction_candidates(board) if board.reaction else candidates(board)
    result = request(board, [] if board.reaction else values)
    if board.reaction:
        result['questions']['action']['criteria'] = {f'a{i}': v['action'] for i, v in enumerate(values)}
    return result


class UnifiedDesktopTests(unittest.TestCase):
    def test_same_single_client_receives_every_action_path_and_public_input_only(self):
        class Model:
            def __init__(self): self.requests = []
            def system_one(self, **kw):
                self.requests.append(kw)
                keys = list(kw['questions']['action']['criteria'])
                return {'answers': {'action': {'probabilities': {k: 1 / len(keys) for k in keys}}}}
        model = Model()
        client = UnifiedPolicyClient(model)
        for name, board in model_cases().items():
            p = payload(board)
            p['_meta'] = {'private_outcomes': 'never submitted'}
            p['questions']['action'].update(label='a0', target={'a0': 1})
            before = copy.deepcopy(p)
            answer, trace = client.system_one_with_trace(**p)
            self.assertEqual(p, before)
            self.assertEqual(trace['route'], 'unified')
            self.assertEqual(set(model.requests[-1]), {'state', 'questions'})
            self.assertEqual(set(model.requests[-1]['questions']['action']), {'type', 'instructions', 'criteria'})
            self.assertIn('winning sooner', model.requests[-1]['questions']['action']['instructions'])
            self.assertEqual(set(answer['answers']['action']['probabilities']), set(p['questions']['action']['criteria']))
        self.assertEqual(len(model.requests), 7)

    @unittest.skipUnless(importlib.util.find_spec('riichienv') and importlib.util.find_spec('jevmahjong.policy.unified'),
        'Native comparison runs in the training checkout')
    def test_portable_formatter_matches_actual_training_formatter_for_all_seven_paths(self):
        from jevmahjong.policy.unified import serving_request as native
        for name, board in model_cases().items():
            with self.subTest(name=name):
                p = payload(board)
                self.assertEqual(serving_request(p), native(p))

    def test_unknown_river_does_not_become_a_known_safe_call(self):
        board = model_cases()['chi']
        board.riichi = [1]
        board.rivers[1] = None
        p = serving_request(payload(board))
        for c in p['questions']['action']['criteria'].values():
            self.assertIsNone(c['call_features']['safe_discard_at_best_shanten'])

    def test_release_identity_is_the_final2088_unified_candidate(self):
        d = identity()
        self.assertEqual(d['model_id'], 'jev-play-v2')
        self.assertEqual(d['source_model_id'], 'jev-unified-attackrisk-v8')
        self.assertEqual(d['source_step'], 2088)
        self.assertTrue(d['parameter_values_unchanged'])
        self.assertFalse(d['independent_improvement_verified'])


if __name__ == '__main__':
    unittest.main()
