import copy
import unittest
import torch
from jevmahjong.policy.kan_tile_context import VERSION, NAMES, vectors
from jevmahjong.policy.kan_progress import VERSION as LEGACY_VERSION
from jevmahjong.policy.kan_context import feature_names, feature_vectors, context_scores, probabilities, KanContextPolicyClient
from jevmahjong.training.context_network import make_network
from tests.core.test_kan_context import request


class KanTileContextTests(unittest.TestCase):
    def test_all_kan_consumption_keeps_the_legacy_prefix_and_request(self):
        for kind, tile, consumed in [('ankan', 0, 1.), ('daiminkan', 27, .75), ('kakan', 27, .25)]:
            req = request(kind); original = copy.deepcopy(req)
            actual = feature_vectors(req, VERSION); names = feature_names(VERSION)
            with self.subTest(kind=kind):
                self.assertEqual(len(names), 600);self.assertEqual(len(NAMES), 442)
                self.assertEqual([row[:158] for row in actual], feature_vectors(req, LEGACY_VERSION))
                self.assertEqual(actual[1][names.index(f'consumed_tiles_{tile}')], consumed)
                self.assertEqual(actual[1][names.index(f'remaining_owned_tiles_{tile}')], 0.)
                self.assertTrue(all(0 <= value <= 1 for row in actual for value in row[158:]))
                self.assertEqual(req, original)

    def test_kakan_consumes_added_tile_without_removing_the_existing_pon(self):
        req = request('kakan');names = feature_names(VERSION)
        row = feature_vectors(req, VERSION)[1]
        self.assertEqual(row[names.index('own_hand_tiles_27')], .25)
        self.assertEqual(row[names.index('consumed_tiles_27')], .25)
        self.assertEqual(row[names.index('meld_relative_0_27')], .75)
        self.assertEqual(row[names.index('incoming_discard_27')], 0.)

    def test_mixed_call_menu_preserves_the_offered_tile_and_owned_consumption(self):
        req = request('daiminkan');req['state']['hand'] = '3m 3m 3m 1m 2m 4p 5p 6p 7s 8s 9s E P'
        req['state']['rivers'][1] = '3m'
        req['questions']['action']['criteria'] = {
            'a0': {'type': 'none'},
            'a1': {'type': 'pon', 'pai': '3m', 'consumed': ['3m'] * 2},
            'a2': {'type': 'chi', 'pai': '3m', 'consumed': ['1m', '2m']},
            'a3': {'type': 'daiminkan', 'pai': '3m', 'consumed': ['3m'] * 3}}
        rows = feature_vectors(req, VERSION);names = feature_names(VERSION)
        self.assertTrue(all(row[names.index('incoming_discard_2')] == 1. for row in rows))
        self.assertEqual(rows[0][names.index('remaining_owned_tiles_2')], .75)
        self.assertEqual(rows[1][names.index('remaining_owned_tiles_2')], .25)
        self.assertEqual(rows[2][names.index('remaining_owned_tiles_2')], .75)
        self.assertEqual(rows[2][names.index('remaining_owned_tiles_0')], 0.)
        self.assertEqual(rows[3][names.index('consumed_tiles_2')], .75)

    def test_reach_declares_without_prematurely_consuming_an_owned_tile(self):
        req = request('ankan');req['questions']['action']['criteria']['a2'] = {'type': 'reach'}
        rows = feature_vectors(req, VERSION);names = feature_names(VERSION)
        for tile in range(34):
            self.assertEqual(rows[2][names.index(f'own_hand_tiles_{tile}')], rows[2][names.index(f'remaining_owned_tiles_{tile}')])
            self.assertEqual(rows[2][names.index(f'consumed_tiles_{tile}')], 0.)

    def test_full_tile_identity_is_available_where_aggregate_features_collapse(self):
        req = request('ankan');other = copy.deepcopy(req)
        swap = lambda value: value.replace('m', 'x').replace('p', 'm').replace('x', 'p')
        other['state']['hand'] = swap(other['state']['hand'])
        for c in other['questions']['action']['criteria'].values():
            if 'pai' in c:c['pai'] = swap(c['pai'])
            if 'consumed' in c:c['consumed'] = [swap(t) for t in c['consumed']]
        self.assertEqual(feature_vectors(req, LEGACY_VERSION), feature_vectors(other, LEGACY_VERSION))
        self.assertNotEqual(feature_vectors(req, VERSION), feature_vectors(other, VERSION))

    def test_relative_channels_rotate_with_seats_and_ignore_hidden_information(self):
        req = request('kakan');req['state']['rivers'][2] = 'W'
        expected = feature_vectors(req, VERSION);hidden = copy.deepcopy(req)
        hidden['state'].update(wall=['hidden'], opponent_hands=['hidden'], future_points=99999)
        hidden['future_seed'] = 12345
        self.assertEqual(feature_vectors(hidden, VERSION), expected)
        rotated = copy.deepcopy(req);state = rotated['state'];original = req['state']
        for field in ('rivers', 'melds', 'scores'):
            state[field] = [original[field][(i - 1) % 4] for i in range(4)]
        state['seat'] = 1;state['dealer'] = 1;state['riichi'] = [2]
        self.assertEqual(feature_vectors(rotated, VERSION), expected)

    def test_inference_matches_training_and_other_routes_keep_the_frozen_policy(self):
        req = request('daiminkan');names = feature_names(VERSION);net = make_network(64, len(names))
        self.assertEqual(sum(p.numel() for p in net.parameters()), 38528)
        context = dict(version=VERSION, names=list(names), model_kind='mlp', hidden=64,
                       state_dict=net.state_dict(), reference_prior_strength=.05)
        self.assertEqual(probabilities(req, [.8, .2], context), [.8, .2])
        with torch.no_grad():net[2].weight.normal_(std=.01)
        expected = net(torch.tensor(feature_vectors(req, VERSION), dtype=torch.float64)).flatten()
        self.assertTrue(torch.allclose(context_scores(req, context), expected, atol=1e-12, rtol=0))
        answer = {'answers': {'action': {'probabilities': {'a0': .8, 'a1': .2}}}}
        class Frozen:
            def system_one(self, **request):return answer
        nonkan = copy.deepcopy(req);nonkan['questions']['action']['criteria']['a1'] = {'type': 'pon', 'pai': 'E', 'consumed': ['E'] * 2}
        self.assertIs(KanContextPolicyClient(Frozen(), context).system_one(**nonkan), answer)
        with self.assertRaises(ValueError):feature_vectors(nonkan, VERSION)
        with self.assertRaises(ValueError):context_scores(req, dict(context, names=names[:-1]))

    def test_invalid_owned_consumption_or_inconsistent_incoming_tile_is_rejected(self):
        req = request('ankan');req['questions']['action']['criteria']['a1']['consumed'] = ['1m'] * 3
        with self.assertRaisesRegex(ValueError, 'Invalid owned consumption'):vectors(req)
        req = request('ankan');req['questions']['action']['criteria']['a0']['pai'] = '9s'
        with self.assertRaisesRegex(ValueError, 'not owned'):vectors(req)
        req = request('daiminkan');req['questions']['action']['criteria']['a2'] = {'type': 'pon', 'pai': 'P', 'consumed': ['P'] * 2}
        with self.assertRaisesRegex(ValueError, 'one observed discard'):vectors(req)


if __name__ == '__main__':
    unittest.main()
