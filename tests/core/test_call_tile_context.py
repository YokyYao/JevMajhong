"""Public tile identity, seat rotation, schema and inference regressions."""
import copy
import unittest
import torch
from jevmahjong.policy.call_tile_context import VERSION,NAMES
from jevmahjong.policy.kan_progress import VERSION as LEGACY_VERSION
from jevmahjong.policy.kan_context import feature_names,feature_vectors,context_scores,probabilities
from jevmahjong.training.context_network import make_network
from tests.core.test_win_context import pair
from tests.core.test_kan_context import request


class CallTileContextTests(unittest.TestCase):
    def test_public_channels_keep_legacy_prefix_and_track_consumed_owned_tiles(self):
        req=pair(100)['public_request'];base=feature_vectors(req,LEGACY_VERSION)
        actual=feature_vectors(req,VERSION);names=feature_names(VERSION)
        self.assertEqual(len(names),600);self.assertEqual(len(NAMES),442)
        self.assertEqual([v[:158] for v in actual],base)
        def value(action,name):return actual[action][names.index(name)]
        self.assertEqual(value(0,'own_hand_tiles_27'),.75)
        self.assertEqual(value(0,'remaining_owned_tiles_27'),.75)
        self.assertEqual(value(1,'remaining_owned_tiles_27'),.25)
        self.assertEqual(value(1,'consumed_tiles_27'),.5)
        self.assertEqual(value(0,'consumed_tiles_27'),0)
        self.assertEqual(value(0,'called_tile_27'),1)
        self.assertEqual(value(1,'river_relative_1_27'),.25)
        self.assertTrue(all(0<=x<=1 for v in actual for x in v[158:]))

    def test_hidden_information_is_ignored_and_observed_rivers_are_distinguished(self):
        req=pair(100)['public_request'];expected=feature_vectors(req,VERSION)
        hidden=copy.deepcopy(req)
        hidden['state'].update(opponent_hands=['arbitrary hidden information'],wall='hidden wall',winner=2)
        hidden['future_seed']=999999
        self.assertEqual(feature_vectors(hidden,VERSION),expected)
        public=copy.deepcopy(req);public['state']['rivers'][2]='W'
        changed=feature_vectors(public,VERSION)
        self.assertNotEqual(changed,expected)
        self.assertEqual(changed[0][feature_names(VERSION).index('river_relative_2_29')],.25)

    def test_relative_public_channels_rotate_with_seats(self):
        req=pair(100)['public_request'];req['state']['rivers'][2]='W'
        rotated=copy.deepcopy(req);state=rotated['state'];original=req['state']
        for name in ('rivers','melds','scores'):
            state[name]=[original[name][(i-1)%4] for i in range(4)]
        state['seat']=(original['seat']+1)%4;state['dealer']=(original['dealer']+1)%4
        state['riichi']=[(p+1)%4 for p in original['riichi']]
        self.assertEqual(feature_vectors(rotated,VERSION),feature_vectors(req,VERSION))

    def test_real_inference_preserves_zero_prior_and_rejects_wrong_routes_or_schema(self):
        req=pair(100)['public_request'];names=feature_names(VERSION);net=make_network(64,len(names))
        self.assertEqual(sum(p.numel() for p in net.parameters()),38528)
        context=dict(version=VERSION,names=list(names),model_kind='mlp',hidden=64,
                     state_dict=net.state_dict(),reference_prior_strength=.05)
        self.assertEqual(probabilities(req,[.9,.1],context),[.9,.1])
        with torch.no_grad():net[2].weight.normal_(std=.01)
        expected=net(torch.tensor(feature_vectors(req,VERSION),dtype=torch.float64)).flatten()
        self.assertTrue(torch.allclose(context_scores(req,context),expected,atol=1e-12,rtol=0))
        with self.assertRaises(ValueError):feature_vectors(request('ankan'),VERSION)
        with self.assertRaises(ValueError):context_scores(req,dict(context,names=names[:-1]))


if __name__=='__main__':unittest.main()
