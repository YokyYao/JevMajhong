import base64
import json
from types import SimpleNamespace
import unittest

from clean_policy_data import pass_outcomes, policy_request, POLICY_INSTRUCTION
from replay_call_repair import post_call_forbidden, UnsafeReplayCall


def observation(kind='Chi', tiles=(0, 4, 8), called=0, drawn=None, last=0):
    data = {'drawn_tile': drawn, 'last_discard': last,
            'melds': [[{'meld_type': kind, 'tiles': list(tiles), 'called_tile': called}], [], [], []]}
    return SimpleNamespace(hand=list(range(11)), player_id=0,
        serialize_to_base64=lambda: base64.b64encode(json.dumps(data).encode()).decode())


class CleanPolicyDataTests(unittest.TestCase):
    def test_screenshot_unknown_keeps_automatic_counters_and_preserves_source(self):
        source = {'state': {'seat':0,'melds':[[],[],[],[]],'honba':3,'sticks':2,'scores':[25000]*4,
            'dealer':1,'round_wind':0,'hand_number':2,'riichi':[1],
            'tsumogiri_flags':[], 'riichi_sutehais':[], 'last_tedashis':[], 'last_discard':10},
            'questions':{'action':{'instructions':'old','criteria':{'a0':{'type':'dahai','pai':'1m'},
                'a1':{'type':'dahai','pai':'2m'}},'label':'a1'}}}
        result=policy_request(source,'screen_unknown')
        self.assertEqual((result['state']['honba'],result['state']['sticks']),(3,2))
        self.assertIsNone(result['state']['scores'])
        self.assertEqual(result['state']['screen_observation']['confirmed_metadata'],['honba','sticks'])
        self.assertEqual(result['questions']['action']['instructions'],POLICY_INSTRUCTION)
        self.assertEqual(result['questions']['action']['label'],'a1')
        self.assertEqual(source['state']['scores'],[25000]*4)
        self.assertEqual(source['questions']['action']['instructions'],'old')

    def test_chi_both_ends_and_middle_respect_suit_boundaries(self):
        self.assertEqual(post_call_forbidden(observation()), {0, 3})
        self.assertEqual(post_call_forbidden(observation(called=8, last=8)), {2})
        self.assertEqual(post_call_forbidden(observation(tiles=(4, 8, 12), called=12, last=12)), {0, 3})
        self.assertEqual(post_call_forbidden(observation(tiles=(24, 28, 32), called=24, last=24)), {6})
        self.assertEqual(post_call_forbidden(observation(called=4, last=4)), {1})

    def test_pon_red_and_ordinary_five_share_forbidden_tile_type(self):
        self.assertEqual(post_call_forbidden(observation('Pon', (16, 17, 18), 17, last=17)), {4})

    def test_ordinary_draw_does_not_inherit_previous_meld_restriction(self):
        self.assertEqual(post_call_forbidden(observation(drawn=50)), set())

    def test_unresolved_context_is_excluded(self):
        with self.assertRaises(UnsafeReplayCall):
            post_call_forbidden(observation(last=40))

    def test_competing_claim_is_censored_while_unclaimed_discard_identifies_pass(self):
        start = {'type':'start_kyoku','bakaze':'E','oya':1,'honba':2}
        outcomes = pass_outcomes([start, {'type':'dahai'}, {'type':'reach_accepted'},
            {'type':'chi'}, {'type':'dahai'}, {'type':'tsumo'},
            {'type':'dahai'}, {'type':'hora'}])
        self.assertEqual(outcomes, {(0,1,2,1):'chi', (0,1,2,2):'tsumo', (0,1,2,3):'hora'})


if __name__ == '__main__':
    unittest.main()
