import copy
import unittest

from call_option_features import enrich_request, option_features
from jevany.data import api_request


def request():
    return {'state':{'seat':0,'dealer':0,'round_wind':0,'honba':0,'sticks':0,'scores':[25000]*4,
        'hand':'5z 5z 1m 2m 3m 2p 3p 4p 6s 7s 9s 9m 9m','melds':[[],[],[],[]],
        'rivers':['','','','5z'],'riichi':[],'dora_indicators':'4p'},
        'questions':{'action':{'type':'choice','instructions':'choose','criteria':{
            'a0':{'type':'none'},'a1':{'type':'pon','pai':'P','consumed':['P','P']}},'label':'a1'}},
        '_meta':{'private_expert_info':'must not influence features'}}


class CallOptionFeatureTests(unittest.TestCase):
    def test_public_speed_yakuhai_and_closed_hand_cost(self):
        values=option_features(request());p=values['a1'];n=values['a0']
        self.assertEqual(p['best_shanten_after'],0)
        self.assertEqual(p['secured_dragon_han'],1)
        self.assertTrue(p['loses_closed_status']);self.assertFalse(p['closed_after'])
        self.assertTrue(n['closed_after']);self.assertEqual(n['shanten_gain'],0)
        self.assertIsNone(p['safe_discard_at_best_shanten'])

    def test_labels_and_private_metadata_do_not_change_served_features(self):
        r=request();other=copy.deepcopy(r);other['questions']['action']['label']='a0';other['_meta']={'answer':'different'}
        self.assertEqual(api_request(enrich_request(r)),api_request(enrich_request(other)))
        original=copy.deepcopy(r);enrich_request(r);self.assertEqual(r,original)

    def test_red_and_dora_locked_in_pon_are_counted_separately(self):
        r=request();r['state']['hand']=r['state']['hand'].replace('5z 5z','0p 5p');r['state']['rivers'][3]='5p'
        r['questions']['action']['criteria']['a1']={'type':'pon','pai':'5p','consumed':['5pr','5p']}
        values=option_features(r)['a1']
        self.assertEqual(values['red_locked_in_melds'],1)
        self.assertEqual(values['dora_locked_in_melds'],3)

    def test_unknown_wind_and_threat_metadata_stays_unknown(self):
        r=request();r['state'].update(dealer=None,round_wind=None,riichi=None)
        values=option_features(r)['a1']
        self.assertIsNone(values['secured_wind_han']);self.assertIsNone(values['riichi_threat_count'])
        self.assertEqual(values['secured_dragon_han'],1)

    def test_safe_discard_requires_intersection_across_all_riichi_players(self):
        r=request();r['state']['riichi']=[1,2];r['state']['rivers'][1]='9s';r['state']['rivers'][2]='6s'
        self.assertFalse(option_features(r)['a1']['safe_discard_at_best_shanten'])
        r['state']['rivers'][2]='9s'
        self.assertTrue(option_features(r)['a1']['safe_discard_at_best_shanten'])

    def test_enrichment_is_idempotent_and_malformed_consumed_is_rejected(self):
        r=enrich_request(request());self.assertEqual(enrich_request(r),r)
        bad=request();bad['questions']['action']['criteria']['a1']['consumed']=['P','P','P']
        with self.assertRaises(ValueError):enrich_request(bad)


if __name__=='__main__':unittest.main()
