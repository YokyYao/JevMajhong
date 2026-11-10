import copy
from types import SimpleNamespace
import unittest
import torch

from guide_call_features import NAMES, VERSION
from guide_call_value import PointValueCallRouter, utility_preferences


class FakeClient:
    def __init__(self, head):
        self.runtime = SimpleNamespace(model=SimpleNamespace(head=head))
        self.decide = torch.tensor([.1,.2,.3,.4])
        self.options = torch.tensor([[.4,.3,.2,.1],[.1,.2,.3,.4]])
        self.fail = False

    def system_one(self, **request):
        if self.fail: raise RuntimeError('Simulated native failure')
        values = self.runtime.model.head(self.decide,self.options).softmax(-1).tolist()
        return {'answers':{'action':{'probabilities':dict(zip(request['questions']['action']['criteria'],values))}}}


class PointValueInterfaceTests(unittest.TestCase):
    def setUp(self):
        from jevany.model import PointerHead
        torch.manual_seed(15)
        self.base = PointerHead(4,2); self.teacher = PointerHead(4,2)
        self.client = FakeClient(self.base)
        self.payload = {'strategy_feature_version':VERSION,'strategy_feature_names':list(NAMES),
                        'scope':['chi','pon','none'],'coefficient':torch.zeros(2+len(NAMES))}
        self.request = {'state':{'seat':1,'dealer':0,'round_wind':0,'honba':3,'sticks':2,
            'scores':[25000]*4,'hand':'5z 5z 2m 3m 4m 4p 0p 6p 6s 7s 8s 2p 2p',
            'rivers':['','','',''],'riichi':[],'melds':[[],[],[],[]],'dora_indicators':'9m'},
            'questions':{'action':{'criteria':{'a0':{'type':'pon','pai':'P','consumed':['P','P']},'a1':{'type':'none'}}}}}

    def test_native_call_uses_point_value_and_restores_original_head(self):
        self.payload['coefficient'][2+NAMES.index('call')] = 2
        router = PointValueCallRouter(self.client,self.teacher,self.payload)
        original = copy.deepcopy(self.base.state_dict())
        result = router.system_one(**self.request)
        self.assertGreater(result['answers']['action']['probabilities']['a0'],result['answers']['action']['probabilities']['a1'])
        self.assertEqual(router.last['chosen'],0); self.assertEqual(router.checks,1)
        self.assertIs(self.client.runtime.model.head,self.base)
        self.assertTrue(all(torch.equal(v,self.base.state_dict()[k]) for k,v in original.items()))
        router.close()

    def test_zero_value_tie_prefers_pass_and_negative_value_declines(self):
        for v in ([0.,0.],[-2.,0.]):
            p, chosen = utility_preferences(torch.tensor(v),1)
            self.assertEqual(chosen,1); self.assertGreater(p[1],p[0])

    def test_nondiscard_scope_and_native_failure_do_not_change_base(self):
        router = PointValueCallRouter(self.client,self.teacher,self.payload)
        request = copy.deepcopy(self.request)
        request['questions']['action']['criteria'] = {'a0':{'type':'dahai','pai':'1m'},'a1':{'type':'dahai','pai':'2m'}}
        expected = self.client.system_one(**request)
        self.assertEqual(router.system_one(**request),expected); self.assertEqual(router.checks,0)
        self.client.fail = True
        with self.assertRaisesRegex(RuntimeError,'native failure'): router.system_one(**self.request)
        self.assertIs(self.client.runtime.model.head,self.base); self.assertFalse(router.active)
        router.close()

    def test_unknown_screenshot_and_wrong_schema_are_rejected(self):
        router = PointValueCallRouter(self.client,self.teacher,self.payload)
        self.request['state']['screen_observation'] = {'metadata_confirmed':False}
        with self.assertRaisesRegex(ValueError,'not been validated'): router.system_one(**self.request)
        router.close(); self.payload['strategy_feature_version'] = 'other'
        with self.assertRaisesRegex(ValueError,'schema mismatch'): PointValueCallRouter(self.client,self.teacher,self.payload)


if __name__ == '__main__': unittest.main()
