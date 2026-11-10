import copy
from types import SimpleNamespace
import unittest

from encoder_call_policy import EncoderCallRouter
from encoder_training_guard import ExactRecallGuard,exact_metrics


class CallEncoderTests(unittest.TestCase):
    def test_wrong_chi_combination_is_not_an_exact_recall_success(self):
        record={'_meta':{'id':'case','view':'native'},'questions':{'action':{'label':'a0','criteria':{
            'a0':{'type':'chi'},'a1':{'type':'chi'},'a2':{'type':'none'}}}}}
        row={'id':'case','keys':['a0','a1','a2'],'label':0,'p':[.1,.8,.1]}
        result=exact_metrics([record],[row])
        self.assertEqual(result['native']['support']['chi'],1)
        self.assertEqual(result['native']['exact_tp'].get('chi',0),0)
        with self.assertRaisesRegex(ValueError,'coverage differs'):exact_metrics([record],[])

    def test_loss_improvement_cannot_select_collapsed_call_recall(self):
        from jevany.train import early_stop_update
        guard=ExactRecallGuard(['native'])
        baseline={'native':{'exact_tp':{'chi':4,'pon':10}}}
        guard.assess(baseline)
        self.assertFalse(guard.assess({'native':{'exact_tp':{'chi':0,'pon':10}}})['admissible'])
        self.assertEqual(guard.update(early_stop_update,.1,.5,2,maximize=False),(.5,3,False))
        self.assertTrue(guard.assess({'native':{'exact_tp':{'chi':2,'pon':5}}})['admissible'])
        self.assertEqual(guard.update(early_stop_update,.4,.5,2,maximize=False),(.4,0,True))

    def test_call_encoder_routes_separately_and_retains_confirmed_counter_fields(self):
        class Client:
            def __init__(self,name):self.name=name;self.seen=[]
            def system_one(self,**r):self.seen.append(r);return self.name
        base,call=Client('incumbent'),Client('call');router=EncoderCallRouter(base,call)
        request={'state':{'honba':3,'sticks':2,'scores':[25000]*4,'dealer':0,'round_wind':0,
            'screen_observation':{'metadata_confirmed':False,'confirmed_metadata':['honba','sticks']}},
            'questions':{'action':{'criteria':{'a0':{'type':'pon'},'a1':{'type':'none'}}}}}
        original=copy.deepcopy(request)
        self.assertEqual(router.system_one(**request),'call')
        self.assertEqual(request,original)
        self.assertEqual((call.seen[-1]['state']['honba'],call.seen[-1]['state']['sticks']),(3,2))
        self.assertIsNone(call.seen[-1]['state']['scores'])
        request['questions']['action']['criteria']={'a0':{'type':'dahai'}}
        self.assertEqual(router.system_one(**request),'incumbent')
        self.assertEqual(base.seen[-1],request)


if __name__=='__main__':unittest.main()
