import copy
import unittest
import torch
from riichienv import MeldType
from jevmahjong.policy.kan_context import VERSION, NAMES, feature_vectors, probabilities, context_scores, KanContextPolicyClient


def request(kind):
    state={'seat':0,'dealer':0,'round_wind':0,'honba':2,'sticks':1,'scores':[25000]*4,
        'riichi':[1],'discards_count':20,'dora_indicators':'4z','rivers':['','','',''],'melds':[[],[],[],[]]}
    if kind=='ankan':
        state['hand']='1m 1m 1m 1m 2m 3m 4m 5p 6p 7p 1s 2s 3s E'
        action={'type':kind,'pai':'1m','consumed':['1m']*4};other={'type':'dahai','pai':'E'}
    elif kind=='daiminkan':
        state['hand']='E E E 2m 3m 4m 4p 5p 6p 6s 7s 8s P'
        action={'type':kind,'pai':'E','consumed':['E']*3};other={'type':'none'}
    else:
        state['hand']='E 2m 3m 4m 4p 5p 6p 6s 7s 8s P'
        state['melds'][0]=[{'type':int(MeldType.Pon),'tiles':'1z 1z 1z'}]
        action={'type':'kakan','pai':'E','consumed':['E']*3};other={'type':'dahai','pai':'P'}
    return {'state':state,'questions':{'action':{'criteria':{'a0':other,'a1':action}}}}


class KanContextTests(unittest.TestCase):
    def test_nonlinear_inference_matches_training_and_zero_start_preserves_play(self):
        from jevmahjong.training.context_network import make_network
        torch.manual_seed(17);net=make_network(3);req=request('daiminkan')
        context={'version':VERSION,'names':list(NAMES),'model_kind':'mlp','hidden':3,'state_dict':net.state_dict()}
        self.assertEqual(probabilities(req,[.8,.2],context),[.8,.2])
        with torch.no_grad():net[2].weight.copy_(torch.tensor([[.1,.2,-.4]],dtype=torch.float64))
        actual=context_scores(req,context);expected=net(torch.tensor(feature_vectors(req),dtype=torch.float64)).flatten()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=0))
        context['hidden']=4
        with self.assertRaises(ValueError):context_scores(req,context)

    def test_all_three_kans_handle_owned_tiles_closed_status_and_known_dora(self):
        for kind in ('ankan','daiminkan','kakan'):
            req=request(kind);old=copy.deepcopy(req);values=feature_vectors(req)[1]
            self.assertEqual(len(values),len(NAMES));self.assertEqual(req,old)
            self.assertEqual(values[NAMES.index('pre_rinshan')],1)
            self.assertEqual(values[NAMES.index('closed_after')],float(kind=='ankan'))
            self.assertEqual(values[NAMES.index('loses_closed')],float(kind=='daiminkan'))
            self.assertEqual(values[NAMES.index('meld_count_after')],.25)
            if kind!='ankan':self.assertEqual(values[NAMES.index('meld_dora')],.5)

    def test_unknown_future_fields_cannot_change_public_features(self):
        req=request('daiminkan');expected=feature_vectors(req)
        req['state'].update(wall=['1m']*70,other_hands=[['1m']*13]*3,future_points=48000)
        self.assertEqual(feature_vectors(req),expected)

    def test_zero_component_preserves_reference_and_nonkan_route_is_unchanged(self):
        context={'version':VERSION,'names':list(NAMES),'coefficients':torch.zeros(len(NAMES))}
        req=request('ankan');self.assertEqual(probabilities(req,[.8,.2],context),[.8,.2])
        answer={'answers':{'action':{'probabilities':{'a0':.8,'a1':.2}}}}
        class Frozen:
            def system_one(self,**request):return answer
        client=KanContextPolicyClient(Frozen(),context)
        nonkan=copy.deepcopy(req);nonkan['questions']['action']['criteria']['a1']={'type':'none'}
        self.assertIs(client.system_one(**nonkan),answer)
        context['coefficients'][NAMES.index('ankan')]=1
        self.assertGreater(client.system_one(**req)['answers']['action']['probabilities']['a1'],.5)
        self.assertEqual(answer['answers']['action']['probabilities']['a1'],.2)

    def test_schema_and_invalid_reference_are_rejected(self):
        req=request('kakan');context={'version':VERSION,'names':list(NAMES),'coefficients':torch.zeros(len(NAMES))}
        with self.assertRaises(ValueError):probabilities(req,[0,0],context)
        context['version']='unknown'
        with self.assertRaises(ValueError):probabilities(req,[.5,.5],context)

    def test_action_panel_uses_context_decision_instead_of_unchanged_pointer(self):
        from jevmahjong.evaluation.action_suite import evaluate
        context={'version':VERSION,'names':list(NAMES),'coefficients':torch.zeros(len(NAMES))}
        context['coefficients'][NAMES.index('ankan')]=1
        row={'public_request':request('ankan'),'scope':'ordinary','category':'ankan',
            'episode_key':(0,0,0),'action':1,'teacher_action':0,'eligible':[0,1],
            'reference':torch.tensor([.8,.2]),'training_advantage':.1,'fast_advantage':.1}
        panel=evaluate({},[row],per_direction=1,context=context)
        self.assertEqual(panel['actions']['ankan']['positive']['pair_order_correct'],1)
        self.assertEqual(panel['actions']['ankan']['positive']['operation_selected_from_full_legal_set'],1)
