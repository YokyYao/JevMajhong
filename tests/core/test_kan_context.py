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
    def test_pure_call_progress_is_versioned_and_keeps_kan_and_discard_routes_frozen(self):
        from jevmahjong.policy.kan_context import feature_names, CallContextPolicyClient
        from jevmahjong.policy.kan_progress import VERSION as v2
        req=request('daiminkan');req['state']['rivers'][1]='E'
        req['questions']['action']['criteria']['a1']={'type':'pon','pai':'E','consumed':['E','E']}
        with self.assertRaises(ValueError):feature_vectors(req)
        self.assertEqual(len(feature_vectors(req,v2)[0]),158)
        names=feature_names(v2)
        context={'version':v2,'names':list(names),'coefficients':torch.zeros(len(names)),
            'reference_prior_strength':.05}
        answer={'answers':{'action':{'probabilities':{'a0':.8,'a1':.2}}}}
        class Frozen:
            def system_one(self,**request):return answer
        client=CallContextPolicyClient(Frozen(),context)
        self.assertEqual(client.system_one(**req),answer)
        context['coefficients'][names.index('pon')]=1.
        self.assertGreater(client.system_one(**req)['answers']['action']['probabilities']['a1'],.5)
        self.assertIs(client.system_one(**request('ankan')),answer)
        self.assertIs(KanContextPolicyClient(Frozen(),context).system_one(**req),answer)
        discard=copy.deepcopy(req)
        discard['questions']['action']['criteria']['a1']={'type':'dahai','pai':'P'}
        self.assertIs(client.system_one(**discard),answer)

    def test_combined_selection_weights_states_and_exposes_source_tradeoff(self):
        from jevmahjong.training.context_policy import selection_profile
        def source(n,regret):
            return {'source':str(n),'states':n,'policy_kl':.5,'label_regret':regret,
                'mean_label_utility_gain_vs_play':-regret,'changed_argmax_states':1}
        first=[source(100,.3),source(1,0)]
        later=[source(100,.1),source(1,.2)]
        self.assertLess(selection_profile(first,'latest_source')['label_regret'],
            selection_profile(later,'latest_source')['label_regret'])
        self.assertGreater(selection_profile(first,'all_sources')['label_regret'],
            selection_profile(later,'all_sources')['label_regret'])
        merged=selection_profile(later,'all_sources')
        self.assertAlmostEqual(merged['label_regret'],10.2/101)
        self.assertEqual(merged['states'],101);self.assertEqual(merged['changed_argmax_states'],2)
        with self.assertRaises(ValueError):selection_profile(later,'test_set')

    def test_progress_distinguishes_equal_legacy_features_with_different_public_ukeire(self):
        from jevmahjong.policy.kan_context import feature_names
        from jevmahjong.policy.kan_progress import VERSION as v2
        req=request('ankan')
        req['questions']['action']['criteria'].update(
            a2={'type':'dahai','pai':'2m','ukeire':12},
            a3={'type':'dahai','pai':'3m','ukeire':28})
        old=feature_vectors(req);new=feature_vectors(req,v2);names=feature_names(v2)
        self.assertEqual(old[2],old[3])
        self.assertEqual(new[2][:len(NAMES)],old[2])
        self.assertEqual(new[3][:len(NAMES)],old[3])
        self.assertEqual(new[2][names.index('ukeire_lower')],12/136)
        self.assertEqual(new[3][names.index('ukeire_upper')],28/136)

    def test_progress_retains_concealed_dora_and_distinguishes_red_discard(self):
        from jevmahjong.policy.kan_context import feature_names
        from jevmahjong.policy.kan_progress import VERSION as v2
        req=request('ankan');req['state']['dora_indicators']='4p'
        req['state']['hand']='1m 1m 1m 1m 2m 3m 4m 0p 5p 6p 7p 1s 2s 3s'
        req['questions']['action']['criteria']['a0']={'type':'dahai','pai':'5pr'}
        req['questions']['action']['criteria']['a2']={'type':'dahai','pai':'5p'}
        old=feature_vectors(req);new=feature_vectors(req,v2);names=feature_names(v2)
        self.assertEqual(old[0],old[2])
        self.assertEqual(new[0][names.index('concealed_red_after')],0)
        self.assertEqual(new[2][names.index('concealed_red_after')],1/3)
        self.assertEqual(new[0][names.index('concealed_dora_after')],1/12)

    def test_visible_progress_bounds_deduplicate_calls_without_guessing_chi_tile(self):
        from jevmahjong.policy.kan_progress import visible_bounds
        s={'hand':'9p','dora_indicators':'9s','rivers':['E','','',''],
            'melds':[[{'type':int(MeldType.Pon),'tiles':'E E E'}],[],[],[]]}
        lo,hi=visible_bounds(s);self.assertEqual((lo[27],hi[27]),(3,3))
        s['rivers']=['1m 2m 3m','','','']
        s['melds'][0]=[{'type':int(MeldType.Chi),'tiles':'1m 2m 3m'}]
        lo,hi=visible_bounds(s)
        self.assertEqual(lo[:3],[1,1,1]);self.assertEqual(hi[:3],[2,2,2])
        s['rivers']=['2m','','','']
        lo,hi=visible_bounds(s)
        self.assertEqual(lo[:3],[1,1,1]);self.assertEqual(hi[:3],[1,1,1])

    def test_progress_counts_only_unseen_copies_of_an_actual_wait(self):
        from collections import Counter
        from jevmahjong.policy.call_helpers import tile_ids
        from jevmahjong.policy.kan_progress import visible_bounds, option_progress
        s={'hand':'1m 2m 3m 4m 5m 6m 7p 8p 9p 2s 3s 4s E',
            'dora_indicators':'N','rivers':['','','',''],'melds':[[],[],[],[]]}
        choices=[(None,tile_ids(s['hand']))]
        before=option_progress(choices,visible_bounds(s),Counter())
        self.assertEqual(before[:4],[3/136,3/136,1/34,1/34])
        s['rivers'][1]='E E E'
        self.assertEqual(option_progress(choices,visible_bounds(s),Counter())[:4],[0,0,0,0])

    def test_progress_schema_zero_start_native_inference_and_hidden_isolation(self):
        from jevmahjong.policy.kan_context import feature_names
        from jevmahjong.policy.kan_progress import VERSION as v2
        from jevmahjong.training.context_network import make_network
        req=request('ankan');names=feature_names(v2);net=make_network(3,len(names))
        context={'version':v2,'names':list(names),'model_kind':'mlp','hidden':3,
            'state_dict':net.state_dict(),'reference_prior_strength':.05}
        self.assertEqual(probabilities(req,[.8,.2],context),[.8,.2])
        expected=feature_vectors(req,v2)
        req['state'].update(wall=['1m']*70,other_hands=[['1m']*13]*3,future_points=48000)
        self.assertEqual(feature_vectors(req,v2),expected)
        with torch.no_grad():net[2].weight.fill_(.1)
        self.assertTrue(torch.allclose(context_scores(req,context),
            net(torch.tensor(expected,dtype=torch.float64)).flatten(),atol=1e-12,rtol=0))
        context['version']=VERSION
        with self.assertRaises(ValueError):context_scores(req,context)

    def test_residual_prior_preserves_confident_reference_until_value_exceeds_log_odds(self):
        context={'version':VERSION,'names':list(NAMES),'coefficients':torch.zeros(len(NAMES)),
            'reference_prior_strength':.05}
        req=request('ankan')
        self.assertEqual(probabilities(req,[.9,.1],context),[.9,.1])
        context['coefficients'][NAMES.index('ankan')]=.01
        self.assertLess(probabilities(req,[.9,.1],context)[1],.5)
        context['coefficients'][NAMES.index('ankan')]=.2
        self.assertGreater(probabilities(req,[.9,.1],context)[1],.5)
        context['reference_prior_strength']=0
        with self.assertRaises(ValueError):probabilities(req,[.9,.1],context)

    def test_residual_training_requires_every_eligible_option_and_uses_full_set_regret(self):
        from jevmahjong.training.context_policy import pack_options,policy_profile,policy_logits
        from jevmahjong.training.context_network import make_network
        req=request('ankan');req['questions']['action']['criteria']['a2']={'type':'dahai','pai':'2m'}
        row={'episode_key':(100,0,0,1),'request_sha256':'one','public_request':req,
            'eligible':[0,1,2],'teacher_action':0,'action':1,'reference':torch.tensor([.8,.15,.05]),
            'training_advantage':.1}
        other=dict(row,episode_key=(100,0,0,2),action=2,training_advantage=-.2)
        with self.assertRaisesRegex(ValueError,'Complete eligible'):pack_options([row])
        with self.assertRaisesRegex(ValueError,'Conflicting'):pack_options([row,row,other])
        group=pack_options([row,other])[0]
        self.assertEqual(group['utility'].tolist(),[0.,.1,-.2])
        net=make_network(3)
        self.assertTrue(torch.allclose(policy_logits(net,group,.05).softmax(-1),row['reference'].double(),atol=1e-7))
        sources=[{'path':'one','seed_start':100,'seeds':1}]
        before=policy_profile(net,[group],sources,.05)[0]
        self.assertAlmostEqual(before['label_regret'],.1);self.assertEqual(before['mean_label_utility_gain_vs_play'],0)
        net=torch.nn.Linear(len(NAMES),1,bias=False).double()
        with torch.no_grad():net.weight.zero_();net.weight[0,NAMES.index('ankan')]=.2
        after=policy_profile(net,[group],sources,.05)[0]
        self.assertEqual(after['label_regret'],0);self.assertAlmostEqual(after['mean_label_utility_gain_vs_play'],.1)
        self.assertEqual(after['changed_argmax_states'],1)

    def test_validation_averages_public_states_instead_of_counting_more_options_as_more_states(self):
        from jevmahjong.training.context_fit import prediction_profile
        rows=[{'episode_key':(100,0,0,1),'request_sha256':'same'},
            {'episode_key':(100,0,0,2),'request_sha256':'same'},
            {'episode_key':(101,0,0,1),'request_sha256':'other'}]
        sources=[{'path':'one','seed_start':100,'seeds':2}]
        p=prediction_profile(torch.zeros(3),rows,torch.tensor([1.,1.,0.]),sources,state_balanced=True)[0]
        self.assertEqual(p['states'],2);self.assertEqual(p['rows'],3);self.assertEqual(p['loss'],.5)

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
        row['request_sha256']='same-state';duplicate=dict(row,episode_key=(0,0,0,2))
        panel=evaluate({},[row,duplicate],per_direction=8,context=context)
        self.assertEqual(panel['actions']['ankan']['positive']['cases'],1)
        self.assertFalse(panel['actions']['ankan']['positive']['coverage_complete'])
