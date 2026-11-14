import copy
import json
from pathlib import Path
import tempfile
import unittest
from jevmahjong.evaluation.unified_contract import MODEL_FILES,bundle,assert_same_candidate,check_recipes
from jevmahjong.evaluation.unified_audit import recount,independent_outcomes
from jevmahjong.policy.unified import serving_request,UnifiedPolicyClient
from jevmahjong.data.base_lora_pool import normalized_request
from jevmahjong.evaluation.unified_run import preflight
from jevmahjong.training.base_lora import validation_profile


class UnifiedEvaluationTests(unittest.TestCase):
    def test_complete_bundle_covers_adapter_and_tokenizer_not_only_pointer_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            a=Path(temporary)/'a';b=Path(temporary)/'b';a.mkdir();b.mkdir()
            for name in MODEL_FILES:(a/name).write_text(name);(b/name).write_text(name)
            first=bundle(a);self.assertEqual(first['identity_sha256'],bundle(b)['identity_sha256'])
            (a/'adapter_model.safetensors').write_text('different LoRA, same head')
            self.assertNotEqual(first['identity_sha256'],bundle(a)['identity_sha256'])
            with self.assertRaisesRegex(ValueError,'different unified'):
                assert_same_candidate(dict(candidate_bundle=first,fit_receipt={}),dict(model=bundle(a),receipt={}))

    def test_one_client_uses_shared_training_format_without_labels_or_metadata(self):
        req=dict(state=dict(seat=0,hand='1m 2m',rivers=['']*4,melds=[[],[],[],[]]),
            questions={'action':dict(type='choice',instructions='old',criteria={'a0':{'type':'dahai','pai':'1m'}},label='a0',target={'a0':1.})},_meta={'future_seed':'not-a-feature'})
        before=copy.deepcopy(req);formatted=serving_request(req)
        self.assertEqual(req,before);self.assertNotIn('_meta',formatted)
        self.assertEqual(set(formatted['questions']['action']),{'type','instructions','criteria'})
        self.assertIn('winning sooner',formatted['questions']['action']['instructions'])
        class Model:
            def system_one(self,**request):return request
        self.assertEqual(UnifiedPolicyClient(Model()).system_one(**req),formatted)

    def test_recipe_links_reject_old_development_pointer_or_overlapping_walls(self):
        common=dict(candidate_run='fresh-adapter',candidate_model_id='fresh-model',reference={'residual':'fixed'},confirmation_objective='hand_win',fast_gamma=.98,max_dealin_rate_increase=.01)
        p=dict(common,out='preflight',stage='development',preflight_only=True,seed_start=100,seeds=8)
        d=dict(common,out='development',stage='development',required_native_preflight='preflight/native-preflight.json',seed_start=100,seeds=8)
        c=dict(common,out='confirmation',stage='independent_confirmation',development_out='development',required_native_preflight='preflight/native-preflight.json',seed_start=200,seeds=32)
        self.assertEqual(check_recipes(p,d,c)['status'],'passed_unified_recipe_links')
        for changes in [dict(development_out='some-other-model'),dict(seed_start=105),dict(reference={'residual':'other'})]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):check_recipes(p,d,dict(c,**changes))
        with self.assertRaises(ValueError):check_recipes(p,d,dict(c,candidate_selection={'manifest':'other'}))

    def test_native_preflight_covers_all_routes_and_compares_each_probability(self):
        from tests.core.test_win_context import pair
        from tests.core.test_kan_context import request
        raw=[request('ankan'),request('daiminkan'),request('kakan'),pair(100)['public_request']]
        for opened in (False,True):
            r=request('kakan')
            if not opened:r['state']['melds'][0]=[]
            r['questions']['action']['criteria']={'a0':{'type':'dahai','pai':'P'},'a1':{'type':'dahai','pai':'2m'}}
            raw.append(r)
        records=[];saved=[]
        for i,r in enumerate(raw):
            r['questions']['action'].update(type='choice',label='a0')
            record=normalized_request(r,'held:'+str(i),'expert_replay',i);records.append(record)
            saved.append(dict(id=record['_meta']['id'],keys=['a0','a1'],p=[.7,.3]))
        weights=dict(closed_discard=.45,open_discard=.20,call_window=.30,kan_window=.05)
        candidate=dict(validation_records=records,selected_validation_rows=saved,
            training_config={'validation_phase_weights':weights},selected_validation=validation_profile(records,saved,weights),
            model={'identity_sha256':'a'*64},selected_step=128)
        class Client:
            def system_one(self,**request):return {'answers':{'action':{'probabilities':{'a0':.7,'a1':.3}}}}
        result,rows=preflight(Client(),candidate)
        self.assertEqual(result['records'],6);self.assertEqual(result['complete_matches_simulated'],0)
        self.assertEqual(result['max_abs_probability_difference'],0.)
        self.assertTrue(all(n>0 for n in result['routes'].values()))
        candidate['selected_validation_rows'][0]['p']=[.6,.4]
        with self.assertRaisesRegex(ValueError,'Native probability differs'):preflight(Client(),candidate)

    def test_independent_recount_handles_multiple_ron_and_terminal_unclaimed_sticks(self):
        events=[dict(type='start_kyoku',scores=[25000]*4,kyotaku=0),dict(type='reach_accepted',actor=0),
            dict(type='tsumo',actor=1),dict(type='hora',actor=1,target=0,deltas=[-2000,3000,0,0]),
            dict(type='hora',actor=2,target=0,deltas=[-1000,0,1000,0]),dict(type='end_kyoku'),dict(type='end_game')]
        m=recount(events,0,.98);self.assertEqual(m['dealin_hands'],1);self.assertEqual(m['scores'],[21000,28000,26000,25000])
        draw=[dict(type='start_kyoku',scores=[25000]*4,kyotaku=0),dict(type='reach_accepted',actor=0),
            dict(type='ryukyoku',deltas=[0]*4),dict(type='end_kyoku'),dict(type='end_game')]
        self.assertEqual(recount(draw,0,.98)['scores'],[24000,26000,25000,25000])
        wrong=copy.deepcopy(events);wrong[3]['deltas'][0]+=1
        with self.assertRaisesRegex(ValueError,'conserve'):recount(wrong,0,.98)

    def test_independent_whole_wall_statistics_validate_grid_scores_and_ranks(self):
        rows=[];cfg=dict(seed_start=100,seeds=2,fast_gamma=.98)
        for seed in (100,101):
            for suite in ('closed','mixed','strong'):
                for seat in range(4):
                    for policy in ('reference','candidate'):
                        winner=seat if policy=='candidate' else (seat+1)%4;target=(winner+1)%4
                        deltas=[0]*4;deltas[winner]=2000;deltas[target]=-2000;scores=[25000+d for d in deltas]
                        events=[dict(type='start_kyoku',scores=[25000]*4,kyotaku=0),dict(type='tsumo',actor=0),dict(type='hora',actor=winner,target=target,deltas=deltas),dict(type='end_kyoku'),dict(type='end_game')]
                        m=recount(events,seat,.98)
                        rows.append(dict(seed=seed,suite=suite,seat=seat,policy=policy,mjai_log=events,final_scores=scores,score=scores[seat],rank=m['rank'],hands=1,hand_wins=m['wins'],fast_reward_sum=m['discount'],calls=0,kan_choices=m['kan_choices']))
        result=independent_outcomes(rows,cfg)
        self.assertEqual(result['hand_win_paired']['gain'],1.)
        self.assertEqual(result['net_points_paired']['gain'],2000.)
        with self.assertRaisesRegex(ValueError,'grid'):independent_outcomes(rows+rows[:1],cfg)
        wrong=copy.deepcopy(rows);wrong[0]['score']+=1000
        with self.assertRaisesRegex(ValueError,'outcome'):independent_outcomes(wrong,cfg)


if __name__=='__main__':unittest.main()
