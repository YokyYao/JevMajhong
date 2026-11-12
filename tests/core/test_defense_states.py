"""Public defense admission and unlabeled native inputs preserve their scope."""
import copy
import unittest
import json
from pathlib import Path
import tempfile
import torch
from jevmahjong.core.io import request_hash
from jevmahjong.data.native_states import METHOD, decision_records, validate
from jevmahjong.policy.defense import is_defense_request, request_eligible


def fixture():
    req={'state':{'seat':0,'riichi':[1],'melds':[[{'type':1}],[],[],[]]},
         'questions':{'action':{'criteria':{
             'a0':{'type':'dahai','sh':0,'safe':[]},
             'a1':{'type':'dahai','sh':1,'safe':[1]},
             'a2':{'type':'dahai','sh':1,'safe':[]}}}}}
    events=[{'type':'start_kyoku','scores':[25000]*4,'kyotaku':0},
            {'type':'ryukyoku','deltas':[0]*4},{'type':'end_kyoku'},{'type':'end_game'}]
    controls=[dict(seed=10,seat=p,score=25000,mjai_log=events,
        all_point_components=[{'draw_settlement':0} for _ in range(4)],
        point_components={'draw_settlement':0},settlement_ledger_passed=True,
        frozen_control=True,environment_steps=8,available_intervention_decisions=int(p==0)) for p in range(4)]
    row=dict(episode_key=(10,0,0),scope='ordinary',public_request=req,
        request_sha256=request_hash(req),eligible=[0,1],teacher_action=0,action=1,
        reference=torch.tensor([.6,.3,.1]),decide=torch.zeros(4),options=torch.zeros(3,4),
        action_keys=['one','two','three'],source_state_digest='digest',hand_index=0,draws_before=5)
    meta=dict(episode_key=(10,0,0),seed=10,seat=0,replica=0,intervention_step=3,
        intervention_state_digest='digest',hand_index=0,draws_before=5)
    return dict(collection_method=METHOD,seed=10,controls=controls,decisions=[meta],rows=[row])


class DefenseStatesTest(unittest.TestCase):
    def test_public_predicate_and_all_known_riichi_safety(self):
        req=fixture()['rows'][0]['public_request']
        self.assertTrue(is_defense_request(req));self.assertEqual(request_eligible(req),[0,1])
        req['state']['riichi']=[1,2]
        self.assertFalse(is_defense_request(req))
        req['questions']['action']['criteria']['a1']['safe']=[1,2]
        self.assertTrue(is_defense_request(req));self.assertEqual(request_eligible(req),[0,1])

    def test_closed_riichi_non_tenpai_and_call_menus_are_outside_scope(self):
        for kind in ['closed','ankan','own_riichi','one_shanten','call']:
            req=copy.deepcopy(fixture()['rows'][0]['public_request'])
            if kind=='closed':req['state']['melds'][0]=[]
            if kind=='ankan':req['state']['melds'][0]=[{'type':3}]
            if kind=='own_riichi':req['state']['riichi'].append(0)
            if kind=='one_shanten':req['questions']['action']['criteria']['a0']['sh']=1
            if kind=='call':req['questions']['action']['criteria']['a2']['type']='pon'
            with self.subTest(kind=kind):self.assertFalse(is_defense_request(req))

    def test_inputs_have_no_invented_branch_outcomes(self):
        payload=fixture();self.assertEqual(len(validate(payload,10,{})),1)
        self.assertEqual(decision_records(payload),payload['decisions']);self.assertNotIn('games',payload)
        self.assertNotIn('training_advantage',payload['rows'][0])
        self.assertEqual(decision_records({'games':[1]}),[1])

    def test_reject_corrupt_control_inputs_and_replay_metadata(self):
        def duplicate(p):p['controls'].append(copy.deepcopy(p['controls'][0]))
        def other_ledger(p):p['controls'][0]['all_point_components'][1]={'ron_payments':0}
        def unsafe(p):p['rows'][0]['eligible'].append(2)
        def metadata(p):p['decisions'][0]['seat']=1
        def shape(p):p['rows'][0]['options']=torch.zeros(2,4)
        def teacher(p):p['rows'][0]['teacher_action']=1;p['rows'][0]['action']=0
        def coverage(p):p['controls'][0]['available_intervention_decisions']=2
        def prefix(p):p['decisions'][0]['intervention_step']=8
        for change in [duplicate,other_ledger,unsafe,metadata,shape,teacher,coverage,prefix]:
            payload=fixture();change(payload)
            with self.subTest(change=change.__name__),self.assertRaises(ValueError):validate(payload,10,{})

    def test_unlabeled_sources_cannot_enter_a_fitter(self):
        from jevmahjong.data.native_pairs import ENCODER_FILES,load_sources
        from jevmahjong.core.io import digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'walls-private').mkdir()
            cfg={'game_mode':'4p-red-east','seed_scheme':'spaced-v1','fast_gamma':.995,
                'training_seed_start':10,'training_seeds_per_stage':4}
            files={role:{name:'fixed-'+name for name in ENCODER_FILES} for role in ('discard','call')}
            (root/'protocol.json').write_text(json.dumps({'config':cfg,'model_files':files,'collection_method':METHOD}))
            for seed in range(10,14):
                payload=fixture();payload['seed']=seed
                for control in payload['controls']:control['seed']=seed
                payload['rows'][0]['episode_key']=(seed,0,0)
                payload['decisions'][0].update(seed=seed,episode_key=(seed,0,0))
                target=root/'walls-private'/f'{seed}.pt';torch.save(payload,target)
                target.with_suffix('.audit.json').write_text(json.dumps({'sha256':digest(target)}))
            (root/'collection-result.json').write_text(json.dumps({'status':'collection_complete','walls':4,'rows':4}))
            with self.assertRaisesRegex(ValueError,'no reward labels'):load_sources(root)
            parts,_,sources=load_sources(root,allow_unlabeled=True)
            self.assertEqual((len(parts['train']),len(parts['held'])),(3,1))
            self.assertEqual((sources[0]['train_walls'],sources[0]['held_walls']),(3,1))
            self.assertTrue(all('training_advantage' not in r for rows in parts.values() for r in rows))

    def test_native_inputs_use_real_paired_future_credits(self):
        from jevmahjong.data.future_pairs import mean_credit,validate_rows,METHOD as FUTURE_METHOD
        from jevmahjong.training.collector import derived_seed
        from jevmahjong.core.io import digest
        from jevmahjong.core.scoring import settlement_ledger
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'walls-private').mkdir();source=root/'walls-private/10.pt'
            original=fixture();old=original['rows'][0];old.update(category='ordinary',draws_before=0)
            original['decisions'][0]['draws_before']=0;torch.save(original,source)
            cfg={'train_scopes':['ordinary'],'decision_filter':'open_tenpai_riichi','source_root':str(root),
                'training_seed_start':10,'fast_gamma':.995,'seed':9,'worlds':2,
                'reward':{'mode':'net_points_with_fast_support_v1','net_weight':1.,'fast_weight':.1,
                    'dealin_penalty':.4,'point_scale':8000.}}
            start=original['controls'][0]['mjai_log'][0]
            logs={'control':original['controls'][0]['mjai_log'],
                'branch':[start,{'type':'tsumo'},{'type':'hora','actor':0,'target':1,
                    'deltas':[1000,-1000,0,0]},{'type':'end_kyoku'}]}
            scores={'control':[25000]*4,'branch':[26000,24000,25000,25000]}
            worlds=[{'future_seed':derived_seed(9,10,0,0,f'future-{i}'),'future_wall_sha256':'a'*64,
                **{name:{'mjai_log':log,'scores_at_stop':scores[name],
                    'point_ledger':settlement_ledger(log,scores[name]),'stopped_after_intervention_hand':True}
                    for name,log in logs.items()}} for i in range(2)]
            row=dict(old,source_split='train',prefix_events=1,counterfactual_worlds=worlds)
            means=mean_credit(worlds,row,cfg,cfg['reward'])
            row.update(training_credit=means,training_advantage=means['branch']['utility']-means['control']['utility'],
                fast_advantage=means['branch']['fast_reward']-means['control']['fast_reward'])
            sha=digest(source);payload={'seed':10,'collection_method':FUTURE_METHOD,'source_path':str(source),
                'source_sha256':sha,'rows':[row]}
            protocol={'source_files':{str(source):sha},'selected_case_keys':[[10,0,0]]}
            validated=validate_rows(payload,10,cfg,protocol)
            self.assertAlmostEqual(validated[0]['training_advantage'],.125+.1*.995)
            self.assertNotIn('counterfactual_worlds',validated[0])
            bad=copy.deepcopy(payload);bad['rows'][0]['training_advantage']+=1
            with self.assertRaisesRegex(ValueError,'advantage changed'):validate_rows(bad,10,cfg,protocol)

    def test_defense_features_and_uniform_weights_preserve_other_routes(self):
        from jevmahjong.policy.defense import VERSION
        from jevmahjong.policy.kan_context import feature_names,feature_vectors,DefenseContextPolicyClient
        from jevmahjong.training.context_policy import training_groups
        from jevmahjong.evaluation.run import candidate_activation,parity_route
        req=copy.deepcopy(fixture()['rows'][0]['public_request'])
        req['state'].update(dealer=0,round_wind=0,honba=0,sticks=1,scores=[25000]*4,
            discards_count=20,dora_indicators='4z',hand='1m 2m 3m 4p 5p 6p 6s 7s 8s P F',
            rivers=['','5p','',''],melds=[[{'type':1,'tiles':'E E E'}],[],[],[]])
        for c,tile in zip(req['questions']['action']['criteria'].values(),['F','5p','P']):c.update(pai=tile,ukeire=3)
        matrix=feature_vectors(req,VERSION);names=feature_names(VERSION)
        self.assertEqual(len(matrix[0]),158)
        self.assertEqual(matrix[0][names.index('unsafe_discard')],1.)
        self.assertEqual(matrix[1][names.index('unsafe_discard')],0.)
        with self.assertRaises(ValueError):feature_vectors(req)
        base=dict(fixture()['rows'][0],public_request=req,eligible=[0,1,2])
        rows=[dict(base,action=a,training_advantage=1000*a) for a in [1,2]]
        rows.append(dict(base,episode_key=(11,0,0),eligible=[0,1],action=1,training_advantage=-1000.))
        groups,coverage=training_groups(rows,{'policy_state_weighting':'uniform'},VERSION)
        self.assertEqual([g['weight'] for g in groups],[1.,1.]);self.assertFalse(coverage['outcome_sign_reweighting'])
        answer={'answers':{'action':{'probabilities':{'a0':.6,'a1':.3,'a2':.1}}}}
        class Frozen:
            def system_one(self,**request):return answer
        context={'version':VERSION,'names':list(names),'coefficients':torch.zeros(len(names)),
            'reference_prior_strength':.05}
        client=DefenseContextPolicyClient(Frozen(),context)
        actual=client.system_one(**req)['answers']['action']['probabilities']
        for key,value in answer['answers']['action']['probabilities'].items():self.assertAlmostEqual(actual[key],value)
        other=copy.deepcopy(req);other['state']['riichi']=[]
        self.assertIs(client.system_one(**other),answer)
        self.assertEqual(parity_route(dict(base),'defense_windows'),'ordinary')
        self.assertEqual(parity_route(dict(base,public_request=other),'defense_windows'),'ordinary_other')
        cfg={'updated_scopes':['ordinary'],'candidate_activation':'defense_windows',
            'candidate_action_admission':'open_tenpai_genbutsu_v1'}
        self.assertEqual(candidate_activation(cfg,'defense_windows'),'defense_windows')
        cfg.pop('candidate_action_admission')
        with self.assertRaises(ValueError):candidate_activation(cfg,'defense_windows')


if __name__=='__main__':unittest.main()
