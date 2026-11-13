"""Kan data keeps native admission and a pinned stronger calling continuation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from riichienv import ActionType
from jevmahjong.core.io import digest, request_hash
from jevmahjong.data.native_states import KAN_METHOD, validate
from jevmahjong.evaluation.agents import guarded_indices
from jevmahjong.policy.call_baseline import identity, load, transform, wrap
from jevmahjong.policy.kan import request_eligible
from jevmahjong.policy.kan_context import feature_names
from jevmahjong.policy.kan_progress import VERSION
from jevmahjong.training.kan_states import KanStateCollector
from tests.core.test_defense_states import fixture
from tests.core.test_kan_context import request


class KanStatesTests(unittest.TestCase):
    def test_public_guard_matches_native_admission_with_reach_and_defense(self):
        for riichi in ([],[1],[1,2]):
            for reach in (False,True):
                for distances in ((0,1,2),(1,1,2),(2,1,0)):
                    req=request('ankan');req['state']['riichi']=riichi
                    menu={'a0':req['questions']['action']['criteria']['a1']}
                    for i,sh in enumerate(distances,1):
                        menu[f'a{i}']=dict(type='dahai',sh=sh,ukeire=10*i,red=i==2,safe=[1] if i==3 else [])
                    if reach:menu['a4']=dict(type='reach')
                    req['questions']['action']['criteria']=menu
                    kinds={'ankan':ActionType.ANKAN,'dahai':ActionType.DISCARD,'reach':ActionType.RIICHI}
                    actions=[SimpleNamespace(action_type=kinds[c['type']]) for c in menu.values()]
                    feats=[c if c['type']=='dahai' else {} for c in menu.values()]
                    obs=SimpleNamespace(player_id=0,to_dict=lambda:dict(riichi_declared=[i in riichi for i in range(4)]))
                    expected=sorted(set(guarded_indices(obs,actions,feats))|{i for i,f in enumerate(feats) if not f})
                    self.assertEqual(request_eligible(req),expected)

    def test_unlabeled_kan_inputs_preserve_all_admissible_alternatives(self):
        for kind in ('ankan','daiminkan','kakan'):
            p=fixture();p['collection_method']=KAN_METHOD;r=p['rows'][0];req=request(kind)
            if kind=='daiminkan':
                req['questions']['action']['criteria']['a2']=dict(type='pon',pai='E',consumed=['E','E'])
            else:
                req['questions']['action']['criteria']['a0'].update(sh=0,ukeire=10,red=False,safe=[])
                req['questions']['action']['criteria']['a2']=dict(type='dahai',pai='2m',sh=1,ukeire=20,red=False,safe=[])
            r.update(public_request=req,request_sha256=request_hash(req),eligible=request_eligible(req))
            self.assertEqual(validate(p,10,{})[0]['eligible'],[0,1,2] if kind=='daiminkan' else [0,1])
            self.assertNotIn('training_credit',r)
            r['eligible']=[0,1,2] if kind!='daiminkan' else [0,1]
            with self.assertRaisesRegex(ValueError,'action set'):validate(p,10,{})

    def test_collector_captures_only_kan_without_changing_native_actions(self):
        collector=KanStateCollector.__new__(KanStateCollector)
        def original(instance,obs,frozen,stochastic,trace):
            if trace is not None:trace.append('captured')
            return 'unchanged-action'
        for req,expected in ((request('ankan'),True),({'questions':{'action':{'criteria':{'a0':dict(type='pon')}}}},False)):
            with patch('jevmahjong.training.kan_states.build_request',return_value=(req,None,None)),patch('jevmahjong.training.collector.NativeCollector.act',original):
                trace=[];self.assertEqual(collector.act(None,frozen=True,trace=trace),'unchanged-action')
                self.assertEqual(bool(trace),expected)

    def test_call_baseline_load_rejects_mutations_and_keeps_noncall_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files={};heads={}
            for scope,role in [('ordinary','discard'),('calling','call')]:
                (root/role).mkdir();path=root/role/'head.pt';state={'weight':torch.ones(2,2)}
                torch.save({'head':state},path);files[role]={str(path):digest(path)};heads[scope]=state
            names=feature_names(VERSION);coef=torch.zeros(len(names));coef[names.index('pon')]=.5
            context=dict(version=VERSION,names=list(names),coefficients=coef,reference_prior_strength=.05)
            saved=dict(model_files=files,heads=heads,base_and_LoRA_frozen=True,trained_scopes=['calling'],decision_filter='call_windows',kan_context=context)
            path=root/'calling.pt';torch.save(saved,path)
            cfg=dict(train_scopes=['ordinary'],decision_filter='kan_windows',call_context_baseline=dict(checkpoint=str(path),sha256=digest(path)))
            loaded=load(cfg,files);req=request('daiminkan');req['questions']['action']['criteria']['a1']=dict(type='pon',pai='E',consumed=['E','E'])
            answer={'answers':{'action':{'probabilities':{'a0':.9,'a1':.1}}}}
            client=wrap(SimpleNamespace(system_one=lambda **kw:answer),loaded)
            self.assertGreater(client.system_one(**req)['answers']['action']['probabilities']['a1'],.99)
            self.assertIs(client.system_one(**request('ankan')),answer)
            self.assertTrue(torch.allclose(transform(req,torch.tensor([.9,.1]),loaded),torch.tensor(list(client.system_one(**req)['answers']['action']['probabilities'].values()))))
            self.assertIsNone(identity({}));self.assertIsNone(load({},{}))
            for change in ('head','encoder','scope'):
                wrong=copy.deepcopy(saved)
                if change=='head':wrong['heads']['ordinary']['weight']+=1
                if change=='encoder':wrong['model_files']['discard']={}
                if change=='scope':wrong['decision_filter']='kan_windows'
                torch.save(wrong,path);spec=dict(cfg,call_context_baseline=dict(checkpoint=str(path),sha256=digest(path)))
                with self.subTest(change=change),self.assertRaises(ValueError):load(spec,files)
            with self.assertRaisesRegex(ValueError,'changed'):load(cfg,files)
            with self.assertRaises(ValueError):identity(dict(call_context_baseline=dict(checkpoint='x',sha256='z'*64)))

    def test_data_rejects_silently_mixing_old_and_new_calling_continuations(self):
        from jevmahjong.data.native_pairs import load_sources, ENCODER_FILES
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'walls-private').mkdir()
            spec=dict(checkpoint='private-calling.pt',sha256='a'*64)
            cfg=dict(game_mode='4p-red-east',seed_scheme='spaced-v1',fast_gamma=.98,training_seed_start=10,training_seeds_per_stage=0,call_context_baseline=spec)
            files={role:{name:name for name in ENCODER_FILES} for role in ('discard','call')}
            (root/'protocol.json').write_text(json.dumps(dict(config=cfg,model_files=files,collection_method=KAN_METHOD)))
            (root/'collection-result.json').write_text(json.dumps(dict(status='collection_complete',walls=0,rows=0)))
            with self.assertRaisesRegex(ValueError,'continuation baseline'):load_sources(root,allow_unlabeled=True)
            load_sources(root,allow_unlabeled=True,expected_call_baseline=spec)
            with self.assertRaisesRegex(ValueError,'continuation baseline'):
                load_sources(root,allow_unlabeled=True,expected_call_baseline=dict(spec,sha256='b'*64))

    def test_reference_cache_identity_includes_frozen_calling_residual_runtime(self):
        from jevmahjong.evaluation.reference_cache import GAMEPLAY_FILES,gameplay_identity
        extra=('policy/call_baseline.py','policy/kan_context.py','policy/kan_progress.py',
               'policy/call_tile_context.py','policy/kan.py','policy/defense.py')
        protocol=dict(call_context_baseline=dict(sha256='a'*64),
                      code_sha256={'/source/jevmahjong/'+p:p for p in (*GAMEPLAY_FILES,*extra)})
        before=gameplay_identity(protocol);other=copy.deepcopy(protocol)
        other['code_sha256']['/source/jevmahjong/policy/call_tile_context.py']='changed'
        self.assertNotEqual(before,gameplay_identity(other))
        legacy=copy.deepcopy(protocol);legacy.pop('call_context_baseline')
        self.assertEqual(set(gameplay_identity(legacy)),set(GAMEPLAY_FILES))


if __name__=='__main__':unittest.main()
