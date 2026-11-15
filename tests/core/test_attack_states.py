"""Attacking inputs retain native admission, complete context and wall splits."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from riichienv import ActionType,GameRule,RiichiEnv
from tests.core.test_defense_states import fixture
from jevmahjong.core.io import digest,request_hash
from jevmahjong.core.seeds import engine_seed
from jevmahjong.data.native_states import ATTACK_METHOD,decision_records,validate
from jevmahjong.data.native_pairs import select_decisions
from jevmahjong.data.future_pairs import future_scope,validate_rows,METHOD
from jevmahjong.evaluation.agents import EfficiencyAgent,guarded_indices
from jevmahjong.policy.ordinary import is_attack_request,request_eligible
from jevmahjong.training.attack_states import AttackStateCollector,enumerate_control,audit_control_coverage


def attack_fixture(opened=False):
    payload=fixture();payload['collection_method']=ATTACK_METHOD;row=payload['rows'][0]
    request=row['public_request'];request['state'].update(riichi=[],scores=[25000]*4,
        rivers=[[] for _ in range(4)],honba=0,sticks=0,hand_number=0,
        melds=[[{'type':1}] if opened else [],[],[],[]])
    for criterion in request['questions']['action']['criteria'].values():
        criterion.update(sh=1,ukeire=10,red=False)
    row.update(eligible=[0,1,2],request_sha256=request_hash(request))
    return payload


class AttackStatesTests(unittest.TestCase):
    def test_public_guard_matches_native_closed_open_reach_and_threats(self):
        for opened in (False,True):
            for riichi in ([],[1],[1,2]):
                for reach in (False,True):
                    for distances in ((0,1,2),(1,1,2),(2,1,0)):
                        request=attack_fixture(opened)['rows'][0]['public_request']
                        request['state']['riichi']=riichi
                        menu={f'a{i}':dict(type='dahai',sh=sh,ukeire=10*i,red=i==2,safe=[1] if i==2 else [])
                            for i,sh in enumerate(distances)}
                        if reach:menu['a3']=dict(type='reach')
                        request['questions']['action']['criteria']=menu
                        actions=[SimpleNamespace(action_type=ActionType.RIICHI if c['type']=='reach'
                            else ActionType.DISCARD) for c in menu.values()]
                        feats=[c if c['type']=='dahai' else {} for c in menu.values()]
                        obs=SimpleNamespace(player_id=0,to_dict=lambda:dict(riichi_declared=[i in riichi for i in range(4)]))
                        actual=sorted(set(guarded_indices(obs,actions,feats))|{i for i,f in enumerate(feats) if not f})
                        self.assertEqual(request_eligible(request),actual)
                        self.assertEqual(is_attack_request(request),not riichi)

    def test_closed_open_selection_and_incomplete_or_other_windows(self):
        closed=attack_fixture()['rows'][0];opened=attack_fixture(True)['rows'][0]
        parts={'train':[closed,opened],'held':[]}
        for mode,expected in [('attacking_closed_discard',closed),('attacking_open_discard',opened)]:
            self.assertEqual(future_scope(dict(train_scopes=['ordinary'],decision_filter=mode)),('ordinary',mode))
            self.assertEqual(select_decisions(parts,mode,['ordinary'])['train'],[expected])
        for kind in ('threat','own_riichi','scores','melds','rivers','honba','sticks','hand_number','kan','call','win'):
            request=copy.deepcopy(closed['public_request'])
            if kind=='threat':request['state']['riichi']=[1]
            elif kind=='own_riichi':request['state']['riichi']=[0]
            elif kind in ('kan','call','win'):
                request['questions']['action']['criteria']['a1']['type']={'kan':'ankan','call':'pon','win':'hora'}[kind]
            else:request['state'][kind]=None
            with self.subTest(kind=kind):self.assertFalse(is_attack_request(request))

    def test_native_inputs_remain_unlabelled_and_preserve_actual_guard(self):
        for opened in (False,True):
            payload=attack_fixture(opened)
            self.assertEqual(len(validate(payload,10,{})),1)
            self.assertEqual(decision_records(payload),payload['decisions'])
            self.assertNotIn('training_advantage',payload['rows'][0])
            bad=copy.deepcopy(payload);bad['rows'][0]['eligible']=[0,1]
            with self.assertRaises(ValueError):validate(bad,10,{})
            bad=copy.deepcopy(payload);bad['rows'][0]['public_request']['state']['riichi']=[1]
            bad['rows'][0]['request_sha256']=request_hash(bad['rows'][0]['public_request'])
            with self.assertRaises(ValueError):validate(bad,10,{})

    def test_collector_filters_inputs_without_changing_native_actions(self):
        collector=AttackStateCollector.__new__(AttackStateCollector)
        def capture(instance,obs,frozen,stochastic,trace):
            if trace is not None:trace.extend(obs)
            return 'same-native-action'
        ordinary=attack_fixture()['rows'][0];threat=copy.deepcopy(ordinary)
        threat['public_request']['state']['riichi']=[1]
        calling=dict(ordinary,scope='calling')
        with patch('jevmahjong.training.collector.NativeCollector.act',capture):
            trace=[];self.assertEqual(collector.act([ordinary,threat,calling],True,trace=trace),'same-native-action')
            self.assertEqual(trace,[ordinary])
            self.assertEqual(collector.act([],True),'same-native-action')

    def test_real_control_replay_rejects_missing_opportunities_and_wrong_score(self):
        cfg=dict(game_mode='4p-red-east',seed_scheme='spaced-v1');seed=36
        env=RiichiEnv(game_mode=cfg['game_mode'],rule=GameRule.default_mjsoul(),seed=engine_seed(seed,cfg['seed_scheme']))
        observations=env.reset();agent=EfficiencyAgent();steps=0
        while not env.done():
            observations=env.step({p:agent.act(o) for p,o in observations.items()});steps+=1
        control=dict(seed=seed,seat=0,score=env.scores()[0],mjai_log=env.mjai_log,environment_steps=steps)
        decisions=enumerate_control(control,cfg)
        self.assertGreater(len(decisions),10)
        payload=dict(controls=[control],decisions=decisions)
        audit_control_coverage(payload,cfg)
        missing=copy.deepcopy(payload);missing['decisions'].pop()
        with self.assertRaisesRegex(ValueError,'coverage differs'):audit_control_coverage(missing,cfg)
        with self.assertRaisesRegex(ValueError,'terminal replay differs'):
            enumerate_control(dict(control,score=control['score']+1000),cfg)

    def test_future_reader_requires_explicit_attacking_source_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'walls-private').mkdir();path=root/'walls-private/10.pt'
            torch.save(attack_fixture(),path);checksum=digest(path)
            payload=dict(seed=10,collection_method=METHOD,source_path=str(path),source_sha256=checksum,rows=[])
            protocol=dict(source_files={str(path):checksum},selected_case_keys=[])
            cfg=dict(train_scopes=['ordinary'],decision_filter='attacking_closed_discard',source_root=str(root),training_seed_start=10)
            self.assertEqual(validate_rows(payload,10,cfg,protocol),[])
            cfg['decision_filter']='closed_discard'
            with self.assertRaisesRegex(ValueError,'declared scope'):validate_rows(payload,10,cfg,protocol)


if __name__=='__main__':unittest.main()
