"""Recovered ordinary decisions preserve native guards and the complete prefix."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from riichienv import ActionType,GameRule,RiichiEnv
from jevmahjong.core.io import digest,request_hash
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request
from jevmahjong.data.native_pairs import select_decisions
from jevmahjong.data.native_states import CLOSED_METHOD,validate
from jevmahjong.data.future_pairs import future_scope
from jevmahjong.evaluation.agents import EfficiencyAgent,guarded_indices
from jevmahjong.policy.closed import is_closed_request,request_eligible
from jevmahjong.training.closed_states import replay_control,audit_control_coverage,control_source_window
from jevmahjong.training.collector import action_key
from tests.core.test_defense_states import fixture


class ClosedStatesTests(unittest.TestCase):
    def test_completed_control_subsets_preserve_opponents_and_held_partition(self):
        source=dict(training_seed_start=11810000,training_seeds_per_stage=128)
        marker=dict(status='collection_complete',walls=128)
        for offset,count in [(0,8),(12,24),(108,20)]:
            cfg=dict(training_seed_start=11810000+offset,training_seeds_per_stage=count)
            self.assertEqual(control_source_window(cfg,source,marker),(cfg['training_seed_start'],count))
            for seed in range(cfg['training_seed_start'],cfg['training_seed_start']+count):
                self.assertEqual((seed-cfg['training_seed_start'])%3,(seed-source['training_seed_start'])%3)
                self.assertEqual((seed-cfg['training_seed_start'])%4,(seed-source['training_seed_start'])%4)
        for offset,count in [(8,24),(3,24),(-12,24),(120,24),(12,0),(12,True)]:
            with self.assertRaises(ValueError):control_source_window(dict(training_seed_start=11810000+offset,training_seeds_per_stage=count),source,marker)
        with self.assertRaises(ValueError):control_source_window(source,source,dict(marker,status='collecting'))

    def test_public_guard_matches_native_reach_and_multi_riichi_risk(self):
        for riichi in ([],[1],[1,2]):
            for reach in (False,True):
                for distances in ((0,1,2),(1,1,2),(2,1,0)):
                    req=copy.deepcopy(fixture()['rows'][0]['public_request']);req['state'].update(riichi=riichi,melds=[[]]*4)
                    menu={f'a{i}':dict(type='dahai',sh=sh,ukeire=10*i,red=i==2,safe=[1] if i==2 else [])
                        for i,sh in enumerate(distances)}
                    if reach:menu['a3']=dict(type='reach')
                    req['questions']['action']['criteria']=menu
                    actions=[SimpleNamespace(action_type=ActionType.RIICHI if c['type']=='reach' else ActionType.DISCARD) for c in menu.values()]
                    feats=[c if c['type']=='dahai' else {} for c in menu.values()]
                    obs=SimpleNamespace(player_id=0,to_dict=lambda:dict(riichi_declared=[i in riichi for i in range(4)]))
                    expected=sorted(set(guarded_indices(obs,actions,feats))|{i for i,f in enumerate(feats) if not f})
                    self.assertEqual(request_eligible(req),expected)

    def test_closed_scope_rejects_other_routes_and_unlabeled_outcomes(self):
        p=fixture();p['collection_method']=CLOSED_METHOD;r=p['rows'][0];req=r['public_request']
        req['state'].update(riichi=[],melds=[[]]*4)
        for i,c in enumerate(req['questions']['action']['criteria'].values()):c.update(sh=1,ukeire=10,red=False)
        r.update(eligible=[0,1,2],request_sha256=request_hash(req))
        self.assertEqual(len(validate(p,10,{})),1);self.assertNotIn('training_advantage',r)
        self.assertEqual(future_scope(dict(train_scopes=['ordinary'],decision_filter='closed_discard')),('ordinary','closed_discard'))
        self.assertEqual(len(select_decisions({'train':[r]},'closed_discard',['ordinary'])['train']),1)
        for kind in ('own_riichi','opened','kan','call','win'):
            bad=copy.deepcopy(req)
            if kind=='own_riichi':bad['state']['riichi']=[0]
            elif kind=='opened':bad['state']['melds'][0]=[dict(type=1)]
            else:bad['questions']['action']['criteria']['a1']['type']={'kan':'ankan','call':'pon','win':'hora'}[kind]
            self.assertFalse(is_closed_request(bad))
        req['state']['melds'][0]=[dict(type=3)];self.assertTrue(is_closed_request(req))

    def test_real_full_control_replay_preserves_timing_and_rejects_omissions(self):
        cfg=dict(game_mode='4p-red-east',seed_scheme='spaced-v1');seed=36
        env=RiichiEnv(game_mode=cfg['game_mode'],rule=GameRule.default_mjsoul(),seed=engine_seed(seed,cfg['seed_scheme']))
        obs=env.reset();agent=EfficiencyAgent();steps=0
        while not env.done():obs=env.step({p:agent.act(o) for p,o in obs.items()});steps+=1
        control=dict(seed=seed,seat=0,score=env.scores()[0],mjai_log=env.mjai_log,environment_steps=steps,
            available_intervention_decisions=0)
        class Capture:
            def act(self,obs,frozen,trace):
                req,actions,feats=build_request(obs)
                eligible=sorted(set(guarded_indices(obs,actions,feats))|{i for i,f in enumerate(feats) if not f})
                trace.append(dict(scope='ordinary',public_request=req,request_sha256=request_hash(req),eligible=eligible,
                    teacher_action=eligible[0],action_keys=[action_key(a) for a in actions]))
        rows,decisions=replay_control(control,cfg,Capture());_,again=replay_control(control,cfg)
        self.assertGreater(len(rows),10);self.assertEqual(decisions,again)
        self.assertTrue(all(r['draws_before']>=0 and 'training_credit' not in r for r in rows))
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'walls-private';source.mkdir();file=source/f'{seed}.pt'
            torch.save(dict(controls=[control]),file);h=digest(file)
            data=dict(seed=seed,source_path=str(file),source_sha256=h,controls=[control],decisions=decisions)
            args=dict(cfg,source_collection=tmp);audit_control_coverage(data,args,{str(file):h})
            missing=copy.deepcopy(data);missing['decisions'].pop()
            with self.assertRaisesRegex(ValueError,'coverage differs'):audit_control_coverage(missing,args,{str(file):h})
            changed=copy.deepcopy(data);changed['controls'][0]['score']+=1000
            with self.assertRaisesRegex(ValueError,'changed original'):audit_control_coverage(changed,args,{str(file):h})
        bad=dict(control,environment_steps=steps-1)
        with self.assertRaisesRegex(ValueError,'saved steps'):replay_control(bad,cfg)


if __name__=='__main__':unittest.main()
