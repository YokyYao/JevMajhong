import copy

import unittest

from tools.build_mortal_base_retention import isolated,terminal_target,normalized_source_request
from jevmahjong.endgame.context import placements
from jevmahjong.endgame.targeted_refresh import world_design


def terminal_fixture():
    outcomes={}
    for key,cut in (('a0',1),('a1',2)):
        values=[]
        for world in range(64):
            scores=[40000,30000,20000,10000] if world%4<cut else [10000,40000,30000,20000]
            values.append(dict(scores=scores,ranks=placements(scores),completed=True,
                ledger_passed=True,world=world,**world_design(world)))
        outcomes[key]=values
    return dict(source=dict(seat=0,request=dict(questions=dict(action=dict(criteria={
        'a0':dict(type='dahai',pai='1m'),'a1':dict(type='dahai',pai='2m')})))),
        outcomes=outcomes,worlds_per_action=64,all_actions_completed=True,
        prior128={'a0':.01,'a1':.99},record={'questions':{'action':{'target':{'a0':.01,'a1':.99}}}})


class MortalBaseRetentionTests(unittest.TestCase):
    def test_source_display_context_is_replaced_by_real_serving_input(self):
        source={'state':{'seat':0,'scores':[25000]*4,'hand':'1m',
            'screen_observation':{'reaction':{'seat':3,'tile':'2m'}}},
            'questions':{'action':{'type':'choice','instructions':'source',
                'criteria':{'a0':{'type':'dahai','pai':'1m'},'a1':{'type':'none'}}}}}
        actual=copy.deepcopy(source);actual['state'].pop('screen_observation')
        actual['questions']['action']['instructions']='actual-serving'
        normalized=normalized_source_request(source,actual)
        self.assertEqual(normalized,actual)
        self.assertIn('screen_observation',source['state'])
        self.assertIsNot(normalized,actual)

    def test_normalization_rejects_changed_native_state_or_action_order(self):
        source={'state':{'seat':0,'scores':[25000]*4},'questions':{'action':{
            'criteria':{'a0':{'type':'dahai','pai':'1m'},'a1':{'type':'dahai','pai':'2m'}}}}}
        changed=copy.deepcopy(source);changed['state']['scores'][0]=26000
        with self.assertRaisesRegex(ValueError,'native public state'):
            normalized_source_request(source,changed)
        changed=copy.deepcopy(source)
        changed['questions']['action']['criteria']=dict(reversed(list(changed['questions']['action']['criteria'].items())))
        with self.assertRaisesRegex(ValueError,'ordered action support'):
            normalized_source_request(source,changed)

    def test_terminal_target_uses_fresh_prior_not_old_targets(self):
        label=terminal_fixture();prior={'a0':.8,'a1':.2}
        actual=terminal_target(label,prior,['a0','a1'])
        changed=copy.deepcopy(label)
        changed['prior128']={'a0':.99,'a1':.01}
        changed['record']['questions']['action']['target']={'a0':1.,'a1':0.}
        self.assertEqual(terminal_target(changed,prior,['a0','a1']),actual)
        target,_,top2,escape=actual
        self.assertEqual(top2,{'a0':.25,'a1':.5})
        self.assertEqual(escape,top2)
        self.assertGreater(target['a1'],prior['a1'])
        self.assertAlmostEqual(sum(target.values()),1.,places=12)


    def test_missing_or_unpaired_world_is_not_admitted(self):
        label=terminal_fixture();label['outcomes']['a0'].pop()
        with self.assertRaisesRegex(ValueError,'all64'):
            terminal_target(label,{'a0':.8,'a1':.2},['a0','a1'])
        label=terminal_fixture();label['outcomes']['a0'][4]['policy']=1
        with self.assertRaisesRegex(ValueError,'World pairing'):
            terminal_target(label,{'a0':.8,'a1':.2},['a0','a1'])


    def test_distinct_decisions_from_same_source_cannot_cross_split(self):
        train=[{'_meta':{'id':'d1','group_id':'source1'}}]
        validation=[{'_meta':{'id':'d2','group_id':'source1'}}]
        with self.assertRaisesRegex(ValueError,'Whole source game'):
            isolated(train,validation)
