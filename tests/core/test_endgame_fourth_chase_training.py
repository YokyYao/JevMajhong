import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from jevmahjong.endgame.fourth_chase_training import source_weights,chase_guards,progress_state,fourth_probabilities
from tools.train_endgame_fourth_chase import validate_config


def initial_metrics():
    return {'legacy':{'original_expert_ce':1.,'utility_by_stratum':{'third':.1,'hold':.97},
        'kl':{k:{'mean':v} for k,v in (('original',.005),('trailing',.2),('leading',0.))},
        'fourth_escape':{'existing':{'candidate':.4,'original_policy':.44},
                         'native_fourth':{'candidate':.35,'original_policy':.35}},
        'third_pursuit':{'candidate':.448,'original_policy':.433}},
        'fourth':{'top2':.20,'original_top2':.20,'soft_top2':.18,'fourth':.5},
        'native_third_kl':{'positions':8,'mean':.03},'native_fourth_kl_report_only':{'positions':8,'mean':.4}}


def guard_config():
    return {'enforce_placement_strata_guard':True,'enforce_win_stratum_guard':False,
        'reference_kl_guard_relative_to_start':True,'fourth_escape_constraint':True,
        'reference_kl_guard':.05,'native_reference_kl_guard':.05,'leading_native_reference_kl_guard':.05}


class FourthChaseTrainingTests(unittest.TestCase):
    def test_risk_can_qualify_despite_lower_escape_and_larger_fourth_kl(self):
        initial=initial_metrics();current=copy.deepcopy(initial)
        current['fourth'].update(top2=.25,fourth=.65)
        current['legacy']['fourth_escape']['existing']['candidate']=.2
        current['legacy']['fourth_escape']['native_fourth']['candidate']=.2
        current['native_fourth_kl_report_only']['mean']=1.2
        result=chase_guards(1,current,initial,guard_config())
        self.assertTrue(result['eligible_nonzero']);self.assertEqual(result['fourth_escape_constraint'],'report_only')
        self.assertFalse(chase_guards(1,current,initial,guard_config(),exposure_met=False)['eligible_nonzero'])

    def test_third_and_leading_protection_remains_active(self):
        initial=initial_metrics();current=copy.deepcopy(initial);current['fourth']['top2']=.3
        current['native_third_kl']['mean']=.06
        self.assertFalse(chase_guards(1,current,initial,guard_config())['eligible_nonzero'])
        current=copy.deepcopy(initial);current['fourth']['top2']=.3
        current['legacy']['utility_by_stratum']['hold']=.92
        self.assertFalse(chase_guards(1,current,initial,guard_config())['eligible_nonzero'])

    def test_soft_progress_prevents_flat_argmax_early_stop_but_cannot_select(self):
        initial=initial_metrics();config=guard_config()
        state=progress_state(0,initial,chase_guards(0,initial,initial,config),None,initial)
        for step in (1,2,3):
            current=copy.deepcopy(initial);current['fourth']['soft_top2']+=step*.00001
            flags=chase_guards(step,current,initial,config)
            state=progress_state(step,current,flags,state,initial)
            self.assertFalse(state['stop']);self.assertEqual(state['selected_step'],0)
        current['fourth']['top2']=.21
        state=progress_state(4,current,chase_guards(4,current,initial,config),state,initial)
        self.assertEqual(state['selected_step'],4)

    def test_unsafe_shared_guards_stop_even_with_soft_improvement(self):
        initial=initial_metrics();config=guard_config()
        state=progress_state(0,initial,chase_guards(0,initial,initial,config),None,initial)
        for step in (1,2):
            current=copy.deepcopy(initial);current['fourth']['soft_top2']+=step*.01
            current['legacy']['utility_by_stratum']['hold']=.9
            state=progress_state(step,current,chase_guards(step,current,initial,config),state,initial)
        self.assertTrue(state['stop']);self.assertEqual(state['selected_step'],0)

    def test_source_cap_preserves_all_records_and_total_fourth_mass(self):
        config=json.loads((Path(__file__).resolve().parents[2]/'configs/training/endgame-fourth-top2-chase-v1.json').read_text())
        counts={'replay_reference':544,'replay_reference_leading':64,'replay_fixed_win':79,
            'endgame_terminal_win_third':17,'endgame_terminal_win_fourth_chase':94,'endgame_terminal_third':29,
            'endgame_terminal_hold':23,'endgame_terminal_fourth_chase_improve':16,'endgame_terminal_fourth_chase_keep':39}
        rows=[{'questions':{'action':{'src':key}}} for key,n in counts.items() for _ in range(n)]
        result=source_weights(rows,config)
        self.assertAlmostEqual(result['weights']['endgame_terminal_fourth_chase_improve'],8.)
        self.assertAlmostEqual(result['mass']['endgame_terminal_fourth_chase_improve']+result['mass']['endgame_terminal_fourth_chase_keep'],.2)
        self.assertEqual(result['counts'],counts);self.assertEqual(result['weights']['replay_fixed_win'],0.)

    def test_registration_requires_user_scoped_risk_and_explicit_fit_enable(self):
        config=json.loads((Path(__file__).resolve().parents[2]/'configs/training/endgame-fourth-top2-chase-v1.json').read_text())
        validate_config(config);validate_config(dict(config,fit_registered=True),fit=True)
        for changes in ({'fourth_escape_constraint':'veto'},{'loo_stability_filter':True},
                        {'max_optimizer_steps':128},{'fit_registered':True}):
            with self.assertRaises(ValueError):validate_config(dict(config,**changes))

    def test_native_fourth_agent_uses_one_model_query_and_real_legal_action_scope(self):
        from jevmahjong.endgame.fourth_policy import FourthChaseAgent
        obs=SimpleNamespace(round_wind=0,kyoku_index=3,player_id=0,to_dict=lambda:{'scores':[15000,35000,25000,25000]})
        actions=[SimpleNamespace(to_mjai=lambda:'{"type":"dahai"}') for _ in range(2)]
        payload={'state':{'scores':[15000,35000,25000,25000],'seat':0,
            'endgame_context':{'late_match_window':True,'placement':4}},
            'questions':{'action':{'criteria':{'safe':{'type':'dahai'},'risk':{'type':'dahai'}}}}}
        model=Mock();model.system_one.return_value={'answers':{'action':{'probabilities':{'safe':.2,'risk':.8}}}}
        agent=FourthChaseAgent(model)
        with patch('jevmahjong.endgame.fourth_policy.legal_candidates',return_value=actions),\
             patch('jevmahjong.endgame.fourth_policy.build_request',return_value=(copy.deepcopy(payload),actions,[{},{}])),\
             patch('jevmahjong.endgame.fourth_policy.add_native_context',side_effect=lambda request,*args:request),\
             patch('jevmahjong.endgame.fourth_policy.settlement_request',side_effect=lambda request:request):
            self.assertIs(agent.act(obs),actions[1]);model.system_one.assert_called_once()
        obs.to_dict=lambda:{'scores':[35000,15000,25000,25000]}
        with patch.object(agent.base,'act',return_value=actions[0]) as fallback:
            self.assertIs(agent.act(obs),actions[0]);fallback.assert_called_once()


if __name__=='__main__':unittest.main()
