import json
from pathlib import Path
import unittest
from unittest.mock import patch

from jevmahjong.endgame.fourth_chase import top2_chase_target,fourth_chase_action_keys
from tools.audit_endgame_fourth_risk import validate_config


class FourthChaseTests(unittest.TestCase):
    def test_higher_top2_can_accept_a_larger_fourth_place_tail(self):
        target=top2_chase_target({'safe':1.,'risk':0.},{'safe':.05,'risk':.30},
            {'safe':1.,'risk':.35},['safe','risk'])
        self.assertEqual(target,{'safe':0.,'risk':1.})

    def test_escape_only_breaks_an_exact_top2_tie(self):
        target=top2_chase_target({'safe':.1,'risk':.9},{'safe':.2,'risk':.2},
            {'safe':.8,'risk':.3},['safe','risk'])
        self.assertEqual(target,{'safe':1.,'risk':0.})

    def test_unseen_reference_actions_can_receive_tied_target_mass(self):
        target=top2_chase_target({'old':1.,'b':0.,'c':0.},{'old':.1,'b':.4,'c':.4},
            {'old':1.,'b':.6,'c':.6},['old','b','c'])
        self.assertEqual(target,{'old':0.,'b':.5,'c':.5})
        with self.assertRaises(ValueError):top2_chase_target({'a':1.},{'a':.8},{'a':.3},['a'])

    def test_only_confirmed_late_fourth_ordinary_actions_bypass_old_guard(self):
        payload={'state':{'endgame_context':{'late_match_window':True,'placement':4}},
            'questions':{'action':{'criteria':{'a':{'type':'dahai'},'b':{'type':'dahai'}}}}}
        with patch('jevmahjong.endgame.public_policy.action_keys',return_value=['a']) as original:
            self.assertEqual(fourth_chase_action_keys(payload),['a','b']);original.assert_not_called()
            payload['state']['endgame_context']['placement']=3
            self.assertEqual(fourth_chase_action_keys(payload),['a'])
            payload['state']['endgame_context']={'placement':4}
            self.assertEqual(fourth_chase_action_keys(payload),['a'])

    def test_win_continue_public_scope_is_preserved(self):
        payload={'state':{'endgame_context':{'late_match_window':True,'placement':4}},
            'questions':{'action':{'criteria':{'win':{'type':'hora'},'pass':{'type':'none'}}}}}
        self.assertEqual(fourth_chase_action_keys(payload),['win'])
        payload['state']['endgame_context']['current_win_settlement']={
            'status':'public_single_winner_baseline_scenarios','winner_is_dealer':False,
            'top2_in_any_baseline_scenario':False}
        self.assertEqual(fourth_chase_action_keys(payload),['win','pass'])

    def test_audit_registration_cannot_be_changed_to_training_or_validation(self):
        path=Path(__file__).resolve().parents[2]/'configs/training/endgame-fourth-risk-audit-v1.json'
        config=json.loads(path.read_text());validate_config(config)
        for changes in ({'validation_data_used':True},{'fit_registered':True},{'new_simulations':1},
                        {'fourth_training_positions':19},{'out':'runs/endgame-fourth-risk-audit-v1'}):
            with self.assertRaises(ValueError):validate_config(dict(config,**changes))


if __name__=='__main__':unittest.main()
