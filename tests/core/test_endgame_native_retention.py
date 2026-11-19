import copy
import unittest
from unittest.mock import patch
from jevmahjong.endgame.native_retention import native_record,reference_preparation_only
from jevmahjong.endgame.reference import apply_reference
from jevmahjong.endgame.fit import original_replay_ce,selection_value
from jevmahjong.endgame.prepare import balance_training_sources


def native(rid):
    frame={'id':rid,'group_id':'group:'+rid,'split':'train','phase':'riichi','rank':4,
        'teacher_request_sha256':'teacher:'+rid,'student_request_sha256':'student:'+rid,
        'student_request':{'state':{},'questions':{'action':{'type':'choice',
            'criteria':{'a':{'type':'none'},'b':{'type':'none'}}}}}}
    return native_record(frame)


class NativeRetentionTests(unittest.TestCase):
    def test_feasible_candidate_outranks_infeasible_anchor_without_promoting_rejected_steps(self):
        anchor=selection_value(0,.6,1.,True,False,True)
        self.assertLess(selection_value(56,.5,1.,True,False,True),anchor)
        self.assertGreater(selection_value(56,.9,.01,False,False,True),anchor)
        feasible_anchor=selection_value(0,.6,1.,True,True,True)
        self.assertGreater(selection_value(56,.5,1.,True,True,True),feasible_anchor)
        self.assertEqual(selection_value(0,.6,1.,True,False,False),-.6)

    def test_rank_balancing_only_applies_after_runtime_fixed_wins_are_removed(self):
        def record(seat,src='endgame_terminal'):
            return {'state':{'seat':seat,'scores':[40000,30000,20000,10000]},
                '_meta':{'bucket':'win'},'questions':{'action':{'src':src}}}
        config={'balance_endgame_buckets':True,'split_win_by_initial_rank':True,'reference_regularization':True}
        rows=[record(0,'replay_fixed_win'),record(2),record(3)]
        balance_training_sources(rows,config)
        self.assertEqual([r['questions']['action']['src'] for r in rows],
            ['replay_fixed_win','endgame_terminal_win_third','endgame_terminal_win_fourth'])
        provisional=[record(0)]
        balance_training_sources(provisional,dict(config,reference_regularization=False))
        self.assertEqual(provisional[0]['questions']['action']['src'],'endgame_terminal_win')

    def test_added_native_validation_cannot_mask_old_expert_regression(self):
        old={'replay_expert':[.2,.4]}
        self.assertAlmostEqual(original_replay_ce(dict(old,replay_native=[0.]*100)),.3)
        self.assertAlmostEqual(original_replay_ce(dict(old,replay_native=[100.]*100)),.3)
        with self.assertRaises(ValueError):original_replay_ce({'replay_native':[.1]})

    def test_raw_native_pair_has_no_invented_target_or_expert_label(self):
        record=native('train')
        self.assertNotIn('label',record['questions']['action'])
        self.assertNotIn('target',record['questions']['action'])
        self.assertEqual(record['questions']['action']['src'],'replay_native')
        self.assertEqual(record['_meta']['label_type'],'frozen_original_policy_pending')
        self.assertTrue(record['_meta']['native_reference_pending'])

    def test_pending_pairs_cannot_be_used_as_registered_training(self):
        config={'internal_reference_preparation':True,'reference_regularization':False,'fit_registered':False}
        mass,weights=reference_preparation_only(config,{'old':3,'native':1})
        self.assertEqual(mass,{'old':.75,'native':.25});self.assertEqual(weights,{'old':1.,'native':1.})
        for changes in ({'fit_registered':True},{'internal_reference_preparation':False},{'reference_regularization':True}):
            with self.assertRaises(ValueError):reference_preparation_only(dict(config,**changes),{'native':1})

    def test_reference_labels_train_and_native_validation_without_changing_old_expert_validation(self):
        train=native('train');val=native('val');val['_meta']['split']='validation'
        terminal={'state':{'endgame_context':{'late_match_window':True,'placement':3,
            'current_win_settlement':{'status':'public_single_winner_baseline_scenarios',
                'winner_is_dealer':False,'top2_in_any_baseline_scenario':False}}},
            '_meta':{'id':'terminal','bucket':'win','target_basis':'complete_terminal_MC_soft_target',
                     'expected_terminal_utilities':{'a':0.,'b':.1}},
            'questions':{'action':{'type':'choice','src':'endgame_terminal',
                'criteria':{'a':{'type':'hora'},'b':{'type':'none'}},'target':{'a':.5,'b':.5},'label':'a'}}}
        expert={'_meta':{'id':'expert'},'questions':{'action':{'src':'replay_expert','label':'a','criteria':{'a':{},'b':{}}}}}
        before=copy.deepcopy(expert)
        cache={'identity_sha256':'baseline','input_version':'endgame-trailing-public-v1','rows':{
            'train':{'probabilities':{'a':.2,'b':.8},'teacher_request_sha256':'teacher:train'},
            'val':{'probabilities':{'a':.9,'b':.1},'teacher_request_sha256':'teacher:val'},
            'terminal':{'probabilities':{'a':1.,'b':0.}},'expert':{'probabilities':{'a':.5,'b':.5}}}}
        config={'reference_mode':'released_original_policy','input_version':'endgame-trailing-public-v1',
                'reference_temperature':.1,'reference_exploration':.1,'win_target_style':'greedy_terminal_best'}
        with patch('jevmahjong.endgame.reference.load_reference',return_value=cache):
            apply_reference(config,{'train':[train,terminal],'validation':[val,expert]})
        self.assertEqual(train['questions']['action']['target'],{'a':.2,'b':.8})
        self.assertEqual(train['questions']['action']['src'],'replay_reference')
        self.assertEqual(val['questions']['action']['target'],{'a':.9,'b':.1})
        self.assertEqual(val['questions']['action']['src'],'replay_native')
        self.assertNotIn('native_reference_pending',train['_meta'])
        self.assertNotIn('native_reference_pending',val['_meta'])
        self.assertEqual(expert,before)


if __name__=='__main__':unittest.main()
