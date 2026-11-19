import copy
import unittest
from jevmahjong.endgame.context import public_context
from jevmahjong.endgame.coverage import public_stratum
from jevmahjong.endgame.reuse import fourth_cell, select_reused_rows
from jevmahjong.endgame.mix import label_set_contract
from jevmahjong.endgame.futures import label
from tools.diagnose_endgame_continuation import select_cases


class ReusedManifestTests(unittest.TestCase):
    def test_continuation_audit_is_training_only_and_future_independent(self):
        records=[{'_meta':{'id':str(i),'group_id':str(i//2),'split':'train','source_initial_placement':4},
                  'questions':{'action':{'src':'endgame_terminal_win','target':{'win':float(i%2)}}}}
                 for i in range(24)]
        records.append({'_meta':{'id':'validation','group_id':'validation','split':'validation',
                               'source_initial_placement':4},'questions':{'action':{'src':'endgame_terminal_win'}}})
        first=select_cases(records)
        self.assertEqual(len(first),8)
        self.assertEqual(len({r['_meta']['group_id'] for r in first}),8)
        self.assertTrue(all(r['_meta']['split']=='train' for r in first))
        changed=copy.deepcopy(records)
        for row in changed:row['future_rank']=1;row['questions']['action']['target']={'win':1.}
        self.assertEqual([r['_meta']['id'] for r in first],[r['_meta']['id'] for r in select_cases(changed)])

    def test_fourth_gap_is_to_third_not_second_and_does_not_use_future(self):
        state={'seat':3,'scores':[40000,35000,13000,12000],'round_wind':0,'hand_number':4,
               'melds':[[],[],[],[]],'riichi':[]}
        context=public_context(state)
        context['current_win_settlement']={'status':'public_single_winner_baseline_scenarios',
            'winner_is_dealer':False,'top2_in_any_baseline_scenario':False,
            'maximum_baseline_scenario':{'deltas':[0,-1000,0,1000]},
            'minimum_baseline_scenario':{'rank_after':3}}
        state['endgame_context']=context
        payload={'state':state}
        row={'public_request':payload,'coverage_stratum':public_stratum(payload),'source_profile':'dama'}
        cell=fourth_cell(row)
        self.assertEqual(cell['gap_to_third'],1000)
        self.assertEqual(cell['gap_to_third_band'],'le3000')
        self.assertEqual(row['coverage_stratum']['gap_band'],'gt8000')
        changed=copy.deepcopy(row);changed['future_scores']=[0,0,0,100000]
        self.assertEqual(fourth_cell(changed),cell)

    def test_source_profiles_cannot_duplicate_a_seed_or_cross_splits(self):
        def row(seed,split,key):
            return {'seed':seed,'split':split,'reuse_fourth':{'cell':key},'reuse_selection_key':str(seed)}
        rows=[row(1,'train','dama'),row(1,'train','standard'),row(2,'train','dama'),row(3,'validation','standard')]
        selected=select_reused_rows(rows,{'train':64,'validation':16})
        self.assertEqual(len(selected),3)
        self.assertEqual(len({r['seed'] for r in selected}),3)
        with self.assertRaises(ValueError):
            select_reused_rows(rows+[row(1,'validation','dama')],{'train':64,'validation':16})

    def test_third_label_set_requires_explicit_count_and_preserves_mc_counts(self):
        original={'legacy4':{'future_worlds':4},'target16':{'future_worlds':16}}
        specs=dict(original,fourth32={'positions':80,'future_worlds':32})
        self.assertEqual(label_set_contract(specs),{'legacy4':(264,4),'target16':(44,16),'fourth32':(80,32)})
        for bad in ({'future_worlds':32},{'positions':80,'future_worlds':33}):
            with self.assertRaises(ValueError):
                label_set_contract(dict(original,fourth32=bad))
        with self.assertRaises(ValueError):
            label_set_contract(dict(specs,legacy4={'future_worlds':32}))

    def test_unregistered_continuation_cannot_query_model_or_read_sources(self):
        with self.assertRaisesRegex(ValueError,'not registered'):
            label({'label_registered':False})


if __name__=='__main__':
    unittest.main()
