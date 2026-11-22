import copy
import unittest
from tools.export_curriculum_riichi_curves_v1 import check,METHOD,TRAIN_FIELDS


class RiichiCurveExportTests(unittest.TestCase):
    def fixture(self,step=2):
        stage='stage1-riichi-class-control-v1'
        train=[dict(local_step=s,stage=stage,loss_method=METHOD,loss=1.7,ce=1.5,reward=None,
            ce_is_original_hard_action_loss=True,lr=2e-5,head_lr_derived=1e-4,records_seen=s*94)
            for s in range(1,step+1)]
        validation=[dict(local_step=s,stage=stage,loss_method=METHOD,loss=1.8,ce=1.6,natural_loss=1.7,
            riichi_by_source={'legacy':dict(tp=20,fn=17,tn=40,fp=30),'teacher':dict(tp=180,fn=107,tn=160,fp=78)})
            for s in range(0,step+1,76)]
        games=[dict(local_step=s,stage=stage,matches=16,seed_groups=2,hand_win_rate=.2,
            dealin_hand_rate=.1,first_rate=.25,top2_rate=.5,fourth_rate=.25,
            net_points=800,reward=.1,reward_definition='terminal_raw_net_points_div8000_v1')
            for s in range(0,step+1,152)]
        return train,validation,games

    def test_changed_objective_and_common_ce_are_allowed_to_differ(self):
        train,validation,games=self.fixture();matched,final=check(train,validation,games,2)
        self.assertEqual(len(matched),1);self.assertEqual(final,[])
        self.assertNotEqual(train[0]['loss'],train[0]['ce'])
        self.assertEqual(games[0]['local_step'],0)

    def test_wrong_method_holes_reward_alias_or_source_counts_rejected(self):
        for change in ('method','stage','hole','reward','CE_alias','counts','natural_nan'):
            train,validation,games=self.fixture()
            if change=='method':train[0]['loss_method']='hard_ce'
            elif change=='stage':train[0]['stage']='another-course'
            elif change=='hole':train.pop(0)
            elif change=='reward':train[0]['reward']=0.
            elif change=='CE_alias':train[0]['ce_is_original_hard_action_loss']=False
            elif change=='counts':validation[0]['riichi_by_source']['legacy']['tp']-=1
            else:validation[0]['natural_loss']=float('nan')
            with self.assertRaises(ValueError,msg=change):check(train,validation,games,2)

    def test_no_synthetic_game_points_or_incomparable_grid(self):
        train,validation,games=self.fixture(152)
        matched,_=check(train,validation,games,152);self.assertEqual([r['local_step'] for r in matched],[0,152])
        for change in ('missing_zero','wrong_grid','reward'):
            altered=copy.deepcopy(games)
            if change=='missing_zero':altered.pop(0)
            elif change=='wrong_grid':altered[-1]['matches']=64
            else:altered[-1]['reward']=.2
            with self.assertRaises(ValueError):check(train,validation,altered,152)

    def test_final64_is_separate_and_matching16_subset_must_exist(self):
        train,validation,games=self.fixture(608)
        subset={k:v for k,v in games[-1].items() if k not in ('local_step','stage')}
        games[-1].update(matches=64,seed_groups=8,net_points=1600,reward=.2,curve_grid_subset=subset)
        matched,final=check(train,validation,games,608,final=True)
        self.assertEqual(len(matched),5);self.assertEqual(final[-1]['net_points'],1600)
        self.assertEqual(matched[-1]['net_points'],800);self.assertEqual(matched[-1]['matches'],16)
        self.assertEqual(games[-1]['matches'],64)
        games[-1].pop('curve_grid_subset')
        with self.assertRaises(ValueError):check(train,validation,games,608,final=True)

    def test_incomplete_is_not_a_final_or_a_private_diagnostic_export(self):
        train,validation,games=self.fixture()
        with self.assertRaises(ValueError):check(train,validation,games,2,final=True)
        self.assertEqual(set(TRAIN_FIELDS),{'local_step','loss','ce','lr','head_lr_derived','records_seen'})
        self.assertNotIn('grad_norm',TRAIN_FIELDS);self.assertNotIn('source_counts',TRAIN_FIELDS)


if __name__=='__main__':unittest.main()
