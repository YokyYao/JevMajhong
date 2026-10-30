import copy
import unittest

from preserved_development import compare_preserved


class PreservedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.config={'development_seed_start':10,'development_seeds':2,'reward_scale':1000.,
                     'target_mean_net_points':0,'target_net_points_gain':0}

    def rows(self,points):
        return [{'seed':s,'seat':seat,'engine_seed':s*(1<<32),'seed_scheme':'spaced-v1',
                 'game_mode':'4p-red-east','reward_mode':'net_points','reward_scale':1000.,
                 'score':points,'rank':2,'reward':(points-25000)/1000.,'sampling_seed':None,
                 'trainable_decisions':0,'old_opponent':False} for s in (10,11) for seat in range(4)]

    def test_incremental_gain_is_against_preserved_not_original(self):
        variants=[{'variant':1,'checkpoint':'new','rows':self.rows(25150)}]
        before=copy.deepcopy(variants)
        report=compare_preserved(self.rows(25000),self.rows(25100),variants,self.config)
        self.assertEqual(report['preserved_gain_vs_original'],100.)
        self.assertEqual(report['variants'][0]['gain_vs_preserved'],50.)
        self.assertEqual(variants,before);self.assertIs(report['main_selection_changed'],False)

    def test_positive_against_original_can_still_regress_against_preserved(self):
        report=compare_preserved(self.rows(25000),self.rows(25100),
                                 [{'variant':1,'checkpoint':'new','rows':self.rows(25050)}],self.config)
        self.assertEqual(report['variants'][0]['gain_vs_preserved'],-50.)
        self.assertIs(report['primary_confirmation_rule_changed'],False)

    def test_mismatched_walls_cannot_be_compared(self):
        bad=self.rows(25100);bad[0]['seed']=99
        with self.assertRaisesRegex(ValueError,'every fixed seed/seat'):
            compare_preserved(self.rows(25000),bad,[],self.config)


if __name__=='__main__':unittest.main()
