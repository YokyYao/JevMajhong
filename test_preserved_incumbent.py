import copy
import unittest
from incremental_score import ORIGINAL_HEAD
from scripts.prepare_preserved_incumbent import study_config,validate_frozen_runtime


class PreservedIncumbentTests(unittest.TestCase):
    def test_only_actor_weights_may_differ(self):
        original=dict(base='model',base_revision='pinned',special_embeddings=False,head={'q':0})
        candidate={**original,'head':{'q':1}}
        validate_frozen_runtime(candidate,original)
        for key,value in [('base_revision','changed'),('special_embeddings',True)]:
            with self.assertRaisesRegex(ValueError,'runtime metadata differs'):
                validate_frozen_runtime({**candidate,key:value},original)

    def test_followup_has_fresh_rule_without_rewriting_original_config(self):
        original=dict(reference='old_training_anchor',target_net_points_gain=250,target_mean_net_points=1454.296875,
                      confirmation_seed_start=3802000,confirmation_seeds=128,
                      game_mode='4p-red-east',reward_mode='net_points',seed_scheme='spaced-v1',
                      final_benchmark_reference='original2000',final_benchmark_head_sha256=ORIGINAL_HEAD)
        previous=copy.deepcopy(original)
        new=study_config(original,4202000,128)
        self.assertEqual(original,previous)
        self.assertEqual(new['reference'],'original2000')
        self.assertEqual(new['confirmation_seed_start'],4202000)
        self.assertEqual(new['target_net_points_gain'],0.)
        self.assertEqual(new['target_mean_net_points'],0.)
        self.assertEqual(new['desired_net_points_gain'],500)
        self.assertEqual(new['confirmation_min_ci95_lower'],0)
        for start,count in [(4202000,32),(-1,128),(2**32-64,128)]:
            with self.assertRaises(ValueError):study_config(original,start,count)


if __name__=='__main__':unittest.main()
