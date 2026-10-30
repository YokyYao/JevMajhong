import copy
import unittest

from diagonal_replication import combine_rows,shard_configs


class FixedReplicationTests(unittest.TestCase):
    def setUp(self):
        self.config={'out':'runs/example','confirmation_seed_start':10,'confirmation_seeds':4,
                     'reward_scale':1000.}
        self.rows=[{'seed':seed,'seat':seat,'engine_seed':seed*(1<<32),'seed_scheme':'spaced-v1',
                    'game_mode':'4p-red-east','reward_mode':'net_points','reward_scale':1000.,
                    'score':25000,'rank':2,'reward':0.,'sampling_seed':None,
                    'trainable_decisions':0,'old_opponent':False} for seed in range(10,14) for seat in range(4)]

    def test_predeclared_shards_partition_walls_without_mutating_protocol(self):
        original=copy.deepcopy(self.config);parts=shard_configs(self.config)
        self.assertEqual(self.config,original)
        walls=[set(range(p['confirmation_seed_start'],p['confirmation_seed_start']+p['confirmation_seeds'])) for p in parts]
        self.assertFalse(walls[0]&walls[1]);self.assertEqual(set.union(*walls),set(range(10,14)))

    def test_join_requires_every_wall_and_seat_and_orders_records(self):
        combined=combine_rows([list(reversed(self.rows[:8])),list(reversed(self.rows[8:]))],self.config)
        self.assertEqual(combined,self.rows)
        with self.assertRaisesRegex(ValueError,'every fixed seed/seat'):combine_rows([self.rows[:8],self.rows[8:-1]],self.config)

    def test_duplicate_shard_cannot_replace_missing_walls(self):
        with self.assertRaisesRegex(ValueError,'every fixed seed/seat'):combine_rows([self.rows[:8],self.rows[:8]],self.config)


if __name__=='__main__':unittest.main()
