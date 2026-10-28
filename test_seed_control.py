import hashlib
import json
import unittest
from riichienv import RiichiEnv, GameRule
from seed_control import engine_seed, SEED_STRIDE, MAX_ENVIRONMENT_STEPS


def initial_deal(env):
    env.reset()
    observations = env.get_obs_py(list(range(4)))
    data = {'hands': [sorted(list(observations[p].hand)) for p in range(4)],
            'dora': observations[0].to_dict()['dora_indicators']}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class SeedIsolationTests(unittest.TestCase):
    def test_runtime_reproduces_raw_neighbor_wall_overlap(self):
        a = RiichiEnv(seed=1301000, rule=GameRule.default_mjsoul())
        initial_deal(a)
        second = initial_deal(a)
        b = RiichiEnv(seed=1301001, rule=GameRule.default_mjsoul())
        self.assertEqual(second, initial_deal(b))

    def test_spaced_episode_streams_do_not_share_adjacent_hand(self):
        a = RiichiEnv(seed=engine_seed(1302000), rule=GameRule.default_mjsoul())
        initial_deal(a)
        second = initial_deal(a)
        b = RiichiEnv(seed=engine_seed(1302001), rule=GameRule.default_mjsoul())
        self.assertNotEqual(second, initial_deal(b))
        self.assertGreater(SEED_STRIDE, MAX_ENVIRONMENT_STEPS+3)
        self.assertGreater(engine_seed(1302001)-engine_seed(1302000), MAX_ENVIRONMENT_STEPS+3)

    def test_mapping_is_stable_and_rejects_overflow(self):
        self.assertEqual(engine_seed(3), 3 << 32)
        self.assertEqual(engine_seed(3, 'raw'), 3)
        with self.assertRaises(ValueError):
            engine_seed(1 << 32)


if __name__ == '__main__':
    unittest.main()
