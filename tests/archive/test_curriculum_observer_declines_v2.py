import unittest

from tools.curriculum_observer_decline_fixtures_v2 import fixture, verify_recovery


class ObserverDeclineNativeTests(unittest.TestCase):
    def test_real_decline_temporarily_blocks_then_allows_ron(self):
        env, obs, proof = fixture('temporary_draw')
        self.assertTrue(proof['legal_ron_declined_by_real_native_action'])
        self.assertTrue(verify_recovery(env, obs, 'temporary_draw')['subsequent_ron_offered'])

    def test_accepted_riichi_decline_stays_furiten(self):
        env, obs, proof = fixture('riichi_draw')
        self.assertTrue(proof['known_riichi_furiten'])
        self.assertFalse(verify_recovery(env, obs, 'riichi_draw')['subsequent_ron_offered'])

    def test_post_pon_retains_known_flag_until_discard(self):
        env, obs, proof = fixture('post_pon')
        self.assertTrue(proof['own_pon_before_discard'])
        self.assertTrue(verify_recovery(env, obs, 'post_pon')['subsequent_ron_offered'])


if __name__ == '__main__':
    unittest.main()
