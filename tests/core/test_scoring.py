import unittest

from jevmahjong.core.scoring import independent_win_score, settlement_ledger


class ScoringAuditTests(unittest.TestCase):
    def test_final_unclaimed_pool_is_awarded_to_leader_with_seat_tiebreak(self):
        log = [{'type': 'start_kyoku', 'scores': [25000] * 4, 'kyotaku': 0},
               {'type': 'reach_accepted', 'actor': 0}, {'type': 'end_game'}]
        result = settlement_ledger(log, [24000, 26000, 25000, 25000])
        self.assertEqual(result['components'][1]['final_unclaimed_pool'], 1000)
        with self.assertRaisesRegex(ValueError, 'Terminal ledger mismatch'):
            settlement_ledger(log, [25000] * 4)

    def test_riichi_pool_carries_across_draw_and_is_paid_once(self):
        log = [
            {'type': 'start_kyoku', 'scores': [25000] * 4, 'kyotaku': 0},
            {'type': 'reach_accepted', 'actor': 0},
            {'type': 'ryukyoku', 'deltas': [-1000, -1000, 3000, -1000]},
            {'type': 'start_kyoku', 'scores': [23000, 24000, 28000, 24000], 'kyotaku': 1},
            {'type': 'hora', 'actor': 2, 'target': 1, 'deltas': [0, -1000, 2000, 0]},
        ]
        result = settlement_ledger(log, [23000, 23000, 30000, 24000])
        self.assertEqual(result['unclaimed_riichi_sticks'], 0)
        self.assertEqual(result['components'][0]['riichi_deposits'], -1000)
        with self.assertRaisesRegex(ValueError, 'Terminal ledger mismatch'):
            settlement_ledger(log, [23000, 23000, 31000, 24000])

    def test_independent_open_yakuhai_score_and_honba(self):
        # Open white-dragon pon; 234m 456p 678s 22p, ron on 4m.
        history = [{'type': 'start_kyoku'}, {'type': 'dahai', 'actor': 1}]
        obs = {'player_id': 1, 'hands': [[], [4, 8, 48, 53, 56, 92, 96, 100, 40, 41], [], []],
               'melds': [[], [{'meld_type': 'Pon', 'tiles': [124, 125, 126], 'opened': True}], [], []],
               'riichi_declared': [False] * 4, 'oya': 0, 'round_wind': 0,
               'honba': 2, 'riichi_sticks': 0, 'dora_indicators': [32]}
        event = {'actor': 1, 'target': 2, 'deltas': [0, 1600, -1600, 0]}
        result = independent_win_score(history, event, {'observation': obs, 'win_tile': 12})
        self.assertEqual(result['status'], 'passed')
        self.assertEqual((result['han'], result['fu'], result['open']), (1, 30, True))
        event['deltas'] = [0, 2600, -2600, 0]
        self.assertEqual(independent_win_score(history, event, {'observation': obs, 'win_tile': 12})['status'], 'mismatch')


if __name__ == '__main__': unittest.main()
