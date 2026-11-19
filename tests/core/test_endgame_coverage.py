import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from riichienv import ActionType
from jevmahjong.endgame.coverage import DamaAgent, public_stratum, select_cells


class SourceCoverageTests(unittest.TestCase):
    def test_source_dama_removes_reach_but_keeps_legal_tsumo(self):
        reach = SimpleNamespace(action_type=ActionType.RIICHI)
        win = SimpleNamespace(action_type=ActionType.TSUMO)
        discard = SimpleNamespace(action_type=ActionType.DISCARD)
        obs = SimpleNamespace(player_id=0, to_dict=lambda: {'riichi_declared': [False] * 4})
        for menu, chosen in (([reach, discard], discard), ([reach, win, discard], win)):
            with patch('jevmahjong.endgame.coverage.legal_candidates', return_value=menu), \
                 patch('jevmahjong.endgame.coverage.describe', return_value=({}, {}, {})), \
                 patch('jevmahjong.endgame.coverage.features', return_value=[{}] * (len(menu) - 1)):
                # Real heuristic prioritizes the legal win; otherwise only discard remains.
                self.assertIs(DamaAgent().act(obs), chosen)

    def test_coverage_uses_gap_to_second_even_for_fourth(self):
        payload = {'state': {'seat': 3, 'scores': [32000, 26000, 22000, 20000],
            'riichi': [], 'melds': [[], [], [], []], 'endgame_context': {
                'late_match_window': True, 'placement': 4,
                'current_win_settlement': {'status': 'public_single_winner_baseline_scenarios',
                    'winner_is_dealer': False, 'top2_in_any_baseline_scenario': False,
                    'maximum_baseline_scenario': {'deltas': [-1000, 0, 0, 1000]}}}}}
        self.assertEqual(public_stratum(payload)['gap_to_second'], 6000)
        self.assertTrue(public_stratum(payload)['target'])
        changed = copy.deepcopy(payload); changed['state']['riichi'] = [3]
        self.assertFalse(public_stratum(changed)['target'])
        changed = copy.deepcopy(payload)
        changed['state']['endgame_context']['current_win_settlement']['top2_in_any_baseline_scenario'] = True
        self.assertIsNone(public_stratum(changed))

    def test_public_hash_selection_ignores_hypothetical_future_outcomes(self):
        rows = [{'selection_key': k, 'request_sha256': k, 'coverage_stratum': {'cell': cell},
                 'future_outcome': future} for k, cell, future in
                [('b', 'near', 1), ('a', 'near', -1), ('c', 'far', 1)]]
        self.assertEqual([r['selection_key'] for r in select_cells(rows, 1)], ['a', 'c'])
        for row in rows:
            row['future_outcome'] *= -1
        self.assertEqual([r['selection_key'] for r in select_cells(rows, 1)], ['a', 'c'])


if __name__ == '__main__':
    unittest.main()
