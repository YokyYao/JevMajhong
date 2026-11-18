import unittest
from jevmahjong.endgame.context import (
    placements, public_context, settled_placement, target_from_utilities, terminal_utility)


class EndgameContextTests(unittest.TestCase):
    def state(self, **changes):
        return dict(seat=2, scores=[32000, 25000, 22000, 21000],
                    round_wind=0, hand_number=4, **changes)

    def test_direct_hit_changes_both_sides_of_gap(self):
        scores = [32000, 25000, 22000, 21000]
        self.assertEqual(settled_placement(scores, [-1000, 0, 1000, 0], 2), 3)
        self.assertEqual(settled_placement(scores, [0, -1000, 1000, 0], 2), 3)
        self.assertEqual(settled_placement(scores, [0, -2000, 2000, 0], 2), 2)

    def test_tie_break_uses_original_seat(self):
        self.assertEqual(placements([25000] * 4), [1, 2, 3, 4])
        self.assertEqual(settled_placement([32000, 25000, 22000, 21000],
                                         [-3000, 0, 3000, 0], 2), 3)

    def test_third_targets_second(self):
        context = public_context(self.state())
        self.assertEqual((context["placement"], context["target_seat"], context["gap_to_target"]), (3, 1, 3000))
        self.assertTrue(context["late_match_window"])
        self.assertTrue(context["late_window_is_not_guaranteed_terminal"])

    def test_fourth_can_target_escape(self):
        state = self.state()
        state["seat"] = 3
        context = public_context(state)
        self.assertEqual((context["target_seat"], context["gap_to_target"]), (2, 1000))

    def test_early_hand_is_not_marked_late(self):
        state = self.state()
        state["hand_number"] = 2
        self.assertFalse(public_context(state)["late_match_window"])
        state["round_wind"] = 1
        self.assertTrue(public_context(state)["late_match_window"])

    def test_utility_preserves_rank_order(self):
        self.assertEqual([terminal_utility(r) for r in (1, 2, 3, 4)], [1., .5, -.5, -1.])
        target = target_from_utilities([-.5, .5, -1.])
        self.assertAlmostEqual(sum(target), 1.)
        self.assertGreater(target[1], target[0])
        self.assertGreater(target[0], target[2])

    def test_invalid_state_is_rejected(self):
        state = self.state()
        state["hand_number"] = None
        with self.assertRaises(ValueError):
            public_context(state)



    def test_neutral_windows_do_not_teach_uniform_passing(self):
        from jevmahjong.endgame.context import terminal_training_target
        criteria = {"win": {"type": "hora"}, "pass": {"type": "none"}}
        target, basis = terminal_training_target(criteria, [-.5, -.5])
        self.assertEqual(target, {"win": 1., "pass": 0.})
        target, basis = terminal_training_target(criteria, [-.5, .5])
        self.assertGreater(target["pass"], target["win"])
        target, basis = terminal_training_target(
            {"a": {"type": "dahai"}, "b": {"type": "dahai"}}, [.5, .5])
        self.assertIsNone(target)


class EndgameLedgerTests(unittest.TestCase):
    def test_records_deposits_settlement_and_next_hand(self):
        from jevmahjong.endgame.ledger import hand_score_history
        start = {"type": "start_kyoku", "bakaze": "E", "kyoku": 4,
                 "honba": 0, "oya": 3, "scores": [25000] * 4, "kyotaku": 0}
        scores = [24000, 28000, 23000, 25000]
        log = [start, {"type": "reach_accepted", "actor": 0},
               {"type": "hora", "actor": 1, "target": 2,
                "deltas": [0, 3000, -2000, 0]}, {"type": "end_kyoku"},
               {"type": "end_game"}]
        result = hand_score_history(log)
        self.assertEqual(result["final_scores"], scores)
        self.assertEqual(result["hands"][0]["scores_after"], scores)
        self.assertEqual(result["hands"][0]["sticks_after"], 0)

    def test_terminal_pool_is_awarded_without_fabricating_a_hand(self):
        from jevmahjong.endgame.ledger import hand_score_history
        log = [{"type": "start_kyoku", "bakaze": "E", "kyoku": 4,
                "honba": 0, "oya": 3, "scores": [25000] * 4, "kyotaku": 0},
               {"type": "reach_accepted", "actor": 3},
               {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
               {"type": "end_kyoku"}, {"type": "end_game"}]
        result = hand_score_history(log)
        self.assertEqual(result["final_scores"], [26000, 25000, 25000, 24000])
        self.assertEqual(len(result["hands"]), 1)


class EndgameWinPathTests(unittest.TestCase):
    def test_late_win_and_pass_reach_the_model(self):
        import json
        from types import SimpleNamespace
        import sys
        from unittest.mock import Mock, patch
        from riichienv import ActionType
        from jevmahjong.endgame.policy import ModelAgent

        class Action:
            def __init__(self, kind, name):
                self.action_type, self.name = kind, name
            def to_mjai(self):
                return json.dumps({"type": self.name})
        win, decline = Action(ActionType.RON, "hora"), Action(ActionType.PASS, "none")
        obs = SimpleNamespace(round_wind=0, kyoku_index=3, legal_actions=lambda: [win, decline])
        payload = {"state": {}, "questions": {"action": {
            "type": "choice", "criteria": {"a0": {"type": "hora"}, "a1": {"type": "none"}}}}}
        model = Mock()
        model.system_one.return_value = {"answers": {"action": {
            "probabilities": {"a0": 0.1, "a1": 0.9}}}}
        validator = Mock(return_value=([0.1, 0.9], {}))
        api = SimpleNamespace(validate_distribution=validator)
        with patch.dict(sys.modules, {"jevany": SimpleNamespace(__path__=[]), "jevany.api": api}), \
             patch("jevmahjong.endgame.policy.build_request", return_value=(payload, [win, decline], [{}, {}])), \
             patch("jevmahjong.endgame.policy.serving_request", side_effect=lambda x: x):
            chosen = ModelAgent(model, placement_aware=True).act(obs)
        self.assertIs(chosen, decline)
        model.system_one.assert_called_once()
        model.reset_mock()
        self.assertIs(ModelAgent(model, placement_aware=False).act(obs), win)
        model.system_one.assert_not_called()



if __name__ == "__main__":
    unittest.main()

