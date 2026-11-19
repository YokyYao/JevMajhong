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



    def test_top_two_objective_does_not_prefer_locked_third_over_a_live_comeback(self):
        from jevmahjong.endgame.context import placement_objective
        locked_third = placement_objective(3)
        one_second_three_fourths = (placement_objective(2) + 3 * placement_objective(4)) / 4
        self.assertGreater(one_second_three_fourths, locked_third)
        self.assertGreater(placement_objective(3), placement_objective(4))
        self.assertGreater(placement_objective(1), placement_objective(2))
        self.assertGreater(placement_objective(2), one_second_three_fourths)


class EndgameReferenceTests(unittest.TestCase):
    def test_original_policy_distribution_restricts_only_to_known_support(self):
        from jevmahjong.endgame.reference import restrict_distribution
        q = restrict_distribution({'bad': .7, 'good': .2, 'kan': .1}, ['good', 'kan'])
        self.assertEqual(q['bad'], 0)
        self.assertAlmostEqual(q['good'], 2 / 3)
        self.assertAlmostEqual(q['kan'], 1 / 3)

    def test_win_tilt_can_learn_a_comeback_but_keeps_an_inferior_continuation_out(self):
        from jevmahjong.endgame.reference import improvement_target
        p = {'win': 1., 'continue': 0.}
        positive = improvement_target(p, {'win': 0., 'continue': .2125}, .05, .1)
        self.assertGreater(positive['continue'], .7)
        self.assertEqual(improvement_target(p, {'win': 0., 'continue': -.05}, .05, .1), p)

    def test_publicly_fixed_wins_become_reference_replay_without_future_selection(self):
        from unittest.mock import patch
        from jevmahjong.endgame.reference import apply_reference
        def row(key, enough):
            return {'state': {'endgame_context': {'late_match_window': True, 'placement': 3,
                        'current_win_settlement': {'status': 'public_single_winner_baseline_scenarios',
                            'winner_is_dealer': False, 'top2_in_any_baseline_scenario': enough}}},
                    '_meta': {'id': key, 'bucket': 'win', 'target_basis': 'complete_terminal_MC_soft_target',
                              'expected_terminal_utilities': {'win': 0., 'continue': .2125}},
                    'questions': {'action': {'src': 'endgame_terminal', 'label': 'continue',
                        'criteria': {'win': {'type': 'hora'}, 'continue': {'type': 'none'}},
                        'target': {'win': 0., 'continue': 1.}}}}
        fixed, active = row('fixed', True), row('active', False)
        cache = {'identity_sha256': 'test', 'input_version': 'endgame-public-v3',
                 'rows': {k: {'probabilities': {'win': 1., 'continue': 0.}} for k in ('fixed', 'active')}}
        cfg = {'reference_mode': 'released_original_policy', 'reference_temperature': .1,
               'reference_win_temperature': .05, 'reference_exploration': .1}
        with patch('jevmahjong.endgame.reference.load_reference', return_value=cache):
            apply_reference(cfg, {'train': [fixed, active], 'validation': []})
        self.assertEqual(fixed['questions']['action']['src'], 'replay_reference')
        self.assertEqual(fixed['questions']['action']['target'], {'win': 1., 'continue': 0.})
        self.assertGreater(active['questions']['action']['target']['continue'], .7)

    def test_equal_returns_keep_frozen_policy_instead_of_uniform_targets(self):
        from jevmahjong.endgame.reference import improvement_target
        p = {"a": .9, "b": .1}
        self.assertEqual(improvement_target(p, {"a": 1., "b": 1.}), p)

    def test_lower_probability_comeback_can_overcome_a_reference_preference(self):
        from jevmahjong.endgame.reference import improvement_target
        p = {"win": .99, "continue": .01}
        q = improvement_target(p, {"win": 0., "continue": .5}, .1, .1)
        self.assertGreater(q["continue"], q["win"])
        self.assertAlmostEqual(sum(q.values()), 1.)

    def test_exploration_does_not_worsen_expected_labeled_utility(self):
        from jevmahjong.endgame.reference import improvement_target
        p = {"a": 1., "b": 0.}
        values = {"a": .99, "b": .98}
        q = improvement_target(p, values, .1, .1)
        self.assertGreaterEqual(sum(q[k] * values[k] for k in p), .99 - 1e-12)

    def test_teacher_input_excludes_targets_and_future_metadata(self):
        from jevmahjong.endgame.reference import inference_request
        record = {"state": {"scores": [25000] * 4}, "_meta": {"future": [1, 2]},
                  "questions": {"action": {"type": "choice", "criteria": {"a": {}},
                    "instructions": "test", "label": "a", "target": {"a": 1.}, "src": "private"}}}
        payload = inference_request(record)
        self.assertNotIn("_meta", payload)
        self.assertEqual(set(payload["questions"]["action"]), {"type", "instructions", "criteria"})

    def test_reference_replay_has_registered_mass_without_reintroducing_hard_labels(self):
        from jevmahjong.endgame.prepare import loss_recipe
        counts = {"replay_reference": 512, "endgame_terminal_win": 94,
                  "endgame_terminal_third": 29, "endgame_terminal_fourth": 23,
                  "endgame_terminal_hold": 23}
        mass, weights = loss_recipe(counts, .2, True, True)
        self.assertAlmostEqual(mass["replay_reference"], .8)
        self.assertAlmostEqual(sum(mass.values()), 1.)
        self.assertLess(max(weights.values()), 8)


class EndgameEvaluationTests(unittest.TestCase):
    def test_report_uses_actual_points_and_explicit_dealin_denominator(self):
        from jevmahjong.endgame.evaluate import point_and_risk_summary
        rows = [{"seed": 123,
                 "baseline": {"net_points_from_window": -3000, "own_wins_after_window": 0,
                              "dealins_after_window": 1, "hands_ended_after_window": 2},
                 "candidate": {"net_points_from_window": 1000, "own_wins_after_window": 1,
                               "dealins_after_window": 0, "hands_ended_after_window": 1}}]
        report = point_and_risk_summary(rows)
        self.assertEqual(report["source_group_mean_net_point_gain"], 4000)
        self.assertEqual(report["source_group_bootstrap_net_point_gain_ci95"], [4000, 4000])
        self.assertEqual(report["per_policy"]["baseline"]["dealin_events_per_completed_hand"], .5)
        self.assertEqual(report["per_policy"]["candidate"]["windows_with_any_dealin_fraction"], 0)


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


class EndgameLossMixTests(unittest.TestCase):
    def test_more_win_positions_do_not_take_mass_from_pursuit_and_guard(self):
        from jevmahjong.endgame.prepare import loss_recipe
        counts = {"replay_expert": 384, "replay_masked": 128,
                  "endgame_terminal_win": 94, "endgame_terminal_third": 29,
                  "endgame_terminal_fourth": 23, "endgame_terminal_hold": 23}
        mass, weights = loss_recipe(counts, .2, True)
        self.assertAlmostEqual(mass["replay_expert"], .6)
        self.assertAlmostEqual(mass["replay_masked"], .2)
        for source in counts:
            self.assertAlmostEqual(counts[source] * weights[source] / sum(counts.values()), mass[source])
        self.assertAlmostEqual(mass["endgame_terminal_win"], mass["endgame_terminal_third"])

    def test_missing_stratum_and_unbounded_weights_are_rejected(self):
        from jevmahjong.endgame.prepare import loss_recipe
        with self.assertRaises(ValueError):
            loss_recipe({"endgame_terminal": 100, "replay_expert": 512}, .2)
        with self.assertRaises(ValueError):
            loss_recipe({"endgame_terminal": 1, "replay_expert": 384, "replay_masked": 128}, .5)


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
             patch("jevmahjong.endgame.policy.model_request", side_effect=lambda x: x):
            chosen = ModelAgent(model, placement_aware=True).act(obs)
        self.assertIs(chosen, decline)
        model.system_one.assert_called_once()
        model.reset_mock()
        self.assertIs(ModelAgent(model, placement_aware=False).act(obs), win)
        model.system_one.assert_not_called()
        self.assertIs(ModelAgent(model, True, force_all_wins=True).act(obs), win)
        model.system_one.assert_not_called()
        payload["state"].update(seat=0, scores=[32000, 25000, 22000, 21000])
        with patch("jevmahjong.endgame.policy.build_request", return_value=(payload, [win, decline], [{}, {}])):
            self.assertIs(ModelAgent(model, True, protect_leading_wins=True).act(obs), win)
        model.system_one.assert_not_called()

    def test_optional_ordinary_guard_preserves_original_discard_eligibility(self):
        import json, sys
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from riichienv import ActionType
        from jevmahjong.endgame.policy import ModelAgent
        actions = [SimpleNamespace(action_type=ActionType.DISCARD,
                   to_mjai=lambda: json.dumps({"type": "dahai"})) for _ in range(2)]
        obs = SimpleNamespace(round_wind=0, kyoku_index=3)
        payload = {"state": {}, "questions": {"action": {"criteria": {"a0": {}, "a1": {}}}}}
        model = Mock()
        model.system_one.return_value = {"answers": {"action": {"probabilities": {"a0": .9, "a1": .1}}}}
        api = SimpleNamespace(validate_distribution=lambda *args: ([.9, .1], {}))
        with patch.dict(sys.modules, {"jevany": SimpleNamespace(__path__=[]), "jevany.api": api}), \
             patch("jevmahjong.endgame.policy.legal_candidates", return_value=actions), \
             patch("jevmahjong.endgame.policy.build_request", return_value=(payload, actions, [{"sh": 2}, {"sh": 1}])), \
             patch("jevmahjong.endgame.policy.model_request", side_effect=lambda x: x), \
             patch("jevmahjong.endgame.policy.guarded_indices", return_value=[1]):
            self.assertIs(ModelAgent(model, True).act(obs), actions[0])
            self.assertIs(ModelAgent(model, True, ordinary_guard=True).act(obs), actions[1])



if __name__ == "__main__":
    unittest.main()

