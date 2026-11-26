import unittest

from tools.curriculum_observer_rollout_v1 import finish, WinReplanner, physical_key
from tools.curriculum_outer_fixture_v1 import late_ron_offer, FixtureAgent


class ObserverOuterTests(unittest.TestCase):
    def test_hook_only_after_actual_initial_decline(self):
        from riichienv import ActionType
        env, _, _, decline = late_ron_offer(); calls = []
        def hook(current, obs, ordinary):
            calls.append((len(current.mjai_log), bool(current.missed_agari_doujun[0])))
            self.assertEqual(ordinary.action_type, ActionType.RON)
            return ordinary
        initial_prefix = len(env.mjai_log)
        result = finish(env, 0, decline, 202611230500, [FixtureAgent() for _ in range(4)],
                        observer=hook, record_events=True)
        self.assertTrue(result['full_match_completed'])
        self.assertGreaterEqual(len(calls), 1)
        self.assertGreater(calls[0][0], initial_prefix)
        self.assertFalse(calls[0][1])

    def test_accepting_hook_matches_old_complete_events(self):
        from tools.curriculum_decision_rollout import finish as original
        env, _, _, decline = late_ron_offer()
        old = original(env.clone(), 0, decline, 202611230501,
                       agents=[FixtureAgent() for _ in range(4)], record_events=True)
        new = finish(env.clone(), 0, decline, 202611230501, [FixtureAgent() for _ in range(4)],
                     observer=lambda e, o, a: a, record_events=True)
        for key in ('terminal_scores', 'rank', 'top2', 'net_from_anchor', 'future_hand_reseeds',
                    'merged_ledger_hands', 'merged_mjai_log'):
            self.assertEqual(new[key], old[key])

    def test_stale_physical_action_is_rejected_before_step(self):
        from riichienv import Action, ActionType
        env, _, _, decline = late_ron_offer()
        with self.assertRaisesRegex(ValueError, 'physically stale or illegal'):
            finish(env, 0, decline, 202611230502, [FixtureAgent() for _ in range(4)],
                   observer=lambda e, o, a: Action(ActionType.DISCARD, 136))

    def test_initial_intervention_never_calls_hook(self):
        from jevmahjong.core.state import build_request
        env, obs, accept, _ = late_ron_offer(); initial_request = build_request(obs)[0]
        def hook(current, view, ordinary):
            self.assertNotEqual(build_request(view)[0], initial_request)
            return ordinary
        result = finish(env, 0, accept, 202611230503, [FixtureAgent() for _ in range(4)],
                        observer=hook)
        self.assertTrue(result['initial_intervention_skips_observer_hook'])

    def test_budget_stops_inner_calls_without_false_partial_values(self):
        from jevmahjong.core.state import build_request
        env, obs, accept, decline = late_ron_offer(); calls = []
        def fake(e, o, c, seed, worlds):
            calls.append(seed)
            return dict(accept=accept, decline=decline,
                        decision=dict(selected=decline), full_terminal_arms=2 * worlds,
                        synthetic_unit_mock_not_actual_terminal_arms=True)
        control = WinReplanner(None, seed=1, worlds=2, max_calls=1, planner=fake)
        request, actions, _ = build_request(obs); keys = list(request['questions']['action']['criteria'])
        win = actions[keys.index(accept)]
        self.assertEqual(physical_key(control(env, obs, win)), physical_key(actions[keys.index(decline)]))
        self.assertEqual(physical_key(control(env, obs, win)), physical_key(win))
        self.assertEqual(len(calls), 1)
        self.assertEqual(control.counts['budget_fallback_wins'], 1)

    def test_nonfinite_or_unbounded_budgets_rejected(self):
        for kwargs in (dict(seed=1, max_calls=0), dict(seed=1, worlds=17),
                       dict(seed=True), dict(seed=1, max_calls=True)):
            with self.assertRaises(ValueError):
                WinReplanner(None, **kwargs)


if __name__ == '__main__':
    unittest.main()
