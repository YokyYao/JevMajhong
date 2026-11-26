import unittest

from tools.curriculum_source_event_replay_v1 import resolve_batch, exact_prefix, replay


class SourceReplayTests(unittest.TestCase):
    def test_alternate_discard_recovers_exact_events_without_mutating_source(self):
        from riichienv import RiichiEnv, GameRule, ActionType
        from jevmahjong.core.state import legal_candidates, signature
        env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=202611230701)
        observations = env.reset(); seat = next(iter(observations))
        menu = [a for a in legal_candidates(observations[seat]) if a.action_type == ActionType.DISCARD]
        chosen = menu[-1]; saved = env.clone(); saved.step({seat: chosen})
        original = (list(env.mjai_log), list(env.scores()), [list(h) for h in env.hands], list(env.wall))
        fork, _, selected, receipt = resolve_batch(env, observations, saved.mjai_log)
        self.assertEqual(fork.mjai_log, saved.mjai_log)
        self.assertEqual(signature(selected[seat]), signature(chosen))
        self.assertEqual(original, (list(env.mjai_log), list(env.scores()), [list(h) for h in env.hands], list(env.wall)))
        self.assertGreaterEqual(receipt['native_trials'], 1)

    def test_legal_ron_cannot_be_reconstructed_as_an_expert_decline(self):
        from riichienv import ActionType
        from jevmahjong.core.state import legal_candidates
        from tools.curriculum_outer_fixture_v1 import late_ron_offer
        env, _, _, _ = late_ron_offer(); observations = env.get_obs_py(env.active_players)
        passes = {p: next(a for a in legal_candidates(o) if a.action_type == ActionType.PASS)
                  for p, o in observations.items()}
        declined = env.clone(); declined.step(passes)
        with self.assertRaisesRegex(ValueError, 'No forced-win canonical batch'):
            resolve_batch(env, observations, declined.mjai_log)
        accepted = env.clone()
        selected = {p: next((a for a in legal_candidates(o) if a.action_type == ActionType.RON), passes[p])
                    for p, o in observations.items()}
        accepted.step(selected)
        actual, _, _, proof = resolve_batch(env, observations, accepted.mjai_log)
        self.assertEqual(actual.mjai_log, accepted.mjai_log)
        self.assertGreater(proof['legal_wins_taken'], 0)

    def test_wrong_prefix_is_rejected(self):
        from tools.curriculum_outer_fixture_v1 import late_ron_offer
        env, _, _, _ = late_ron_offer()
        self.assertFalse(exact_prefix(env.mjai_log, []))
        with self.assertRaisesRegex(ValueError, 'prefix'):
            resolve_batch(env, env.get_obs_py(env.active_players), [])

    def test_bounded_replay_contract(self):
        for budget in (0, True, 30001):
            with self.assertRaisesRegex(ValueError, 'Bounded'):
                replay(None, None, [], max_steps=budget)


if __name__ == '__main__': unittest.main()
