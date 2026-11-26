"""Synthetic late native opportunity after an actual earlier ron decline."""


class FixtureAgent:
    def act(self, obs):
        from riichienv import ActionType
        menu = list(obs.legal_actions())
        wins = [a for a in menu if a.action_type in (ActionType.RON, ActionType.TSUMO)]
        if wins:
            return wins[0]
        passing = [a for a in menu if a.action_type == ActionType.PASS]
        if passing:
            return passing[0]
        discards = [a for a in menu if a.action_type == ActionType.DISCARD]
        preferred = {0: 30, 1: 18, 2: 18, 3: 29}[obs.player_id]
        return next((a for a in discards if a.tile // 4 == preferred), discards[0])


def late_ron_offer():
    from riichienv import RiichiEnv, GameRule, ActionType
    from jevmahjong.core.state import build_request
    from tools.curriculum_observer_decline_fixtures_v3 import physical_wall, discard, pass_responses
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=2026112304)
    observations = env.reset(oya=0, round_wind=1, scores=[31000, 26000, 24000, 19000],
                             wall=physical_wall('temporary_draw'))
    observations = discard(env, observations, 0, 112)
    observations = pass_responses(env, observations)
    observations = discard(env, observations, 1, 73)
    obs = observations[0]; request, actions, _ = build_request(obs)
    keys = list(request['questions']['action']['criteria'])
    accept = next(k for k, a in zip(keys, actions) if a.action_type == ActionType.RON)
    decline = next(k for k, a in zip(keys, actions) if a.action_type == ActionType.PASS)
    return env, obs, accept, decline
