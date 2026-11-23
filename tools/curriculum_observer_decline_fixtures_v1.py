"""Real legal-ron declines for the pinned native world constructor.

Scripted physical walls are integration fixtures, never training examples.
No missed-win flag is injected: each is caused by an actual legal PASS.
"""
import random

KINDS = ('temporary_draw', 'riichi_draw', 'temporary_tsumo', 'riichi_tsumo', 'post_pon')
SEED = 2026112301


def physical_wall(kind):
    if kind not in KINDS:
        raise ValueError('An explicitly registered decline fixture is required')
    own = [108, 109, 110, 0, 4, 8, 12, 16, 20, 36, 40, 44, 72]
    draws = ([112, 73, 111, 74] if kind == 'post_pon' else
             [112, 73, 74, 116, 75] if kind.endswith('tsumo') else
             [112, 73, 74, 116, 120, 75])
    pool = sorted(set(range(136)) - set(own) - set(draws))
    random.Random(SEED + KINDS.index(kind)).shuffle(pool)
    hands = {0: own, 1: pool[:13], 2: pool[13:26], 3: pool[26:39]}
    queues = {p: list(h) for p, h in hands.items()}
    dealt = []
    for _ in range(3):
        for player in range(4):
            dealt.extend(queues[player][:4]); del queues[player][:4]
    for player in range(4):
        dealt.append(queues[player][0])
    wall = [*dealt, *draws, *pool[39:]]
    if len(wall) != 136 or set(wall) != set(range(136)):
        raise ValueError('Fixture wall does not contain each physical tile exactly once')
    return wall


def offered(env, player, kind):
    return any(a.action_type == kind for a in env.current_claims.get(player, []))


def discard(env, observations, player, tile, reach=False):
    from riichienv import ActionType, Phase
    from jevmahjong.core.state import legal_candidates
    if set(observations) != {player} or env.phase != Phase.WaitAct:
        raise ValueError('A scripted draw/discard decision was skipped or reordered')
    kind = ActionType.RIICHI if reach else ActionType.DISCARD
    action = next((a for a in legal_candidates(observations[player])
                   if a.action_type == kind and a.tile == tile), None)
    if action is None:
        raise ValueError('Scripted physical discard/reach is not a legal native action')
    return env.step({player: action})


def pass_responses(env, observations, allow_own_ron=False):
    from riichienv import ActionType, Phase
    from jevmahjong.core.state import legal_candidates
    marker = (env.round_wind, env.kyoku_idx, env.honba)
    while env.phase == Phase.WaitResponse:
        actions = {}
        for player, obs in sorted(observations.items()):
            menu = legal_candidates(obs)
            wins = [a for a in menu if a.action_type == ActionType.RON]
            if wins and not (player == 0 and allow_own_ron):
                raise ValueError('Fixture would unknowingly decline another legal ron')
            actions[player] = next(a for a in menu if a.action_type == ActionType.PASS)
        observations = env.step(actions)
        if env.done() or (env.round_wind, env.kyoku_idx, env.honba) != marker:
            raise ValueError('Decline fixture silently settled/changed the hand')
    return observations


def accept_ron_control(env):
    from riichienv import ActionType
    from jevmahjong.core.state import legal_candidates
    from tools.curriculum_endgame_ledger import ledger
    control = env.clone(); before = len(control.mjai_log)
    views = control.get_obs_py(list(control.active_players))
    actions = {p: next(a for a in legal_candidates(o)
                       if a.action_type == (ActionType.RON if p == 0 else ActionType.PASS))
               for p, o in views.items()}
    control.step(actions)
    if not any(e['type'] == 'hora' and e['actor'] == 0 for e in control.mjai_log[before:]):
        raise ValueError('Positive legal-ron control failed to execute the win')
    ledger(control.mjai_log, control.scores())


def cleared_flags_ron_control(env):
    """Diagnostic only: show a blocked ron is caused by the known flags."""
    from riichienv import ActionType
    from tools.curriculum_native_worlds import refresh_discard_claims
    control = env.clone()
    for field in ('missed_agari_doujun', 'missed_agari_riichi'):
        flags = list(getattr(control, field)); flags[0] = False
        setattr(control, field, flags)
    trigger = control.mjai_log[-1]
    refresh_discard_claims(control, trigger, control.last_discard[1])
    if not offered(control, 0, ActionType.RON):
        raise ValueError('Clearing only the known observer flags did not restore ron')


def fixture(kind):
    from importlib.metadata import version
    from riichienv import RiichiEnv, GameRule, ActionType
    from jevmahjong.core.state import legal_candidates
    if version('riichienv') != '0.4.11':
        raise ValueError('Decline controls require the pinned native engine')
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(),
                    seed=SEED + KINDS.index(kind))
    observations = env.reset(oya=0, wall=physical_wall(kind))
    reached = kind.startswith('riichi')
    observations = discard(env, observations, 0, 112, reach=reached)
    observations = pass_responses(env, observations)
    if bool(env.riichi_declared[0]) != reached:
        raise ValueError('Actual riichi declaration/acceptance differs from the fixture')
    observations = discard(env, observations, 1, 73)
    if not offered(env, 0, ActionType.RON):
        raise ValueError('The first scripted opportunity is not an actual legal ron')
    accept_ron_control(env)
    observations = pass_responses(env, observations, allow_own_ron=True)
    if not env.missed_agari_doujun[0] or bool(env.missed_agari_riichi[0]) != reached:
        raise ValueError('Actual ron decline did not establish the expected known flags')
    if kind == 'post_pon':
        observations = discard(env, observations, 2, 111)
        if offered(env, 0, ActionType.RON) or not offered(env, 0, ActionType.PON):
            raise ValueError('The post-decline pon opportunity differs')
        actions = {}
        for player, obs in sorted(observations.items()):
            menu = legal_candidates(obs)
            if any(a.action_type == ActionType.RON for a in menu):
                raise ValueError('Post-pon setup would decline an unregistered legal win')
            actions[player] = next(a for a in menu
                                   if a.action_type == (ActionType.PON if player == 0 else ActionType.PASS))
        observations = env.step(actions)
        if set(observations) != {0} or env.mjai_log[-1]['type'] != 'pon':
            raise ValueError('A real post-pon observer decision is required')
    else:
        observations = discard(env, observations, 2, 74)
        if offered(env, 0, ActionType.RON):
            raise ValueError('Missed ron illegally reappeared before the observer turn')
        cleared_flags_ron_control(env)
        observations = pass_responses(env, observations)
        observations = discard(env, observations, 3, 116)
        observations = pass_responses(env, observations)
    obs = observations[0]
    if not env.missed_agari_doujun[0] or bool(env.missed_agari_riichi[0]) != reached:
        raise ValueError('Known missed-win flags did not survive until the observer decision')
    tsumo = any(a.action_type == ActionType.TSUMO for a in legal_candidates(obs))
    if tsumo != kind.endswith('tsumo'):
        raise ValueError('Furiten fixture has a different legal-tsumo status')
    return env, obs, dict(kind=kind, legal_ron_declined_by_real_native_action=True,
                         accepted_positive_control_ledger=True,
                         known_temporary_furiten=True, known_riichi_furiten=reached,
                         legal_tsumo_despite_furiten=tsumo,
                         own_pon_before_discard=kind == 'post_pon',
                         synthetic_physical_wall_not_training_source=True)


def verify_recovery(env, obs, kind):
    """Verify the pinned engine's actual discard-based temporary reset."""
    from riichienv import ActionType
    tile = next(t for t in obs.hand if t // 4 == 27) if kind == 'post_pon' else 120
    observations = discard(env, {0: obs}, 0, tile)
    if env.missed_agari_doujun[0] or bool(env.missed_agari_riichi[0]) != kind.startswith('riichi'):
        raise ValueError('Temporary reset or permanent riichi furiten differs after discard')
    observations = pass_responses(env, observations)
    observations = discard(env, observations, 1, 74 if kind == 'post_pon' else 75)
    ron = offered(env, 0, ActionType.RON)
    if ron == kind.startswith('riichi'):
        raise ValueError('Later ron availability ignores the actual known decline state')
    if ron:
        accept_ron_control(env)
    else:
        cleared_flags_ron_control(env)
    return dict(temporary_flag_clears_on_own_discard=True,
                permanent_flag_survives_own_discard=kind.startswith('riichi'),
                subsequent_ron_offered=ron, later_ron_uses_original_fixture_world=True)
