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
    own = ([40, 41, 42, 44, 48, 53, 92, 96, 124, 125, 128, 129, 130]
           if kind == 'post_pon' else
           [108, 109, 110, 0, 4, 8, 12, 16, 20, 36, 40, 44, 72])
    draws = ([112, 89, 126, 100] if kind == 'post_pon' else
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
    if set(observations) != {player} or env.phase != Phase.WaitAct:
        raise ValueError('A scripted draw/discard decision was skipped or reordered')
    if player != 0 and any(a.action_type == ActionType.TSUMO
                           for a in observations[player].legal_actions()):
        raise ValueError('A fixture opponent would decline a legal self-draw win')
    if reach:
        action = next(a for a in observations[player].legal_actions()
                      if a.action_type == ActionType.RIICHI)
        observations = env.step({player: action})
        if set(observations) != {player} or env.phase != Phase.WaitAct:
            raise ValueError('Untiled reach did not require its own tenpai discard')
    # The normalized model menu merges equal physical copies. Script the real
    # native physical action instead; model intervention keys remain normalized.
    action = next((a for a in observations[player].legal_actions()
                   if a.action_type == ActionType.DISCARD and a.tile == tile), None)
    if action is None:
        raise ValueError('Scripted physical discard is not a legal native action')
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


def cleared_temporary_ron_control(env, observations, actor, tile, expect_ron):
    """Before-discard diagnostic; the native engine may skip empty responses."""
    from riichienv import ActionType
    control = env.clone()
    flags = list(control.missed_agari_doujun); flags[0] = False
    control.missed_agari_doujun = flags
    views = control.get_obs_py(list(control.active_players))
    discard(control, views, actor, tile)
    if offered(control, 0, ActionType.RON) != expect_ron:
        raise ValueError('Native legal-ron behavior after temporary-only clearing differs')


def planned_discard(obs, kind):
    from riichienv import ActionType
    tile_type = 10 if kind == 'post_pon' else 18 if kind.endswith('tsumo') else 30
    choices = [a.tile for a in obs.legal_actions()
               if a.action_type == ActionType.DISCARD and a.tile // 4 == tile_type]
    if not choices:
        raise ValueError('The registered followup discard is not legal')
    return min(choices)


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
    observations = discard(env, observations, 1, 89 if kind == 'post_pon' else 73)
    if not offered(env, 0, ActionType.RON):
        raise ValueError('The first scripted opportunity is not an actual legal ron')
    accept_ron_control(env)
    observations = pass_responses(env, observations, allow_own_ron=True)
    if not env.missed_agari_doujun[0] or bool(env.riichi_declared[0]) != reached:
        raise ValueError('Actual ron decline did not establish the expected known flags')
    if kind == 'post_pon':
        observations = discard(env, observations, 2, 126)
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
        cleared_temporary_ron_control(env, observations, 2, 74, expect_ron=not reached)
        observations = discard(env, observations, 2, 74)
        if offered(env, 0, ActionType.RON):
            raise ValueError('Missed ron illegally reappeared before the observer turn')
        observations = pass_responses(env, observations)
        observations = discard(env, observations, 3, 116)
        observations = pass_responses(env, observations)
    obs = observations[0]
    if not env.missed_agari_doujun[0] or bool(env.riichi_declared[0]) != reached:
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
    tile = planned_discard(obs, kind)
    observations = discard(env, {0: obs}, 0, tile)
    if env.missed_agari_doujun[0] or bool(env.riichi_declared[0]) != kind.startswith('riichi'):
        raise ValueError('Temporary reset or permanent riichi furiten differs after discard')
    observations = pass_responses(env, observations)
    target = 100 if kind == 'post_pon' else 75
    cleared_temporary_ron_control(env, observations, 1, target, expect_ron=not kind.startswith('riichi'))
    observations = discard(env, observations, 1, target)
    ron = offered(env, 0, ActionType.RON)
    if ron == kind.startswith('riichi'):
        raise ValueError('Later ron availability ignores the actual known decline state')
    if ron:
        accept_ron_control(env)
    return dict(temporary_flag_clears_on_own_discard=True,
                permanent_flag_survives_own_discard=kind.startswith('riichi'),
                subsequent_ron_offered=ron, later_ron_uses_original_fixture_world=True)


def verify_sampled_recovery(env, obs, kind):
    """Separate diagnostic clone, not a draw from the prior or a label.

    Put one publicly unseen waiting tile in next player's hand by swapping
    unknown physical slots. Own hand/public tiles stay unchanged. This makes
    the latent permanent flag observable after temporary furiten clears.
    """
    from riichienv import ActionType, HandEvaluator
    from collections import Counter
    from tools.curriculum_native_decisions import decision_inventory
    if kind.endswith('tsumo'):
        raise ValueError('The controlled recovery probe is for nonwinning decisions')
    control = env.clone()
    hands = [list(h) for h in control.hands]; wall = list(control.wall)
    before = Counter(t for h in hands for t in h); before.update(wall)
    inventory = decision_inventory(obs, 'post_call' if kind == 'post_pon' else 'draw')
    after_discard = list(obs.hand); after_discard.remove(planned_discard(obs, kind))
    waits = set(HandEvaluator(after_discard, list(obs.to_dict()['melds'][0])).get_waits())
    waiting_tiles = [t for t in inventory['unseen_tiles'] if t // 4 in waits]
    if not waiting_tiles:
        raise ValueError('Recovery probe requires a publicly unseen waiting tile')
    target = min(waiting_tiles)
    if target in hands[0] or target in control.dora_indicators:
        raise ValueError('Recovery probe target must be publicly unseen')
    if target not in hands[1]:
        replacement = hands[1][0]
        locations = [(p, h.index(target)) for p, h in enumerate(hands)
                     if p not in (0, 1) and target in h]
        if locations:
            player, index = locations[0]; hands[player][index] = replacement
        elif target in wall:
            wall[wall.index(target)] = replacement
        else:
            raise ValueError('Recovery probe target is not in an unknown physical slot')
        hands[1][0] = target
    after = Counter(t for h in hands for t in h); after.update(wall)
    if before != after or hands[0] != list(obs.hand):
        raise ValueError('Diagnostic swap changed physical inventory or observer hand')
    control.hands = hands; control.wall = wall
    own = control.get_obs_py([0])[0]
    tile = planned_discard(own, kind)
    observations = discard(control, {0: own}, 0, tile)
    if control.missed_agari_doujun[0]:
        raise ValueError('Sampled recovery did not clear temporary furiten on discard')
    observations = pass_responses(control, observations)
    cleared_temporary_ron_control(control, observations, 1, target, expect_ron=not kind.startswith('riichi'))
    observations = discard(control, observations, 1, target)
    ron = offered(control, 0, ActionType.RON)
    if ron == kind.startswith('riichi'):
        raise ValueError('Resampling lost latent permanent RI-furiten behavior')
    return dict(diagnostic_unknown_slot_swap_not_prior_sample=True,
                temporary_cleared_by_actual_discard=True,
                subsequent_ron_offered=ron,
                latent_RI_flag_checked_by_behavior=kind.startswith('riichi'))
