"""Exact native prefix replay and paired future-wall sampling for training."""
from collections import Counter
import json
import random
from riichienv import ActionType
from .collector import call_observation_digest
from .rewards import hand_outcomes
from jevmahjong.core.scoring import settlement_ledger


def replay_prefix(env, log, steps):
    observations = env.reset()
    current = env.mjai_log
    if current != log[:len(current)]:
        raise ValueError('Initial native log differs from source')
    for _ in range(steps):
        pending = log[len(current):]
        if not pending or env.done(): raise ValueError('Source prefix ended early')
        # Deferred kan dora can be emitted immediately before the committing
        # discard. These are engine events, not the player's next action.
        while pending and pending[0]['type'] in ('dora', 'reach_accepted'):
            pending = pending[1:]
        if not pending: raise ValueError('Source has no following decision event')
        events = []
        for event in pending:
            if events and (event['type'] != 'hora' or events[0]['type'] != 'hora'): break
            events.append(event)
            if event['type'] != 'hora': break
        selected = {}
        for player, obs in observations.items():
            action = None
            for event in events:
                if event.get('actor') == player or (len(observations) == 1 and event['type'] == 'ryukyoku'):
                    action = obs.select_action_from_mjai(json.dumps(event))
                    if action is not None: break
            if action is None:
                legal = obs.legal_actions()
                action = next((a for a in legal if a.action_type == ActionType.PASS), None)
                if action is None and len(legal) == 1: action = legal[0]
            if action is None:
                kinds = sorted({json.loads(a.to_mjai())['type'] for a in obs.legal_actions()})
                raise ValueError(f'Source action cannot be reconstructed: step={_}, player={player}, '
                    f'next={events[0]["type"]}, actor={events[0].get("actor")}, legal_types={kinds}')
            selected[player] = action
        observations = env.step(selected)
        previous_size = len(current); current = env.mjai_log
        if current[previous_size:] != log[previous_size:len(current)]:
            raise ValueError('Reconstructed actions changed the source event prefix')
    if current != log[:len(current)]: raise ValueError('Reconstructed prefix differs')
    return observations


def sample_future(env, seed, wall_scope='live_only'):
    """Keep hands and revealed indicators; sample declared hidden-wall scope.

    RiichiEnv v0.4.11 draws live tiles from the vector end, rinshan from
    its start, and exposes indicators inside its first 14 positions.
    """
    # Observation retrieval consumes pending per-player events. Compare two
    # copies at the same cursor, without advancing the parent environment.
    before = call_observation_digest(env.clone().get_observations())
    result = env.clone(); wall = list(env.wall)
    if len(wall) < 14: raise ValueError('Dead wall missing')
    if wall_scope == 'live_only': fixed = set(range(14))
    elif wall_scope == 'unseen': fixed = {wall.index(t) for t in env.dora_indicators}
    else: raise ValueError('Unknown future-wall scope')
    positions = [i for i in range(len(wall)) if i not in fixed]
    values = [wall[i] for i in positions]; random.Random(seed).shuffle(values)
    sampled = list(wall)
    for i,tile in zip(positions,values): sampled[i] = tile
    result.wall = sampled
    if (Counter(result.wall) != Counter(wall) or any(result.wall[i] != wall[i] for i in fixed)
            or call_observation_digest(result.get_observations()) != before):
        raise ValueError('Future sampling changed visible state or tile inventory')
    return result


def finish_intervention_hand(env, actions, policy, seat, suite, hand_index):
    marker = (env.kyoku_idx, env.honba, env.round_wind)
    observations = env.step(actions); steps = 1
    while not env.done() and (env.kyoku_idx, env.honba, env.round_wind) == marker:
        selected = {p:policy.act(o, frozen=True) if p == seat else policy.opponent(p, o, seat, suite)
            for p,o in observations.items()}
        observations = env.step(selected); steps += 1
        if steps > 1000: raise RuntimeError('Intervention hand did not terminate')
    log = env.mjai_log
    if not hand_outcomes(log)[hand_index]['complete']:
        raise ValueError('Stopped before the intervention hand settled')
    ledger = settlement_ledger(log, env.scores())
    return {'mjai_log':log, 'scores_at_stop':list(env.scores()),
        'point_ledger':ledger, 'stopped_after_intervention_hand':True}
