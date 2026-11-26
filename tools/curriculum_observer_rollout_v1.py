"""Bounded observer-aware outer continuation; inner plans never recurse here."""
from collections import Counter
import os
import socket

from tools.audit_curriculum_history_worlds import derived_seed


def physical_key(action):
    return int(action.action_type), action.tile, tuple(sorted(action.consume_tiles))


class WinReplanner:
    def __init__(self, client, seed, worlds=4, max_calls=4, planner=None):
        if (type(seed) is not int or type(worlds) is not int or not 1 <= worlds <= 16
                or type(max_calls) is not int or not 1 <= max_calls <= 16):
            raise ValueError('Explicit finite public planning budgets are required')
        if planner is None:
            from tools.curriculum_public_win_planner_v1 import plan
            planner = plan
        self.client = client; self.seed = seed; self.worlds = worlds
        self.max_calls = max_calls; self.planner = planner
        self.counts = Counter(); self.records = []

    def __call__(self, env, obs, ordinary):
        from riichienv import ActionType
        from jevmahjong.core.state import build_request, legal_candidates
        from tools.curriculum_endgame import in_endgame_scope
        actions = legal_candidates(obs)
        wins = [a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO)]
        if not wins:
            return ordinary
        self.counts['legal_win_offers'] += 1
        request, actions, _ = build_request(obs)
        if not in_endgame_scope(request['state']):
            self.counts['outside_late_scope'] += 1; return ordinary
        trigger = env.mjai_log[-1]
        if trigger['type'] not in ('dahai', 'tsumo'):
            self.counts['unsupported_current_kan_offer'] += 1; return ordinary
        if self.counts['planned_calls'] >= self.max_calls:
            self.counts['budget_fallback_wins'] += 1
            if ordinary.action_type not in (ActionType.RON, ActionType.TSUMO):
                raise ValueError('Budget fallback must be the ordinary legal win')
            return ordinary
        result = self.planner(env, obs, self.client, seed=self.seed, worlds=self.worlds)
        keys = list(request['questions']['action']['criteria'])
        accept, decline = result['accept'], result['decline']
        selected = result['decision']['selected']
        if (accept == decline or accept not in keys or decline not in keys
                or selected not in (accept, decline)
                or actions[keys.index(accept)].action_type not in (ActionType.RON, ActionType.TSUMO)
                or actions[keys.index(decline)].action_type in (ActionType.RON, ActionType.TSUMO)
                or result['full_terminal_arms'] != 2 * self.worlds):
            raise ValueError('Planner changed its legal two-arm contract')
        self.counts['planned_calls'] += 1
        self.counts['inner_terminal_arms'] += result['full_terminal_arms']
        self.counts['selected_declines'] += selected == decline
        self.records.append(result)
        return actions[keys.index(selected)]


def finish(env, seat, first_key, future_seed, agents, observer=None, record_events=False):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
            or socket.gethostname().startswith(('lo-', 'login'))):
        raise RuntimeError('Observer-aware continuation requires scheduled GPU resources')
    from riichienv import RiichiEnv, GameRule, ActionType
    from jevmahjong.core.state import build_request
    from tools.curriculum_endgame_ledger import ledger
    if type(seat) is not int or seat not in range(4) or len(agents) != 4:
        raise ValueError('An observer and four public-observation agents are required')
    anchor = list(env.scores()); combined = list(env.mjai_log); cursor = len(combined)
    observations = env.get_obs_py(list(env.active_players)); initial = True
    hands = 0; resets = 0; hook_calls = 0
    for step in range(30000):
        if env.done():
            break
        marker = (env.round_wind, env.kyoku_idx, env.honba)
        selected = {p: agents[p].act(obs) for p, obs in sorted(observations.items())}
        if initial:
            if seat not in observations:
                raise ValueError('The initial intervention observer is not active')
            request, actions, _ = build_request(observations[seat])
            keys = list(request['questions']['action']['criteria'])
            if first_key not in keys:
                raise ValueError('Initial intervention is absent from its exact legal menu')
            selected[seat] = actions[keys.index(first_key)]; initial = False
        elif observer is not None and seat in observations:
            obs = observations[seat]
            if any(a.action_type in (ActionType.RON, ActionType.TSUMO) for a in obs.legal_actions()):
                before = build_request(obs)[0]
                selected[seat] = observer(env, obs, selected[seat]); hook_calls += 1
                if seat not in env.active_players or build_request(env.get_obs_py([seat])[seat])[0] != before:
                    raise ValueError('Observer planning mutated the actual outer public state')
        for player, obs in observations.items():
            menu = list(obs.legal_actions())
            if physical_key(selected[player]) not in {physical_key(a) for a in menu}:
                raise ValueError('A continuation action is physically stale or illegal')
            wins = any(a.action_type in (ActionType.RON, ActionType.TSUMO) for a in menu)
            if (player != seat and wins
                    and selected[player].action_type not in (ActionType.RON, ActionType.TSUMO)):
                raise ValueError('A sampled opponent declined a legal win; source certificate invalid')
        observations = env.step(selected)
        log = list(env.mjai_log); changed = (env.round_wind, env.kyoku_idx, env.honba) != marker
        if changed and not env.done():
            new_start = next(i for i in range(cursor, len(log)) if log[i]['type'] == 'start_kyoku')
            combined.extend(log[cursor:new_start]); hands += 1
            if hands > 64:
                raise RuntimeError('Bounded observer continuation exceeded64 subsequent hands')
            context = dict(oya=env.oya, round_wind=env.round_wind, scores=list(env.scores()),
                           honba=env.honba, kyotaku=env.riichi_sticks)
            fresh = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(),
                              seed=derived_seed(future_seed, hands))
            observations = fresh.reset(**context)
            if (fresh.oya != context['oya'] or fresh.round_wind != context['round_wind']
                    or list(fresh.scores()) != context['scores'] or fresh.honba != context['honba']
                    or fresh.riichi_sticks != context['kyotaku']):
                raise ValueError('Later-hand reseeding changed public match context')
            env = fresh; initial_log = list(env.mjai_log)
            combined.extend(e for e in initial_log if e['type'] != 'start_game')
            cursor = len(initial_log); resets += 1
        else:
            combined.extend(log[cursor:]); cursor = len(log)
        if not observations and not env.done():
            raise RuntimeError('A live outer match has no next native decision')
    if not env.done():
        raise RuntimeError('Outer continuation did not finish within its finite bound')
    checked = ledger(combined, env.scores())
    result = dict(terminal_scores=list(env.scores()), rank=int(env.ranks()[seat]),
                  top2=env.ranks()[seat] <= 2, net_from_anchor=env.scores()[seat] - anchor[seat],
                  full_match_completed=True, future_hand_reseeds=resets,
                  merged_ledger_hands=checked['hands'], next_hand_context_preserved=True,
                  observer_hook_calls=hook_calls, initial_intervention_skips_observer_hook=True,
                  opponents_no_legal_win_declines_verified=True,
                  continuation='bounded_observer_aware_outer_v1',
                  independent_han_fu_verification=False, teacher_admitted=False, training_label=False)
    if record_events:
        result['merged_mjai_log'] = combined
    return result
