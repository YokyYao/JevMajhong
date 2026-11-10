"""CPU live/replay candidate audit for clean full-policy training.

Scripted games are diagnostics only, never training labels or strength evidence.
The chosen action is used only to pair repeated public states, never to rebuild
legal candidates. Detailed states and MJAI stay in the private run directory.
"""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riichienv import ActionType, GameRule, MjaiReplay, RiichiEnv
from arena import EfficiencyAgent
from head_rl import write
from replay_call_repair import (UnsafeReplayCall, canonical_signature,
                                build_replay_request)
from semantic import describe, features, legal_candidates
from seed_control import engine_seed
from clean_policy_data import pass_is_identified, pass_outcomes
from replay_dora_repair import DoraChronology, UnsafeDoraPrefix


def kind(action):
    return json.loads(action.to_mjai())['type']


def row(obs, chosen):
    state, _, _ = describe(obs)
    public = json.loads(base64.b64decode(obs.serialize_to_base64()))
    # Public tile names avoid irrelevant physical-copy IDs assigned by replay.
    fields = ('seat', 'dealer', 'round_wind', 'honba', 'sticks', 'scores',
              'hand', 'rivers', 'melds', 'dora_indicators', 'riichi')
    key = json.dumps({k: state[k] for k in fields}, sort_keys=True)
    return {'key': key, 'chosen': canonical_signature(chosen),
            'type': kind(chosen), 'open_hand': bool(state['melds'][obs.player_id]),
            'drawn_tile': public['drawn_tile'], 'own_melds': public['melds'][obs.player_id],
            'candidates': sorted({canonical_signature(a) for a in legal_candidates(obs)})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--games', type=int, default=4)
    p.add_argument('--seed-start', type=int, default=9813100)
    p.add_argument('--exercise-kans',action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    totals, excluded, mismatch_types = Counter(), Counter(), Counter()
    coverage = Counter()
    details = []
    baseline = EfficiencyAgent()
    for offset in range(a.games):
        env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(),
                       seed=engine_seed(a.seed_start + offset, 'spaced-v1'))
        observations = env.reset()
        native = defaultdict(list)
        relaxed = defaultdict(list)
        step = 0
        while not env.done():
            chosen = {}
            for pid, obs in observations.items():
                actions = legal_candidates(obs)
                wins = [x for x in actions if x.action_type in (ActionType.TSUMO, ActionType.RON)]
                calls = [x for x in actions if kind(x) in ('chi', 'pon')]
                reaches = [x for x in actions if x.action_type == ActionType.RIICHI]
                kans = [x for x in actions if kind(x) in ('ankan','kakan','daiminkan')]
                if wins:
                    action = wins[0]
                elif kans and a.exercise_kans:
                    action=kans[0];totals['live_kan_actions']+=1
                elif calls and (pid + offset) % 2 == 0:
                    action = calls[0]
                elif reaches and (step + pid) % 2 == 0:
                    _, visible, d = describe(obs)
                    fs = features(obs, actions, visible, d)
                    indices = [i for i, f in enumerate(fs) if f]
                    index = min(indices, key=lambda i: (fs[i]['sh'], -fs[i]['ukeire'], i))
                    action = actions[index]
                    totals['live_declined_riichi'] += 1
                else:
                    action = baseline.act(obs)
                chosen[pid] = action
                if len(actions) > 1:
                    r = row(obs, action)
                    native[(r['key'], r['chosen'])].append(r)
                    st = json.loads(r['key'])
                    relaxed[json.dumps([st['seat'], st['hand'], st['rivers'], r['chosen']])].append(r)
                    totals['live_multi_option_states'] += 1
            observations = env.step(chosen)
            step += 1
            if step > 100000:
                raise RuntimeError('Diagnostic game did not terminate')
        path = a.out / f'private-scripted-{offset}.jsonl'
        path.write_text(''.join(json.dumps(e) + '\n' for e in env.mjai_log))
        replay = MjaiReplay.from_jsonl(str(path), rule='mjsoul')
        outcomes = pass_outcomes(env.mjai_log)
        chronology = DoraChronology(env.mjai_log)
        for hand in replay.take_kyokus():
            for pid, obs, expert in hand.steps(seat=None, skip_single_action=True):
                try:obs,repair=chronology.repair(obs)
                except UnsafeDoraPrefix as exc:
                    excluded[str(exc)]+=1
                    continue
                totals.update(repair)
                kinds = {kind(x) for x in legal_candidates(obs)}
                if kinds & {'hora', 'ankan', 'kakan', 'daiminkan', 'ryukyoku'}:
                    excluded['win_kan_abort_handled_outside_learned_scope'] += 1
                    continue
                if kind(expert) == 'none' and not pass_is_identified(obs, outcomes):
                    excluded['censored_or_unresolved_pass'] += 1
                    continue
                r = row(obs, expert)
                if len(r['candidates']) < 2:
                    totals['replay_canonical_single_option_skipped'] += 1
                    continue
                try:
                    payload, signatures, audit = build_replay_request(obs)
                except UnsafeReplayCall as exc:
                    excluded[str(exc)] += 1
                    continue
                r['candidates'] = sorted(signatures)
                serialized = sorted(canonical_signature({k: v for k, v in choice.items()
                    if k not in ('sh', 'ukeire', 'safe', 'red')})
                    for choice in payload['questions']['action']['criteria'].values())
                if serialized != r['candidates']:
                    raise ValueError('Serialized criteria do not match returned legal actions')
                if len(r['candidates']) < 2:
                    totals['repaired_single_option_skipped'] += 1
                    continue
                matches = native.get((r['key'], r['chosen']), [])
                if len(matches) != 1:
                    totals['unmatched' if not matches else 'ambiguous'] += 1
                    st = json.loads(r['key'])
                    nearest = relaxed.get(json.dumps([st['seat'], st['hand'], st['rivers'], r['chosen']]), [])
                    details.append({'game': offset, 'status': 'unpaired', 'replay': r, 'nearby': nearest})
                    continue
                live = matches[0]
                totals['matched'] += 1
                label = r['type']
                if label == 'dahai':
                    label = 'discard_open' if r['open_hand'] else 'discard_closed'
                coverage[label] += 1
                if live['candidates'] != r['candidates']:
                    totals['mismatches'] += 1
                    mismatch_types[label] += 1
                    details.append({'game': offset, 'status': 'mismatch', 'live': live, 'replay': r})
        print(json.dumps({'games': offset + 1, 'totals': dict(totals),
                          'coverage': dict(coverage), 'exclusions': dict(excluded)}), flush=True)
    adequate = all(coverage[x] >= 5 for x in ('discard_closed', 'discard_open', 'reach', 'chi', 'pon', 'none'))
    # Every matched record must agree; unsupported/unpaired states are explicit.
    report = {'complete': True, 'passed_matched_candidate_sets': totals['mismatches'] == 0,
              'required_phase_coverage_met': adequate, 'games': a.games,
              'totals': dict(totals), 'coverage': dict(coverage),
              'mismatch_types': dict(mismatch_types), 'explicit_exclusions': dict(excluded),
              'scope': 'scripted diagnostic; paired multi-option states only; not expert data or strength evidence',
              'repair_sha256': hashlib.sha256((ROOT/'replay_call_repair.py').read_bytes()).hexdigest()}
    report['dora_repair_sha256']=hashlib.sha256((ROOT/'replay_dora_repair.py').read_bytes()).hexdigest()
    report['kan_chronology_coverage_met']=not a.exercise_kans or totals['live_kan_actions']>=4
    write(a.out/'private-details.json', details)
    write(a.out/'summary.json', report)
    print(json.dumps(report), flush=True)
    if not report['passed_matched_candidate_sets'] or not adequate or not report['kan_chronology_coverage_met'] or totals['unmatched']:
        raise RuntimeError('Candidate audit failed or phase coverage insufficient')


if __name__ == '__main__':
    main()
