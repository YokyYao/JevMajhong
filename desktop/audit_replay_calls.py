"""Compare live reaction candidates with replayed calls from the same game."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from riichienv import RiichiEnv, GameRule, MjaiReplay
    from arena import EfficiencyAgent
    from semantic import build_request, legal_candidates
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--policy', choices=('call', 'pass', 'mixed'), default='call')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=9813000)
    observations = env.reset()
    baseline = EfficiencyAgent()
    live = []
    step = 0
    while not env.done():
        chosen = {}
        for player, obs in observations.items():
            actions = legal_candidates(obs)
            calls = [a for a in actions if json.loads(a.to_mjai())['type'] in ('chi', 'pon')]
            wins = [a for a in actions if json.loads(a.to_mjai())['type'] == 'hora']
            take_call = calls and (args.policy == 'call' or (args.policy == 'mixed' and step % 2 == 0))
            chosen[player] = wins[0] if wins else calls[0] if take_call else baseline.act(obs)
            if calls and not wins:
                payload, _, _ = build_request(obs)
                live.append({'player': player, 'request': payload,
                             'chosen': json.loads(chosen[player].to_mjai())})
        observations = env.step(chosen)
        step += 1
    path = args.out / 'private-simulated-replay.jsonl'
    path.write_text(''.join(json.dumps(e) + '\n' for e in env.mjai_log), encoding='utf8')
    replay = MjaiReplay.from_jsonl(str(path), rule='mjsoul')
    replayed = []
    for hand in replay.take_kyokus():
        for player, obs, expert in hand.steps(seat=None, skip_single_action=True):
            action = json.loads(expert.to_mjai())
            payload, _, _ = build_request(obs)
            if any(v['type'] in ('chi', 'pon') for v in payload['questions']['action']['criteria'].values()):
                from replay_call_repair import build_replay_request, UnsafeReplayCall
                row = {'player': player, 'request': payload, 'chosen': action}
                try:
                    repaired, _, audit = build_replay_request(obs)
                    row['repaired_criteria'] = repaired['questions']['action']['criteria']
                    row['repair_audit'] = dict(audit)
                except UnsafeReplayCall as exc:
                    row['repair_exclusion'] = str(exc)
                replayed.append(row)
    def counts(rows):
        from collections import Counter
        formats = Counter((r['chosen']['type'], v['type'], len(v.get('consumed', [])))
                          for r in rows for v in r['request']['questions']['action']['criteria'].values()
                          if v['type'] in ('chi', 'pon'))
        return {'n': len(rows), 'with_pass': sum(any(v['type'] == 'none' for v in r['request']['questions']['action']['criteria'].values()) for r in rows),
                'candidate_consumed_lengths_by_target': [{'target': k[0], 'candidate': k[1], 'consumed': k[2], 'n': n} for k, n in formats.items()]}
    report = {'live_calls': counts(live), 'replayed_calls': counts(replayed),
              'note': 'Scripted diagnostic game, not expert strategy or training labels.'}
    from replay_call_repair import canonical_signature
    from collections import Counter
    def key(row):
        state = row['request']['state']
        return row['player'], state['hand'], state['last_discard'] // 4, state['discards_count']
    index = {key(row): row for row in live}
    comparison, excluded = Counter(), Counter()
    for row in replayed:
        if 'repair_exclusion' in row:
            excluded[row['repair_exclusion']] += 1
            continue
        original = index.get(key(row))
        if original is None:
            comparison['unmatched'] += 1
            continue
        expected = {canonical_signature(a) for a in original['request']['questions']['action']['criteria'].values()}
        actual = {canonical_signature(a) for a in row['repaired_criteria'].values()}
        comparison['matched'] += 1
        comparison['mismatches'] += expected != actual
        comparison['alternatives_restored'] += row['repair_audit'].get('missing_alternatives_restored', 0)
    report['repaired_comparison'] = dict(comparison)
    report['explicit_exclusions'] = dict(excluded)
    (args.out / 'visible-call-pairs.json').write_text(json.dumps({'summary': report, 'live': live, 'replay': replayed}, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
