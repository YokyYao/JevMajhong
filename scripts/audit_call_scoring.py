"""Deterministically replay opened development games; audit their native scores."""
import argparse
import base64
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena import EfficiencyAgent
from call_policy import CallRouter, digest
from head_rl import write
from riichienv import ActionType, GameRule, RiichiEnv
from scoring_audit import independent_win_score, settlement_ledger
from scripts.evaluate_call_repair import Agent, CallingOpponent
from seed_control import engine_seed


def replay(agent, original):
    seed, seat = original['seed'], original['seat']
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed, 'spaced-v1'))
    obs = env.reset(); ordinary, caller = EfficiencyAgent(), CallingOpponent()
    win_checks = []; steps = 0
    while not env.done():
        selected, snapshots = {}, {}
        for player, observation in obs.items():
            policy = agent if player == seat else caller if original['suite'] == 'mixed' and player == (seat + 1) % 4 else ordinary
            action = policy.act(observation); selected[player] = action
            if action.action_type in (ActionType.RON, ActionType.TSUMO):
                snapshots[player] = {'observation': json.loads(base64.b64decode(observation.serialize_to_base64())),
                                     'win_tile': action.tile}
        history = list(env.mjai_log)
        obs = env.step(selected); steps += 1
        new_events = env.mjai_log[len(history):]
        wins = [e for e in new_events if e['type'] == 'hora']
        for event in wins:
            check = independent_win_score(history, event, snapshots[event['actor']], len(wins))
            if check['status'] == 'mismatch': check['snapshot_private'] = snapshots[event['actor']]
            win_checks.append({'actor': event['actor'], 'target': event['target'], **check})
        if steps > 100000: raise RuntimeError('Replay did not finish')
    log = env.mjai_log
    actual = {'score': env.scores()[seat], 'rank': env.ranks()[seat], 'steps': steps,
        'hands': sum(e['type'] == 'start_kyoku' for e in log),
        'hand_wins': sum(e['type'] == 'hora' and e['actor'] == seat for e in log),
        'dealins': sum(e['type'] == 'hora' and e['target'] == seat and e['actor'] != seat for e in log)}
    if any(actual[k] != original[k] for k in actual): raise ValueError(f'Replay differed: {original} -> {actual}')
    try:
        ledger = settlement_ledger(log, env.scores())
    except ValueError as error:
        # Persist native evidence before the caller fails the audit.
        ledger = {'error': str(error), 'final_scores': list(env.scores())}
    actions = Counter(e['type'] for e in log if e.get('actor') == seat)
    return {'original': original, 'replay_matches': True, 'ledger': ledger,
            'focal_actions': dict(actions), 'win_checks': win_checks, 'log_private': log}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); args = p.parse_args()
    cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']; out.mkdir(parents=True, exist_ok=False)
    trial = json.loads((ROOT / cfg['source_config']).read_text())
    source = ROOT / trial['out'] / 'native-development/games-private.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows = [r for r in rows if cfg['seed_start'] <= r['seed'] < cfg['seed_start'] + cfg['seeds']]
    if len(rows) != cfg['seeds'] * 4 * 3 * 2: raise ValueError('Opened development slice incomplete')
    reference = ROOT / trial['reference']; ref_protocol = json.loads((ROOT / 'runs/call-repair-20261109/protocol.json').read_text())
    for name, h in ref_protocol['reference_files'].items():
        if digest(reference / name) != h: raise ValueError('Reference changed')
    write(out / 'protocol.json', {'config': cfg, 'source_games_sha256': digest(source),
        'reference_files': ref_protocol['reference_files'], 'mahjong_version': importlib.metadata.version('mahjong'),
        'riichienv_version': importlib.metadata.version('riichienv'), 'new_strength_cases': False,
        'code_sha256': {n: digest(ROOT / n) for n in ['scoring_audit.py', 'scripts/audit_call_scoring.py',
            'scripts/evaluate_call_repair.py', 'arena.py', 'call_policy.py', 'semantic.py', 'seed_control.py']}})
    import torch
    from jevany import JevModel
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    from scripts.train_call_repair import make_head
    base = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
    client = JevModel.from_pretrained(str(reference), device='cuda', dtype='bf16',
        options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
            fused_kernels=False, base_load_path=str(base)),
        inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192, max_packed_tokens=8192, prefix_cache_size=0))
    meta = torch.load(reference / 'head.pt', map_location='cpu', weights_only=True)
    agents = {'reference': Agent(client)}; heads = {}
    for name, checkpoint in [('previous', trial['previous_call_checkpoint']),
                             ('candidate', trial['out'] + '/selected-call-head.pt')]:
        payload = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=True)
        if payload['reference_head_sha256'] != trial['reference_head_sha256']: raise ValueError('Call source mismatch')
        head = make_head(meta, 'cuda'); head.load_state_dict(payload['head']); head.eval().requires_grad_(False)
        agents[name] = Agent(CallRouter(client, head)); heads[checkpoint] = digest(ROOT / checkpoint)
    results = []
    with (out / 'replays-private.jsonl').open('w') as stream:
        for row in rows:
            result = replay(agents[row['policy']], row); results.append(result)
            stream.write(json.dumps(result) + '\n'); stream.flush()
            if 'error' in result['ledger']: raise ValueError(result['ledger']['error'])
            status = {'status': 'auditing_scoring', 'games': len(results), 'total_games': len(rows)}
            write(out / 'status.json', status); print(json.dumps(status), flush=True)
    checks = [c for r in results for c in r['win_checks']]
    by_policy = {}
    for name in agents:
        subset = [r for r in results if r['original']['policy'] == name]
        totals = Counter(); actions = Counter()
        for r in subset:
            totals.update(r['ledger']['components'][r['original']['seat']]); actions.update(r['focal_actions'])
        by_policy[name] = {'games': len(subset), 'mean_net_points': sum(r['original']['score'] - 25000 for r in subset) / len(subset),
                          'point_components_total': dict(totals), 'actions': dict(actions)}
    for checkpoint, h in heads.items():
        if digest(ROOT / checkpoint) != h: raise ValueError('Call source changed during replay')
    for name, h in ref_protocol['reference_files'].items():
        if digest(reference / name) != h: raise ValueError('Reference changed during replay')
    result = {'status': 'complete', 'games': len(results), 'replay_matches': len(results), 'ledger_passed': len(results),
        'hand_score_checks': dict(Counter(c['status'] for c in checks)),
        'skipped_reasons': dict(Counter(c['reason'] for c in checks if c['status'] == 'skipped')),
        'mismatches_private': [c for c in checks if c['status'] == 'mismatch'],
        'policies': by_policy, 'training_updates': 0, 'unopened_test_accessed': False,
        'not_new_strength_evidence': True}
    write(out / 'result.json', result); write(out / 'status.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'mismatches_private'}), flush=True)
    if result['mismatches_private']: raise ValueError('Independent score mismatches; inspect private results')


if __name__ == '__main__': main()
