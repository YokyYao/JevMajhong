"""Paired full-East call-head evaluation, with closed and call-aware opponents."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from arena import EfficiencyAgent, guarded_indices, ci
from call_policy import CallRouter, digest
from head_rl import summarize, write
from riichienv import ActionType, GameRule, RiichiEnv
from semantic import legal_candidates, build_request, shanten
from seed_control import engine_seed


class CallingOpponent(EfficiencyAgent):
    """Rule opponent only; never a supervision source for the learner."""
    def act(self, obs):
        actions = legal_candidates(obs)
        if len(actions) == 1: return actions[0]
        wins = [a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO)]
        if wins: return wins[0]
        d = obs.to_dict(); before = shanten(tuple(sorted(t // 4 for t in obs.hand)))
        for action in actions:
            if action.action_type not in (ActionType.CHI, ActionType.PON): continue
            tile = action.tile // 4
            valued = action.action_type == ActionType.PON and (tile >= 31 or tile == 27 + d['round_wind']
                         or tile == 27 + (obs.player_id - d['oya']) % 4)
            visible_own = list(obs.hand) + [action.tile] + [t for meld in d['melds'][obs.player_id] for t in meld.tiles]
            tanyao = all(t // 4 < 27 and (t // 4) % 9 not in (0, 8) for t in visible_own)
            if not (valued or tanyao): continue
            remaining = list(obs.hand)
            for tile_id in action.consume_tiles: remaining.remove(tile_id)
            after = min(shanten(tuple(sorted(t // 4 for j, t in enumerate(remaining) if j != i)))
                        for i in range(len(remaining)))
            if after < before: return action
        return super().act(obs)


class Agent:
    def __init__(self, client):
        self.client = client
        self.choices = Counter()

    def act(self, obs):
        from jevany.api import validate_distribution
        actions = legal_candidates(obs)
        if len(actions) == 1: return actions[0]
        for action in actions:
            if action.action_type in (ActionType.RON, ActionType.TSUMO): return action
        request, actions, feats = build_request(obs)
        result = self.client.system_one(**request)
        p, _ = validate_distribution(result['answers']['action']['probabilities'], list(request['questions']['action']['criteria']))
        # Guard discards, without silently dropping legal reach/kan with empty features.
        eligible = sorted(set(guarded_indices(obs, actions, feats)) | {i for i, f in enumerate(feats) if not f})
        chosen = actions[max(eligible, key=lambda i: p[i])]
        self.choices[json.loads(chosen.to_mjai())['type']] += 1
        return chosen


def game(agent, seed, seat, mixed):
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed, 'spaced-v1'))
    observations = env.reset(); ordinary, caller = EfficiencyAgent(), CallingOpponent()
    steps = 0
    while not env.done():
        selected = {}
        for player, obs in observations.items():
            policy = agent if player == seat else caller if mixed and player == (seat + 1) % 4 else ordinary
            selected[player] = policy.act(obs)
        observations = env.step(selected); steps += 1
        if steps > 100000: raise RuntimeError('Simulation failed to terminate')
    log = env.mjai_log
    return {'seed': seed, 'seat': seat, 'score': env.scores()[seat], 'rank': env.ranks()[seat],
        'hands': sum(e['type'] == 'start_kyoku' for e in log),
        'hand_wins': sum(e['type'] == 'hora' and e.get('actor') == seat for e in log),
        'dealins': sum(e['type'] == 'hora' and e.get('target') == seat and e.get('actor') != seat for e in log),
        'steps': steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text()); run = ROOT / cfg['out']
    selection = json.loads((run / 'selected.json').read_text())
    if not selection['checkpoint']:
        write(run / 'development-skipped.json', {'reason': 'No improved validation checkpoint'}); return
    if selection['validation']['macro_f1'] <= selection['baseline']['macro_f1']:
        write(run / 'development-skipped.json', {'reason': 'NLL-only calibration without validation decision improvement',
            'candidate_preserved': True, 'deployed': False}); return
    reference, candidate = ROOT / cfg['reference'], ROOT / selection['checkpoint']
    if digest(reference / 'head.pt') != cfg['reference_head_sha256']: raise ValueError('Reference changed')
    from scripts.train_call_repair import make_head
    from jevany import JevModel
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    import os
    base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
    client = JevModel.from_pretrained(str(reference), device='cuda', dtype='bf16',
        options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
                            fused_kernels=False, base_load_path=str(base)),
        inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
                                          max_packed_tokens=8192, prefix_cache_size=0))
    meta = torch.load(reference / 'head.pt', map_location='cpu', weights_only=True)
    repair = torch.load(candidate, map_location='cpu', weights_only=True)
    if repair['reference_head_sha256'] != cfg['reference_head_sha256']: raise ValueError('Call-head source changed')
    head = make_head(meta, 'cuda'); head.load_state_dict(repair['head']); head.eval().requires_grad_(False)
    agents = {'reference': Agent(client), 'candidate': Agent(CallRouter(client, head))}
    out = run / 'native-development'; out.mkdir(exist_ok=False)
    write(out / 'protocol.json', {'config': cfg, 'selected': selection, 'candidate_sha256': digest(candidate),
        'code_sha256': {p: digest(ROOT / p) for p in ['scripts/evaluate_call_repair.py', 'call_policy.py', 'semantic.py', 'seed_control.py']},
        'opponents': {'closed': '3 EfficiencyAgent', 'mixed': '2 EfficiencyAgent plus1 valued-honor/open-tanyao caller'},
        'guard': 'same discard guard for both; retain legal nondiscard actions including reach',
        'historical_legacy_guard_results_not_directly_comparable': True,
        'primary': 'paired terminal score difference; cluster by whole logical wall', 'deployed': False})
    all_rows = []; started = time.monotonic()
    with (out / 'games-private.jsonl').open('w') as stream:
        for suite in ('closed', 'mixed'):
            rows = []
            for seed in range(cfg['development_seed_start'], cfg['development_seed_start'] + cfg['development_seeds']):
                for seat in range(4):
                    for name, agent in agents.items():
                        row = {'suite': suite, 'policy': name, **game(agent, seed, seat, suite == 'mixed')}
                        rows.append(row); all_rows.append(row)
                        stream.write(json.dumps(row) + '\n'); stream.flush()
                        status = {'status': 'native_call_development', 'suite': suite, 'games': len(all_rows),
                            'total_games': cfg['development_seeds'] * 4 * 2 * 2, 'seconds': time.monotonic() - started}
                        write(run / 'strength-status.json', status); print(json.dumps(status), flush=True)
            summaries = {name: summarize([r for r in rows if r['policy'] == name]) for name in agents}
            for name, summary in summaries.items():
                subset = [r for r in rows if r['policy'] == name]
                hands = max(1, sum(r['hands'] for r in subset))
                summary.update(hand_win_rate=sum(r['hand_wins'] for r in subset) / hands,
                               dealin_rate=sum(r['dealins'] for r in subset) / hands)
            delta = []
            for seed in range(cfg['development_seed_start'], cfg['development_seed_start'] + cfg['development_seeds']):
                delta.append(sum((1 if r['policy'] == 'candidate' else -1) * r['score']
                                 for r in rows if r['seed'] == seed) / 4)
            write(out / f'{suite}-summary.json', {'policies': summaries,
                'paired_net_points_gain': sum(delta) / len(delta), 'paired_seed_cluster_ci95': ci(delta),
                'independent_wall_streams': len(delta), 'is_final_confirmation': False})
    write(run / 'strength-status.json', {'status': 'development_complete', 'games': len(all_rows),
        'deployed': False, 'independent_confirmation_pending': True})


if __name__ == '__main__': main()
