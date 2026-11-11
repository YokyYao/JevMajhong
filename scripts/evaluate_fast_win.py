"""Fresh complete-East games for both fast-win heads, with score and danger checks."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from arena import EfficiencyAgent
from call_policy import digest
from clean_policy_evaluation import aggregate, POLICIES, SUITES
from fast_win import FastHeadClient, fast_game_stats
from fast_win_collector import FastWinCollector
from guide_development_audit import interval
from head_rl import write
from riichienv import GameRule, RiichiEnv
from scoring_audit import settlement_ledger
from scripts.collect_fast_win import verify_source
from scripts.evaluate_call_repair import Agent, CallingOpponent
from scripts.train_call_repair import make_head
from seed_control import engine_seed
from semantic import legal_candidates

CODE = ['scripts/evaluate_fast_win.py', 'fast_win.py', 'fast_win_collector.py',
    'scripts/collect_fast_win.py', 'scripts/fit_fast_win.py', 'clean_policy_evaluation.py',
    'guide_development_audit.py', 'scripts/evaluate_call_repair.py', 'semantic.py',
    'arena.py', 'scoring_audit.py', 'seed_control.py']


def play(agent, strong, seed, seat, suite, gamma):
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed, 'spaced-v1'))
    observations = env.reset(); ordinary, caller = EfficiencyAgent(), CallingOpponent()
    steps = 0; opportunities = Counter()
    while not env.done():
        chosen = {}
        for p, obs in observations.items():
            policy = agent if p == seat else (strong if suite == 'strong' else caller) if p == (seat + 1) % 4 and suite != 'closed' else ordinary
            if p == seat:
                kinds = {json.loads(a.to_mjai())['type'] for a in legal_candidates(obs)}
                opportunities.update(kinds & {'ankan', 'daiminkan', 'kakan'})
            chosen[p] = policy.act(obs)
        observations = env.step(chosen); steps += 1
        if steps > 100000:
            raise RuntimeError('Evaluation did not terminate')
    log = env.mjai_log; ledger = settlement_ledger(log, env.scores())
    counts = Counter(e['type'] for e in log if e.get('actor') == seat)
    return {'suite': suite, 'seed': seed, 'seat': seat, 'score': env.scores()[seat], 'rank': env.ranks()[seat],
        'hands': ledger['hands'], 'hand_wins': counts['hora'],
        'dealins': sum(e['type'] == 'hora' and e['target'] == seat and e['actor'] != seat for e in log),
        'calls': counts['chi'] + counts['pon'], 'steps': steps,
        'point_components': ledger['components'][seat], 'settlement_ledger_passed': True,
        'kan_opportunities': dict(opportunities),
        'kan_choices': {k: counts[k] for k in ('ankan', 'daiminkan', 'kakan')},
        'final_scores': list(env.scores()), 'mjai_log': log,
        **fast_game_stats(log, seat, gamma)}


def summarize(rows, cfg):
    result = aggregate(rows, cfg)
    indexed = {(r['suite'], r['seed'], r['seat'], r['policy']): r for r in rows}
    for r in rows:
        ledger = settlement_ledger(r['mjai_log'], r['final_scores'])
        actual = fast_game_stats(r['mjai_log'], r['seat'], cfg['fast_gamma'])
        if (ledger['final_scores'][r['seat']] != r['score']
                or ledger['components'][r['seat']] != r['point_components']
                or any(abs(actual[k] - r[k]) > 1e-12 for k in actual)):
            raise ValueError('Reward or point summary differs from actual native event log')
        if r['fast_hand_count'] != r['hands'] or r['fast_hand_wins'] != r['hand_wins']:
            raise ValueError('Fast reward and arena hand counts differ')
        if not 0 <= r['fast_reward_sum'] <= r['hands']:
            raise ValueError('Invalid discounted wins')
    gains = [statistics.mean(indexed[s, n, p, 'candidate']['fast_reward_sum'] / indexed[s, n, p, 'candidate']['hands']
        - indexed[s, n, p, 'reference']['fast_reward_sum'] / indexed[s, n, p, 'reference']['hands']
        for s in SUITES for p in range(4)) for n in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds'])]
    result['fast_reward_paired'] = {'gain': statistics.mean(gains), 'ci95': interval(gains),
        'unit': 'discounted own wins per hand, equal-weight match means, whole-wall paired CI'}
    for name in POLICIES:
        subset = [r for r in rows if r['policy'] == name]; wins = sum(r['hand_wins'] for r in subset)
        result['overall']['policies'][name].update(
            discounted_own_wins_per_hand=statistics.mean(r['fast_reward_sum'] / r['hands'] for r in subset),
            mean_table_draws_when_winning=sum(r['winning_table_draws'] for r in subset) / wins if wins else None,
            kan_opportunities={k: sum(r['kan_opportunities'].get(k, 0) for r in subset) for k in ('ankan', 'daiminkan', 'kakan')},
            kan_choices={k: sum(r['kan_choices'].get(k, 0) for r in subset) for k in ('ankan', 'daiminkan', 'kakan')})
    # Keep net points as the overarching improvement objective. Positive speed
    # with negative points is evidence to adjust risk, not a deployment claim.
    result['eligible_for_confirmation'] = (result['fast_reward_paired']['gain'] > 0
        and result['overall']['candidate_minus_reference']['gain'] > 0)
    result['not_deployed'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out / 'evaluation.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_cfg = json.loads((ROOT / cfg['source_config']).read_text()); source = ROOT / source_cfg['out']
    protocol = json.loads((source / 'protocol.json').read_text()); verify_source(protocol, source_cfg)
    fit = json.loads((source / 'fit-result.json').read_text())
    if fit['status'] != 'fast_heads_trained':
        write(out / 'evaluation-skipped.json', {'reason': fit['status'], 'deployed': False}); return
    prefix = 'development' if cfg['stage'] == 'development' else 'confirmation'
    if cfg['fast_gamma'] != source_cfg['fast_gamma']:
        raise ValueError('Evaluation reward discount differs from training')
    if cfg['seed_start'] != source_cfg[prefix + '_seed_start'] or cfg['seeds'] != source_cfg[prefix + '_seeds']:
        raise ValueError('Unregistered evaluation walls')
    if cfg['stage'] != 'development':
        development = ROOT / cfg['development_out']
        if (development / 'evaluation-skipped.json').exists():
            write(out / 'evaluation-skipped.json', {'reason': 'development_skipped'}); return
        prior = json.loads((development / 'result.json').read_text())
        audit = json.loads((development / 'evaluation-audit.json').read_text())
        if audit['status'] != 'passed_complete' or digest(development / 'games-private.jsonl') != audit['games_sha256']:
            raise ValueError('Audited complete development required')
        if not prior['eligible_for_confirmation']:
            write(out / 'evaluation-skipped.json', {'reason': 'no_joint_speed_and_point_gain'}); return
        if json.loads((development / 'protocol.json').read_text())['checkpoint_sha256'] != fit['selected']['checkpoint_sha256']:
            raise ValueError('Independent confirmation must freeze the same candidate')
    checkpoint = ROOT / fit['selected']['checkpoint']
    if digest(checkpoint) != fit['selected']['checkpoint_sha256']:
        raise ValueError('Selected heads changed')
    registration = {'config': cfg, 'source_config': source_cfg, 'fit_result_sha256': digest(source / 'fit-result.json'),
        'checkpoint_sha256': digest(checkpoint), 'code_sha256': {n: digest(ROOT / n) for n in CODE},
        'human_test_opened': False, 'candidate_frozen': True, 'deployed': False}
    target = out / 'protocol.json'
    if target.exists() and json.loads(target.read_text()) != registration:
        raise ValueError('Resume evaluation source changed')
    write(target, registration)
    # Load exactly the same frozen encoders and unchanged oldB strong opponent.
    policy = FastWinCollector(source_cfg)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['reference_files'] != protocol['reference_files'] or payload['teacher_files'] != protocol['teacher_files']:
        raise ValueError('Updated heads have different source encoders')
    heads = {}
    for scope, key in [('ordinary', 'reference'), ('calling', 'teacher_checkpoint')]:
        meta = torch.load(ROOT / source_cfg[key] / 'head.pt', map_location='cpu', weights_only=True)
        heads[scope] = make_head(meta, 'cuda').eval().requires_grad_(False)
        heads[scope].load_state_dict(payload['heads'][scope])
    client = FastHeadClient(policy.ordinary, policy.client, heads)
    # Validate updated native probabilities against the cached CPU head, per scope.
    checked = Counter(); maximum = 0.
    for shard in sorted((source / 'walls-private').glob('*.pt')):
        if digest(shard) != fit['training_shards'][str(shard.relative_to(ROOT))]:
            raise ValueError('Native parity source shard changed')
        for row in torch.load(shard, map_location='cpu', weights_only=False)['rows']:
            scope = row['scope']
            if checked[scope] >= 8:
                continue
            cpu = make_head(torch.load(ROOT / source_cfg['reference' if scope == 'ordinary' else 'teacher_checkpoint'] /
                'head.pt', map_location='cpu', weights_only=True), 'cpu').eval().requires_grad_(False)
            cpu.load_state_dict(payload['heads'][scope])
            with torch.no_grad():
                expected = cpu(row['decide'], row['options']).softmax(-1)
            before = policy.ordinary.runtime.model.head
            output = client.system_one(**row['public_request'])['answers']['action']['probabilities']
            actual = torch.tensor(list(output.values()))
            error = float((actual - expected).abs().max()); maximum = max(maximum, error)
            if error > 2e-5 or actual.argmax() != expected.argmax() or policy.ordinary.runtime.model.head is not before:
                raise ValueError('Updated native head parity or frozen opponent restoration failed')
            checked[scope] += 1
        if checked == {'ordinary': 8, 'calling': 8}:
            break
    if checked != {'ordinary': 8, 'calling': 8}:
        raise ValueError('Insufficient two-scope native parity')
    write(out / 'native-parity.json', {'checked': dict(checked), 'maximum_probability_error': maximum,
        'frozen_strong_head_restored': True})
    agents = {'reference': Agent(policy.router), 'candidate': Agent(client)}
    raw = out / 'games-private.jsonl'; rows = [json.loads(l) for l in raw.read_text().splitlines()] if raw.exists() else []
    done = {(r['suite'], r['seed'], r['seat'], r['policy']) for r in rows}
    expected = {(s, n, p, k) for s in SUITES for n in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds'])
        for p in range(4) for k in POLICIES}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError('Foreign or duplicate resumed game')
    started = time.monotonic()
    try:
        with raw.open('a') as stream:
            for seed in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds']):
                for suite in SUITES:
                    for seat in range(4):
                        for name in POLICIES:
                            if (suite, seed, seat, name) in done:
                                continue
                            row = dict(policy=name, **play(agents[name], policy.strong, seed, seat, suite, source_cfg['fast_gamma']))
                            stream.write(json.dumps(row) + '\n'); stream.flush(); rows.append(row)
                            write(out / 'status.json', {'status': 'fast_win_full_East_evaluation',
                                'games': len(rows), 'expected_games': len(expected), 'seconds': time.monotonic() - started})
        result = summarize(rows, cfg)
        verify_source(protocol, source_cfg)
        for name, h in registration['code_sha256'].items():
            if digest(ROOT / name) != h:
                raise ValueError('Evaluation code changed')
        if digest(checkpoint) != registration['checkpoint_sha256']:
            raise ValueError('Candidate changed during evaluation')
        write(out / 'result.json', result)
        # Re-read the actual JSONL, not the in-memory accumulator, and recompute.
        reread = [json.loads(l) for l in raw.read_text().splitlines()]
        if summarize(reread, cfg) != result:
            raise ValueError('Independent file reaggregation differs')
        audit = {'status': 'passed_complete', 'games': len(rows), 'games_sha256': digest(raw),
            'checkpoint_sha256': digest(checkpoint), 'fast_reward_paired': result['fast_reward_paired'],
            'net_point_paired': result['overall']['candidate_minus_reference']}
        write(out / 'evaluation-audit.json', audit); write(out / 'status.json', audit)
        print(json.dumps(audit), flush=True)
    finally:
        policy.close()


if __name__ == '__main__':
    main()
