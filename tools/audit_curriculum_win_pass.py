"""Check native legal win-decline paths before admitting an endgame course."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]


class NoInference:
    def __init__(self):
        self.calls = 0

    def system_one(self, **kwargs):
        self.calls += 1
        raise AssertionError('Current forced-win path unexpectedly consulted the model')


def audit(registered, out, seed_start=2026400000, games=8):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    assert games > 0 and not set(range(seed_start, seed_start+games)) & set(range(2026300000, 2026300032))
    reg = json.loads(Path(registered).read_text())
    sys.path.insert(0, reg['source'])
    sys.path.insert(0, reg['training_config']['upstream_source'])
    from riichienv import ActionType, GameRule, RiichiEnv
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import build_request, legal_candidates, signature
    from jevmahjong.evaluation.agents import EfficiencyAgent, PolicyAgent

    records = []
    total = Counter()
    for seed in range(seed_start, seed_start+games):
        env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed, 'spaced-v1'))
        observations = env.reset()
        baseline = EfficiencyAgent()
        forced_types = set()
        counts = Counter()
        for step in range(30000):
            if env.done():
                break
            chosen = {}
            for seat, obs in observations.items():
                actions = legal_candidates(obs)
                wins = [a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO)]
                if not wins:
                    chosen[seat] = baseline.act(obs)
                    continue
                request, encoded, _ = build_request(obs)
                assert {signature(a) for a in encoded} == {signature(a) for a in actions}
                assert any(c['type'] == 'hora' for c in request['questions']['action']['criteria'].values())
                spy = NoInference()
                guarded = PolicyAgent(spy).act(obs)
                assert guarded.action_type in (ActionType.RON, ActionType.TSUMO) and spy.calls == 0
                counts['current_policy_forced_win_without_inference'] += 1
                kind = 'ron' if wins[0].action_type == ActionType.RON else 'tsumo'
                alternatives = [a for a in actions if a.action_type ==
                    (ActionType.PASS if kind == 'ron' else ActionType.DISCARD)]
                counts[kind+'_offers'] += 1
                counts[kind+'_offers_with_native_decline'] += bool(alternatives)
                state = request['state']
                if state['round_wind'] == 0 and state['hand_number'] == 4:
                    counts['scheduled_east4_win_offers'] += 1
                    rank = sorted(range(4), key=lambda p: (-state['scores'][p], p)).index(seat)+1
                    counts['scheduled_east4_trailing_win_offers'] += rank >= 3
                if alternatives and kind not in forced_types:
                    chosen[seat] = alternatives[0]
                    forced_types.add(kind)
                    counts[kind+'_declines_executed'] += 1
                else:
                    chosen[seat] = wins[0]
            observations = env.step(chosen)
        assert env.done(), 'Game truncated during native decline checks'
        ledger = settlement_ledger(env.mjai_log, env.scores())
        row = dict(seed=seed, completed=True, action_steps=step,
                   hands=ledger['hands'], counts=dict(counts), internal_score_ledger_passed=True)
        records.append(row)
        total.update(counts)
    coverage = all(total[k+'_declines_executed'] > 0 for k in ('ron', 'tsumo'))
    result = dict(status='native_win_decline_paths_observed' if coverage else 'native_win_decline_coverage_incomplete',
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], games=games,
        seed_start=seed_start, final_test_seeds_not_used=True, totals=dict(total),
        aggregate_game_receipts=records, both_ron_and_tsumo_declines_executed=coverage,
        current_policy_win_guard_bypasses_model=True, student_or_v8_model_loaded=False,
        new_training_labels_created=False, optimizer_updates=0, current_fit_or_policy_modified=False,
        independent_han_fu_scoring=False,
        note='Native path/score-ledger audit only. Deliberately declining a win is not an optimality label or a strength evaluation. East4 offers are not assumed to terminate the match: continuations/extensions need explicit stage4 checks. Preserve practical v8 forced-win behavior in its reference arm; do not weaken the baseline routing to claim a student gain. Endgame data generation and fitting remain unadmitted.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed-start', type=int, default=2026400000)
    p.add_argument('--games', type=int, default=8)
    a = p.parse_args()
    audit(a.registered, a.out, a.seed_start, a.games)
