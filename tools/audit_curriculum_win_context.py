"""Native public win-source contract audit, with no model or training labels."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(registered, out, seed_start=2026400200, games=8):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    assert games > 0 and not set(range(seed_start,seed_start+games)) & set(range(2026300000,2026300032))
    if out.exists(): raise ValueError('Never overwrite an audit')
    reg = json.loads(registered.read_text())
    sys.path.insert(0, reg['source'])
    from riichienv import RiichiEnv, GameRule, ActionType
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.core.state import legal_candidates, build_request, signature
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.evaluation.agents import EfficiencyAgent
    from tools.curriculum_win_context import build_win_request, project_trigger, SCHEMA
    assert package_digest(Path(reg['source'])/'jevmahjong') == reg['source_sha256']
    totals = Counter()
    for seed in range(seed_start, seed_start+games):
        env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed,'spaced-v1'))
        observations = env.reset()
        agent = EfficiencyAgent()
        for step in range(30000):
            if env.done(): break
            # Collector supplies only the current actual event. No synthetic
            # replay history or hidden environment field reaches the adapter.
            raw = env.mjai_log[-1]
            event = json.loads(raw) if isinstance(raw,str) else raw
            for seat, obs in observations.items():
                actions = legal_candidates(obs)
                if not any(a.action_type in (ActionType.RON, ActionType.TSUMO) for a in actions): continue
                trigger = project_trigger(event, seat)
                request, encoded, _ = build_win_request(obs, event)
                original, ordinary, _ = build_request(obs)
                assert request['questions'] == original['questions']
                assert [signature(a) for a in encoded] == [signature(a) for a in ordinary]
                state = dict(request['state'])
                context = state.pop('win_context')
                assert state == original['state'] and context['schema'] == SCHEMA
                for key, value in context['candidates'].items():
                    assert request['questions']['action']['criteria'][key]['type'] == 'hora'
                    assert value['source_seat'] == trigger['actor']
                    totals[value['kind']] += 1
                    totals[value['source']] += 1
                    totals['red_winning_tile'] += value['tile'][0] == '0'
                    totals['honor_winning_tile'] += value['tile'][1] == 'z'
                totals['native_menu_parity_checks'] += 1
            observations = env.step({seat:agent.act(obs) for seat,obs in observations.items()})
        assert env.done(), 'Truncated match'
        settlement_ledger(env.mjai_log, env.scores())
        totals['complete_match_ledgers'] += 1
    assert totals['ron'] and totals['tsumo']
    result = dict(status='public_win_source_native_audit_passed', schema=SCHEMA,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'], seed_start=seed_start, games=games, totals=dict(totals),
        real_model_loaded=False, optimizer_updates=0, labels_generated=0,
        actual_robbed_kan_observed=bool(totals['robbed_kakan'] or totals['robbed_ankan']),
        robbed_kan_logic_fixture_only_if_unobserved=True, full_scoring_verification=False,
        stage4_admitted=False, earlier_input_contracts_unchanged=True, final_reserved_seeds_unused=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed-start',type=int,default=2026400200)
    p.add_argument('--games',type=int,default=8)
    a = p.parse_args()
    audit(a.registered,a.out,a.seed_start,a.games)
