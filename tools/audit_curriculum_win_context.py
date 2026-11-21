"""Native public win-source contract audit, with no model or training labels."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(registered, out, seed_start=2026400200, games=8, inventory_check=False, calling_profile=False, prior_trials=0, prior_worlds=0):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    assert games > 0 and not set(range(seed_start,seed_start+games)) & set(range(2026300000,2026300032))
    if type(prior_trials) is not int or not 0 <= prior_trials <= 4096 or (prior_trials and not inventory_check):
        raise ValueError('Prior feasibility probes require inventory checking and at most4096 trials')
    if type(prior_worlds) is not int or not 0 <= prior_worlds <= 16 or (prior_worlds and not inventory_check):
        raise ValueError('Prior allocation checks require inventory and at most16 draws per window')
    if out.exists(): raise ValueError('Never overwrite an audit')
    reg = json.loads(registered.read_text())
    sys.path.insert(0, reg['source'])
    from riichienv import RiichiEnv, GameRule, ActionType, calculate_shanten, HandEvaluator
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.core.state import legal_candidates, build_request, signature
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.evaluation.agents import EfficiencyAgent, CallingOpponent
    from tools.curriculum_win_context import build_win_request, project_trigger, SCHEMA
    from tools.curriculum_public_inventory import from_observation
    from tools.curriculum_public_worlds import sample_allocations, SamplingExhausted
    assert package_digest(Path(reg['source'])/'jevmahjong') == reg['source_sha256']
    totals = Counter()
    for seed in range(seed_start, seed_start+games):
        env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed,'spaced-v1'))
        observations = env.reset()
        agents = [CallingOpponent() if calling_profile and seat == 1 else EfficiencyAgent() for seat in range(4)]
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
                    if inventory_check:
                        pool = from_observation(obs, value['kind'])
                        assert len(pool['known_tiles'])+len(pool['unseen_tiles']) == 136
                        assert not set(pool['known_tiles']) & set(pool['unseen_tiles'])
                        totals['public_inventory_checks'] += 1
                        totals['inventory_riichi_constraints'] += len(pool['riichi_tenpai_required'])
                        totals['inventory_called_tile_pairs'] += pool['public_called_tile_duplicates']
                        # Diagnostic only: unconstrained hands violating riichi
                        # tenpai are NOT admitted as worlds or training labels.
                        for other in pool['riichi_tenpai_required']:
                            rng = random.Random(f'{seed}:{step}:{seat}:{other}:public-prior-v1')
                            for _ in range(prior_trials):
                                candidate = rng.sample(pool['unseen_tiles'],pool['opponent_concealed_counts'][other])
                                totals['uniform_riichi_candidate_trials'] += 1
                                totals['uniform_riichi_candidate_tenpai'] += calculate_shanten(candidate) == 0
                        if prior_worlds:
                            try:
                                drawn = sample_allocations(pool,seed*1000000+step*16+seat,prior_worlds,200000,calculate_shanten)
                            except SamplingExhausted as error:
                                totals['public_prior_exhausted_windows'] += 1
                                totals['public_prior_attempts'] += error.attempts
                            else:
                                totals['public_prior_complete_windows'] += 1
                                totals['public_prior_attempts'] += drawn['attempts']
                                totals['public_prior_allocations'] += len(drawn['allocations'])
                                for allocation in drawn['allocations']:
                                    hands = allocation['opponent_hands']
                                    assert {p:len(h) for p,h in hands.items()} == pool['opponent_concealed_counts']
                                    assert all(calculate_shanten(hands[p]) == 0 for p in pool['riichi_tenpai_required'])
                                    assert all(HandEvaluator(hands[p],list(obs.to_dict()['melds'][p])).is_tenpai()
                                        for p in pool['riichi_tenpai_required'])
                totals['native_menu_parity_checks'] += 1
            observations = env.step({seat:agents[seat].act(obs) for seat,obs in observations.items()})
        assert env.done(), 'Truncated match'
        settlement_ledger(env.mjai_log, env.scores())
        totals['complete_match_ledgers'] += 1
    assert totals['ron'] and totals['tsumo']
    result = dict(status='public_win_source_native_audit_passed', schema=SCHEMA,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'], seed_start=seed_start, games=games, totals=dict(totals),
        real_model_loaded=False, optimizer_updates=0, labels_generated=0,
        actual_robbed_kan_observed=bool(totals['robbed_kakan'] or totals['robbed_ankan']),
        inventory_checked=inventory_check, calling_profile=calling_profile,
        inventory_is_not_a_hidden_world_sampler=True,
        uniform_prior_trials_per_constraint=prior_trials,
        uniform_prior_is_not_a_posterior_or_accepted_world=True,
        public_prior_requested_allocations_per_window=prior_worlds,
        public_prior_all_cases_complete=bool(prior_worlds and not totals['public_prior_exhausted_windows']),
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
    p.add_argument('--inventory-check',action='store_true')
    p.add_argument('--calling-profile',action='store_true')
    p.add_argument('--prior-trials',type=int,default=0)
    p.add_argument('--prior-worlds',type=int,default=0)
    a = p.parse_args()
    audit(a.registered,a.out,a.seed_start,a.games,a.inventory_check,a.calling_profile,a.prior_trials,a.prior_worlds)
