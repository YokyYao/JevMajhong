"""Small reusable conditional first-hand benchmark, never a natural win rate."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha


def evaluate(config):
    gpu_only(); c = read(config)
    if (c['method'] != 'fixed_validation_yaku92_first_hand_public_mortal_opponents_v1'
            or c['source_partition'] != 'validation' or c['expected_cases'] != 92
            or c['maximum_action_batches'] != 2000
            or c['policy_method'] not in ('original_policy_agent', 'complete_legal_fixed_RI_class')):
        raise ValueError('Exact fixed conditional validation scope required')
    reg_path = ROOT/c['runtime_registration']; reg = read(reg_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(2)
    from riichienv import RiichiEnv, GameRule, ActionType
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.agents import PolicyAgent
    from tools.curriculum_complete_legal_class_policy_v1 import CompleteLegalClassPolicyAgent
    from tools.curriculum_mortal_source import MortalSource
    from tools.curriculum_case_order import ordered_seats
    from tools.curriculum_endgame_ledger import ledger
    from tools.audit_curriculum_endgame_cases import public_solo_win
    from tools.evaluate_curriculum_riichi_class_route_v2 import status_write
    if package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']:
        raise ValueError('Pinned reference runtime changed')
    if c['checkpoint_from_reference_registration']:
        checkpoint = Path(reg['curriculum']['reference_checkpoint'])
        expected = reg['curriculum']['reference_identity']
        if c['policy_method'] != 'original_policy_agent':
            raise ValueError('Keep the actual frozen v8 reference execution route')
    else:
        checkpoint = ROOT/c['checkpoint']; expected = c['checkpoint_identity']
    identity = bundle(checkpoint)
    if identity['identity_sha256'] != expected:
        raise ValueError('Expected exact checkpoint identity differs')
    source_path = ROOT/c['source_summary']; source = read(source_path)
    if source['status'] != 'actual_native_initial_yaku_scenarios_generated_unlabelled_not_fit_ready':
        raise ValueError('Actual fixed source scenarios required')
    spec = source['files']['validation']; path = ROOT/spec['path']
    if sha(path) != spec['sha256'] or spec['records'] != 92 or spec['groups'] != 45:
        raise ValueError('All original92 validation cases and45 source families required')
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    client = UnifiedPolicyClient(load_model(checkpoint))
    agent = (PolicyAgent if c['policy_method'] == 'original_policy_agent' else CompleteLegalClassPolicyAgent)(client)
    # load_model fixes TF32 flags; initialize every comparator's opponents after
    # that same runtime setup. Original all-Mortal source outcomes are not reused
    # as a formally matched comparator with different numerical settings.
    teacher = MortalSource((ROOT/c['teacher_root']).resolve())
    inputs = {str(p.resolve()): sha(p) for p in (Path(__file__), config, reg_path, source_path, path,
        ROOT/'tools/curriculum_case_order.py', ROOT/'tools/curriculum_endgame_ledger.py',
        ROOT/'tools/audit_curriculum_endgame_cases.py', ROOT/'tools/curriculum_complete_legal_class_policy_v1.py',
        ROOT/'cluster/curriculum-yaku-v8-cases-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-yaku-validation-cases-20261123.md')}
    inputs.update(teacher.bindings)
    write(out/'registered-private.json', dict(method=c['method'], config=c, inputs=inputs, checkpoint=identity,
        source_partition='validation', source_cases=92, source_families=45,
        three_fixed_public_history_Mortal_opponents=True, policy_method=c['policy_method'],
        targeted_distribution_not_natural_game_strength=True,
        source_is_development_not_final_confirmation=True,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID']))
    counts = Counter(); routes = defaultdict(Counter); groups = defaultdict(set)
    started = time.monotonic(); raw = (out/'case-outcomes-private.jsonl').open('x')
    try:
        for line in path.open():
            case = json.loads(line); original = case['source']; plan = original['plan']
            if plan['split'] != 'validation':
                raise ValueError('A TRAIN case enters the fixed heldout benchmark')
            env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=original['native_seed'])
            observations = env.reset(oya=0, wall=original['wall'])
            if env.mjai_log != original['native_prefix']:
                raise ValueError('Source initial physical prefix differs')
            opponents = {p: teacher.player(p) for p in (1, 2, 3)}
            selected_counts = Counter(); settled = False; winning_score = None
            initial_win_shortcut = any(a.action_type in (ActionType.RON, ActionType.TSUMO)
                                      for a in observations[0].legal_actions())
            for step in range(2000):
                selected = {}; offered_score = None
                for player, obs in ordered_seats(observations):
                    selected[player] = agent.act(obs) if player == 0 else opponents[player].act(obs)
                    if player == 0:
                        selected_counts[json.loads(selected[player].to_mjai())['type']] += 1
                        if selected[player].action_type in (ActionType.RON, ActionType.TSUMO):
                            offered_score = public_solo_win(env, obs)
                before = len(env.mjai_log); observations = env.step(selected)
                suffix = env.mjai_log[before:]
                if any(e['type'] in ('hora', 'ryukyoku') for e in suffix):
                    settled = True
                    if any(e['type'] == 'hora' and e['actor'] == 0 for e in suffix):
                        winning_score = offered_score
                    break
                if env.done():
                    raise ValueError('Conditional episode ended without a real settlement')
            if not settled:
                raise ValueError('Conditional episode exceeded finite action budget')
            financial = ledger(env.mjai_log, env.scores())
            settlements = [e for e in env.mjai_log if e['type'] in ('hora', 'ryukyoku')]
            won = any(e['type'] == 'hora' and e['actor'] == 0 for e in settlements)
            dealin = any(e['type'] == 'hora' and e['actor'] != 0 and e['target'] == 0 for e in settlements)
            delta = env.scores()[0]-25000; route = plan['route']; stat = routes[route]
            stat['cases'] += 1; stat['observer_wins'] += won; stat['observer_dealins'] += dealin
            stat['first_hand_net_points_sum'] += delta
            stat['initial_rule_win_shortcut_cases'] += initial_win_shortcut
            stat['observer_wins_excluding_initial_shortcut'] += won and not initial_win_shortcut
            if winning_score is not None:
                if winning_score['supported']:
                    stat['public_no_ura_supported_wins'] += 1
                    for yaku in winning_score['yaku']:
                        stat['supported_actual_yaku:'+yaku] += 1
                else:
                    stat['public_score_unsupported:'+winning_score['reason']] += 1
            for kind, n in selected_counts.items():
                stat['selected:'+kind] += n
            groups[route].add(plan['group_id']); counts['completed_cases'] += 1
            raw.write(json.dumps(dict(source_case_id=plan['id'], group_id=plan['group_id'], route=route,
                source_seed=original['native_seed'], completed_first_hand=True,
                observer_won=won, observer_dealt_in=dealin, first_hand_net_points=delta,
                initial_rule_win_shortcut=initial_win_shortcut, supported_public_score=winning_score,
                selected_operations=dict(selected_counts), final_scores=list(env.scores()),
                actual_mjai=env.mjai_log, ledger=financial, full_match_ended=env.done()),
                separators=(',', ':'))+'\n'); raw.flush()
            status_write(out/'status.json', dict(status='running', completed=counts['completed_cases'], total=92,
                seconds=time.monotonic()-started))
            if counts['completed_cases'] % 16 == 0:
                print(json.dumps(dict(completed_cases=counts['completed_cases'], total=92)), flush=True)
        if counts['completed_cases'] != 92 or bundle(checkpoint) != identity:
            raise ValueError('Incomplete benchmark or changed reference parameters')
        if any(sha(p) != h for p, h in inputs.items()):
            raise ValueError('Bound benchmark inputs changed')
        aggregate = {route: dict(stat, source_families=len(groups[route]),
            conditional_hand_win_rate=stat['observer_wins']/stat['cases'],
            conditional_dealin_rate=stat['observer_dealins']/stat['cases'],
            first_hand_mean_net_points=stat['first_hand_net_points_sum']/stat['cases']) for route, stat in routes.items()}
        result = dict(status='actual_fixed_validation_yaku92_first_hand_benchmark_complete',
            method=c['method'], model_role=c['model_role'], checkpoint_identity=expected,
            policy_method=c['policy_method'], completed_cases=92, source_families=45,
            by_route=aggregate, three_fixed_stateful_Mortal_opponents=True,
            source_route_and_hidden_world_not_model_inputs=True,
            conditional_hand_outcomes_not_natural_or_complete_match_rates=True,
            forced_initial_wins_reported_separately=True, public_yaku_scorer_has_declared_limits=True,
            model_parameters_unchanged=True, optimizer_updates=0,
            original64_match_reference_not_replayed=True, final_confirmation_seeds_unused=True,
            raw_trajectories_private=True, node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
        write(out/'summary-private.json', result)
        write(out/'summary-public.json', result)
        write(out/'completion-binding-private.json', dict(files=inputs))
        status_write(out/'status.json', dict(status='complete', completed=92))
        print(json.dumps(dict(status=result['status'], completed_cases=92)), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error), counts=dict(counts)))
        raise
    finally:
        raw.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); evaluate(a.config.resolve())
