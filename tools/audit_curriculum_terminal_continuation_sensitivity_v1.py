"""Matched cached outer worlds: ordinary wins versus bounded future replanning."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write


def audit(config_path):
    gpu_only(); c = read(config_path)
    expected = dict(method='matched_cached_worlds_future_win_policy_sensitivity_v1',
        all_original_selected_views=True, expected_views=16, expected_families=8,
        block_indices=[0, 1], world_indices_per_block=[0, 1], expected_baseline_reproductions=128,
        expected_changed_outer_arms=128, future_planner_seed=202611231200, future_planner_worlds=4,
        maximum_future_plans_per_outer_arm=4, maximum_new_inner_terminal_arms=4096,
        initial_intervention_replanned=False, teacher_target_or_student_admission=False)
    if any(c.get(k) != v for k, v in expected.items()): raise ValueError('Frozen matched continuation diagnostic required')
    prior = ROOT/c['world_stability_root']; pilot = prior/'pilot-v1'
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    scheduler_exit(read(ROOT/'logs/curriculum-terminal-world-stability-exit-317775.json'),
        (prior/'pilot-terminal-scheduler-private.txt').read_text(), ROOT/'cluster/curriculum-terminal-world-stability-v1.slurm')
    completed = read(pilot/'summary-private.json'); previous = read(pilot/'registered-private.json')
    if (completed['status'] != 'train_terminal_world_stability_diagnostic_complete'
            or completed['views'] != 16 or completed['source_families'] != 8 or completed['actual_terminal_arms'] != 1024
            or not all(sha(p) == h for p, h in previous['files'].items())):
        raise ValueError('Actual complete larger-world diagnostic required')
    registered = ROOT/c['source_registration']; reg = read(registered)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.agents import PolicyAgent, EfficiencyAgent, CallingOpponent
    from tools.audit_curriculum_reused_terminal_v1 import restore
    from tools.audit_curriculum_observer_rollout_v1 import native_allocation, projection
    from tools.audit_curriculum_history_worlds import derived_seed
    from tools.curriculum_native_decisions import construct_world
    from tools.curriculum_observer_rollout_v1 import finish, WinReplanner
    from tools.curriculum_public_win_planner_v1 import choose
    root = ROOT/c['data_root']; ready = read(root/'collection-v1/summary-private.json')
    rawpath = ROOT/ready['files']['train']['path']
    if sha(rawpath) != ready['files']['train']['sha256']: raise ValueError('Original raw TRAIN cases changed')
    rows = {r['_meta']['id']: r for r in (json.loads(line) for line in rawpath.open())}
    cache_path = pilot/'plans-private.jsonl'; cases = [json.loads(line) for line in cache_path.open()]
    caches = {item['source_id']: item for item in cases}; specs = previous['selected']
    if (len(cases) != 16 or len(caches) != 16 or {s['id'] for s in specs} != set(caches)
            or len({s['group'] for s in specs}) != 8): raise ValueError('Exact all16-view/eight-family cohort required')
    paths = [Path(__file__), config_path, registered, rawpath, cache_path, pilot/'registered-private.json',
        pilot/'summary-private.json', prior/'pilot-terminal-scheduler-private.txt',
        ROOT/'logs/curriculum-terminal-world-stability-exit-317775.json',
        ROOT/'cluster/curriculum-terminal-continuation-sensitivity-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-terminal-continuation-sensitivity-20261123.md']
    paths += [ROOT/'tools'/name for name in ('curriculum_observer_rollout_v1.py',
        'audit_curriculum_observer_rollout_v1.py', 'curriculum_public_win_planner_v1.py',
        'curriculum_terminal_scout_stats.py', 'curriculum_native_decisions.py', 'curriculum_native_worlds.py',
        'curriculum_decision_history.py', 'curriculum_hierarchical_prior.py', 'curriculum_public_inventory.py',
        'curriculum_decision_rollout.py', 'curriculum_endgame_ledger.py', 'audit_curriculum_reused_terminal_v1.py')]
    files = {str(p.resolve()): sha(p) for p in paths}
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    checkpoint = Path(reg['curriculum']['reference_checkpoint']); identity = bundle(checkpoint)
    if identity != previous['frozen_teacher_identity']: raise ValueError('Frozen continuation teacher changed')
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(config=c, files=files, selected=specs,
        subset_fixed_without_view_selection_by_simulated_values=True,
        changed_and_ordinary_values_use_identical_four_world_subset=True, teacher_identity=identity))
    client = UnifiedPolicyClient(load_model(checkpoint))
    def agents(seat):
        result = [CallingOpponent() if p == (seat+1) % 4 else EfficiencyAgent() for p in range(4)]
        result[seat] = PolicyAgent(client); return result
    def world(spec, block_index, world_index):
        env, obs = restore(rows[spec['id']]); block = caches[spec['id']]['blocks'][block_index]
        fork, _ = construct_world(env, obs, env.mjai_log[-1], native_allocation(block['allocations'][world_index]), True)
        return fork, obs.player_id, block
    baseline_count = 0
    # Finish all exact observer-disabled event reproductions before comparing
    # any changed policy. A mismatch fails; it never becomes a new baseline.
    with (out/'baseline-reproductions-private.jsonl').open('x') as stream:
        for spec in specs:
            for block_index in (0, 1):
                for world_index in (0, 1):
                    fork, seat, block = world(spec, block_index, world_index)
                    for arm in (block['accept'], block['decline']):
                        result = finish(fork.clone(), seat, arm, derived_seed(block['public_seed'], world_index), agents(seat), record_events=True)
                        if projection(result) != projection(block['pairs'][world_index][arm]):
                            raise ValueError('Observer-disabled cached full events/outcomes do not reproduce')
                        baseline_count += 1
                        stream.write(json.dumps(dict(source_id=spec['id'], block=block_index, world=world_index, arm=arm, result=result))+'\n')
                    stream.flush()
            print(json.dumps(dict(stage='sensitivity_baseline_reproduction', complete_arms=baseline_count)), flush=True)
    if baseline_count != 128: raise ValueError('Incomplete baseline reproduction grid')
    counts = Counter(); cohort_counts = defaultdict(Counter); view_summaries = []
    with (out/'changed-outer-arms-private.jsonl').open('x') as stream:
        for spec in specs:
            changed_pairs = []; original_pairs = []
            for block_index in (0, 1):
                for world_index in (0, 1):
                    fork, seat, block = world(spec, block_index, world_index); changed = {}
                    for arm in (block['accept'], block['decline']):
                        control = WinReplanner(client, seed=c['future_planner_seed'], worlds=4, max_calls=4)
                        result = finish(fork.clone(), seat, arm, derived_seed(block['public_seed'], world_index),
                                        agents(seat), observer=control, record_events=True)
                        old = block['pairs'][world_index][arm]; changed[arm] = result
                        counts['changed_outer_arms'] += 1
                        counts['changed_complete_event_streams'] += projection(result) != projection(old)
                        counts['changed_own_terminal_rank_or_net'] += result['rank'] != old['rank'] or result['net_from_anchor'] != old['net_from_anchor']
                        counts.update(control.counts)
                        stream.write(json.dumps(dict(source_id=spec['id'], block=block_index, world=world_index,
                            initial_arm=arm, result=result, plans=control.records, planner_counts=dict(control.counts)))+'\n'); stream.flush()
                    changed_pairs.append(changed); original_pairs.append(block['pairs'][world_index])
            accept, decline = block['accept'], block['decline']
            ordinary = choose(original_pairs, accept, decline); adaptive = choose(changed_pairs, accept, decline)
            item = dict(cohort=spec['cohort'], bucket=spec['bucket'], conditional_worlds=4,
                ordinary_subset_declines=ordinary['selected'] == decline, replanned_subset_declines=adaptive['selected'] == decline,
                same_matched_subset_choice=ordinary['selected'] == adaptive['selected'],
                full32_ordinary_declines=caches[spec['id']]['pooled']['decision']['selected'] == decline,
                ordinary_subset_utility_difference=ordinary['paired_decline_minus_accept_utility'],
                replanned_subset_utility_difference=adaptive['paired_decline_minus_accept_utility'])
            view_summaries.append(item); cc = cohort_counts[spec['cohort']]; cc['views'] += 1
            cc['subset_choice_changes'] += not item['same_matched_subset_choice']
            cc['ordinary_subset_declines'] += item['ordinary_subset_declines']; cc['replanned_subset_declines'] += item['replanned_subset_declines']
            print(json.dumps(dict(stage='sensitivity_view_complete', **item)), flush=True)
    if counts['changed_outer_arms'] != 128 or counts['inner_terminal_arms'] > 4096:
        raise ValueError('Changed outer grid incomplete or inner budget exceeded')
    if bundle(checkpoint) != identity or not all(sha(p) == h for p, h in files.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Frozen teacher, diagnostic or active fit artifacts changed')
    result = dict(status='matched_future_win_continuation_sensitivity_complete',
        exact_cached_baseline_reproductions=128, counts=dict(counts),
        cohorts={k: dict(v) for k, v in cohort_counts.items()}, views=view_summaries,
        same_subset_comparison_not_four_world_vs_full32_confound=True,
        source_families=8, conditional_arms_not_population_policy_games=True,
        inner_depth=1, budget_fallback_ordinary_win=True, prior_not_calibrated=True,
        teacher_target_or_student_admission=False, new_training_labels=0, student_updates=0,
        active_fit_dependencies_unchanged=len(active), raw_cases_events_plans_private=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), goal_achieved=False)
    write(out/'summary-private.json', result); print(json.dumps({k: v for k, v in result.items() if k != 'views'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); audit(a.config.resolve())
