"""More-world/seed stability on eight TRAIN families and matched leading controls."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write


def audit(config_path):
    gpu_only(); c = read(config_path)
    if (c['method'] != 'reused_train_small_win_paired_world_stability_v1'
            or c['contexts'] != ['third_gap3000', 'fourth_gap5000'] or c['win_kinds'] != ['ron', 'tsumo']
            or c['families_per_bucket'] != 2 or c['unique_families_across_buckets'] is not True
            or c['maximum_opponent_riichi'] != 1 or c['matched_leading_control'] != 'hold_second'
            or c['worlds_per_block'] != 16 or c['planner_block_seeds'] != [202611231100, 202611231101]
            or c['expected_views'] != 16 or c['expected_source_families'] != 8 or c['expected_terminal_arms'] != 1024
            or c['continuation'] != 'ordinary_frozen_v8_future_wins_forced'
            or c['teacher_or_target_admission'] or c['student_fit']):
        raise ValueError('Fixed TRAIN-only stability diagnostic required')
    root = ROOT/c['data_root']; prior = root/'input-audit-v1'
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    scheduler_exit(read(ROOT/'logs/curriculum-terminal-reuse-audit-exit-317734.json'),
        (root/'audit-terminal-scheduler-private.txt').read_text(), ROOT/'cluster/curriculum-reused-terminal-audit-v1.slurm')
    proof = read(prior/'summary-private.json'); old_binding = read(prior/'completion-binding-private.json')
    if proof['status'] != 'reused_terminal_native_encoder_split_and_public_score_audit_passed' or proof['rows'] != 4580:
        raise ValueError('Actual completed full input audit required')
    if not all(sha(p) == h for p, h in old_binding['files'].items()): raise ValueError('Completed input audit bindings changed')
    registered = ROOT/c['source_registration']; reg = read(registered)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from tools.audit_curriculum_reused_terminal_v1 import restore
    from tools.curriculum_public_win_planner_v1 import plan, choose
    from tools.audit_curriculum_terminal_objective_v1 import decompose
    ready = read(root/'collection-v1/summary-private.json')
    rawpath = ROOT/ready['files']['train']['path']; eligible_path = prior/'eligible-small-win-proxies-private.json'
    if sha(rawpath) != ready['files']['train']['sha256']: raise ValueError('Raw TRAIN inputs changed')
    rows = {r['_meta']['id']: r for r in (json.loads(line) for line in rawpath.open())}
    controls = {(r['_meta']['physical_id'], r['_meta']['score_context']): r for r in rows.values()}
    eligible = {r['id'] for r in read(eligible_path) if r['split'] == 'train'}
    selected = []; used = set()
    for context in c['contexts']:
        for kind in c['win_kinds']:
            candidates = []
            for identifier in sorted(eligible):
                row = rows[identifier]; m, s = row['_meta'], row['state']
                kinds = {v['kind'] for v in s['win_context']['candidates'].values()}
                if (m['score_context'] == context and kinds == {kind} and m['group_id'] not in used
                        and sum(p != s['seat'] for p in s['riichi']) <= 1): candidates.append(row)
            taken = []
            for row in candidates:
                m = row['_meta']
                if m['group_id'] in used: continue
                control = controls.get((m['physical_id'], c['matched_leading_control']))
                if control is None: raise ValueError('Missing same-physical leading control')
                used.add(m['group_id']); taken.append(row)
                selected.extend([('pursuit', context, kind, row), ('leading_control', context, kind, control)])
                if len(taken) == c['families_per_bucket']: break
            if len(taken) != 2: raise ValueError('A predeclared TRAIN/kind/context bucket has fewer than two independent families')
    if len(selected) != 16 or len(used) != 8: raise ValueError('Fixed16-view/eight-family cohort required')
    paths = [Path(__file__), config_path, registered, rawpath, eligible_path, prior/'summary-private.json',
        prior/'completion-binding-private.json', root/'audit-terminal-scheduler-private.txt',
        ROOT/'logs/curriculum-terminal-reuse-audit-exit-317734.json',
        ROOT/'cluster/curriculum-terminal-world-stability-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-terminal-world-stability-20261123.md']
    paths += [ROOT/'tools'/name for name in ('curriculum_public_win_planner_v1.py', 'curriculum_terminal_scout_stats.py',
        'audit_curriculum_terminal_objective_v1.py', 'audit_curriculum_reused_terminal_v1.py',
        'curriculum_native_decisions.py', 'curriculum_native_worlds.py', 'curriculum_decision_history.py',
        'curriculum_hierarchical_prior.py', 'curriculum_public_inventory.py', 'curriculum_decision_rollout.py',
        'curriculum_endgame_ledger.py', 'audit_curriculum_history_worlds.py')]
    files = {str(p.resolve()): sha(p) for p in paths}
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    checkpoint = Path(reg['curriculum']['reference_checkpoint']); identity = bundle(checkpoint)
    if identity['identity_sha256'] != reg['curriculum']['reference_identity']: raise ValueError('Frozen v8 changed')
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(config=c, files=files,
        selected=[dict(cohort=a, bucket=[b, k], id=r['_meta']['id'], group=r['_meta']['group_id']) for a, b, k, r in selected],
        public_small_win_proxy_selection_not_future_value_selection=True, train_only=True,
        frozen_teacher_identity=identity, labels_generated=False))
    started = time.monotonic(); client = UnifiedPolicyClient(load_model(checkpoint))
    counts = Counter(); cohorts = {}; arm_count = 0; details = []
    with (out/'plans-private.jsonl').open('x') as stream:
        for cohort, context, kind, row in selected:
            env, obs = restore(row)
            blocks = [plan(env, obs, client, seed=seed, worlds=16) for seed in c['planner_block_seeds']]
            if any((p['accept'], p['decline']) != (blocks[0]['accept'], blocks[0]['decline']) for p in blocks):
                raise ValueError('Frozen ordinary decline action changed across independent sampling seeds')
            accept, decline = blocks[0]['accept'], blocks[0]['decline']
            pairs = blocks[0]['pairs']+blocks[1]['pairs']; decision = choose(pairs, accept, decline)
            pooled = dict(accept=accept, decline=decline, pairs=pairs, decision=decision)
            components = decompose(pooled)
            loo = [choose(pairs[:i]+pairs[i+1:], accept, decline)['selected'] for i in range(32)]
            stable_blocks = blocks[0]['decision']['selected'] == blocks[1]['decision']['selected']
            stable_loo = all(key == decision['selected'] for key in loo)
            item = dict(cohort=cohort, bucket=[context, kind], selected_decline=decision['selected'] == decline,
                independent_block_choices_agree=stable_blocks, every_leave_one_world_choice_agrees=stable_loo,
                pooled_component_cause=components['cause'], worlds=32, terminal_arms=64)
            cc = cohorts.setdefault(cohort, Counter()); cc['views'] += 1; cc['declines'] += item['selected_decline']
            cc['stable_blocks'] += stable_blocks; cc['stable_leave_one_world'] += stable_loo; cc[components['cause']] += 1
            arm_count += sum(p['full_terminal_arms'] for p in blocks); counts['completed_views'] += 1
            stream.write(json.dumps(dict(source_id=row['_meta']['id'], blocks=blocks, pooled=pooled,
                                         components=components, diagnostic=item))+'\n'); stream.flush()
            details.append(item); print(json.dumps(dict(stage='terminal_world_stability', **item)), flush=True)
    if arm_count != 1024 or bundle(checkpoint) != identity: raise ValueError('Incomplete arm budget or changed teacher')
    if not all(sha(p) == h for p, h in files.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Bound diagnostic or active training changed')
    result = dict(status='train_terminal_world_stability_diagnostic_complete', views=16, source_families=8,
        actual_terminal_arms=1024, cohorts={k: dict(v) for k, v in cohorts.items()}, actual_seconds=time.monotonic()-started,
        confidence_or_last_place_veto_used=False, diagnostic_worlds_not_independent_population_matches=True,
        future_inner_wins_forced=True, adaptive_outer_continuation_sensitivity_pending=True,
        prior_not_calibrated=True, teacher_admitted=False, labels_generated=0, student_updates=0,
        active_fit_dependencies_unchanged=len(active), raw_case_plans_and_events_private=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), goal_achieved=False)
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); audit(a.config.resolve())
