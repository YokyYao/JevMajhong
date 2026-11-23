"""Generate actual32-world rank-aware candidate labels for one fixed shard."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_rank_nonwin_teacher_v2 import propose, soft_target, METHOD
from tools.curriculum_rank_win_teacher_v1 import inner_plan


def collect(shard):
    gpu_only(); root = ROOT/'runs/curriculum-rank-nonwin-labels-v1'
    registered = root/'registered-private.json'; reg = read(registered); c = reg['config']
    if reg['status'] != 'public_nonwin120_views_and24_6_families_frozen_before_values' or shard not in range(c['shards']):
        raise ValueError('Actual fixed selected population and shard required')
    if any(sha(p) != h for p, h in {**reg['files'], **reg['active_fit_files']}.items()):
        raise ValueError('A frozen source or active3A input changed')
    source = read(reg['source_registration'])
    sys.path.insert(0, source['source']); sys.path.insert(0, source['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.agents import PolicyAgent, EfficiencyAgent, CallingOpponent
    from tools.audit_curriculum_reused_terminal_v1 import restore, projection
    from tools.curriculum_native_decisions import construct_world
    from tools.curriculum_observer_rollout_v1 import finish, WinReplanner
    from tools.audit_curriculum_history_worlds import derived_seed
    from tools.curriculum_endgame import public_placement
    from tools.curriculum_endgame_ledger import ledger
    from tools.curriculum_public_worlds import SamplingExhausted
    checkpoint = Path(reg['continuation_model']['directory'])
    if bundle(checkpoint) != reg['continuation_model']: raise ValueError('Frozen continuation changed')
    client = UnifiedPolicyClient(load_model(checkpoint))
    specs = reg['selected'][shard*c['views_per_shard']:(shard+1)*c['views_per_shard']]
    if not specs: raise ValueError('Empty fixed shard')
    wanted = {s['id'] for s in specs}; rows = {}
    for spec in reg['source_files'].values():
        path = ROOT/spec['path']
        if sha(path) != spec['sha256']: raise ValueError('Raw source changed')
        for line in path.open():
            row = json.loads(line)
            if row['_meta']['id'] in wanted: rows[row['_meta']['id']] = row
    if set(rows) != wanted: raise ValueError('Selected source rows missing')
    out = root/'shards'/f'shard-{shard:02d}'; out.mkdir(parents=True, exist_ok=False)
    case_dir = out/'cases-private'; case_dir.mkdir()
    write(out/'registered-private.json', dict(parent_registration_sha256=sha(registered),
        shard=shard, selected=specs, model=reg['continuation_model'],
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname()))
    counts = Counter(); contexts = {}; started = time.monotonic()
    with (out/'labels-private.jsonl').open('x') as labels, (out/'quarantine-private.jsonl').open('x') as quarantine:
        for index, spec in enumerate(specs):
            row = rows[spec['id']]; env, obs = restore(row); seat = obs.player_id
            pairs = []; blocks = []; planner_counts = Counter()
            try:
                for block_index, seed in enumerate(c['outer_seeds']):
                    planned = propose(env, obs, client, seed=seed)
                    if (planned['complete_riichi_class_menu']!=spec['full_RI_menu']
                        or planned['tested_menu_size']!=spec['prospective_candidate_count']):
                        raise ValueError('Actual native action menu differs from its public preregistered budget')
                    if public_placement(row['state'])['rank'] != planned['initial_public_rank']:
                        raise ValueError('Public initial rank changed during proposal')
                    if blocks and (planned['baseline'], planned['candidates']) != (blocks[0]['baseline'], blocks[0]['candidates']):
                        raise ValueError('Frozen current alternatives changed across blocks')
                    block_pairs = []; block_plans = []
                    for world, allocation in enumerate(planned['proposal']['allocations']):
                        fork, receipt = construct_world(env, obs, env.mjai_log[-1], allocation, True)
                        paired = {}; plans = {}
                        for arm in planned['candidates']:
                            agents = [CallingOpponent() if p == (seat+1) % 4 else EfficiencyAgent() for p in range(4)]
                            agents[seat] = PolicyAgent(client)
                            control = WinReplanner(client, seed=c['future_planner_seed'], worlds=4, max_calls=4, planner=inner_plan)
                            result = finish(fork.clone(), seat, arm, derived_seed(planned['public_seed'], world),
                                agents, observer=control, record_events=True)
                            ledger(result['merged_mjai_log'], result['terminal_scores'])
                            paired[arm] = result; plans[arm] = control.records; planner_counts.update(control.counts)
                            counts['complete_outer_arms'] += 1
                        block_pairs.append(paired); block_plans.append(dict(receipt=receipt, future_plans=plans))
                    blocks.append(dict(**planned, pairs=block_pairs, receipts_and_plans=block_plans))
                    pairs.extend(block_pairs)
                baseline, candidates = blocks[0]['baseline'], blocks[0]['candidates']
                value = soft_target(pairs, candidates, spec['public_rank'],
                    seed=derived_seed(blocks[0]['public_seed'], 512))
                normalized = projection(row); normalized['_meta'] = copy.deepcopy(row['_meta'])
                q = normalized['questions']['action']
                q.update(src='curriculum4:rank_nonwin:'+spec['category']+':'+spec['context'], label=value['label'],
                    target={k: value['target'].get(k, 0.) for k in q['criteria']})
                normalized['_meta']['rank_teacher'] = dict(method=METHOD, worlds=32,
                    bootstrap_samples=512, source_family_unchanged=True,
                    teacher_preference_not_expert_optimality=True, current_at_most_four_action_values_only=not blocks[0]['complete_riichi_class_menu'],
                    complete_RI_legal_menu=blocks[0]['complete_riichi_class_menu'],
                    tested_actions=blocks[0]['tested_menu_size'],legal_actions=blocks[0]['legal_menu_size'])
                case_path = case_dir/f'case-{index:03d}.json'
                write(case_path, dict(source_id=spec['id'], group=spec['group'], split=spec['split'],
                    blocks=blocks, soft_label=value, planner_counts=dict(planner_counts),
                    candidate_label=normalized, teacher_model=reg['continuation_model']['identity_sha256']))
                labels.write(json.dumps(normalized, separators=(',', ':'))+'\n'); labels.flush()
                counts['labelled_views'] += 1; counts['alternative_soft_argmax'] += value['label'] != baseline
                counts['block_agreement'] += value['independent_block_choices_agree']
                counts['leave_one_agreement'] += value['every_leave_one_world_choice_agrees']
                cc = contexts.setdefault(spec['split']+':'+spec['category']+':'+spec['context'], Counter())
                cc['views'] += 1; cc['alternative_soft_argmax'] += value['label'] != baseline
                cc['point_alternative'] += value['point_choice'] != baseline
                counts.update({'future:'+k: n for k, n in planner_counts.items()})
            except SamplingExhausted as error:
                # A bounded sampling failure is explicit; never use partial worlds
                # or replace this case after seeing the sampled action advantage.
                counts['sampling_quarantine'] += 1
                quarantine.write(json.dumps(dict(source_id=spec['id'], split=spec['split'],
                    group=spec['group'], reason='public_prior_sampling_exhausted',
                    attempts=error.attempts, accepted=error.accepted, requested=error.requested,
                    no_partial_target_admitted=True))+'\n'); quarantine.flush()
            print(json.dumps(dict(stage='rank_nonwin_label', shard=shard, processed=index+1, total=len(specs),
                labels=counts['labelled_views'], quarantine=counts['sampling_quarantine'],
                outer_arms=counts['complete_outer_arms'], seconds=time.monotonic()-started)), flush=True)
    if counts['labelled_views']+counts['sampling_quarantine'] != len(specs):
        raise ValueError('Every selected view must be labelled or explicitly quarantined')
    if bundle(checkpoint) != reg['continuation_model'] or any(
            sha(p) != h for p, h in {**reg['files'], **reg['active_fit_files']}.items()):
        raise ValueError('Teacher or source/active fit inputs changed during label collection')
    outputs = {name: sha(out/name) for name in ('labels-private.jsonl', 'quarantine-private.jsonl')}
    result = dict(status='fixed_rank_nonwin_label_shard_complete_candidate_targets_not_fit_admitted',
        shard=shard, selected_views=len(specs), counts=dict(counts), contexts={k: dict(v) for k, v in contexts.items()},
        files=outputs, actual_seconds=time.monotonic()-started, teacher_model=reg['continuation_model']['identity_sha256'],
        uncertainty_retained=True, last_place_rate_veto=False, positive_confidence_gate=False,
        policy_and_input_contract_admission_pending=True, student_updates=0, raw_traces_private=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
    write(out/'summary-private.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': collect(int(os.environ['SLURM_ARRAY_TASK_ID']))

