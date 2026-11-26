"""Merge fixed terminal candidate labels with exact lineage and outcome checks."""
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.curriculum_rank_win_teacher_v1 import soft_target, METHOD
from tools.curriculum_endgame_ledger import ledger


def merge():
    gpu_only(); root = ROOT/'runs/curriculum-rank-win-labels-v1'
    reg = read(root/'registered-private.json'); c = reg['config']
    if any(sha(p) != h for p, h in {**reg['files'], **reg['active_fit_files']}.items()):
        raise ValueError('Frozen source/active fit changed')
    selected = {s['id']: s for s in reg['selected']}; labels = {}; quarantined = {}; contexts = defaultdict(Counter)
    total = Counter(); bound = dict(reg['files'])
    for shard in range(c['shards']):
        folder = root/'shards'/f'shard-{shard:02d}'; summary = read(folder/'summary-private.json')
        launch = read(folder/'registered-private.json')
        if (summary['status'] != 'fixed_rank_win_label_shard_complete_candidate_targets_not_fit_admitted'
                or launch['parent_registration_sha256'] != sha(root/'registered-private.json')
                or summary['teacher_model'] != reg['continuation_model']['identity_sha256']):
            raise ValueError('All exact completed shards with unchanged teacher required')
        expected = reg['selected'][shard*c['views_per_shard']:(shard+1)*c['views_per_shard']]
        if launch['selected'] != expected: raise ValueError('Shard selection changed')
        for name, digest in summary['files'].items():
            path = folder/name
            if sha(path) != digest: raise ValueError('Completed shard output changed')
            bound[str(path.resolve())] = digest
        for name in ('summary-private.json', 'registered-private.json'):
            bound[str((folder/name).resolve())] = sha(folder/name)
        for line in (folder/'labels-private.jsonl').open():
            row = json.loads(line); rid = row['_meta']['id']; spec = selected[rid]; q = row['questions']['action']
            if (rid in labels or row['_meta']['group_id'] != spec['group'] or row['_meta']['split'] != spec['split']
                    or row['_meta']['score_context'] != spec['context']
                    or q['src'] != 'curriculum4:rank_win:'+spec['context']
                    or set(q['target']) != set(q['criteria'])
                    or any(not math.isfinite(v) or v < 0 for v in q['target'].values())
                    or abs(sum(q['target'].values())-1) > 1e-12
                    or sum(v > 0 for v in q['target'].values()) != 2
                    or q['label'] != max(q['target'], key=q['target'].get)):
                raise ValueError('Candidate target, public source identity or group changed')
            labels[rid] = row
        for line in (folder/'quarantine-private.jsonl').open():
            row = json.loads(line); rid = row['source_id']
            if rid in quarantined or row['reason'] != 'public_prior_sampling_exhausted' or not row['no_partial_target_admitted']:
                raise ValueError('Explicit bounded sampler quarantine required')
            quarantined[rid] = row
        for index, spec in enumerate(expected):
            path = folder/'cases-private'/f'case-{index:03d}.json'
            if spec['id'] in quarantined:
                if path.exists(): raise ValueError('Quarantined view has an admitted case')
                continue
            case = read(path); pairs = [p for block in case['blocks'] for p in block['pairs']]
            if (case['source_id'] != spec['id'] or case['group'] != spec['group']
                    or len(case['blocks']) != 2 or len(pairs) != 32
                    or case['teacher_model'] != reg['continuation_model']['identity_sha256']):
                raise ValueError('Complete paired32-world case required')
            for pair in pairs:
                for outcome in pair.values():
                    ledger(outcome['merged_mjai_log'], outcome['terminal_scores'])
                    if not outcome['full_match_completed'] or not outcome['opponents_no_legal_win_declines_verified']:
                        raise ValueError('Complete native observer-aware terminal outcome required')
            first = case['blocks'][0]
            from tools.audit_curriculum_history_worlds import derived_seed
            actual = soft_target(pairs, first['accept'], first['decline'], spec['public_rank'],
                seed=derived_seed(first['public_seed'], 512))
            if actual != case['soft_label'] or labels[spec['id']] != case['candidate_label']:
                raise ValueError('Saved target does not reconstruct from actual paired worlds')
            cc = contexts[spec['split']+':'+spec['context']]; cc['views'] += 1
            cc['decline_soft_argmax'] += actual['label'] == first['decline']
            cc['block_agreement'] += actual['independent_block_choices_agree']
            cc['leave_one_agreement'] += actual['every_leave_one_world_choice_agrees']
            bound[str(path.resolve())] = sha(path)
        total.update(summary['counts'])
    if set(labels) & set(quarantined) or set(labels) | set(quarantined) != set(selected):
        raise ValueError('Every fixed view must appear exactly once, including quarantine')
    out = root/'merged-v1'; out.mkdir(parents=True, exist_ok=False); files = {}
    with (out/'catalog-private.jsonl').open('x') as catalog:
        for split in ('train', 'validation'):
            pool = [labels[s['id']] for s in reg['selected'] if s['split'] == split and s['id'] in labels]
            path = out/f'{split}-rank-win-targets-private.jsonl'
            with path.open('x') as stream:
                for row in pool:
                    m = row['_meta']; q = row['questions']['action']
                    stream.write(json.dumps(row, separators=(',', ':'))+'\n')
                    catalog.write(json.dumps(dict(id=m['id'], group=m['group_id'], physical=m['physical_id'],
                        split=split, context=m['score_context'], public_rank=m['placement']['rank'],
                        source=q['src'], phase='late_legal_win', label=q['label'],
                        target_support=[k for k, v in q['target'].items() if v > 0],
                        teacher_preference_uncertainty=1-max(q['target'].values())))+'\n')
            files[split] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                records=len(pool), groups=len({r['_meta']['group_id'] for r in pool}))
    if any(sha(p) != h for p, h in reg['files'].items()): raise ValueError('Frozen recipe changed')
    ready = dict(status='actual_rank32_adaptive_soft_targets_generated_input_policy_and_fit_admission_pending',
        method=METHOD, files=files, selected_views=263, labelled_views=len(labels), quarantined_views=len(quarantined),
        counts=dict(total), by_context={k: dict(v) for k, v in contexts.items()},
        all_original_families_and_partitions_preserved=True, uncertainty_not_positive_only_filtered=True,
        calibrated_teacher_optimality=False, student_updates=0, ordinary_v8_continuation_with_public_heuristic_opponents=True,
        non_win_terminal_labels_still_pending=True, exact_win_input_and_serving_contract_still_pending=True,
        raw_rows_walls_outcomes_private=True, final_grid_unused=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
    write(out/'data-ready-private.json', ready)
    bound[str((root/'registered-private.json').resolve())] = sha(root/'registered-private.json')
    write(out/'completion-binding-private.json', dict(files=bound))
    write(out/'summary-public.json', {k: v for k, v in ready.items() if k != 'files'})
    print(json.dumps(read(out/'summary-public.json')), flush=True)


if __name__ == '__main__': merge()
