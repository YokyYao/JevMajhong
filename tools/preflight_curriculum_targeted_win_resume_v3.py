"""Bind complete-world recovery after roundtripping actual saved case targets.

This performs native reconstruction and ledger/data checks in a scheduled GPU
allocation. It neither loads a policy nor runs any continuation or optimization.
"""
from collections import Counter
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import curriculum_targeted_win_resume_v2 as v2
from tools import curriculum_targeted_win_resume_v3 as v3
from tools.curriculum_targeted_world_cache_v1 import CompleteWorldCache, METHOD, normalized
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields

PUBLIC = ('tools/curriculum_targeted_world_cache_v1.py',
    'tools/curriculum_targeted_win_resume_v3.py',
    'tools/preflight_curriculum_targeted_win_resume_v3.py',
    'tests/core/test_curriculum_targeted_world_cache_v1.py',
    'cluster/curriculum-targeted-win-resume-v3.slurm',
    'docs/experiments/plans/curriculum-targeted-win-resume-worlds-20261124.md')


def observer_counts(paired, plans):
    """Reconstruct only cases whose excluded diagnostic branches are zero."""
    result = Counter()
    for arm, outcome in paired.items():
        offers = outcome['observer_hook_calls']; records = plans[arm]
        if offers: result['legal_win_offers'] += offers
        if records:
            result['planned_calls'] += len(records)
            result['inner_terminal_arms'] += sum(p['full_terminal_arms'] for p in records)
            result['selected_declines'] += sum(p['decision']['selected'] == p['decline'] for p in records)
        if offers-len(records): result['budget_fallback_wins'] += offers-len(records)
    return result


def main():
    allocation = resource(); started = time.monotonic()
    if v3.BINDING.exists(): raise ValueError('Never overwrite a frozen recovery entry')
    old = read(v2.BINDING)
    if (old['status'] != 'targeted_win_resume_entry_bound_not_started'
            or any(sha(p) != h for p, h in old['files'].items())):
        raise ValueError('Previous source and recovery proof must remain frozen')
    reg_path = v2.RUN/'registered-private.json'; reg = read(reg_path)
    source = read(reg['source_registration'])
    sys.path[:0] = [source['source'], source['training_config']['upstream_source']]
    from tools.curriculum_targeted_win_source_v1 import restore
    from tools.curriculum_native_decisions import construct_world
    from tools.curriculum_endgame_ledger import ledger
    from tools.curriculum_rank_win_controls_v1 import soft_target
    from tools.audit_curriculum_history_worlds import derived_seed
    log_path = ROOT/'logs/curriculum-targeted-world-cache-tests-v1.log'
    log = log_path.read_text()
    if 'Ran 7 tests' not in log or not log.endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Seven cache and unchanged-calculation tests required')
    files = {**old['files'], str(v2.BINDING): sha(v2.BINDING)}
    for p in (log_path, *[ROOT/p for p in PUBLIC]): files[str(p.resolve())] = sha(p)
    for name in PUBLIC:
        if name.endswith('.py'): ast.parse((ROOT/name).read_text())
    subprocess.run(['bash', '-n', str(v3.ENTRY)], check=True, timeout=10)
    compile(v3.collector_tree(v2.COLLECTOR.read_text()), str(v2.COLLECTOR), 'exec')
    compile(v3.driver_tree(Path(v2.__file__).read_text()), str(v2.__file__), 'exec')
    out = v3.RUN/'entry-preflight-v2'; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files, allocation=allocation,
        proof_scope='first_three_fixed_complete_training_cases_no_new_rollouts', student_updates=0))
    try:
        specs = reg['selected'][:3]
        if len(specs) != 3 or any(s['split'] != 'train' for s in specs):
            raise ValueError('Only fixed first three training cases may enter this software proof')
        rows = {}
        for entry in reg['source_files'].values():
            path = ROOT/entry['path']
            if sha(path) != entry['sha256']: raise ValueError('Original source changed')
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['_meta']['id'] in {s['id'] for s in specs}: rows[row['_meta']['id']] = row
        identity = dict(purpose='software_roundtrip_only_not_production_reusable',
            collection_sha256=sha(reg_path), test_registration_sha256=sha(out/'registered-private.json'))
        writer = CompleteWorldCache(out/'world-cache-private', [], identity, ledger)
        reader = CompleteWorldCache(out/'readback-cache-private', [out/'world-cache-private'], identity, ledger)
        worlds = 0; arms = 0; proofs = []
        for index, spec in enumerate(specs):
            path = v2.RUN/'shards/shard-00/cases-private'/f'case-{index:03d}.json'
            case = read(path); files[str(path.resolve())] = sha(path)
            v2.audit_case(case, spec, reg)
            if any(case['planner_counts'].get(k, 0) for k in ('outside_late_scope', 'unsupported_current_kan_offer')):
                raise ValueError('Cannot infer per-world excluded observer diagnostics from aggregate counters')
            env, obs = restore(rows[spec['id']]); counts = Counter(); recovered = []
            for block_index, block in enumerate(case['blocks']):
                for world, paired in enumerate(block['pairs']):
                    plan = block['receipts_and_plans'][world]
                    saved_allocation = block['proposal']['allocations'][world]
                    # JSON case serialization turns the native integer player
                    # keys into strings. Restore those two typed maps only.
                    allocation_row = dict(saved_allocation)
                    for field in ('opponent_hands', 'opponent_temporary_furiten'):
                        allocation_row[field] = {int(k): v for k, v in saved_allocation[field].items()}
                    fork, receipt = construct_world(env, obs, env.mjai_log[-1], allocation_row, True)
                    if normalized(receipt) != normalized(plan['receipt']):
                        raise ValueError('Actual restored physical world differs from original receipt')
                    context = writer.make_context(spec, block_index, world, allocation_row, receipt,
                        block, counts, obs.player_id, list(fork.scores()))
                    counts.update(observer_counts(paired, plan['future_plans']))
                    writer.publish(context, paired, plan['future_plans'], counts)
                    packet = reader.lookup(context)
                    if packet['paired'] != normalized(paired) or packet['plans'] != normalized(plan['future_plans']):
                        raise ValueError('Serialized complete-world outcomes or plans changed')
                    if packet['planner_after'] != dict(counts): raise ValueError('Observer counters changed')
                    recovered.append(packet['paired']); worlds += 1; arms += len(paired)
            if dict(counts) != case['planner_counts']: raise ValueError('Complete original observer diagnostics differ')
            first = case['blocks'][0]
            target = soft_target(recovered, first['candidates'], spec['public_rank'],
                seed=derived_seed(first['public_seed'], 512))
            if target != case['soft_label']: raise ValueError('Original full32-world soft target changed')
            proofs.append(dict(source_id=spec['id'], original_case_sha256=sha(path),
                full32_world_target_equal=True, complete_diagnostic_counters_equal=True))
        writer.seal(out/'written-cache-binding-private.json'); reader.seal(out/'readback-cache-binding-private.json')
        actual = scheduler('319759_0'); values = fields(actual.stdout)
        if actual.returncode or values.get('JobState') != 'RUNNING':
            raise ValueError('Observe original live writer; this proof never restarts it')
        summary = dict(status='complete_world_recovery_bound_after_actual_saved_target_roundtrip',
            unit_tests_passed=7, actual_complete_cases=3, actual_complete_worlds=worlds,
            actual_complete_outer_arms=arms, actual_physical_world_receipts_equal=True,
            original_full32_world_soft_targets_equal=True, exact_observer_diagnostics_equal=True,
            original_per_world_calculation_AST_unchanged=True, source_case_proofs=proofs,
            excluded_diagnostic_reconstruction_scope='outside_late_scope_and_unsupported_current_kan_zero',
            production_cache_validates_all_seven_counters=True, new_rollouts=0, model_forwards=0,
            model_loaded=False, student_updates=0, original_writer_live=True,
            original_writer_not_restarted=True, continuation_jobs_submitted=False,
            requires_actual_prior_terminals=True, final_merge_and_projection_pending=True,
            actual_seconds=time.monotonic()-started)
        write(out/'summary-private.json', summary)
        write(out/'live-writer-observation-private.json', dict(raw=actual.stdout))
        for p in (out/'registered-private.json', out/'summary-private.json',
                out/'written-cache-binding-private.json', out/'readback-cache-binding-private.json',
                out/'live-writer-observation-private.json'):
            files[str(p.resolve())] = sha(p)
        if any(sha(p) != h for p, h in files.items()): raise ValueError('A frozen proof source changed')
        write(v3.BINDING, dict(status='targeted_win_resume_entry_bound_not_started',
            world_cache_method=METHOD, files=files, original_collection_registration_sha256=sha(reg_path),
            proof=str(out/'summary-private.json'), no_jobs_submitted=True, raw_records_private=True))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(error=repr(error))); raise


if __name__ == '__main__': main()
