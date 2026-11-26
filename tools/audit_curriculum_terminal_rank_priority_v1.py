"""Cached-only proposed objective comparison; no new teacher/student execution."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import socket
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write


def audit(out):
    gpu_only()
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    base = ROOT/'runs/curriculum-terminal-world-stability-v1'; changed = ROOT/'runs/curriculum-terminal-continuation-sensitivity-v1'
    for root, job, prefix in ((base, 317775, 'curriculum-terminal-world-stability'),
                              (changed, 317824, 'curriculum-terminal-future-sensitivity')):
        scheduler_exit(read(ROOT/f'logs/{prefix}-exit-{job}.json'),
            (root/'pilot-terminal-scheduler-private.txt').read_text())
        if not all(sha(p) == h for p, h in read(root/'pilot-v1/registered-private.json')['files'].items()):
            raise ValueError('Completed diagnostic source/input changed')
    registration = read(base/'pilot-v1/registered-private.json'); summary = read(changed/'pilot-v1/summary-private.json')
    cache_path = base/'pilot-v1/plans-private.jsonl'; changed_path = changed/'pilot-v1/changed-outer-arms-private.jsonl'
    plans = [json.loads(line) for line in cache_path.open()]; cached = {r['source_id']: r for r in plans}
    cases = [json.loads(line) for line in changed_path.open()]
    adaptive = {(r['source_id'], r['block'], r['world'], r['initial_arm']): r['result'] for r in cases}
    if len(cached) != 16 or len(adaptive) != 128 or len(cases) != 128: raise ValueError('All16-view/128-changed-arm caches required')
    data_root = ROOT/'runs/curriculum-reused-terminal-v1/collection-v1'; ready = read(data_root/'summary-private.json')
    rawpath = ROOT/ready['files']['train']['path']
    if sha(rawpath) != ready['files']['train']['sha256']: raise ValueError('Public score-view data changed')
    rows = {r['_meta']['id']: r for r in (json.loads(line) for line in rawpath.open())}
    files = {str(p.resolve()): sha(p) for p in (Path(__file__), ROOT/'tools/curriculum_terminal_rank_priority_v1.py',
        ROOT/'tests/integration/test_curriculum_terminal_rank_priority_v1.py',
        ROOT/'cluster/curriculum-terminal-rank-priority-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-terminal-rank-priority-20261123.md',
        base/'pilot-v1/registered-private.json', changed/'pilot-v1/summary-private.json', cache_path, changed_path, rawpath)}
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    reg = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from tools.curriculum_endgame import public_placement
    from tools.curriculum_public_win_planner_v1 import choose as scalar
    from tools.curriculum_terminal_rank_priority_v1 import choose, METHOD
    from tools.audit_curriculum_terminal_objective_v1 import decompose
    out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files, method=METHOD,
        source_views='all16 previously fixed views, no outcome-based case selection',
        scenarios=['ordinary32', 'ordinary4', 'adaptive4'],
        no_new_rollouts_or_student_updates=True, not_new_future_planning_policy=True))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests/integration'), pattern='test_curriculum_terminal_rank_priority_v1.py')
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful() or tested.testsRun != 3: raise ValueError('Independent objective priority checks failed')
    totals = defaultdict(Counter); details = []
    if len(registration['selected']) != len(summary['views']): raise ValueError('Original view order differs')
    for spec, old_summary in zip(registration['selected'], summary['views']):
        cache = cached[spec['id']]; row = rows[spec['id']]
        initial_rank = public_placement(row['state'])['rank']
        if initial_rank != row['_meta']['placement']['rank'] or (spec['cohort'] == 'leading_control') != (initial_rank == 2):
            raise ValueError('Public initial rank or original cohort differs')
        accept, decline = cache['pooled']['accept'], cache['pooled']['decline']
        ordinary = [cache['blocks'][b]['pairs'][w] for b in (0, 1) for w in (0, 1)]
        changed_pairs = [{a: adaptive[spec['id'], b, w, a] for a in (accept, decline)} for b in (0, 1) for w in (0, 1)]
        for scenario, pairs in (('ordinary32', cache['pooled']['pairs']), ('ordinary4', ordinary), ('adaptive4', changed_pairs)):
            old = scalar(pairs, accept, decline); new = choose(pairs, accept, decline, initial_rank)
            if scenario == 'ordinary32' and old != cache['pooled']['decision']: raise ValueError('Original32 scalar decision does not reproduce')
            if scenario != 'ordinary32':
                prefix = 'ordinary' if scenario == 'ordinary4' else 'replanned'
                if (old_summary[prefix+'_subset_declines'] != (old['selected'] == decline)
                        or abs(old_summary[prefix+'_subset_utility_difference']-old['paired_decline_minus_accept_utility']) > 1e-12):
                    raise ValueError('Matched four-world original decision/values differ')
            cause = decompose(dict(accept=accept, decline=decline, pairs=pairs, decision=old))['cause']
            cc = totals[scenario+':'+spec['cohort']]; cc['views'] += 1
            cc['scalar_declines'] += old['selected'] == decline; cc['proposed_declines'] += new['selected'] == decline
            cc['changed_choices'] += old['selected'] != new['selected']
            if old['selected'] != new['selected']: cc['changed_old_cause:'+cause] += 1
            if new['selected'] == decline: cc['proposed_decline_component:'+new['decision_component']] += 1
            item = dict(source_id=spec['id'], scenario=scenario, old=old, proposed=new, original_scalar_cause=cause)
            if scenario == 'ordinary32':
                blocks = [choose(p['pairs'], accept, decline, initial_rank)['selected'] for p in cache['blocks']]
                loo = [choose(pairs[:i]+pairs[i+1:], accept, decline, initial_rank)['selected'] for i in range(32)]
                cc['proposed_two_blocks_agree'] += blocks[0] == blocks[1]
                cc['proposed_all_leave_one_agree'] += all(key == new['selected'] for key in loo)
            details.append(item)
    if not all(sha(p) == h for p, h in files.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Cached objective audit changed a bound artifact')
    write(out/'objective-comparisons-private.json', details)
    result = dict(status='cached_public_rank_priority_objective_comparison_complete', method=METHOD,
        aggregate={k: dict(v) for k, v in totals.items()}, numerical_tests=3,
        source_views=16, source_families=8, unique_existing_ordinary_arms=1024, existing_changed_outer_arms=128,
        new_model_calls=0, new_terminal_arms=0, student_updates=0, labels_generated=0,
        proposed_objective_not_executed_in_future_planning=True, not_proof_of_teacher_or_policy_improvement=True,
        active_fit_dependencies_unchanged=len(active), raw_details_private=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), goal_achieved=False)
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); audit(a.out.resolve())
