"""Read-only utility decomposition and public small-win coverage; no new games."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_curriculum_endgame_cases import sha
from tools.curriculum_public_win_planner_v1 import choose


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def decompose(value):
    accept, decline = value['accept'], value['decline']; pairs = value['pairs']
    check = choose(pairs, accept, decline)
    if check['selected'] != value['decision']['selected'] or check['worlds'] != value['decision']['worlds']:
        raise ValueError('Saved paired worlds do not reconstruct the observed choice')
    components = {}
    for arm in (accept, decline):
        components[arm] = dict(
            top2=statistics.mean(float(pair[arm]['top2']) for pair in pairs),
            rank=statistics.mean(.02 * (4 - pair[arm]['rank']) / 3 for pair in pairs),
            net=statistics.mean(.005 * math.tanh(pair[arm]['net_from_anchor'] / 8000) for pair in pairs))
        if (abs(sum(components[arm].values()) - value['decision']['mean_utility'][arm]) > 1e-12
                or abs(check['mean_utility'][arm] - value['decision']['mean_utility'][arm]) > 1e-12):
            raise ValueError('Utility components do not reproduce actual saved means')
    delta = {key: components[decline][key] - components[accept][key] for key in ('top2', 'rank', 'net')}
    if abs(sum(delta.values()) - value['decision']['paired_decline_minus_accept_utility']) > 1e-12:
        raise ValueError('Paired component differences do not reconstruct saved utility')
    declined = check['selected'] == decline
    if not declined:
        cause = 'accept'
    elif delta['top2'] > 1e-12:
        cause = 'sampled_top2_gain'
    elif delta['top2'] < -1e-12:
        cause = 'sampled_top2_loss_overridden_by_secondary'
    elif abs(delta['rank']) <= 1e-12:
        cause = 'top2_and_rank_tie_net_only'
    else:
        cause = 'top2_tie_secondary_rank_and_net'
    return dict(worlds=len(pairs), declined=declined, cause=cause,
                components=components, decline_minus_accept=delta)


def audit(out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
            or socket.gethostname().startswith(('lo-', 'login'))):
        raise RuntimeError('Finite scheduled GPU resources are required')
    out.mkdir(parents=True, exist_ok=False)
    first = ROOT / 'runs/curriculum-public-win-planner-v1'
    outer = ROOT / 'runs/curriculum-observer-rollout-v1'
    source = ROOT / 'runs/curriculum-stage4-placement-mortal-data-v4'
    ready_path = source / 'data/data-ready-private.json'; ready = read(ready_path)
    coverage_path = source / 'input-audit-v1/summary-private.json'; coverage = read(coverage_path)
    if coverage['data_ready_sha256'] != sha(ready_path):
        raise ValueError('Existing public-score audit refers to different data')
    for root, job, prefix in ((first, 317376, 'curriculum-public-win-planner'),
                              (outer, 317406, 'curriculum-observer-outer')):
        text = (root / 'pilot-terminal-scheduler-private.txt').read_text()
        if ('JobState=COMPLETED' not in text or 'ExitCode=0:0' not in text
                or read(ROOT / f'logs/{prefix}-exit-{job}.json')['exit_code'] != 0):
            raise ValueError('An actual successful planner prerequisite is missing')
        if not all(sha(p) == h for p, h in read(root / 'pilot-v1/registered-private.json')['files'].items()):
            raise ValueError('Bound completed planner source/input changed')
    active_path = ROOT / 'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json'
    active = read(active_path)['files'] if active_path.exists() else {}
    if not all(sha(p) == h for p, h in active.items()):
        raise ValueError('An active training dependency changed')
    details_path = source / 'input-audit-v1/public-score-details-private.jsonl'
    paths = [Path(__file__), ready_path, coverage_path, details_path,
        ROOT / 'tools/curriculum_public_win_planner_v1.py', ROOT / 'tools/curriculum_terminal_scout_stats.py',
        ROOT / 'cluster/curriculum-terminal-objective-v1.slurm',
        ROOT / 'docs/experiments/plans/curriculum-terminal-objective-20261123.md']
    for root in (first, outer):
        paths += [root / 'pilot-v1/registered-private.json', root / 'pilot-v1/summary-private.json',
                  root / 'pilot-terminal-scheduler-private.txt']
    paths += [first / 'pilot-v1/plans-private.jsonl', outer / 'pilot-v1/real-future-plan-fixture-private.json',
              outer / 'pilot-v1/changed-outer-matches-private.jsonl',
              ROOT / 'logs/curriculum-public-win-planner-exit-317376.json',
              ROOT / 'logs/curriculum-observer-outer-exit-317406.json']
    dataset = {}; groups = defaultdict(set)
    for split, spec in ready['files'].items():
        path = ROOT / spec['path']; paths.append(path)
        if sha(path) != spec['sha256']:
            raise ValueError('Canonical input data changed')
        for row in rows(path):
            meta = row['_meta']
            if meta['split'] != split or meta['id'] in dataset:
                raise ValueError('Changed split or duplicate canonical source ID')
            dataset[meta['id']] = row; groups[split].add(meta['group_id'])
    if len(dataset) != 925 or groups['train'] & groups['validation']:
        raise ValueError('Canonical source population/split differs')
    bindings = {str(p.resolve()): sha(p) for p in paths}
    write(out / 'registered-private.json', dict(method='read_only_terminal_utility_and_coverage_v1',
        files=bindings, sources='all four initial plans, all future fixture/candidate plans and925 canonical rows',
        expected_plans=9, expected_cached_terminal_arm_records=68,
        small_win_proxy='supported public no-ura sole win income<=3000, start rank3/4, immediate solo rank>2',
        proxy_not_guaranteed_final_rank_or_actual_ura_value=True,
        no_inference_games_labels_updates=True))
    planned = []
    for record in rows(first / 'pilot-v1/plans-private.jsonl'):
        if record['original'] != record['changed_hidden']:
            raise ValueError('Saved original hidden-world control differs')
        planned.append(('initial_four_contexts', record['original']))
    fixture = read(outer / 'pilot-v1/real-future-plan-fixture-private.json')
    if len(fixture['plans']) != fixture['counts']['planned_calls']:
        raise ValueError('Fixture plan count differs')
    planned.extend(('synthetic_future_fixture', value) for value in fixture['plans'])
    for record in rows(outer / 'pilot-v1/changed-outer-matches-private.jsonl'):
        if len(record['plans']) != record['aggregate']['planner_counts'].get('planned_calls', 0):
            raise ValueError('Saved later-plan count differs')
        planned.extend(('TRAIN_future_plans', value) for value in record['plans'])
    plan_counts = defaultdict(Counter); private_plans = []; arms = 0
    for cohort, value in planned:
        result = decompose(value); c = plan_counts[cohort]
        c['plans'] += 1; c['declines'] += result['declined']; c[result['cause']] += 1
        arms += 2 * result['worlds']; private_plans.append(dict(cohort=cohort, result=result))
    if len(planned) != 9 or arms != 68:
        raise ValueError('The complete actual saved-plan population differs')
    write(out / 'utility-components-private.json', private_plans)
    score_rows = rows(details_path); scores = {r['id']: r for r in score_rows}
    win_ids = {identifier for identifier, r in dataset.items() if r['_meta']['category'] == 'win'}
    if len(scores) != len(score_rows) or set(scores) != win_ids:
        raise ValueError('Existing public-score details do not cover every canonical win view')
    counts = defaultdict(Counter); physical = defaultdict(set); families = defaultdict(set); eligible = []
    for identifier in sorted(win_ids):
        row = dataset[identifier]; meta = row['_meta']; detail = scores[identifier]
        if detail['group_id'] != meta['group_id'] or detail['split'] != meta['split']:
            raise ValueError('Public score detail changed its source family/split')
        key = meta['split'] + ':' + meta['score_context']; c = counts[key]
        c['win_views'] += 1; score = detail['public_solo_score']
        if not score['supported']:
            c['unsupported:' + score['reason']] += 1; continue
        c['supported_no_ura_solo_views'] += 1
        rank = meta['placement']['rank']
        identity = tuple(meta[k] for k in ('source_logical_seed', 'source_hand_index', 'source_step', 'seat'))
        if score['own_income'] <= 3000:
            c['all_supported_income_le3000'] += 1
            if rank <= 2 and score['immediate_rank'] <= 2:
                c['leading_small_win_retains_immediate_top2_proxy'] += 1
                physical[meta['split'] + ':LEADING_SMALL'].add(identity)
                families[meta['split'] + ':LEADING_SMALL'].add(meta['group_id'])
        if rank not in (3, 4):
            continue
        c['supported_start_rank3or4'] += 1
        if score['own_income'] <= 3000:
            c['income_le3000'] += 1
        if score['own_income'] > 3000 or score['immediate_rank'] <= 2:
            continue
        c['insufficient_small_win_proxy'] += 1
        contexts = row['state']['win_context']['candidates']
        kinds = {v['kind'] for v in contexts.values()}
        if len(kinds) != 1:
            raise ValueError('Ambiguous canonical native win kind')
        c['proxy_kind:' + next(iter(kinds))] += 1
        c['proxy_open' if score['open'] else 'proxy_closed'] += 1
        threats = sum(p != meta['seat'] for p in row['state']['riichi'])
        c['proxy_opponent_RI:' + str(threats)] += 1
        c['proxy_income_le1000' if score['own_income'] <= 1000 else
          'proxy_income_le2000' if score['own_income'] <= 2000 else 'proxy_income_le3000'] += 1
        physical[key].add(identity); families[key].add(meta['group_id'])
        physical[meta['split'] + ':ALL'].add(identity); families[meta['split'] + ':ALL'].add(meta['group_id'])
        eligible.append(dict(id=identifier, split=meta['split'], group=meta['group_id'],
                             context=meta['score_context'], public_solo_score=score,
                             public_proxy_only_not_action_target=True))
    if not all(sha(p) == h for p, h in bindings.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Read-only audit changed a bound input')
    write(out / 'eligible-source-inventory-private.json', eligible)
    summary = dict(status='terminal_objective_components_and_public_coverage_audited',
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
        plans=len(planned), cached_terminal_arm_records=arms,
        plan_cohorts={k: dict(v) for k, v in plan_counts.items()},
        canonical_rows=925, canonical_win_views=len(win_ids),
        coverage={k: dict(v) for k, v in counts.items()},
        proxy_distinct_physical_decisions={k: len(v) for k, v in physical.items()},
        proxy_distinct_source_families={k: len(v) for k, v in families.items()},
        active_fit_dependencies_unchanged=len(active),
        old_public_no_ura_score_cache_used_not_recomputed=True,
        no_ura_solo_proxy_is_not_guaranteed_final_placement=True,
        sampled_mean_utility_is_not_calibrated_or_expert_truth=True,
        new_inference_calls=0, new_games=0, new_training_labels=0, optimizer_updates=0,
        reward_changed=False, stage4_fit_admitted=False, final_grid_unused=True,
        raw_details_private=True)
    write(out / 'summary-private.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); audit(args.out.resolve())
