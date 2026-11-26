"""Generate native initial decisions for later yaku-route teaching.

These deliberately biased initial deals are unlabeled source cases, not SFT
examples, natural-game samples, completed games or optimality evidence.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
from itertools import permutations
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha

ROUTES = ('kokushi', 'chuuren', 'suuankou', 'daisangen', 'chiitoitsu')
METHOD = 'constrained_full_wall_native_initial_yaku_decisions_unlabelled_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def canonical_shape(tiles):
    """Keep suit permutations and rank-reflection variants in the same family."""
    counts = [0]*34
    for tile in tiles:
        counts[tile] += 1
    variants = []
    for order in permutations(range(3)):
        for flips in range(8):
            suited = []
            for i, old in enumerate(order):
                block = counts[old*9:old*9+9]
                suited.extend(reversed(block) if flips & (1 << i) else block)
            variants.append(tuple(suited+counts[27:]))
    return min(variants)


def complete_target(route, rng):
    if route == 'kokushi':
        tiles = [0, 8, 9, 17, 18, 26, *range(27, 34)]
        return tiles+[rng.choice(tiles)]
    if route == 'chuuren':
        suit = rng.randrange(3)*9
        return [suit+i for i, n in enumerate((3, 1, 1, 1, 1, 1, 1, 1, 3)) for _ in range(n)]+[suit+rng.randrange(9)]
    if route == 'suuankou':
        values = rng.sample(range(34), 5)
        return [t for t in values[:4] for _ in range(3)]+[values[4]]*2
    if route == 'daisangen':
        start = rng.randrange(3)*9+rng.randrange(7)
        pair = rng.choice([t for t in range(31) if t not in range(start, start+3)])
        return [31]*3+[32]*3+[33]*3+list(range(start, start+3))+[pair]*2
    if route == 'chiitoitsu':
        return [t for t in rng.sample(range(34), 7) for _ in range(2)]
    raise ValueError('Unregistered target route')


def noise_tile(tiles, excluded, rng):
    counts = Counter(tiles)
    return rng.choice([t for t in range(34) if t not in excluded and counts[t] < 4])


def scenario_plans(c):
    plans = []
    for route_no, route in enumerate(ROUTES):
        for template in range(c['templates_per_route']):
            base_seed = c['seed_start']+10000*route_no+100*template
            target = complete_target(route, random.Random(base_seed))
            for damage in c['removed_target_tiles']:
                rng = random.Random(base_seed+damage)
                shuffled = list(target); rng.shuffle(shuffled)
                own = shuffled[damage:]
                while len(own) < 13:
                    own.append(noise_tile(own, set(target), rng))
                for world in range(c['world_variants_per_public13']):
                    seed = base_seed+damage*10+world
                    first = noise_tile(own, set(target), random.Random(seed))
                    plans.append(dict(route=route, template=template, removed_target_tiles=damage,
                        initial13=own, first_draw_type=first, seed=seed, world_variant=world))
    for template in range(c['natural_control_templates']):
        seed = c['seed_start']+100000+100*template
        own = [t//4 for t in random.Random(seed).sample(range(136), 13)]
        for world in range(c['world_variants_per_public13']):
            first = noise_tile(own, set(), random.Random(seed+world+1))
            plans.append(dict(route='unselected_random_initial_hand_control', template=template,
                removed_target_tiles=None, initial13=own, first_draw_type=first,
                seed=seed+world+1, world_variant=world))
    if len(plans) != 512:
        raise ValueError('Exact source pilot has512 planned decisions')

    # Build family components BEFORE native collection/teacher inference.
    # Shared initial13 worlds and identical/symmetry-equivalent14 shapes cannot
    # land on opposite sides merely because the draw/dora/physical copy differs.
    parents = list(range(len(plans))); seen = {}
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    keys = []
    for i, plan in enumerate(plans):
        own = plan['initial13']
        shape_keys = (('initial13', canonical_shape(own)),
                      ('decision14', canonical_shape(own+[plan['first_draw_type']])))
        keys.append(shape_keys)
        for key in shape_keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    components = defaultdict(set)
    for i, shapes in enumerate(keys):
        components[find(i)].update(shapes)
    for i, plan in enumerate(plans):
        family = digest(sorted(components[find(i)]))
        plan['group_id'] = 'yaku-v1:'+family
        plan['split'] = 'validation' if int(family[:16], 16) % 5 == 0 else 'train'
        plan['id'] = 'yaku-v1:'+digest(plan)
    return plans


def physical_wall(plan):
    rng = random.Random(plan['seed'])
    pool = list(range(136)); rng.shuffle(pool)
    owned = []
    for kind in plan['initial13']+[plan['first_draw_type']]:
        tile = next(t for t in pool if t//4 == kind)
        pool.remove(tile); owned.append(tile)
    own, first = owned[:13], owned[13]
    hands = {0: own, 1: pool[:13], 2: pool[13:26], 3: pool[26:39]}
    queues = {p: list(h) for p, h in hands.items()}; dealt = []
    for _ in range(3):
        for player in range(4):
            dealt.extend(queues[player][:4]); del queues[player][:4]
    for player in range(4):
        dealt.append(queues[player][0])
    wall = dealt+[first]+pool[39:]
    if len(wall) != 136 or sorted(wall) != list(range(136)):
        raise ValueError('Full physical wall must contain every tile exactly once')
    return wall, own+[first]


def generate(config):
    gpu_only(); c = read(config)
    if (c['method'] != METHOD or c['routes'] != list(ROUTES)
            or c['templates_per_route'] != 16 or c['removed_target_tiles'] != [1, 3, 6]
            or c['world_variants_per_public13'] != 2 or c['natural_control_templates'] != 16
            or c['seed_start'] != 2026560000 or c['automatic_training'] or c['new_teacher_labels']):
        raise ValueError('Exact fixed unlabeled512-scenario pilot required')
    if version('riichienv') != '0.4.11':
        raise ValueError('Use existing pinned native engine')
    reg = read(ROOT/c['source_registration'])
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from riichienv import RiichiEnv, GameRule, Phase
    import riichienv._riichienv as native
    from jevmahjong.core.state import build_request
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    files = {str(p.resolve()): sha(p) for p in (Path(__file__), config, Path(native.__file__),
        ROOT/c['source_registration'], Path(reg['source'])/'jevmahjong/core/state.py',
        ROOT/'cluster/curriculum-yaku-scenarios-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-yaku-scenarios-20261123.md')}
    plans = scenario_plans(c)
    write(out/'registered-plans-private.json', dict(method=METHOD, files=files, config=c, plans=plans,
        partitions_fixed_before_native_collection=True, no_labels=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname()))
    streams = {}; paths = {}; counts = {s: Counter() for s in ('train', 'validation')}
    groups = {s: set() for s in counts}; public_ids = {s: set() for s in counts}
    shape_families = {s: set() for s in counts}; rows = Counter()
    try:
        for split in counts:
            paths[split] = out/f'{split}-unlabelled-scenarios-private.jsonl'
            streams[split] = paths[split].open('x')
        for number, plan in enumerate(plans, 1):
            wall, expected = physical_wall(plan)
            env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=plan['seed'])
            views = env.reset(oya=0, wall=wall)
            if env.phase != Phase.WaitAct or set(views) != {0} or Counter(views[0].hand) != Counter(expected):
                raise ValueError('Native initial deal/draw does not match registered physical source')
            request, menu, _ = build_request(views[0])
            q = request['questions']['action']
            if 'label' in q or 'target' in q or 'src' in q or len(menu) < 2:
                raise ValueError('A genuine unlabeled multi-option native decision is required')
            public_id = digest(request)
            split = plan['split']; groups[split].add(plan['group_id'])
            public_ids[split].add(public_id)
            shape_families[split].add(canonical_shape(plan['initial13']+[plan['first_draw_type']]))
            item = dict(request=request, source=dict(plan=plan, wall=wall,
                native_prefix=env.mjai_log, native_seed=plan['seed'], observer=0,
                public_request_sha256=public_id, physical_full_wall=True,
                no_hidden_fields_in_request=True, targeted_not_natural_game_distribution=True),
                status='unlabelled_not_SFT_admitted')
            streams[split].write(json.dumps(item, separators=(',', ':'))+'\n')
            counts[split]['route:'+plan['route']] += 1
            counts[split]['damage:'+str(plan['removed_target_tiles'])] += 1
            counts[split].update('offered:'+t for t in {a['type'] for a in q['criteria'].values()})
            rows[split] += 1
            if number % 64 == 0:
                for stream in streams.values(): stream.flush()
                print(json.dumps(dict(generated=number, total=len(plans))), flush=True)
        if (groups['train'] & groups['validation'] or public_ids['train'] & public_ids['validation']
                or shape_families['train'] & shape_families['validation']):
            raise ValueError('Source family/public decision/canonical shape crosses partitions')
        for stream in streams.values(): stream.flush()
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError('Bound generator/native/runtime changed')
        result = dict(status='actual_native_initial_yaku_scenarios_generated_unlabelled_not_fit_ready',
            method=METHOD, files={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p), records=rows[s],
                groups=len(groups[s])) for s, p in paths.items()},
            coverage={s: dict(v) for s, v in counts.items()}, planned_decisions=512,
            full_physical_walls_and_native_initial_deals_checked=512,
            actual_public_input_and_shape_cross_partition_overlap=0,
            family_partitions_fixed_before_collection=True, labels_generated=0, student_updates=0,
            completed_hands=0, completed_matches=0, native_legal_win_not_student_mastery=True,
            limitations=['dealer0_initial_turn_only', 'no_threat_or_midhand_abandonment_yet',
                'no_teacher_or_rollout_labels', 'biased_case_distribution_not_game_strength'],
            raw_walls_requests_and_plans_private=True, active_stage3A_unchanged=True,
            job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
        write(out/'summary-private.json', result)
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps(dict(status=result['status'], records=dict(rows))), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error), records=dict(rows)))
        raise
    finally:
        for stream in streams.values(): stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); generate(args.config.resolve())
