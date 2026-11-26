"""Native, unlabelled kan opportunities with take/skip context controls.

Scripted prefixes construct source observations, not expert actions. No
teacher, student, action value or sampled future is used in generation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.generate_curriculum_yaku_scenarios_v1 import canonical_shape

OPERATIONS = ('ankan', 'daiminkan', 'kakan')
MOTIFS = ('yakuhai_ready_shape', 'tanyao_ready_shape', 'closed_route_control', 'pair_heavy_control')
METHOD = 'native_scripted_kan_opportunities_unlabelled_192_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rest_hand(motif, template, seed):
    rng = random.Random(seed)
    for _ in range(10000):
        if motif == 'yakuhai_ready_shape':
            tile = 31+template % 3
        elif motif == 'tanyao_ready_shape':
            tile = rng.randrange(3)*9+rng.randrange(1, 8)
        elif motif == 'closed_route_control':
            tile = 29+template % 2
        else:
            tile = rng.randrange(34)
        if motif == 'pair_heavy_control':
            rest = [t for t in rng.sample([t for t in range(34) if t != tile], 5) for _ in range(2)]
        else:
            starts = [rng.randrange(3)*9+rng.randrange(1, 6) if motif == 'tanyao_ready_shape'
                      else rng.randrange(3)*9+rng.randrange(7) for _ in range(3)]
            rest = [s+i for s in starts for i in range(3)]
            choices = [t for t in range(34) if t != tile]
            if motif == 'tanyao_ready_shape':
                choices = [t for t in choices if t < 27 and 0 < t % 9 < 8]
            rest.append(rng.choice(choices))
        if tile not in rest and max(Counter(rest).values()) <= 4:
            bridge = next(t for t in range(34) if t != tile and Counter(rest)[t] < 4)
            return tile, rest, bridge
    raise ValueError('Bounded source-shape construction exhausted')


def plans(config):
    result = []
    for mi, motif in enumerate(MOTIFS):
        for template in range(8):
            seed = config['seed_start']+mi*10000+template*100
            tile, rest, bridge = rest_hand(motif, template, seed)
            for operation in OPERATIONS:
                own = [tile]*(2 if operation == 'kakan' else 3)+rest
                if operation == 'kakan':
                    own.append(bridge)
                for world in range(2):
                    result.append(dict(operation=operation, motif=motif, template=template,
                        tile_type=tile, initial13=own, bridge_discard_type=bridge,
                        effective14=[tile]*4+rest, world_variant=world,
                        seed=seed+OPERATIONS.index(operation)*10+world))
    # Link actual initial shapes AND effective decision shapes before any
    # native construction. Suit/rank symmetries and cross-operation controls
    # cannot enter opposite partitions under different source names.
    parents = list(range(len(result))); seen = {}; keys = []
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    for i, plan in enumerate(result):
        shapes = [('initial13', canonical_shape(plan['initial13'])),
                  ('effective14', canonical_shape(plan['effective14']))]
        keys.append(shapes)
        for shape in shapes:
            if shape in seen:
                a, b = find(i), find(seen[shape])
                if a != b:
                    parents[max(a, b)] = min(a, b)
            else:
                seen[shape] = i
    components = defaultdict(set)
    for i, shapes in enumerate(keys):
        components[find(i)].update(shapes)
    for i, plan in enumerate(result):
        family = digest(sorted(components[find(i)]))
        plan['group_id'] = 'kan-source-v1:'+family
        plan['split'] = 'validation' if int(family[:16], 16) % 5 == 0 else 'train'
        plan['id'] = 'kan-source-v1:'+digest(plan)
    if len(result) != 192 or len({p['id'] for p in result}) != 192:
        raise ValueError('Exact unique prospective192 source plans required')
    return result


def physical_wall(plan):
    rng = random.Random(plan['seed']); pool = list(range(136)); rng.shuffle(pool)
    def take(kind):
        value = next(t for t in pool if t//4 == kind)
        pool.remove(value); return value
    seat = 0 if plan['operation'] == 'ankan' else 1
    own = [take(t) for t in plan['initial13']]
    first = take(plan['tile_type'])
    fourth = take(plan['tile_type']) if plan['operation'] == 'kakan' else None
    hands = {seat: own}
    for p in range(4):
        if p != seat:
            hands[p] = pool[:13]; del pool[:13]
    queue = {p: list(h) for p, h in hands.items()}; dealt = []
    for _ in range(3):
        for p in range(4):
            dealt.extend(queue[p][:4]); del queue[p][:4]
    dealt.extend(queue[p][0] for p in range(4))
    draws = [first]
    if fourth is not None:
        draws.extend(pool[:3]); del pool[:3]; draws.append(fourth)
    wall = dealt+draws+pool
    if len(wall) != 136 or sorted(wall) != list(range(136)):
        raise ValueError('Every physical tile must occur exactly once')
    return wall, seat


def signature(action):
    return int(action.action_type), action.tile, tuple(sorted(action.consume_tiles))


def choose(obs, kind, tile_type=None):
    from jevmahjong.core.state import build_request
    request, menu, _ = build_request(obs)
    options = [(key, action) for (key, value), action in
               zip(request['questions']['action']['criteria'].items(), menu)
               if value['type'] == kind and (tile_type is None or action.tile//4 == tile_type)]
    if not options:
        raise ValueError('Scripted source action unavailable: '+kind)
    return options[0][1]


class PrefixWin(Exception):
    pass


def construct(plan):
    from riichienv import RiichiEnv, GameRule, Phase
    from jevmahjong.core.state import build_request
    wall, seat = physical_wall(plan)
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=plan['seed'])
    views = env.reset(oya=0, wall=wall); trace = []
    def step(selected):
        if set(selected) != set(views):
            raise ValueError('Scripted prefix must specify every active actor')
        for p, obs in views.items():
            kinds = {a['type'] for a in build_request(obs)[0]['questions']['action']['criteria'].values()}
            if kinds & {'hora', 'ron', 'tsumo'}:
                raise PrefixWin('A scripted prefix would decline a legal win')
        trace.append([dict(player=p, type=signature(a)[0], tile=a.tile,
                           consumed=list(a.consume_tiles)) for p, a in sorted(selected.items())])
        return env.step(selected)
    if plan['operation'] != 'ankan':
        views = step({0: choose(views[0], 'dahai', plan['tile_type'])})
    if plan['operation'] == 'kakan':
        selected = {p: choose(obs, 'pon' if p == seat else 'none',
                              plan['tile_type'] if p == seat else None) for p, obs in views.items()}
        views = step(selected)
        if set(views) != {seat}:
            raise ValueError('Expected immediate post-pon discard')
        views = step({seat: choose(views[seat], 'dahai', plan['bridge_discard_type'])})
        for _ in range(24):
            if env.done():
                raise ValueError('Source ended before its kan opportunity')
            if env.phase == Phase.WaitAct and set(views) == {seat}:
                break
            if env.phase == Phase.WaitResponse:
                views = step({p: choose(obs, 'none') for p, obs in views.items()})
            elif env.phase == Phase.WaitAct and len(views) == 1:
                p = next(iter(views))
                views = step({p: choose(views[p], 'dahai', env.drawn_tile//4)})
            else:
                raise ValueError('Unexpected native source phase')
        else:
            raise ValueError('Bounded scripted source prefix exhausted')
    if seat not in views:
        raise ValueError('Observer not active at source opportunity')
    request, menu, _ = build_request(views[seat])
    kinds = {value['type'] for value in request['questions']['action']['criteria'].values()}
    if plan['operation'] not in kinds or len(menu) < 2:
        raise ValueError('Requested kan and at least one alternative must be genuinely legal')
    if any(key in request['questions']['action'] for key in ('src', 'label', 'target')):
        raise ValueError('Generated source must remain unlabelled')
    return dict(request=request, source=dict(plan=plan, wall=wall, observer=seat,
        native_seed=plan['seed'], native_prefix=env.mjai_log, action_trace=trace,
        public_request_sha256=digest(request), physical_full_wall=True,
        scripted_prefix_not_teacher_actions=True, no_prior_declined_legal_wins=True,
        hidden_wall_and_motif_not_model_inputs=True), status='unlabelled_not_SFT_admitted')


def restore(case):
    from riichienv import RiichiEnv, GameRule
    from jevmahjong.core.state import build_request, legal_candidates
    source = case['source']
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=source['native_seed'])
    views = env.reset(oya=0, wall=source['wall'])
    for batch in source['action_trace']:
        if set(views) != {a['player'] for a in batch}:
            raise ValueError('Replay actor membership differs')
        selected = {}
        for saved in batch:
            identity = saved['type'], saved['tile'], tuple(sorted(saved['consumed']))
            options = {signature(a): a for a in legal_candidates(views[saved['player']])}
            if identity not in options:
                raise ValueError('Replay physical action is not legal')
            selected[saved['player']] = options[identity]
        views = env.step(selected)
    obs = views[source['observer']]
    if env.mjai_log != source['native_prefix'] or build_request(obs)[0] != case['request']:
        raise ValueError('Exact native source prefix/public input/menu replay failed')
    return env, obs


def generate(config_path):
    allocation = resource(); c = read(config_path)
    if (c['method'] != METHOD or c['operations'] != list(OPERATIONS) or c['motifs'] != list(MOTIFS)
            or c['templates_per_motif'] != 8 or c['worlds_per_operation'] != 2
            or c['seed_start'] != 2026680000 or c['new_teacher_labels'] or c['automatic_training']):
        raise ValueError('Exact prospective unlabelled192 source pilot required')
    if version('riichienv') != '0.4.11':
        raise ValueError('Pinned native engine required')
    reg_path = ROOT/c['source_registration']; reg = read(reg_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import riichienv._riichienv as native
    paths = [Path(__file__), config_path, reg_path, Path(native.__file__),
        ROOT/'tools/generate_curriculum_yaku_scenarios_v1.py',
        ROOT/'tools/capture_curriculum_paired_terminal_v2.py',
        ROOT/'cluster/curriculum-kan-sources-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-kan-sources-20261124.md',
        Path(reg['source'])/'jevmahjong/core/state.py']
    files = {str(p.resolve()): sha(p) for p in paths}
    source_plans = plans(c); out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-plans-private.json', dict(method=METHOD, config=c, files=files,
        plans=source_plans, partitions_fixed_before_native_generation=True, allocation=allocation,
        teacher_model_loaded=False, values_or_labels_generated=False))
    streams = {}; outputs = {}; counts = {s: Counter() for s in ('train', 'validation')}
    groups = {s: set() for s in counts}; inputs = {s: set() for s in counts}; quarantined = []
    try:
        for split in counts:
            outputs[split] = out/f'{split}-unlabelled-kan-sources-private.jsonl'
            streams[split] = outputs[split].open('x')
        for i, plan in enumerate(source_plans, 1):
            split = plan['split']; counts[split]['planned'] += 1
            try:
                case = construct(plan)
            except PrefixWin as error:
                quarantined.append(dict(id=plan['id'], group_id=plan['group_id'], split=split,
                    reason=str(error), source_not_labelled=True, replacement_not_drawn=True))
                counts[split]['prefix_win_quarantine'] += 1
                continue
            restore(case)
            groups[split].add(plan['group_id']); inputs[split].add(case['source']['public_request_sha256'])
            counts[split]['generated'] += 1; counts[split]['exact_native_replays'] += 1
            counts[split]['operation:'+plan['operation']] += 1
            counts[split]['motif:'+plan['motif']] += 1
            counts[split].update('offered:'+a for a in
                {v['type'] for v in case['request']['questions']['action']['criteria'].values()})
            streams[split].write(json.dumps(case, separators=(',', ':'))+'\n')
            if i % 24 == 0:
                print(json.dumps(dict(plans_processed=i, planned=192,
                    sources=sum(v['generated'] for v in counts.values()), quarantined=len(quarantined))), flush=True)
        for stream in streams.values():
            stream.flush()
        if groups['train'] & groups['validation'] or inputs['train'] & inputs['validation']:
            raise ValueError('Family or public input crosses pilot partitions')
        if sum(v['generated'] for v in counts.values())+len(quarantined) != 192:
            raise ValueError('Every prospective case needs a source or explicit quarantine')
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError('Bound native/source generator changed')
        write(out/'quarantine-private.json', quarantined)
        summary = dict(status='native_kan_source_pilot_generated_unlabelled_not_fit_admitted',
            method=METHOD, coverage={s: dict(v) for s, v in counts.items()},
            files={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p), records=counts[s]['generated'],
                families=len(groups[s])) for s, p in outputs.items()},
            planned=192, quarantined=len(quarantined), labels=0, model_forwards=0, student_updates=0,
            exact_native_replay_for_every_generated_source=True, early_unthreatened_context_only=True,
            cross_pilot_split_families_and_public_inputs=0,
            full_existing_corpus_and_reserved_family_overlap_audit_pending=True,
            conditional_public_world_labels_and_uncertainty_pending=True,
            teacher_optimality_or_current_student_improvement_not_established=True,
            active_training_inputs_and_policy_unchanged=True, raw_sources_private=True,
            job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
        write(out/'summary-private.json', summary)
        files[str((out/'registered-plans-private.json').resolve())] = sha(out/'registered-plans-private.json')
        files[str((out/'summary-private.json').resolve())] = sha(out/'summary-private.json')
        for path in outputs.values():
            files[str(path.resolve())] = sha(path)
        files[str((out/'quarantine-private.json').resolve())] = sha(out/'quarantine-private.json')
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error)))
        raise
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    generate(p.parse_args().config.resolve())
