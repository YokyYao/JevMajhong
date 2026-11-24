"""Unlabelled native daiminkan controls across meld state and discarder seat."""
import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha
from tools.generate_curriculum_kan_sources_v1 import (
    plans as previous_plans, digest, choose, signature, PrefixWin, restore)
from tools.generate_curriculum_yaku_scenarios_v1 import canonical_shape

METHOD = 'native_daiminkan_meld_state_by_discarder_context576_v1'
CONTEXTS = ('closed', 'open_value_pon', 'open_simple_pon')
DISCARDERS = (0, 3, 2)  # observer1: upper, opposite, lower


def plans():
    old = previous_plans(dict(seed_start=2026680000))
    bases = [p for p in old if p['operation'] == 'daiminkan' and p['world_variant'] == 0]
    result = []
    for base_no, base in enumerate(bases):
        tile = base['tile_type']; rest = base['initial13'][3:]
        for ci, context in enumerate(CONTEXTS):
            opener = bridge = None
            own = list(base['initial13'])
            if context != 'closed':
                remainder = rest[:6]+rest[-1:]
                choices = range(31, 34) if context == 'open_value_pon' else (
                    t for t in range(27) if 0 < t % 9 < 8)
                opener = next(t for t in choices if t != tile and t not in remainder)
                bridge = next(t for t in range(34) if t not in (tile, opener) and t not in remainder)
                own = [tile]*3+remainder+[opener]*2+[bridge]
                opportunity = [tile]*4+remainder+[opener]*3
            else:
                opportunity = list(base['effective14'])
            for discarder in DISCARDERS:
                for world in range(2):
                    plan = dict(operation='daiminkan', motif=base['motif'], template=base['template'],
                        context=context, discarder=discarder, observer=1, tile_type=tile,
                        initial13=own, effective14=opportunity, opener_type=opener,
                        bridge_discard_type=bridge, world_variant=world,
                        seed=2026720000+base_no*1000+ci*100+discarder*10+world,
                        group_id=base['group_id'], split=base['split'],
                        original_pilot_family_anchor=base['group_id'])
                    plan['id'] = 'daiminkan-context-v1:'+digest(plan)
                    result.append(plan)
    # Keep each original family and every contextual/physical variant together.
    # If a new shape bridges opposite existing partitions, quarantine its whole
    # family prospectively. Never repair a split after observing teacher choices.
    shapes = defaultdict(list)
    for p in old+result:
        for name in ('initial13', 'effective14'):
            shapes[name, canonical_shape(p[name])].append((p['group_id'], p['split']))
    bad = {group for values in shapes.values() if len({s for _, s in values}) > 1
           for group, _ in values}
    aliases = {key: sorted({g for g, _ in values}) for key, values in shapes.items()
               if len({g for g, _ in values}) > 1}
    for p in result:
        p['prospective_cross_partition_shape_quarantine'] = p['group_id'] in bad
    if len(result) != 576 or len({p['id'] for p in result}) != 576:
        raise ValueError('Exactly576 distinct pre-value contextual plans required')
    return result, dict(conflicting_original_families=sorted(bad),
        same_partition_shape_alias_keys=len(aliases), global_family_normalization_still_required=True)


def physical_wall(plan):
    rng = random.Random(plan['seed']); pool = list(range(136)); rng.shuffle(pool)
    def take(kind):
        tile = next(t for t in pool if t//4 == kind)
        pool.remove(tile)
        return tile
    hands = {1: [take(t) for t in plan['initial13']]}
    target = take(plan['tile_type'])
    opener = take(plan['opener_type']) if plan['opener_type'] is not None else None
    for player in (0, 2, 3):
        hands[player] = pool[:13]; del pool[:13]
    queued = {p: list(h) for p, h in hands.items()}; dealt = []
    for _ in range(3):
        for p in range(4):
            dealt.extend(queued[p][:4]); del queued[p][:4]
    dealt.extend(queued[p][0] for p in range(4))
    # After observer1's pon/discard the next draw belongs to player2.
    actors = [0, 1, 2, 3] if opener is None else [2, 3, 0]
    draws = [] if opener is None else [opener]
    for p in actors:
        if p == plan['discarder']:
            draws.append(target)
            break
        draws.append(pool.pop(0))
    wall = dealt+draws+pool
    if len(wall) != 136 or sorted(wall) != list(range(136)):
        raise ValueError('Physical wall must contain every tile exactly once')
    return wall


def construct(plan):
    from riichienv import RiichiEnv, GameRule, Phase
    from jevmahjong.core.state import build_request
    wall = physical_wall(plan)
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=plan['seed'])
    views = env.reset(oya=0, wall=wall); trace = []
    def step(selected):
        if set(selected) != set(views): raise ValueError('All active prefix actors required')
        for view in views.values():
            kinds = {v['type'] for v in build_request(view)[0]['questions']['action']['criteria'].values()}
            if kinds & {'hora', 'ron', 'tsumo'}:
                raise PrefixWin('Scripted contextual prefix would decline a legal win')
        trace.append([dict(player=p, type=signature(a)[0], tile=a.tile,
            consumed=list(a.consume_tiles)) for p, a in sorted(selected.items())])
        return env.step(selected)
    if plan['context'] != 'closed':
        views = step({0: choose(views[0], 'dahai', plan['opener_type'])})
        views = step({p: choose(obs, 'pon' if p == 1 else 'none',
            plan['opener_type'] if p == 1 else None) for p, obs in views.items()})
        if set(views) != {1}: raise ValueError('Actual observer pon must precede its discard')
        views = step({1: choose(views[1], 'dahai', plan['bridge_discard_type'])})
    for _ in range(32):
        if env.done(): raise ValueError('Source ended before its planned opportunity')
        if env.phase == Phase.WaitAct and set(views) == {plan['discarder']} and env.drawn_tile//4 == plan['tile_type']:
            views = step({plan['discarder']: choose(views[plan['discarder']], 'dahai', plan['tile_type'])})
            break
        if env.phase == Phase.WaitResponse:
            views = step({p: choose(obs, 'none') for p, obs in views.items()})
        elif env.phase == Phase.WaitAct and len(views) == 1:
            player = next(iter(views))
            views = step({player: choose(views[player], 'dahai', env.drawn_tile//4)})
        else: raise ValueError('Unexpected native source phase')
    else: raise ValueError('Finite contextual prefix budget exhausted')
    if 1 not in views: raise ValueError('Observer must genuinely respond to the target discard')
    request, menu, _ = build_request(views[1])
    offered = {v['type'] for v in request['questions']['action']['criteria'].values()}
    melds = request['state']['melds'][1]
    if ('daiminkan' not in offered or len(menu) < 2
            or len(melds) != (0 if plan['context'] == 'closed' else 1)):
        raise ValueError('Actual legal daiminkan menu and registered meld context required')
    events = [json.loads(v) if isinstance(v, str) else v for v in env.mjai_log]
    last = next(e for e in reversed(events) if e['type'] == 'dahai')
    if last['actor'] != plan['discarder']:
        raise ValueError('Actual final discarder differs from the planned relative seat')
    return dict(request=request, source=dict(plan=plan, wall=wall, observer=1,
        native_seed=plan['seed'], native_prefix=env.mjai_log, action_trace=trace,
        public_request_sha256=digest(request), physical_full_wall=True,
        scripted_prefix_not_teacher_actions=True, no_prior_declined_legal_wins=True,
        hidden_wall_and_motif_not_model_inputs=True), status='unlabelled_not_SFT_admitted')


def generate(config):
    allocation = resource(); c = read(config)
    if (c['method'] != METHOD or c['planned'] != 576 or c['contexts'] != list(CONTEXTS)
            or c['discarders'] != list(DISCARDERS) or c['automatic_fit'] or c['teacher_labels']):
        raise ValueError('Exact prospective contextual576 source contract required')
    if version('riichienv') != '0.4.11': raise ValueError('Pinned native engine required')
    reg_path = ROOT/c['source_registration']; reg = read(reg_path)
    sys.path[:0] = [reg['source'], reg['training_config']['upstream_source']]
    import riichienv._riichienv as native
    plans_all, collision = plans(); out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    files = {str(p.resolve()): sha(p) for p in (Path(__file__), config, reg_path,
        Path(native.__file__), Path(reg['source'])/'jevmahjong/core/state.py',
        ROOT/'tools/generate_curriculum_kan_sources_v1.py',
        ROOT/'tools/generate_curriculum_yaku_scenarios_v1.py',
        ROOT/'tests/core/test_curriculum_daiminkan_context_v1.py',
        ROOT/'docs/experiments/plans/curriculum-daiminkan-context-20261124.md')}
    write(out/'registered-plans-private.json', dict(method=METHOD, plans=plans_all,
        collision=collision, files=files, allocation=allocation, labels=0, student_updates=0,
        all_contexts_and_family_partitions_fixed_before_generation=True))
    paths = {s: out/f'{s}-unlabelled-contexts-private.jsonl' for s in ('train', 'validation')}
    streams = {s: p.open('x') for s, p in paths.items()}
    counts = {s: Counter() for s in paths}; groups = {s: set() for s in paths}
    inputs = {s: set() for s in paths}; quarantine = []
    try:
        for plan in plans_all:
            split = plan['split']; counts[split]['planned'] += 1
            try:
                if plan['prospective_cross_partition_shape_quarantine']:
                    raise PrefixWin('Prospective source family bridges opposite existing shape partitions')
                case = construct(plan)
            except PrefixWin as error:
                quarantine.append(dict(id=plan['id'], group_id=plan['group_id'], split=split,
                    context=plan['context'], discarder=plan['discarder'], reason=str(error), replacement_drawn=False))
                counts[split]['quarantined'] += 1
                continue
            restore(case)
            groups[split].add(plan['group_id']); inputs[split].add(case['source']['public_request_sha256'])
            counts[split]['generated'] += 1
            counts[split][plan['context']+':discarder'+str(plan['discarder'])] += 1
            streams[split].write(json.dumps(case, separators=(',', ':'))+'\n')
        for stream in streams.values(): stream.flush()
        if groups['train'] & groups['validation'] or inputs['train'] & inputs['validation']:
            raise ValueError('A family or exact public input crosses partitions')
        if sum(v['generated'] for v in counts.values())+len(quarantine) != 576:
            raise ValueError('Every registered plan needs source or explicit quarantine')
        if any(sha(p) != h for p, h in files.items()): raise ValueError('Source implementation changed')
        write(out/'quarantine-private.json', quarantine)
        summary = dict(status='native_daiminkan_context_sources_generated_unlabelled_not_fit_admitted',
            method=METHOD, planned=576, quarantined=len(quarantine), coverage={s: dict(v) for s, v in counts.items()},
            files={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p), records=counts[s]['generated'],
                families=len(groups[s])) for s, p in paths.items()},
            original_pilot_family_anchors_and_splits_preserved=True,
            all_generated_native_prefixes_inputs_and_menus_replayed=True,
            only_early_unthreatened_equal_score_contexts=True,
            full_corpus_overlap_normalization_and_encoder_admission_pending=True,
            labels=0, model_forwards=0, student_updates=0, raw_private=True,
            actual_allocation=allocation)
        write(out/'summary-private.json', summary)
        for p in (*paths.values(), out/'registered-plans-private.json', out/'quarantine-private.json', out/'summary-private.json'):
            files[str(p.resolve())] = sha(p)
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps({k: v for k, v in summary.items() if k != 'actual_allocation'}), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(error=repr(error)))
        raise
    finally:
        for stream in streams.values(): stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    generate(p.parse_args().config.resolve())
