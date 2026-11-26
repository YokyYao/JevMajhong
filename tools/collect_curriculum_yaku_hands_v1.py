"""Continue targeted native source hands with actual stateful Mortal play.

Keep every observer decision after the already collected initial window, not
only successful yaku attempts. Source-family partitions remain inherited.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha


def collect(config):
    gpu_only(); c = read(config)
    if (c['method'] != 'same_yaku512_sources_actual_mortal_first_hand_continuations_v1'
            or c['source_cases'] != 512 or c['maximum_action_batches_per_hand'] != 2000
            or not c['preserve_source_partitions'] or c['filter_by_yaku_success']
            or c['capture_initial_window_again'] or c['automatic_student_fit']):
        raise ValueError('Fixed all-outcome source continuation recipe required')
    source_path = ROOT/c['source_summary']; source = read(source_path)
    parent_path = ROOT/c['initial_teacher_summary']; parent = read(parent_path)
    if (source['status'] != 'actual_native_initial_yaku_scenarios_generated_unlabelled_not_fit_ready'
            or source['planned_decisions'] != 512
            or parent['status'] != 'actual_mortal_yaku_initial_choices_collected_not_stage3B_fit_admitted'):
        raise ValueError('Complete original source and actual initial labels required')
    reg_path = ROOT/c['source_registration']; reg = read(reg_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(2)
    from riichienv import RiichiEnv, GameRule, ActionType
    import riichienv._riichienv as native
    from jevmahjong.core.state import build_request, legal_candidates, signature
    from jevmahjong.data.base_lora_pool import phase
    from tools.curriculum_mortal_source import MortalSource
    from tools.curriculum_case_order import ordered_seats
    from tools.curriculum_endgame_ledger import ledger
    from tools.audit_curriculum_endgame_cases import public_solo_win
    teacher = MortalSource((ROOT/c['teacher_root']).resolve())
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    inputs = {str(p.resolve()): sha(p) for p in (Path(__file__), config, source_path, parent_path,
        reg_path, Path(native.__file__), ROOT/'tools/curriculum_case_order.py',
        ROOT/'tools/curriculum_endgame_ledger.py', ROOT/'tools/audit_curriculum_endgame_cases.py',
        ROOT/'cluster/curriculum-yaku-hands-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-yaku-hands-20261123.md')}
    inputs.update(teacher.bindings)
    first_choices = {}; original_cases = []
    for split in ('train', 'validation'):
        for manifest in (source, parent):
            spec = manifest['files'][split]; path = ROOT/spec['path']
            actual = sha(path)
            if actual != spec['sha256']:
                raise ValueError('Original source/initial labels changed')
            inputs[str(path.resolve())] = actual
        for line in (ROOT/parent['files'][split]['path']).open():
            row = json.loads(line)
            first_choices[row['_meta']['id']] = row['questions']['action']['label']
        for line in (ROOT/source['files'][split]['path']).open():
            row = json.loads(line)
            if row['source']['plan']['split'] != split:
                raise ValueError('Source case partition changed')
            original_cases.append(row)
    if len(original_cases) != 512 or len(first_choices) != 511:
        raise ValueError('Exact512 cases and511 original network choices required')
    write(out/'registered-private.json', dict(method=c['method'], config=c, inputs=inputs,
        cases=[dict(id=r['source']['plan']['id'], group_id=r['source']['plan']['group_id'],
                    split=r['source']['plan']['split']) for r in original_cases],
        new_independent_source_families=0, initial_labels_reused_not_duplicated=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname()))
    paths = {s: out/f'{s}-continuation-teacher-private.jsonl' for s in ('train', 'validation')}
    streams = {s: p.open('x') for s, p in paths.items()}
    raw = (out/'completed-source-hands-private.jsonl').open('x')
    stats = Counter(); counts = {s: Counter() for s in paths}; groups = {s: set() for s in paths}
    route_counts = {s: defaultdict(Counter) for s in paths}; identifiers = set()
    try:
        for case in original_cases:
            original = case['source']; plan = original['plan']; split = plan['split']; route = plan['route']
            env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=original['native_seed'])
            observations = env.reset(oya=0, wall=original['wall'])
            if build_request(observations[0])[0] != case['request'] or env.mjai_log != original['native_prefix']:
                raise ValueError('Initial source request/prefix did not reproduce')
            agents = [teacher.player(p) for p in range(4)]
            trace = []; settled = False; csplit = counts[split]; cr = route_counts[split][route]
            for step in range(2000):
                selected = {p: agents[p].act(obs) for p, obs in ordered_seats(observations)}
                if step == 0:
                    request, menu, _ = build_request(observations[0])
                    actual_key = next(k for k, a in zip(request['questions']['action']['criteria'], menu)
                                      if signature(a) == signature(selected[0]))
                    if plan['id'] in first_choices:
                        if actual_key != first_choices[plan['id']]:
                            raise ValueError('Original initial teacher decision changed')
                        stats['original_initial_network_choices_exact'] += 1
                    elif selected[0].action_type not in (ActionType.RON, ActionType.TSUMO):
                        raise ValueError('Missing original initial label is not the known forced win')
                for seat, obs in ordered_seats(observations):
                    if seat != 0:
                        continue
                    menu = legal_candidates(obs)
                    wins = [a for a in menu if a.action_type in (ActionType.RON, ActionType.TSUMO)]
                    if wins:
                        # This is a source outcome, not a network supervision row.
                        score = public_solo_win(env, obs)
                        cr['observer_legal_win_offers'] += 1
                        if score['supported']:
                            cr['source_wins_public_no_ura_score_supported'] += 1
                            for yaku in score['yaku']:
                                cr['source_win_yaku:'+yaku] += 1
                        else:
                            cr['public_score_unsupported:'+score['reason']] += 1
                        continue
                    if step == 0 or len(menu) < 2:
                        continue
                    request, menu, _ = build_request(obs)
                    keys = list(request['questions']['action']['criteria'])
                    label = next(k for k, a in zip(keys, menu) if signature(a) == signature(selected[0]))
                    slot = phase(request); q = request['questions']['action']; kinds = {a['type'] for a in q['criteria'].values()}
                    q.update(label=label, src='curriculum3B:teacher:trajectory:'+slot)
                    identifier = plan['id']+':step:'+str(step)
                    if identifier in identifiers:
                        raise ValueError('Duplicate continuation decision identity')
                    identifiers.add(identifier)
                    request['_meta'] = dict(id=identifier, group_id=plan['group_id'], split=split,
                        phase=slot, label_type='expert_replay', source='mortal_targeted_native_first_hand_continuation',
                        scenario_source_id=plan['id'], scenario_route=route,
                        scenario_removed_target_tiles=plan['removed_target_tiles'],
                        source_step=step, source_engine_seed=original['native_seed'], seat=0,
                        source_hand_reference='completed-source-hands-private.jsonl',
                        actual_teacher_intent_even_if_simultaneous_priority_overrides=True,
                        teacher_choice_not_optimality_proof=True)
                    streams[split].write(json.dumps(request, separators=(',', ':'))+'\n')
                    groups[split].add(plan['group_id']); csplit['rows'] += 1; cr['rows'] += 1
                    chosen_kind = q['criteria'][label]['type']
                    csplit['phase:'+slot] += 1; csplit['selected:'+chosen_kind] += 1
                    csplit['own_discard_number:'+str(len(obs.to_dict()['discards'][0])+1)] += 1
                    threats = sum(p != 0 for p in request['state']['riichi'])
                    csplit['opponent_riichi:'+str(threats)] += 1; cr['opponent_riichi:'+str(threats)] += 1
                    if request['state']['riichi'] and 0 in request['state']['riichi']:
                        csplit['own_post_riichi'] += 1
                    if any(m.opened for m in obs.to_dict()['melds'][0]):
                        csplit['own_open_meld'] += 1
                    for kind in kinds & {'chi', 'pon', 'ankan', 'daiminkan', 'kakan'}:
                        csplit['offered:'+kind] += 1
                        csplit[kind+':selected_same' if chosen_kind == kind else kind+':selected_pass'
                               if chosen_kind == 'none' else kind+':selected_other'] += 1
                before = len(env.mjai_log)
                trace.append([dict(player=p, type=int(a.action_type), tile=a.tile, consumed=list(a.consume_tiles))
                              for p, a in sorted(selected.items())])
                observations = env.step(selected)
                new_events = env.mjai_log[before:]
                if any(e['type'] in ('hora', 'ryukyoku') for e in new_events):
                    settled = True
                    break
                if env.done():
                    raise ValueError('Source ended without a hand settlement')
            if not settled:
                raise ValueError('Bounded source hand did not settle')
            financial = ledger(env.mjai_log, env.scores())
            settlements = [e for e in env.mjai_log if e['type'] in ('hora', 'ryukyoku')]
            observer_win = any(e['type'] == 'hora' and e['actor'] == 0 for e in settlements)
            observer_dealin = any(e['type'] == 'hora' and e['target'] == 0 and e['actor'] != 0 for e in settlements)
            cr['settled_source_hands'] += 1
            cr['observer_won'] += observer_win; cr['observer_dealt_in'] += observer_dealin
            cr['drawn_source_hand'] += any(e['type'] == 'ryukyoku' for e in settlements)
            stats['settled_source_hands'] += 1; stats['native_action_batches'] += len(trace)
            raw.write(json.dumps(dict(source_id=plan['id'], group_id=plan['group_id'], split=split,
                source_route=route, original_scenario_reference=c['source_summary'],
                source_engine_seed=original['native_seed'], source_trace=trace,
                mjai=env.mjai_log, final_scores=list(env.scores()), ledger=financial,
                stopped_after_first_actual_hand_settlement=True, complete_match=env.done()),
                separators=(',', ':'))+'\n')
            for stream in streams.values(): stream.flush()
            raw.flush()
            if stats['settled_source_hands'] % 32 == 0:
                print(json.dumps(dict(settled_source_hands=stats['settled_source_hands'],
                    new_continuation_labels={s: counts[s]['rows'] for s in counts})), flush=True)
        if (stats['settled_source_hands'] != 512 or stats['original_initial_network_choices_exact'] != 511
                or groups['train'] & groups['validation']):
            raise ValueError('Complete inherited source scope and fixed initial choices required')
        if any(sha(p) != h for p, h in inputs.items()):
            raise ValueError('Frozen source or teacher changed')
        result = dict(status='actual_yaku512_first_hands_settled_and_observed_continuation_labels_collected_not_fit_ready',
            method=c['method'], stats=dict(stats), coverage={s: dict(v) for s, v in counts.items()},
            by_route={s: {k: dict(v) for k, v in data.items()} for s, data in route_counts.items()},
            files={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p), records=counts[s]['rows'], groups=len(groups[s]))
                   for s, p in paths.items()},
            full_source_hands_sha256=sha(out/'completed-source-hands-private.jsonl'),
            teacher_counts=dict(teacher.counts), source_family_partitions_preserved=True,
            new_independent_source_families=0, initial_labels_not_duplicated=True,
            unfiltered_first_hand_outcomes=True, observer0_labels_only=True,
            public_source_yaku_not_student_mastery=True,
            role_choice_and_abandonment_optimality_not_certified=True,
            native_score_ledgers_checked_not_independent_full_han_fu=True,
            no_native_wall_resampling_with_reused_teacher_state=True,
            normalization_encoder_and_stage3B_fit_recipe_pending=True,
            active_stage3A_unchanged=True, student_updates=0, raw_private=True,
            job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
        write(out/'summary-private.json', result)
        write(out/'completion-binding-private.json', dict(files=inputs))
        print(json.dumps(dict(status=result['status'], stats=dict(stats))), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error), stats=dict(stats)))
        raise
    finally:
        raw.close()
        for stream in streams.values(): stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); collect(args.config.resolve())
