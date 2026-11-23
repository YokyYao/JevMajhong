"""Observed single-actor discards from existing complete source logs.

Unlike suppressed response intentions, an executed WaitAct discard is present
in MJAI. Only that directly observed action becomes a new expert replay label.
"""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_source_event_replay_v1 import resolve_batch


def collect(config_path):
    gpu_only(); c = read(config_path)
    if (c['method'] != 'observed_waitact_discard_completion_from_original256_v1'
            or c['source_games'] != 256 or c['source_scope'] != 'own_draw_no_meld_no_RI_no_opponent_RI_D_only_missing_context'
            or any(c[k] is not True for k in ('observed_executed_discard_required',
                'response_intentions_are_never_labels', 'preserve_original_family_splits'))
            or any(c[k] for k in ('new_model_inference', 'new_source_games', 'automatic_student_fit'))
            or c['maximum_native_batches_per_match'] != 30000
            or c['existing_teacher_sources'] != ['runs/curriculum-basic-teacher-data-v1/data',
                'runs/curriculum-advanced-teacher-data-v1/data', 'runs/curriculum-risk-teacher-data-v1/data',
                'runs/curriculum-post-reach-teacher-data-v2/data']):
        raise ValueError('Fixed observed-discard-only completion recipe required')
    proof_root = ROOT/c['verified_replay_root']; proof = read(proof_root/'collection-v1/summary-private.json')
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    scheduler_exit(read(ROOT/'logs/curriculum-terminal-reuse256-exit-317726.json'),
        (proof_root/'collection-terminal-scheduler-private.txt').read_text(), ROOT/'cluster/curriculum-reused-terminal-v1.slurm')
    binding = read(proof_root/'collection-v1/completion-binding-private.json')
    if (proof['counts']['exact_complete_sources'] != 256 or not proof['all45351_old_teacher_inputs_exact']
            or not all(sha(p) == h for p, h in binding['files'].items())):
        raise ValueError('Actual full source replay proof required')
    source_root = ROOT/c['source_root']; parent = read(source_root/'registered-private.json')
    ready = read(source_root/'summary-private.json'); rawpath = source_root/'source-matches-private.jsonl'
    if sha(rawpath) != ready['full_source_matches_sha256']: raise ValueError('Original complete source trajectories changed')
    registered = ROOT/parent['config']['source_registration']; reg = read(registered)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from riichienv import RiichiEnv, GameRule, Phase, ActionType
    import riichienv._riichienv as native
    from jevmahjong.core.state import build_request, legal_candidates, signature
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.agents import guarded_indices, heuristic_index
    from tools.curriculum_risk_teacher import guard_diagnostic
    from tools.curriculum_case_order import canonical_trace
    from tools.curriculum_win_context import public_tile
    from tools.curriculum_endgame_ledger import ledger
    from tools.collect_curriculum_basic_teacher import sources
    families = sources(parent['config'])
    if (families != parent['sources'] or version('riichienv') != '0.4.11'
            or package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']):
        raise ValueError('Original split/native/project changed')
    originals = {r['source']['seed']: r for r in (json.loads(line) for line in rawpath.open())}
    if set(originals) != {s['seed'] for s in families}: raise ValueError('All256 complete source records required')
    paths = [Path(__file__), config_path, registered, rawpath, source_root/'registered-private.json',
        source_root/'summary-private.json', proof_root/'collection-v1/summary-private.json',
        proof_root/'collection-v1/completion-binding-private.json', proof_root/'collection-terminal-scheduler-private.txt',
        ROOT/'logs/curriculum-terminal-reuse256-exit-317726.json', ROOT/'tools/curriculum_source_event_replay_v1.py',
        ROOT/'tools/curriculum_win_context.py', ROOT/'tools/curriculum_case_order.py',
        ROOT/'tools/curriculum_endgame_ledger.py', ROOT/'tools/collect_curriculum_basic_teacher.py',
        ROOT/'tools/curriculum_risk_teacher.py', ROOT/'cluster/curriculum-ordinary-completion-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-ordinary-completion-20261123.md', Path(native.__file__)]
    files = {str(p.resolve()): sha(p) for p in paths}; existing = set(); existing_count = 0
    for directory in c['existing_teacher_sources']:
        path = ROOT/directory/'summary-private.json'; data = read(path); files[str(path.resolve())] = sha(path)
        for spec in data['files'].values():
            path = ROOT/spec['path']; files[str(path.resolve())] = sha(path)
            if files[str(path.resolve())] != spec['sha256']: raise ValueError('Existing actual teacher records changed')
            for line in path.open():
                m = json.loads(line)['_meta']; existing.add((m['source_logical_seed'], m['source_step'], m['seat'])); existing_count += 1
    if existing_count != 45351: raise ValueError('All45351 original teacher contexts required')
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(config=c, sources=families, files=files,
        input_scope_selected_before_reading_executed_action=True,
        observed_single_actor_discard_labels_not_inferred_response_negatives=True,
        source_reference=str(rawpath), node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID']))
    output_paths = {s: out/f'{s}-observed-discards-private.jsonl' for s in ('train', 'validation')}
    streams = {s: p.open('x') for s, p in output_paths.items()}
    counts = Counter(); coverage = {s: Counter() for s in streams}; groups = {s: set() for s in streams}
    try:
        for family in families:
            record = originals[family['seed']]
            if record['source'] != family: raise ValueError('Source-family metadata differs')
            env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=record['engine_seed'])
            observations = env.reset(); trace = []; marker = (env.round_wind, env.kyoku_idx, env.honba); hand_index = 0
            for step in range(30000):
                if env.done(): break
                current = (env.round_wind, env.kyoku_idx, env.honba)
                if current != marker: marker = current; trace = []; hand_index += 1
                pending = None; before = len(env.mjai_log)
                if env.phase == Phase.WaitAct and len(observations) == 1:
                    seat, obs = next(iter(observations.items())); d = obs.to_dict()
                    actions = legal_candidates(obs); trigger = env.mjai_log[-1]
                    eligible = (len(actions) >= 2 and {a.action_type for a in actions} == {ActionType.DISCARD}
                        and not d['melds'][seat] and not any(d['riichi_declared'])
                        and trigger['type'] == 'tsumo' and trigger['actor'] == seat
                        and (family['seed'], step, seat) not in existing)
                    if eligible:
                        request, menu, feats = build_request(obs)
                        pending = seat, obs, request, menu, feats
                fork, fresh, selected, receipt = resolve_batch(env, observations, record['mjai'])
                if pending is not None:
                    seat, obs, request, menu, feats = pending
                    event = record['mjai'][before]
                    actual = json.loads(selected[seat].to_mjai())
                    if (set(selected) != {seat} or event['type'] != 'dahai' or event['actor'] != seat
                            or actual['type'] != 'dahai' or public_tile(actual['pai']) != public_tile(event['pai'])
                            or fork.mjai_log[before] != event):
                        raise ValueError('Directly observed executed own discard was not verified')
                    keys = list(request['questions']['action']['criteria'])
                    label = next(k for k, a in zip(keys, menu) if signature(a) == signature(selected[seat]))
                    diagnostic = guard_diagnostic(obs, menu, feats, request, label, guarded_indices, heuristic_index)
                    request['questions']['action'].update(label=label, src='ordinary_completion_teacher:plain')
                    request['_meta'] = dict(id=f'ordinary_completion:{family["seed"]}:{hand_index}:{step}:{seat}',
                        group_id=family['group_id'], split=family['split'], phase='closed_discard', label_type='expert_replay',
                        source='mortal_executed_discard_observed_in_original_complete_log',
                        curriculum_stratum='ordinary_closed_no_threat', source_logical_seed=family['seed'],
                        source_engine_seed=record['engine_seed'], source_hand_index=hand_index, source_step=step, seat=seat,
                        observed_source_event_offset=before, source_trace=canonical_trace([[dict(player=p,
                            type=int(a.action_type), tile=a.tile, consumed=list(a.consume_tiles)) for p, a in batch.items()] for batch in trace]),
                        source_initial_current_hand_scores=list(next(e['scores'] for e in reversed(env.mjai_log) if e['type'] == 'start_kyoku')),
                        direct_executed_action_not_suppressed_claim_inference=True,
                        teacher_choice_not_optimality_proof=True, guard_diagnostic=diagnostic)
                    split = family['split']; streams[split].write(json.dumps(request)+'\n')
                    csplit = coverage[split]; csplit['rows'] += 1
                    csplit['own_discard_number_'+str(len(d['discards'][seat])+1)] += 1
                    csplit['guard:'+diagnostic['reason']] += 1; groups[split].add(family['group_id'])
                    counts['observed_new_discard_labels'] += 1
                trace.append(selected); env, observations = fork, fresh
                counts['native_batches'] += 1; counts['native_trials'] += receipt['native_trials']
            if not env.done() or env.mjai_log != record['mjai'] or list(env.scores()) != record['terminal_scores']:
                raise ValueError('Complete original source events/scores did not reproduce')
            ledger(env.mjai_log, env.scores()); counts['exact_complete_sources'] += 1
            for stream in streams.values(): stream.flush()
            if counts['exact_complete_sources'] % 16 == 0:
                print(json.dumps(dict(stage='ordinary_discard_completion', counts=dict(counts))), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error), counts=dict(counts)))
        raise
    finally:
        for stream in streams.values(): stream.close()
    if groups['train'] & groups['validation']: raise ValueError('Original source split crossed')
    if not all(sha(p) == h for p, h in files.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Bound collection or current training artifacts changed')
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    result = dict(status='observed_ordinary_completion_collected_not_fit_admitted', counts=dict(counts),
        coverage={s: dict(v) for s, v in coverage.items()}, groups={s: len(v) for s, v in groups.items()},
        original_family_splits_preserved=True, all256_source_events_scores_and_ledgers_exact=True,
        files={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for s, p in output_paths.items()},
        new_model_calls=0, additional_source_games=0, student_updates=0,
        only_explicit_executed_waitact_discards_are_labels=True,
        original_response_intentions_remain_unknown=True, existing_teacher_contexts_excluded=existing_count,
        independent_native_label_encoder_split_and_guard_audit_pending=True, training_admitted=False,
        active_fit_dependencies_unchanged=len(active), raw_labels_and_traces_private=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), goal_achieved=False)
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); collect(a.config.resolve())
