"""Reuse original256 forced-win trajectories for unlabelled terminal contexts."""
import argparse
from collections import Counter, defaultdict
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


def visible(row):
    value = dict(row); value.pop('_meta')
    q = dict(value['questions']['action']); q.pop('label'); q.pop('src')
    return dict(value, questions=dict(value['questions'], action=q))


def collect(config_path):
    gpu_only(); config = read(config_path)
    if (config['method'] != 'all256_verified_actor_free_terminal_inputs_v1' or config['source_games'] != 256
            or config['preserve_original_family_splits'] is not True
            or any(config[k] for k in ('new_model_inference', 'new_source_games',
                'expert_labels_from_inferred_actions', 'automatic_student_fit'))
            or config['maximum_native_batches_per_match'] != 30000
            or config['late_scope'] != 'scheduled_east4_and_extensions'
            or config['simultaneous_order'] != 'ascending_absolute_seat'
            or config['per_family_category_caps'] != {'progression_draw': 1, 'post_call': 1, 'call_response': 1, 'win': 1}
            or config['score_contexts'] != ['hold_first', 'hold_second', 'third_gap3000', 'fourth_gap5000', 'fourth_gap13000']
            or config['existing_teacher_sources'] != ['runs/curriculum-basic-teacher-data-v1/data',
                'runs/curriculum-advanced-teacher-data-v1/data', 'runs/curriculum-risk-teacher-data-v1/data',
                'runs/curriculum-post-reach-teacher-data-v2/data']):
        raise ValueError('Fixed full256 unlabelled source-reuse recipe required')
    preflight = ROOT/config['preflight_root']; proof = read(preflight/'preflight-v1/summary-private.json')
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    scheduler_exit(read(ROOT/'logs/curriculum-source-replay4-exit-317718.json'),
        (preflight/'preflight-terminal-scheduler-private.txt').read_text(), ROOT/'cluster/curriculum-source-event-replay-v1.slurm')
    if (proof['status'] != 'four_family_actor_free_native_replay_preflight_passed'
            or proof['counts']['exact_complete_sources'] != 4 or proof['counts']['known_teacher_inputs_exact'] != 87
            or proof['integration_tests'] != 4 or proof['model_calls'] or proof['labels_generated']):
        raise ValueError('Actual successful bounded four-family proof required')
    prior = read(preflight/'preflight-v1/completion-binding-private.json')
    if not all(sha(p) == h for p, h in prior['files'].items()): raise ValueError('Completed preflight bindings changed')
    parent_root = ROOT/config['source_root']; parent = read(parent_root/'registered-private.json')
    completed = read(parent_root/'summary-private.json'); oldpath = parent_root/'source-matches-private.jsonl'
    if sha(oldpath) != completed['full_source_matches_sha256'] or not all(sha(p) == h for p, h in parent['files'].items()):
        raise ValueError('Original source artifacts changed')
    regpath = ROOT/parent['config']['source_registration']; reg = read(regpath)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from riichienv import RiichiEnv, GameRule, Phase, ActionType
    import riichienv._riichienv as native
    from jevmahjong.core.state import build_request, legal_candidates
    from jevmahjong.core.seeds import engine_seed
    from tools.curriculum_endgame import in_endgame_scope, public_placement
    from tools.curriculum_endgame_contexts_v2 import score_contexts, replay_score_context
    from tools.curriculum_case_order import canonical_trace
    from tools.curriculum_win_context import build_win_request
    from tools.curriculum_endgame_ledger import ledger
    from tools.collect_curriculum_basic_teacher import sources
    families = sources(parent['config']); source_by_seed = {s['seed']: s for s in families}
    if parent['sources'] != families or version('riichienv') != '0.4.11':
        raise ValueError('Native runtime or original family split differs')
    originals = {r['source']['seed']: r for r in (json.loads(line) for line in oldpath.open())}
    if set(originals) != set(source_by_seed): raise ValueError('Original256 complete trajectories required')
    paths = [Path(__file__), config_path, ROOT/'tools/curriculum_source_event_replay_v1.py', regpath,
             ROOT/'cluster/curriculum-reused-terminal-v1.slurm',
             ROOT/'docs/experiments/plans/curriculum-reused-terminal-20261123.md',
             preflight/'preflight-v1/summary-private.json', preflight/'preflight-v1/completion-binding-private.json',
             preflight/'preflight-terminal-scheduler-private.txt', ROOT/'logs/curriculum-source-replay4-exit-317718.json',
             parent_root/'registered-private.json', parent_root/'summary-private.json', oldpath,
             ROOT/'tools/curriculum_endgame_contexts_v2.py', Path(native.__file__)]
    files = {str(p.resolve()): sha(p) for p in paths}
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    existing = defaultdict(list); reference_rows = 0
    for directory in config['existing_teacher_sources']:
        summary_path = ROOT/directory/'summary-private.json'; summary = read(summary_path)
        files[str(summary_path.resolve())] = sha(summary_path)
        for spec in summary['files'].values():
            path = ROOT/spec['path']; files[str(path.resolve())] = sha(path)
            if sha(path) != spec['sha256']: raise ValueError('Known teacher source decisions changed')
            for line in path.open():
                row = json.loads(line); m = row['_meta']; family = source_by_seed.get(m['source_logical_seed'])
                if family is None or family['split'] != m['split'] or family['group_id'] != m['group_id']:
                    raise ValueError('A known physical source family was renamed or crossed splits')
                existing[m['source_logical_seed'], m['source_step'], m['seat']].append(visible(row)); reference_rows += 1
    if reference_rows != 45351: raise ValueError('All45351 actual old teacher public observations required')
    out = ROOT/config['out']; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(config=config, sources=families, files=files,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'],
        prospective_caps_and_family_splits=True, input_selection_before_source_action_and_value=True,
        reconstructed_actions_not_expert_labels=True, source_package=reg['source_sha256']))
    counts = Counter(); coverage = {s: Counter() for s in ('train', 'validation')}
    groups = {s: set() for s in coverage}; seen_references = set(); native_counts = Counter()
    rawfiles = {s: out/f'{s}-unlabelled-private.jsonl' for s in coverage}
    streams = {s: p.open('x') for s, p in rawfiles.items()}
    try:
        for family in families:
            record = originals[family['seed']]; seed = engine_seed(family['seed'], 'spaced-v1')
            if record['source'] != family or record['engine_seed'] != seed: raise ValueError('Source metadata changed')
            env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=seed)
            observations = env.reset(); trace = []; marker = (env.round_wind, env.kyoku_idx, env.honba); hand_index = 0
            seen = set()
            for step in range(30000):
                if env.done(): break
                current = (env.round_wind, env.kyoku_idx, env.honba)
                if current != marker: trace = []; marker = current; hand_index += 1
                for seat, obs in sorted(observations.items()):
                    coordinate = family['seed'], step, seat
                    if coordinate in existing:
                        actual = build_request(obs)[0]
                        if any(actual != expected for expected in existing[coordinate]):
                            raise ValueError('Replayed actual teacher public input/menu differs')
                        seen_references.add(coordinate); counts['known_teacher_inputs_exact'] += len(existing[coordinate])
                    if not (env.round_wind > 0 or env.oya == 3): continue
                    actions = legal_candidates(obs); kinds = {a.action_type for a in actions}; trigger = env.mjai_log[-1]
                    if kinds & {ActionType.RON, ActionType.TSUMO}: category = 'win'
                    elif len(actions) < 2: continue
                    elif env.phase == Phase.WaitAct and trigger['type'] in ('chi', 'pon'): category = 'post_call'
                    elif env.phase == Phase.WaitAct and ActionType.DISCARD in kinds: category = 'progression_draw'
                    elif env.phase == Phase.WaitResponse and kinds & {ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN}: category = 'call_response'
                    else: continue
                    if category in seen: continue
                    request = build_request(obs)[0]
                    if not in_endgame_scope(request['state']): raise ValueError('Public late scope differs')
                    contexts = score_contexts(seat, env.riichi_sticks)
                    if not contexts: counts['no_admissible_score_views'] += 1; continue
                    physical_id = f's4-reuse:{family["seed"]}:{hand_index}:{step}:{seat}:{category}'
                    canonical = canonical_trace([[dict(player=p, type=int(a.action_type), tile=a.tile,
                        consumed=list(a.consume_tiles)) for p, a in batch.items()] for batch in trace])
                    for name, context in contexts.items():
                        fork, proof = replay_score_context(env, seed, hand_index, trace, seat, context['scores'])
                        own = fork.get_obs_py([seat])[seat]
                        value = build_win_request(own, fork.mjai_log[-1])[0] if category == 'win' else build_request(own)[0]
                        placement = public_placement(value['state'])
                        if placement['rank'] != context['rank']: raise ValueError('Score-context public rank differs')
                        value['_meta'] = dict(id=physical_id+':'+name, physical_id=physical_id,
                            group_id=family['group_id'], split=family['split'], source_logical_seed=family['seed'],
                            source_engine_seed=seed, source_step=step, source_hand_index=hand_index, seat=seat,
                            category=category, score_context=name, placement=placement,
                            initial_scores=proof['initial_scores'], initial_sticks=proof['initial_sticks'], source_trace=canonical,
                            source_actions_reconstructed_not_expert_labels=True, label_type='unlabelled_terminal_context',
                            labels_generated=False)
                        streams[family['split']].write(json.dumps(value)+'\n')
                        coverage[family['split']][category] += 1; counts['score_views'] += 1
                        groups[family['split']].add(family['group_id'])
                    seen.add(category); counts['physical_'+category] += 1; native_counts[family['split']+':'+category] += 1
                env, observations, selected, receipt = resolve_batch(env, observations, record['mjai'])
                trace.append(selected); counts['native_trials'] += receipt['native_trials']; counts['native_batches'] += 1
                counts['forced_wins'] += receipt['legal_wins_taken']
            if not env.done() or env.mjai_log != record['mjai'] or list(env.scores()) != record['terminal_scores']:
                raise ValueError('Complete original match events/scores not exactly reproduced')
            ledger(env.mjai_log, env.scores()); counts['exact_complete_sources'] += 1
            for stream in streams.values(): stream.flush()
            if counts['exact_complete_sources'] % 16 == 0:
                print(json.dumps(dict(stage='actor_free_terminal_harvest', counts=dict(counts))), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error), counts=dict(counts)))
        raise
    finally:
        for stream in streams.values(): stream.close()
    if seen_references != set(existing) or counts['known_teacher_inputs_exact'] != reference_rows:
        raise ValueError('Some actual teacher reference inputs were not reached')
    if groups['train'] & groups['validation']: raise ValueError('A physical source family crosses splits')
    if not all(sha(p) == h for p, h in files.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Bound collection or active training artifacts changed')
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    result = dict(status='all256_actor_free_terminal_inputs_collected_not_admitted', counts=dict(counts),
        coverage={s: dict(v) for s, v in coverage.items()}, physical_coverage=dict(native_counts),
        output_groups={s: len(v) for s, v in groups.items()},
        registered_family_splits=dict(Counter(s['split'] for s in families)),
        files={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for s, p in rawfiles.items()},
        original256_events_scores_and_ledgers_exact=True, all45351_old_teacher_inputs_exact=True,
        no_model_calls=True, additional_independent_source_games=0,
        inferred_suppressed_claims_are_not_teacher_labels=True, score_views_share_physical_source_group=True,
        native_tokenizer_ancestral_overlap_and_score_coverage_audit_pending=True,
        teacher_target_sensitivity_and_student_fit_pending=True, labels_generated=0, student_updates=0,
        active_fit_bound_dependencies_unchanged=len(active), raw_unlabelled_inputs_private=True,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], goal_achieved=False)
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); collect(a.config.resolve())
