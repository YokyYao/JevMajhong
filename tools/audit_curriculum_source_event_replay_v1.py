"""Finite four-family proof before broader actor-free terminal input reuse."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_source_event_replay_v1 import resolve_batch


def audit(config_path):
    gpu_only(); config = read(config_path)
    if (config['method'] != 'four_family_actor_free_source_event_replay_preflight_v1'
            or config['family_offsets'] != [0, 4, 5, 9]
            or any(config[k] for k in ('new_model_inference', 'new_source_games',
                'expert_labels_from_inferred_actions', 'automatic_full_collection_or_student_fit'))
            or config['maximum_native_batches_per_match'] != 30000
            or config['per_family_category_caps'] != {'progression_draw': 1, 'post_call': 1, 'call_response': 1, 'win': 1}
            or config['score_contexts'] != ['hold_first', 'hold_second', 'third_gap3000', 'fourth_gap5000', 'fourth_gap13000']):
        raise ValueError('Fixed prospective four-family preflight required')
    parent_root = ROOT/config['source_root']; parent = read(parent_root/'registered-private.json')
    completed = read(parent_root/'summary-private.json')
    oldpath = parent_root/'source-matches-private.jsonl'
    if sha(oldpath) != completed['full_source_matches_sha256']:
        raise ValueError('Completed original trajectories changed')
    if not all(sha(p) == h for p, h in parent['files'].items()):
        raise ValueError('Original source dependencies changed')
    reg = read(ROOT/parent['config']['source_registration'])
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
    expected = sources(parent['config'])
    if parent['sources'] != expected or version('riichienv') != '0.4.11':
        raise ValueError('Native runtime or source-family splits differ')
    original = {r['source']['seed']: r for r in (json.loads(line) for line in oldpath.open())}
    families = [expected[n] for n in config['family_offsets']]
    if Counter(s['split'] for s in families) != {'train': 2, 'validation': 2}:
        raise ValueError('Two training/two validation source families required')
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    paths = [Path(__file__), config_path, ROOT/'tools/curriculum_source_event_replay_v1.py',
             ROOT/'tests/integration/test_curriculum_source_event_replay_v1.py',
             ROOT/'cluster/curriculum-source-event-replay-v1.slurm',
             ROOT/'docs/experiments/plans/curriculum-source-event-replay-20261123.md',
             parent_root/'registered-private.json', parent_root/'summary-private.json', oldpath,
             ROOT/'tools/curriculum_endgame_contexts_v2.py', Path(native.__file__)]
    files = {str(p.resolve()): sha(p) for p in paths}
    out = ROOT/config['out']; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(config=config, source_families=families, files=files,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'],
        prospective_unlabelled_caps=True, inferred_actions_never_expert_labels=True))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests/integration'), pattern='test_curriculum_source_event_replay_v1.py')
    checked = unittest.TextTestRunner(verbosity=2).run(suite)
    if not checked.wasSuccessful() or checked.testsRun != 4: raise ValueError('Native replay integration checks failed')
    # Existing actual teacher rows provide an independent observation/menu
    # reference at their original physical decision coordinates, never labels
    # for a suppressed action inferred by the new replay.
    existing = {}
    for spec in completed['files'].values():
        p = ROOT/spec['path']; files[str(p.resolve())] = sha(p)
        if sha(p) != spec['sha256']: raise ValueError('Known teacher observations changed')
        for line in p.open():
            row = json.loads(line); m = row['_meta']
            if m['source_logical_seed'] in {s['seed'] for s in families}:
                existing[m['source_logical_seed'], m['source_step'], m['seat']] = row
    stats = Counter(); coverage = {s: Counter() for s in ('train', 'validation')}; checked_inputs = set()
    with (out/'unlabelled-inputs-private.jsonl').open('x') as stream:
        for family in families:
            record = original[family['seed']]; seed = engine_seed(family['seed'], 'spaced-v1')
            if record['source'] != family or record['engine_seed'] != seed: raise ValueError('Source metadata changed')
            env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=seed)
            observations = env.reset(); trace = []; marker = (env.round_wind, env.kyoku_idx, env.honba); hand_index = 0
            seen = set()
            for step in range(30000):
                if env.done(): break
                current = (env.round_wind, env.kyoku_idx, env.honba)
                if current != marker: trace = []; marker = current; hand_index += 1
                for seat, obs in sorted(observations.items()):
                    known = existing.get((family['seed'], step, seat))
                    if known is not None:
                        actual = build_request(obs)[0]; expected_input = dict(known); expected_input.pop('_meta')
                        expected_q = dict(expected_input['questions']['action'])
                        expected_q.pop('label'); expected_q.pop('src')
                        expected_input = dict(expected_input, questions=dict(expected_input['questions'], action=expected_q))
                        if actual != expected_input: raise ValueError('Replayed input differs from actual saved teacher observation')
                        checked_inputs.add((family['seed'], step, seat)); stats['known_teacher_inputs_exact'] += 1
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
                    for name, context in score_contexts(seat, env.riichi_sticks).items():
                        fork, proof = replay_score_context(env, seed, hand_index, trace, seat, context['scores'])
                        own = fork.get_obs_py([seat])[seat]
                        value = build_win_request(own, fork.mjai_log[-1])[0] if category == 'win' else build_request(own)[0]
                        if public_placement(value['state'])['rank'] != context['rank']:
                            raise ValueError('Score-context public rank differs')
                        value['_meta'] = dict(group_id=family['group_id'], split=family['split'],
                            source_logical_seed=family['seed'], source_engine_seed=seed,
                            source_step=step, source_hand_index=hand_index, seat=seat, category=category,
                            score_context=name, initial_scores=proof['initial_scores'], initial_sticks=proof['initial_sticks'],
                            source_trace=canonical_trace([[dict(player=p, type=int(a.action_type), tile=a.tile,
                                consumed=list(a.consume_tiles)) for p, a in batch.items()] for batch in trace]),
                            source_actions_reconstructed_not_expert_labels=True, labels_generated=False)
                        stream.write(json.dumps(value)+'\n'); coverage[family['split']][category] += 1
                    seen.add(category); stats['physical_'+category] += 1
                env, observations, selected, receipt = resolve_batch(env, observations, record['mjai'])
                trace.append(selected); stats['native_trials'] += receipt['native_trials']; stats['native_batches'] += 1
                stats['forced_wins'] += receipt['legal_wins_taken']
            if not env.done() or env.mjai_log != record['mjai'] or list(env.scores()) != record['terminal_scores']:
                raise ValueError('Complete original match events/scores not exactly reproduced')
            ledger(env.mjai_log, env.scores()); stats['exact_complete_sources'] += 1
            stream.flush()
    if checked_inputs != set(existing): raise ValueError('Some independent source observations were not reached')
    if not all(sha(p) == h for p, h in files.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Bound preflight or active training dependency changed')
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active,
        raw_output_sha256=sha(out/'unlabelled-inputs-private.jsonl')))
    result = dict(status='four_family_actor_free_native_replay_preflight_passed', counts=dict(stats),
        coverage={s: dict(v) for s, v in coverage.items()}, integration_tests=4,
        original_family_splits_preserved=True, existing_teacher_inputs_all_exact=True,
        all_source_events_scores_and_ledgers_exact=True, same_wall_score_variants_verified=True,
        active_fit_bound_dependencies_unchanged=len(active), model_calls=0, new_source_games=0,
        labels_generated=0, student_updates=0, larger_collection_admitted=False,
        raw_unlabelled_inputs_private=True, goal_achieved=False,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); audit(a.config.resolve())
