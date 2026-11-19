"""Reuse the fixed ordinary fourth-place manifest for paired terminal labels."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_config(config):
    if (config.get('schema') != 'native-fourth-terminal16-v1'
            or config.get('future_worlds') != 16 or config.get('workers') != 2
            or config.get('manifest_limits') != {'train': 32, 'validation': 8}
            or config.get('input_version') != 'endgame-trailing-public-v1'
            or config.get('continuation_policy') != 'released_original_v1'
            or config.get('target_temperature') != .1 or config.get('terminal_step_cap') != 30000
            or config.get('max_legal_actions') != 800
            or config.get('label_registered') is not True
            or any(config.get(k) is not False for k in
                   ('fit_registered', 'promotion_registered', 'raw_publication'))):
        raise ValueError('Only the fixed-manifest ordinary-fourth label stage is registered')


def select_rows(rows):
    """Take ALL current-fourth native frames; never choose by their outcomes."""
    selected = [r for r in rows if r['rank'] == 4]
    if (Counter(r['split'] for r in selected) != {'train': 32, 'validation': 8}
            or len({r['seed'] for r in selected}) != 40
            or len({r['id'] for r in selected}) != 40):
        raise ValueError('Original distinct-source 32/8 ordinary-fourth split changed')
    for row in selected:
        if (row['split'] != ('validation' if (row['seed'] - 2026116900) % 4 == 3 else 'train')
                or row['group_id'] != 'endgame-native:' + str(row['seed'])
                or any(a['type'] == 'hora' for a in row['teacher_request']['questions']['action']['criteria'].values())):
            raise ValueError('Ordinary source identity, split or no-win condition changed')
    return selected


def partition(rows):
    bins, costs = [[], []], [0, 0]
    for row in sorted(rows, key=lambda r: (-len(r['teacher_request']['questions']['action']['criteria']), r['id'])):
        index = min(range(2), key=lambda i: (costs[i], i))
        bins[index].append(row)
        costs[index] += len(row['teacher_request']['questions']['action']['criteria'])
    return bins


def source_path(config, row):
    if row['source_profile'] not in ('standard', 'dama'):
        raise ValueError('Unregistered cached-source profile')
    return Path(config['coverage_out']) / row['source_profile'] / 'sources' / f"{row['seed']}.json"


def replay_row(config, row):
    from riichienv import GameRule, RiichiEnv
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import build_request
    from jevmahjong.evaluation.agents import guarded_indices
    from jevmahjong.training.futures import replay_prefix
    from jevmahjong.endgame.collect import digest
    from jevmahjong.endgame.context import annotate_record, placements
    from jevmahjong.endgame.ledger import hand_score_history
    from jevmahjong.endgame.native_public import add_native_context
    from desktop.jev_sidecar.unified_policy import serving_request
    source = json.loads(source_path(config, row).read_text())
    if (source['seed'] != row['seed'] or source['profile'] != row['source_profile']
            or hand_score_history(source['mjai_log'])['final_scores'] != source['final_scores']):
        raise ValueError('Original source identity or full terminal ledger changed')
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(),
                    seed=engine_seed(row['seed'], 'spaced-v1'))
    observations = replay_prefix(env, source['mjai_log'], row['step'])
    obs = observations[row['seat']]
    raw, actions, features = build_request(obs)
    raw['state']['hand_number'] = int(obs.kyoku_index) + 1
    teacher = serving_request(raw)
    student = annotate_record(add_native_context(raw, obs, actions), config['input_version'])
    keys = list(teacher['questions']['action']['criteria'])
    native = sorted(set(guarded_indices(obs, actions, features)) | {i for i, f in enumerate(features) if not f})
    if (raw != row['raw_request'] or teacher != row['teacher_request'] or student != row['student_request']
            or digest(teacher) != row['teacher_request_sha256']
            or digest(student) != row['student_request_sha256']
            or [keys[i] for i in native] != row['effective_action_keys']
            or placements(list(env.scores()))[row['seat']] != 4):
        raise ValueError('Native prefix, original/student input, support or current rank changed')
    return env, actions


def protocol(config):
    from jevmahjong.endgame.collect import digest
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.native_retention import _load_native_set
    from desktop.jev_sidecar.unified_policy import identity
    validate_config(config)
    rows = select_rows(_load_native_set(config['native_context_root'], config['native_context_positions_sha256']))
    action_count = sum(len(r['teacher_request']['questions']['action']['criteria']) for r in rows)
    if action_count > config['max_legal_actions']:
        raise ValueError('Public all-action budget exceeded; do not prune using outcomes')
    baseline = bundle_identity(config['checkpoint'])
    if baseline['identity_sha256'] not in (identity()['source_identity_sha256'], identity()['identity_sha256']):
        raise ValueError('Only the released v8 may supply continuations')
    paths = set()
    for directory in ('jevmahjong/core', 'jevmahjong/policy', 'jevmahjong/evaluation',
                      'jevmahjong/endgame', 'desktop/jev_sidecar'):
        paths.update((ROOT / directory).rglob('*.py'))
    paths.update(ROOT / name for name in ('jevmahjong/training/futures.py',
                                         'tools/check_gpu.py', 'tools/label_native_endgame.py'))
    upstream = Path(config['upstream_source']) / 'jevany'
    if not upstream.is_dir():
        raise ValueError('Pinned JevAny source is missing')
    return {'configuration': config, 'positions_sha256': digest(rows), 'baseline_identity': baseline,
        'native_manifest_file_sha256': hashlib.sha256((Path(config['native_context_root']) / 'paired-contexts-private.json').read_bytes()).hexdigest(),
        'source_sha256': {str(source_path(config, r)): hashlib.sha256(source_path(config, r).read_bytes()).hexdigest() for r in rows},
        'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
        'upstream_sha256': {str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(upstream.rglob('*.py'))},
        'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'riichienv', 'mahjong')},
        'worker_ids': [[r['id'] for r in part] for part in partition(rows)],
        'legal_actions': action_count, 'terminal_rollout_budget': action_count * 16,
        'optimizer_updates': 0}, rows


def checked_protocol(config):
    expected = json.loads((Path(config['out']) / 'protocol-private.json').read_text())
    current, rows = protocol(config)
    if expected != current:
        raise ValueError('Frozen ordinary-fourth label protocol changed')
    if json.loads((Path(config['out']) / 'selected-positions-private.json').read_text()) != rows:
        raise ValueError('Original fixed40 manifest changed')
    return expected, rows


def label_path(config, row):
    return Path(config['out']) / 'labels' / (row['id'].split(':', 1)[1] + '.json')


def validate_label(label, row, config, baseline_identity):
    from jevmahjong.endgame.context import placements, terminal_utility, target_from_utilities
    keys = list(row['student_request']['questions']['action']['criteria'])
    if (label['status'] != 'complete_terminal_menu' or label['source'] != row
            or label['source_request_sha256'] != row['student_request_sha256']
            or label['baseline_identity_sha256'] != baseline_identity
            or label['worlds_per_action'] != 16 or not label['continued_until_native_match_done']
            or label['actual_actions'] != len(keys) or len(label['outcomes']) != len(keys)
            or label['record']['state'] != row['student_request']['state']
            or label['record']['questions']['action']['criteria'] != row['student_request']['questions']['action']['criteria']):
        raise ValueError('Ordinary terminal label provenance changed')
    means = []
    for outcomes in label['outcomes']:
        if len(outcomes) != 16:
            raise ValueError('Incomplete paired native worlds')
        for outcome in outcomes:
            if (not outcome['ledger_passed'] or placements(outcome['scores']) != outcome['ranks']
                    or terminal_utility(outcome['ranks'][row['seat']]) != outcome['utility']
                    or outcome['hand_history']['final_scores'] != outcome['scores']):
                raise ValueError('Terminal score/rank/utility/history does not reconcile')
        means.append(sum(o['utility'] for o in outcomes) / 16)
    q = label['record']['questions']['action']
    expected = dict(zip(keys, target_from_utilities(means, config['target_temperature'])))
    if q['target'] != expected or label['record']['_meta']['expected_terminal_utilities'] != dict(zip(keys, means)):
        raise ValueError('Terminal target differs from complete outcomes')
    return len(keys) * 16


def prepare(config):
    from jevmahjong.endgame.collect import write, digest
    frozen, rows = protocol(config)
    for row in rows:
        replay_row(config, row)
    if protocol(config)[0] != frozen:
        raise ValueError('Sources or runtime changed during native-prefix verification')
    out = Path(config['out'])
    for name, value in (('selected-positions-private.json', rows), ('protocol-private.json', frozen)):
        path = out / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Existing native label registration differs')
        write(path, value)
    summary = {'status': 'fixed40_native_prefixes_verified_and_frozen', 'positions': 40,
        'splits': dict(Counter(r['split'] for r in rows)),
        'phases': dict(Counter(r['split'] + ':' + r['phase'] for r in rows)),
        'positions_sha256': digest(rows), 'protocol_sha256': digest(frozen),
        'worker_positions': [len(p) for p in partition(rows)],
        'worker_actions': [sum(len(r['teacher_request']['questions']['action']['criteria']) for r in p) for p in partition(rows)],
        'legal_actions': frozen['legal_actions'], 'terminal_rollout_budget': frozen['terminal_rollout_budget'],
        'model_queries': 0, 'optimizer_updates': 0, 'raw_publication': False}
    write(out / 'preparation-private.json', summary)
    return summary


def worker(config, index):
    from jevmahjong.core.state import legal_candidates, signature
    from jevmahjong.evaluation.agents import EfficiencyAgent, CallingOpponent
    from jevmahjong.training.futures import sample_future
    from jevmahjong.endgame.collect import write, digest
    from jevmahjong.endgame.context import target_from_utilities
    from jevmahjong.endgame.futures import finish_match
    from jevmahjong.endgame.policy import ModelAgent
    from jevmahjong.policy.routing import load_model
    if index not in (0, 1):
        raise ValueError('Expected worker 0 or 1')
    frozen_protocol, all_rows = checked_protocol(config)
    rows = partition(all_rows)[index]
    baseline_identity = frozen_protocol['baseline_identity']['identity_sha256']
    model = load_model(config['checkpoint'])
    frozen = ModelAgent(model, placement_aware=False)
    ordinary, caller = EfficiencyAgent(), CallingOpponent()
    out = Path(config['out']); start = time.monotonic()
    for number, row in enumerate(rows):
        path = label_path(config, row)
        if path.exists():
            validate_label(json.loads(path.read_text()), row, config, baseline_identity)
            continue
        env, actions = replay_row(config, row)
        signatures = [signature(a) for a in actions]
        outcomes = [[] for _ in actions]
        for world in range(16):
            world_seed = int(digest([row['id'], world, 'native-ordinary-terminal-wall-v1'])[:16], 16)
            sampled = sample_future(env, world_seed, 'unseen')
            initial = {p: obs for p, obs in sampled.clone().get_observations().items() if obs.legal_actions()}
            other_actions = {p: (caller if p == row['seed'] % 4 else ordinary).act(obs)
                             for p, obs in initial.items() if p != row['seat']}
            for action_index, action_signature in enumerate(signatures):
                branch = sampled.clone()
                branch_obs = branch.get_observations()[row['seat']]
                matching = {signature(a): a for a in legal_candidates(branch_obs)}
                if set(matching) != set(signatures):
                    raise ValueError('Sampled unseen wall changed the current action support')
                initial_actions = {**other_actions, row['seat']: matching[action_signature]}
                outcomes[action_index].append(finish_match(branch, initial_actions, row['seat'],
                    row['seed'], frozen, config['terminal_step_cap']))
            write(out / f'worker-{index}-status-private.json', {'status': 'labeling_native_fourth',
                'positions_complete': number, 'positions_total': len(rows), 'world_complete': world + 1,
                'current_actions': len(actions), 'elapsed_seconds': time.monotonic() - start,
                'optimizer_updates': 0})
        means = [sum(o['utility'] for o in values) / 16 for values in outcomes]
        record = copy.deepcopy(row['student_request'])
        keys = list(record['questions']['action']['criteria'])
        probability = target_from_utilities(means, config['target_temperature'])
        record['questions']['action'].update(src='endgame_terminal', target=dict(zip(keys, probability)),
                                             label=keys[max(range(len(keys)), key=lambda i: probability[i])])
        record['_meta'] = {'id': row['id'], 'group_id': row['group_id'], 'split': row['split'],
            'source': 'native_fourth_terminal', 'bucket': 'fourth', 'phase': row['phase'],
            'variant': 'clean', 'source_initial_placement': 4,
            'expected_terminal_utilities': dict(zip(keys, means))}
        label = {'status': 'complete_terminal_menu', 'source': row,
            'source_request_sha256': row['student_request_sha256'], 'baseline_identity_sha256': baseline_identity,
            'worlds_per_action': 16, 'outcomes': outcomes, 'record': record, 'actual_actions': len(actions),
            'sampling_scope': 'unseen current-hand wall; fixed current concealed hands and cloned future-hand RNG',
            'continued_until_native_match_done': True}
        validate_label(label, row, config, baseline_identity)
        write(path, label)
        print(json.dumps({'worker': index, 'positions_complete': number + 1, 'total': len(rows),
                          'elapsed_seconds': time.monotonic() - start}), flush=True)
    checked_protocol(config)
    result = {'status': 'native_fourth_worker_complete', 'worker': index, 'positions': len(rows),
              'protocol_sha256': digest(frozen_protocol), 'optimizer_updates': 0}
    write(out / f'worker-{index}-complete-private.json', result)
    return result


def finalize(config):
    from jevmahjong.endgame.collect import write, digest
    frozen, rows = checked_protocol(config)
    for index, part in enumerate(partition(rows)):
        receipt = json.loads((Path(config['out']) / f'worker-{index}-complete-private.json').read_text())
        if (receipt['status'] != 'native_fourth_worker_complete' or receipt['positions'] != len(part)
                or receipt['protocol_sha256'] != digest(frozen) or receipt['worker'] != index):
            raise ValueError('Incomplete or mismatched native labeling worker')
    total = 0; hashes = {}
    for row in rows:
        path = label_path(config, row)
        total += validate_label(json.loads(path.read_text()), row, config, frozen['baseline_identity']['identity_sha256'])
        hashes[row['id']] = hashlib.sha256(path.read_bytes()).hexdigest()
    if total != frozen['terminal_rollout_budget']:
        raise ValueError('All-action terminal rollout total differs')
    result = {'status': 'all40_native_fourth_terminal_menus_verified', 'positions': 40,
        'splits': {'train': 32, 'validation': 8}, 'future_worlds': 16, 'terminal_rollouts': total,
        'positions_sha256': digest(rows), 'protocol_sha256': digest(frozen), 'labels_sha256': hashes,
        'continuation_policy': 'released_original_v1', 'new_source_groups': 0, 'optimizer_updates': 0,
        'fit_registered': False, 'raw_publication': False}
    write(Path(config['out']) / 'label-verification-private.json', result)
    return {k: v for k, v in result.items() if k != 'labels_sha256'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'worker', 'finalize'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--worker-index', type=int)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-', 'login')):
        raise RuntimeError('Native labeling requires an approved GPU allocation')
    import subprocess
    from tools.prepare_native_context_replay import allocation_metadata
    description = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '--oneliner'], text=True)
    allocation = allocation_metadata(description, socket.gethostname())
    print(json.dumps({'hostname': socket.gethostname(), 'allocation': allocation}), flush=True)
    config = json.loads(args.config.read_text()); validate_config(config)
    sys.path.insert(0, config['upstream_source'])
    result = prepare(config) if args.stage == 'prepare' else worker(config, args.worker_index) if args.stage == 'worker' else finalize(config)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
