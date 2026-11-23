"""Independent native/encoder/split/score audit of reused terminal inputs.

The win-context normalization here is an explicit future-input projection,
not an extension of current ordinary UnifiedPolicyClient serving.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tomllib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write


def restore(row):
    from riichienv import RiichiEnv, GameRule
    from jevmahjong.core.state import legal_candidates, build_request
    from tools.curriculum_win_context import build_win_request
    m, s = row['_meta'], row['state']
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=m['source_engine_seed']+m['source_hand_index'])
    observations = env.reset(oya=s['dealer'], round_wind=s['round_wind'], honba=s['honba'],
                             scores=m['initial_scores'], kyotaku=m['initial_sticks'])
    key = lambda a: (int(a.action_type), a.tile, tuple(sorted(a.consume_tiles)))
    for batch in m['source_trace']:
        if {a['player'] for a in batch} != set(observations): raise ValueError('Reconstructed trace omitted an active actor')
        actions = {}
        for saved in batch:
            identity = saved['type'], saved['tile'], tuple(sorted(saved['consumed']))
            menu = {key(a): a for a in legal_candidates(observations[saved['player']])}
            if identity not in menu: raise ValueError('A stored physical replay action is illegal')
            actions[saved['player']] = menu[identity]
        observations = env.step(actions)
        if env.done(): raise ValueError('Reconstruction settled before the current decision')
    obs = env.get_obs_py([m['seat']])[m['seat']]
    value = (build_win_request(obs, env.mjai_log[-1]) if m['category'] == 'win' else build_request(obs))[0]
    if value != {k: v for k, v in row.items() if k != '_meta'}:
        raise ValueError('Stored raw input differs from native physical reconstruction')
    return env, obs


def projection(row):
    from jevmahjong.policy.unified import serving_request
    value = copy.deepcopy(row); value.pop('_meta', None)
    context = value['state'].pop('win_context', None)
    for q in value['questions'].values():
        q.pop('label', None); q.pop('target', None); q.pop('src', None)
    result = serving_request(value)
    if context is not None: result['state']['win_context'] = context
    return result


def fingerprint(row, remove_win_context=False):
    value = copy.deepcopy(row)
    if remove_win_context: value['state'].pop('win_context', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit(data_root, out, terminal_path, exit_path):
    gpu_only()
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    scheduler_exit(read(exit_path), terminal_path.read_text(), ROOT/'cluster/curriculum-reused-terminal-v1.slurm')
    ready_path = data_root/'summary-private.json'; ready = read(ready_path)
    binding_path = data_root/'completion-binding-private.json'; binding = read(binding_path)
    registration = read(data_root/'registered-private.json')
    if (ready['status'] != 'all256_actor_free_terminal_inputs_collected_not_admitted'
            or ready['counts']['exact_complete_sources'] != 256 or not ready['all45351_old_teacher_inputs_exact']
            or ready['labels_generated'] or ready['student_updates']): raise ValueError('Complete unlabelled256 collection required')
    if not all(sha(p) == h for p, h in binding['files'].items()): raise ValueError('Completed collection bindings changed')
    source = read(ROOT/'runs/curriculum-basic-teacher-data-v1/data/registered-private.json')
    reg = read(ROOT/source['config']['source_registration'])
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevany.api import SystemOneRequest, to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel, load_preprocessor
    from jevmahjong.training.base_lora import package_digest
    from tools.audit_curriculum_endgame_cases import public_solo_win
    from tools.curriculum_endgame import in_endgame_scope, public_placement
    import torch
    torch.set_num_threads(4)
    if (package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
            or package_digest(Path(reg['training_config']['upstream_source'])/'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ValueError('Pinned model-input runtime or encoder source changed')
    recipe = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    tokenizer = load_preprocessor(reg['training_config']['base_load_path'], backbone_adapter='text')
    proxy = SimpleNamespace(decision_mode='pointer', option_isolation=False)
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    paths = [Path(__file__), ready_path, binding_path, data_root/'registered-private.json', terminal_path, exit_path,
             ROOT/'cluster/curriculum-reused-terminal-audit-v1.slurm',
             ROOT/'docs/experiments/plans/curriculum-reused-terminal-audit-20261123.md',
             ROOT/'tools/audit_curriculum_endgame_cases.py', ROOT/'tools/curriculum_win_context.py']
    files = {str(p.resolve()): sha(p) for p in paths}
    # Available earlier corpora plus canonical terminal cases. Compare both
    # full projected inputs and projections without the new win-source field.
    refs = []; clean_path = ROOT/'runs/curriculum-reproduction-v1/clean-data-v1/data-ready.json'
    clean = read(clean_path); reserved_path = clean_path.parent/'reserved-public-groups.json'; reserved = set(read(reserved_path))
    files[str(clean_path)] = sha(clean_path); files[str(reserved_path)] = sha(reserved_path)
    for stage in clean['stages'].values(): refs.extend(stage.items())
    for path in (ROOT/'runs/curriculum-calls-post-reach-data-v1/data/data-ready-private.json',
                 ROOT/'runs/curriculum-risk-teacher-data-v1/audit-v1/summary-private.json',
                 ROOT/'runs/curriculum-stage4-placement-mortal-data-v4/data/data-ready-private.json'):
        value = read(path); files[str(path)] = sha(path); refs.extend(value['files'].items())
    old_inputs = {s: set() for s in ('train', 'validation')}; old_groups = {s: set() for s in old_inputs}
    physical_splits = {}; read_paths = set()
    for split, spec in refs:
        path = ROOT/spec['path']
        if (split, str(path)) in read_paths: continue
        read_paths.add((split, str(path))); files[str(path)] = sha(path)
        if files[str(path)] != spec['sha256']: raise ValueError('Ancestral data changed')
        for line in path.open():
            row = json.loads(line); m = row['_meta']; public = projection(row)
            old_groups[split].add(m['group_id'])
            old_inputs[split].update((fingerprint(public), fingerprint(public, True)))
            if 'source_logical_seed' in m:
                seed = m['source_logical_seed']
                if seed in physical_splits and physical_splits[seed] != split:
                    raise ValueError('An ancestral physical source seed crosses splits under aliases')
                physical_splits[seed] = split
    if len(reserved) != 675: raise ValueError('Original675 reserved groups required')
    families = {s['seed']: s for s in registration['sources']}
    counts = {s: Counter() for s in old_inputs}; lengths = {s: Counter() for s in old_inputs}
    groups = {s: set() for s in old_inputs}; inputs = {s: set() for s in old_inputs}
    proxy_counts = defaultdict(Counter); proxy_physical = defaultdict(set); proxy_families = defaultdict(set)
    physical = {s: set() for s in old_inputs}; physical_caps = defaultdict(set)
    unique_ids = set(); score_rows = []; ids = []
    out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files, collection_sha256=sha(ready_path),
        proposed_win_normalization='ordinary_normalization_plus_exact_native_public_win_source_v1',
        projection_not_current_ordinary_serving_contract=True, no_labels_or_model_weights_loaded=True))
    outputs = {}
    for split, spec in ready['files'].items():
        path = ROOT/spec['path']; files[str(path)] = sha(path)
        if files[str(path)] != spec['sha256']: raise ValueError('New unlabelled collection changed')
        dest = out/f'{split}-unlabelled-normalized-private.jsonl'
        with dest.open('x') as stream:
            for line in path.open():
                row = json.loads(line); m = row['_meta']; family = families[m['source_logical_seed']]
                if (m['id'] in unique_ids or family['split'] != split or m['split'] != split
                        or family['group_id'] != m['group_id'] or m['group_id'] in reserved
                        or physical_splits.get(m['source_logical_seed']) != split
                        or any(k in row['questions']['action'] for k in ('label', 'target', 'src'))):
                    raise ValueError('Changed family/ID, reserved data or labelled input')
                env, obs = restore(row); public = projection(row)
                physical_caps[m['source_logical_seed'], m['category']].add(m['physical_id'])
                if (len(physical_caps[m['source_logical_seed'], m['category']]) > 1
                        or m['score_context'] not in registration['config']['score_contexts']):
                    raise ValueError('Registered per-family/category cap or score context changed')
                if not in_endgame_scope(public['state']) or public_placement(public['state']) != m['placement']:
                    raise ValueError('Late scope or public placement changed')
                signatures = {fingerprint(public), fingerprint(public, True)}; other = 'validation' if split == 'train' else 'train'
                if signatures & old_inputs[other] or signatures & inputs[other] or m['group_id'] in old_groups[other]:
                    raise ValueError('New input or physical group overlaps the opposite split')
                if fingerprint(public) in inputs[split]: raise ValueError('Duplicate new normalized observed input')
                inputs[split].update(signatures); groups[split].add(m['group_id']); unique_ids.add(m['id']); physical[split].add(m['physical_id'])
                request = dict(public, _meta=copy.deepcopy(m))
                record, mapping = to_record(SystemOneRequest.model_validate(api_request(request)))
                encoded = DecisionModel.encode(proxy, tokenizer, record, max_state=recipe['max_state'], max_branch=recipe['max_branch'], strict=True)
                if (encoded['state_truncated'] or len(encoded['ids']) > recipe['max_packed']
                        or mapping[0]['keys'] != list(public['questions']['action']['criteria'])):
                    raise ValueError('Future public input truncates or legal menu is reordered')
                c = counts[split]; c['rows'] += 1; c[m['category']] += 1
                lengths[split]['max_packed'] = max(lengths[split]['max_packed'], len(encoded['ids']))
                lengths[split]['max_state'] = max(lengths[split]['max_state'], encoded['seg'].count(0))
                if m['category'] == 'win':
                    score = public_solo_win(env, obs); score_rows.append(dict(id=m['id'], group_id=m['group_id'], split=split, public_solo_score=score))
                    key = split+':'+m['score_context']; pc = proxy_counts[key]; pc['win_views'] += 1
                    if score['supported']:
                        pc['supported_no_ura_solo'] += 1
                        for yaku in score['yaku']: c['win_yaku:'+yaku] += 1
                        if score['own_income'] <= 3000 and m['placement']['rank'] in (3, 4) and score['immediate_rank'] > 2:
                            pc['insufficient_small_win_proxy'] += 1
                            for tag in (key, split+':ALL'):
                                proxy_physical[tag].add(m['physical_id']); proxy_families[tag].add(m['group_id'])
                            ids.append(dict(id=m['id'], split=split, score=score, context=m['score_context'],
                                            not_a_decline_target=True))
                        if score['own_income'] <= 3000 and m['placement']['rank'] <= 2 and score['immediate_rank'] <= 2:
                            pc['leading_small_win_proxy'] += 1
                    else: pc['unsupported:'+score['reason']] += 1
                stream.write(json.dumps(request)+'\n')
                if len(unique_ids) % 500 == 0: print(json.dumps(dict(stage='reused_input_audit', rows=len(unique_ids))), flush=True)
        outputs[split] = dict(path=str(dest.relative_to(ROOT)), sha256=sha(dest), rows=counts[split]['rows'])
    if len(unique_ids) != ready['counts']['score_views'] or groups['train'] & groups['validation']:
        raise ValueError('Collected rows or source split not preserved')
    if not all(sha(p) == h for p, h in files.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Audit changed a bound source or active fit dependency')
    with (out/'public-score-details-private.jsonl').open('x') as stream:
        for item in score_rows: stream.write(json.dumps(item)+'\n')
    write(out/'eligible-small-win-proxies-private.json', ids)
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    result = dict(status='reused_terminal_native_encoder_split_and_public_score_audit_passed',
        rows=len(unique_ids), exact_native_reconstructions=len(unique_ids),
        physical_decisions={s: len(v) for s, v in physical.items()}, groups={s: len(v) for s, v in groups.items()},
        counts={s: dict(v) for s, v in counts.items()}, tokens={s: dict(v) for s, v in lengths.items()}, files=outputs,
        public_small_win_coverage={k: dict(v) for k, v in proxy_counts.items()},
        insufficient_small_win_distinct_physical={k: len(v) for k, v in proxy_physical.items()},
        insufficient_small_win_distinct_families={k: len(v) for k, v in proxy_families.items()},
        all_available_ancestral_opposite_split_inputs_groups_and_seed_aliases_checked=True,
        reserved_groups_excluded=675, no_backbone_or_policy_weights_loaded=True,
        proposed_win_input_projection_not_current_serving=True, legal_menu_order_preserved=True,
        encoder_default_label_unused_for_training=True, no_ura_sole_win_proxy_not_terminal_target=True,
        labels_generated=0, student_updates=0, teacher_and_serving_admission_pending=True,
        active_fit_dependencies_unchanged=len(active), raw_details_private=True,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], goal_achieved=False)
    write(out/'summary-private.json', result); print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'counts')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data-root', 'out', 'terminal', 'exit'): p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); audit(a.data_root.resolve(), a.out.resolve(), a.terminal.resolve(), a.exit.resolve())
