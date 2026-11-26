"""Normalize new yaku teacher decisions and retain the exact existing replay.

Build data and an aggregate catalog, not a loss mixture, parent admission or
student fit. Original source files and their family partitions stay immutable.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import socket
import os
import sys
import tomllib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.build_curriculum_stage1 import public_input

ROUTES = {'kokushi', 'chuuren', 'suuankou', 'daisangen', 'chiitoitsu', 'unselected_random_initial_hand_control'}


def fingerprint(row):
    return hashlib.sha256(json.dumps(public_input(row), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def action_key(row):
    q = row['questions']['action']
    return q['label']


def skeleton(candidate):
    return {k: candidate[k] for k in ('type', 'pai', 'consumed') if k in candidate}


def prepare(config):
    gpu_only(); c = read(config)
    if (c['method'] != 'exact_progression_replay_plus_normalized_targeted_yaku_teacher_v2'
            or not c['original_replay_objects_unchanged'] or c['automatic_fit']
            or c['source_route_is_visible_input']):
        raise ValueError('Fixed private data-only normalization recipe required')
    replay_path = ROOT/c['replay_audit']; replay = read(replay_path)
    if replay['status'] != 'exact_progression_union_original_rows_encoder_masks_and_split_audit_passed_not_fit_admitted':
        raise ValueError('Use the already audited unchanged progression union')
    reg_path = ROOT/c['source_registration']; reg = read(reg_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(2)
    from jevmahjong.data.base_lora_pool import normalized_request
    from jevany.data import materialize
    from jevany.model import DecisionModel, load_preprocessor
    recipe = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    limits = {k: recipe[k] for k in ('max_state', 'max_branch', 'max_packed')}
    tokenizer = load_preprocessor(reg['training_config']['base_load_path'], backbone_adapter='text')
    proxy = SimpleNamespace(decision_mode='pointer', option_isolation=False)
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    inputs = {str(p.resolve()): sha(p) for p in (Path(__file__), config, replay_path, reg_path,
        ROOT/'tools/build_curriculum_stage1.py', ROOT/'cluster/curriculum-yaku-data-v2.slurm',
        ROOT/'docs/experiments/plans/curriculum-yaku-data-v2-20261123.md',
        ROOT/'tools/prepare_curriculum_yaku_data_v1.py',
        ROOT/'runs/curriculum-yaku-scenarios-v1/data-v1/error-private.json',
        Path(reg['source'])/'jevmahjong/data/base_lora_pool.py',
        Path(reg['source'])/'jevmahjong/policy/call_features.py')}
    old_rows = {}; old_inputs = {}; old_ids = {}; old_groups = {}
    for split in ('train', 'validation'):
        spec = replay['files'][split]; path = ROOT/spec['path']; digest = hashlib.sha256(); rows = []
        with path.open('rb') as stream:
            for raw in stream:
                digest.update(raw); rows.append(json.loads(raw))
        if digest.hexdigest() != spec['sha256'] or len(rows) != spec['records']:
            raise ValueError('Already audited replay file changed')
        inputs[str(path.resolve())] = spec['sha256']; old_rows[split] = rows
        old_inputs[split] = {fingerprint(r) for r in rows}
        old_ids[split] = {r['_meta']['id'] for r in rows}
        old_groups[split] = {r['_meta']['group_id'] for r in rows}
    source_specs = []; normalized = {s: [] for s in old_rows}; raw_counts = Counter()
    for source_name, summary_name, expected_status in c['new_teacher_sources']:
        path = ROOT/summary_name; source = read(path); inputs[str(path.resolve())] = sha(path)
        if source['status'] != expected_status:
            raise ValueError('Incomplete actual new teacher source')
        for split, spec in source['files'].items():
            file = ROOT/spec['path']; actual = sha(file)
            if actual != spec['sha256']:
                raise ValueError('New teacher source bytes changed')
            inputs[str(file.resolve())] = actual
            n = 0
            for n, line in enumerate(file.open(), 1):
                raw = json.loads(line); meta = raw['_meta']; q = raw['questions']['action']
                route = meta['scenario_route']
                if meta['split'] != split or route not in ROUTES or q.get('target') is not None:
                    raise ValueError('Preserve original hard labels and family assignments')
                row = normalized_request(raw, meta['group_id'], 'expert_replay', meta['id'])
                newq = row['questions']['action']
                if (newq['label'] != q['label'] or list(newq['criteria']) != list(q['criteria'])
                        or any(skeleton(newq['criteria'][k]) != skeleton(value) for k, value in q['criteria'].items())):
                    raise ValueError('Normalization changed legal action identities or observed label')
                row['_meta'].update(id=meta['id'], split=split, scenario_route=route,
                    original_teacher_source=source_name, original_teacher_src=q['src'],
                    original_teacher_file=spec['path'], original_teacher_line=n,
                    scenario_source_id=meta.get('scenario_source_id', meta['id']),
                    source_step=meta.get('source_step', 0),
                    teacher_choice_not_optimality_proof=True)
                newq['src'] = 'curriculum3B:teacher:'+route
                normalized[split].append(row); raw_counts[split] += 1
            if n != spec['records']:
                raise ValueError('Source teacher count differs')
            source_specs.append(dict(name=source_name, split=split, path=spec['path'], records=n, sha256=actual))
    if dict(raw_counts) != {'train': 3652, 'validation': 828}:
        raise ValueError('Exact511 initial plus3969 continuation labels required')
    new_groups = {s: {r['_meta']['group_id'] for r in rows} for s, rows in normalized.items()}
    if new_groups['train'] & new_groups['validation']:
        raise ValueError('An inherited new source family crosses partitions')
    inherited_ids = old_ids['train'] | old_ids['validation']
    buckets = {s: defaultdict(list) for s in normalized}
    for split, rows in normalized.items():
        for row in rows:
            if row['_meta']['id'] in inherited_ids:
                raise ValueError('New source identity duplicates inherited replay')
            buckets[split][fingerprint(row)].append(row)
    # Quarantine complete NEW families if they collide with opposite-partition
    # visible inputs. Existing held-outs/replay are never moved or rewritten.
    blocked = set()
    for split, opposite in (('train', 'validation'), ('validation', 'train')):
        collisions = set(buckets[split]) & (old_inputs[opposite] | set(buckets[opposite]))
        for key in collisions:
            blocked.update(r['_meta']['group_id'] for r in buckets[split][key])
    outputs = {}; coverage = {}; token_stats = {}; quarantine = Counter(); all_groups = {}; all_public = {}
    quarantined = (out/'normalization-quarantine-private.jsonl').open('x')
    try:
        for split in ('train', 'validation'):
            admitted = []
            def reject(row, reason):
                quarantine[split+':'+reason] += 1
                quarantined.write(json.dumps(dict(reason=reason, row=row), separators=(',', ':'))+'\n')
            for key, rows in buckets[split].items():
                unblocked = []
                for row in rows:
                    if row['_meta']['group_id'] in blocked:
                        reject(row, 'new_family_cross_partition_public_input_collision')
                    elif key in old_inputs[split]:
                        reject(row, 'already_represented_by_unchanged_replay')
                    else:
                        unblocked.append(row)
                if not unblocked:
                    continue
                if len({action_key(r) for r in unblocked}) != 1:
                    for row in unblocked: reject(row, 'conflicting_hard_choices_in_equivalent_public_input')
                    continue
                admitted.append(unblocked[0])
                for row in unblocked[1:]: reject(row, 'same_split_duplicate_public_input_same_choice')
            tokens = Counter(); categories = Counter(); routes = Counter(); phases = Counter()
            for row in admitted:
                record = materialize(row)
                encoded = DecisionModel.encode(proxy, tokenizer, record,
                    max_state=limits['max_state'], max_branch=limits['max_branch'], strict=True)
                question = record['questions'][0]; rawq = row['questions']['action']
                if (encoded['state_truncated'] or len(encoded['ids']) > limits['max_packed']
                        or question['keys'] != list(rawq['criteria'])
                        or question['keys'][question['label']] != rawq['label']
                        or encoded['labels'] != [question['label']]
                        or len(encoded['opt_idx']) != 1
                        or len(encoded['opt_idx'][0]) != len(question['keys'])):
                    raise ValueError('Actual new packed inputs/actions truncate or lose label alignment')
                tokens['encoded_new_rows'] += 1
                tokens['max_new_packed'] = max(tokens['max_new_packed'], len(encoded['ids']))
                tokens['max_new_state'] = max(tokens['max_new_state'], encoded['seg'].count(0))
                routes[row['_meta']['scenario_route']] += 1; phases[row['_meta']['phase']] += 1
            rows = old_rows[split]+admitted
            target = out/f'{split}-replay-and-yaku-private.jsonl'
            with target.open('x') as stream:
                for row in rows:
                    categories[row['questions']['action']['src']] += 1
                    stream.write(json.dumps(row, separators=(',', ':'))+'\n')
            all_groups[split] = {r['_meta']['group_id'] for r in rows}
            all_public[split] = old_inputs[split] | {fingerprint(r) for r in admitted}
            if set(routes) != ROUTES:
                raise ValueError('A source route is absent after explicit quarantine')
            outputs[split] = dict(path=str(target.relative_to(ROOT)), sha256=sha(target),
                records=len(rows), groups=len(all_groups[split]))
            coverage[split] = dict(replay_records=len(old_rows[split]), new_source_records=raw_counts[split],
                new_admitted_records=len(admitted), new_unique_groups=len({r['_meta']['group_id'] for r in admitted}),
                sources=dict(categories), new_routes=dict(routes), new_phases=dict(phases))
            token_stats[split] = dict(tokens)
        if all_groups['train'] & all_groups['validation'] or all_public['train'] & all_public['validation']:
            raise ValueError('Merged corpus crosses authoritative families/public inputs')
        # Existing native/token/mask audits remain authoritative for old replay.
        # Only the newly normalized records were encoded here; no model loaded.
        if any(sha(p) != h for p, h in inputs.items()):
            raise ValueError('Bound preparation inputs changed')
        result = dict(status='exact_replay_and_normalized_targeted_yaku_corpus_prepared_not_fit_admitted',
            method=c['method'], files=outputs, coverage=coverage, tokens=token_stats,
            quarantine=dict(quarantine), blocked_new_families=len(blocked), source_collections=source_specs,
            original_replay_objects_labels_and_partitions_unchanged=True,
            original_replay_native_encoder_and_soft_mask_audit_reused=True,
            original_quarantine_preserved=replay['original_quarantine_preserved'],
            new_source_routes_are_private_metadata_not_inputs=True,
            actual_new_labels_preserved_and_encoded=True,
            cross_split_family_and_public_input_overlap=0,
            teacher_labels_not_optimality_proof=True, exact_stage3B_mix_and_parent_not_registered=True,
            backbone_loaded=False, optimizer_updates=0, active_stage3A_unchanged=True,
            raw_records_and_quarantine_private=True,
            job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
        write(out/'data-ready-private.json', result)
        write(out/'completion-binding-private.json', dict(files=inputs))
        public = {k: result[k] for k in ('status', 'method', 'coverage', 'tokens', 'quarantine',
            'blocked_new_families', 'original_replay_objects_labels_and_partitions_unchanged',
            'original_replay_native_encoder_and_soft_mask_audit_reused',
            'new_source_routes_are_private_metadata_not_inputs', 'actual_new_labels_preserved_and_encoded',
            'cross_split_family_and_public_input_overlap', 'teacher_labels_not_optimality_proof',
            'exact_stage3B_mix_and_parent_not_registered', 'backbone_loaded', 'optimizer_updates',
            'active_stage3A_unchanged', 'job_id', 'node')}
        write(out/'summary-public.json', public)
        print(json.dumps(dict(status=result['status'], files=outputs, quarantine=dict(quarantine))), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error)))
        raise
    finally:
        quarantined.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); prepare(args.config.resolve())
