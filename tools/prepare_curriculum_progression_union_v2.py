"""Exact union v2: historical source-file splits need no invented row field."""
import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.audit_curriculum_basic_teacher import fingerprint

PUBLIC_FILES = (
    'tools/prepare_curriculum_progression_union_v2.py',
    'tests/core/test_curriculum_progression_union_v2.py',
    'configs/diagnostics/curriculum-progression-union-v2.json',
    'cluster/curriculum-progression-union-v2.slurm',
    'docs/experiments/plans/curriculum-progression-union-v2-20261123.md',
)


def mc_support(row):
    q = row['questions']['action']; target = q.get('target')
    if (row['_meta']['label_type'] != 'paired_future_masked' or not isinstance(target, dict)
            or not target or not set(target) <= set(q['criteria'])
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in target.values())
            or abs(sum(target.values())-1) > 1e-6):
        raise ValueError('Unchanged normalized masked future target required')
    return sum(v > 0 for v in target.values())


def restore_source_tag(row, original):
    restored = copy.deepcopy(row)
    restored['questions']['action']['src'] = original['questions']['action']['src']
    if restored != original: raise ValueError('Prepared MC differs beyond its private loss-only tag')
    return restored


def require_partition(meta, split):
    if split not in ('train', 'validation'):
        raise ValueError('Bound source-file TRAIN/validation partition required')
    if meta.get('split', split) != split or ('source_logical_seed' in meta and meta.get('split') != split):
        raise ValueError('Per-record split contradicts source file or teacher family')
    return 'split' not in meta


def prepare(out):
    gpu_only(); out.mkdir(parents=True, exist_ok=False)
    files = {str(ROOT/p): sha(ROOT/p) for p in PUBLIC_FILES}; config = read(ROOT/PUBLIC_FILES[2])
    active_path = ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json'
    active = read(active_path)['files']; files[str(active_path)] = sha(active_path)
    def pin(path, expected=None):
        path = (ROOT/path).resolve(); actual = sha(path)
        if expected is not None and actual != expected: raise ValueError('Immutable input changed: '+str(path))
        if str(path) in files and files[str(path)] != actual: raise ValueError('Conflicting file identity')
        files[str(path)] = actual
        return path
    def load(path): return read(pin(path))
    def rows(spec):
        path = pin(spec['path'], spec['sha256']); result = [json.loads(line) for line in path.open()]
        if len(result) != spec.get('records', spec.get('rows')): raise ValueError('Source row count differs')
        return result
    def bind(path):
        for p, h in load(path)['files'].items(): pin(p, h)
    old = load(config['calls_manifest']); risk = load(config['risk_manifest']); ordinary = load(config['ordinary_manifest'])
    if (old['status'] != 'calls_post_reach13_replay_split_encoder_and_sampler_audit_passed'
            or risk['status'] != 'risk_teacher_native_normalized_encoder_replay_and_guard_audit_passed'
            or ordinary['status'] != 'ordinary_completion_native_event_label_encoder_split_audit_passed_not_fit_admitted'):
        raise ValueError('Complete independently audited source cohorts required')
    bind(Path(config['calls_manifest']).parent/'audit-binding-private.json')
    bind(Path(config['risk_manifest']).parent/'audit-binding-private.json')
    bind(Path(config['ordinary_manifest']).parent/'completion-binding-private.json')
    clean = load(config['clean_manifest']); reserved = set(load(config['reserved_groups']))
    prepared = load(config['legacy_mc_manifest']); provenance = load(config['legacy_mc_world_audit'])
    legacy_ready_path = pin(prepared['source_data_ready'], prepared['source_data_ready_sha256']); legacy_ready = read(legacy_ready_path)
    if (prepared['status'] != 'stage3_attack_data_prepared_pending_parent_and_fit_registration'
            or provenance['status'] != 'all_current_train_mc_targets_reproduced_from_paired_worlds'
            or provenance['records'] != 2722 or provenance['maximum_target_reproduction_error'] >= 1e-10
            or provenance['data_ready_sha256'] != sha(legacy_ready_path)
            or provenance['labels_changed'] or provenance['validation_rollout_payloads_opened']):
        raise ValueError('Exact prior TRAIN conditional-world provenance required')
    pin(prepared['source_world_audit']['path'], prepared['source_world_audit']['sha256'])
    # Recheck the prior hash-bound TRAIN evidence, without opening held-out
    # rollout payloads or rerunning/reinterpreting their labels.
    for p, h in provenance['inputs'].items(): pin(p, h)
    cohorts = {}; quarantines = {}; mc_counts = {}; original_validation_reference = clean['stages']['stage3']['validation']
    for split in ('train', 'validation'):
        earlier = {r['_meta']['id']: r for r in rows(clean['stages']['stage2'][split])}
        stage3 = rows(clean['stages']['stage3'][split]); full = {r['_meta']['id']: r for r in stage3}
        if len(full) != len(stage3) or any(full.get(k) != v for k, v in earlier.items()):
            raise ValueError('Old stage3 is not the exact original replay superset')
        extras = {k: r for k, r in full.items() if k not in earlier}
        kept = {k: r for k, r in extras.items() if mc_support(r) >= 2}
        quarantine = [r for r in extras.values() if mc_support(r) == 1]
        if len(extras) != len(kept)+len(quarantine): raise ValueError('An MC target has empty support')
        prepared_mc = [r for r in rows(prepared['files'][split]) if r['_meta']['label_type'] == 'paired_future_masked']
        if {r['_meta']['id'] for r in prepared_mc} != set(kept) or len(prepared_mc) != len(kept):
            raise ValueError('Prepared MC is not exactly all informative original targets')
        for row in prepared_mc:
            original = kept[row['_meta']['id']]; restore_source_tag(row, original)
            if mc_support(row) < 2 or row['questions']['action']['src'] != 'curriculum3:mc:'+row['_meta']['phase']:
                raise ValueError('MC support or loss-only source identity changed')
        old_quarantine = rows(legacy_ready['files'][split+'_quarantined'])
        if {r['_meta']['id']: r for r in old_quarantine} != {r['_meta']['id']: r for r in quarantine}:
            raise ValueError('Original structural quarantine changed')
        cohorts[split] = dict(calls_exact_replay=rows(old['files'][split]), risk_teacher=rows(risk['files'][split]),
            ordinary_completion=rows(ordinary['files'][split]), legacy_conditional_mc=prepared_mc)
        quarantines[split] = quarantine
        mc_counts[split] = dict(informative=len(kept), structural_zero_gradient=len(quarantine),
            phases=dict(Counter(r['_meta']['phase'] for r in prepared_mc)))
    ids = {s: set() for s in cohorts}; groups = {s: set() for s in cohorts}; inputs = {s: set() for s in cohorts}
    duplicates = {}; totals = {}; outputs = {}; source_tags = {}
    for split, sources in cohorts.items():
        totals[split] = {key: len(value) for key, value in sources.items()}; duplicate_count = 0; tag_count = Counter(); file_split_rows = 0
        dest = out/f'{split}-exact-union-private.jsonl'
        with dest.open('x') as stream:
            for cohort, pool in sources.items():
                for row in pool:
                    meta = row['_meta']; q = row['questions']['action']
                    file_split_rows += require_partition(meta, split)
                    if meta['id'] in ids[split]: raise ValueError('Duplicate decision ID')
                    if q.get('target') is None and q.get('label') not in q['criteria']: raise ValueError('Hard target lost')
                    ids[split].add(meta['id']); groups[split].add(meta['group_id']); key = fingerprint(row)
                    duplicate_count += key in inputs[split]; inputs[split].add(key); tag_count[q['src']] += 1
                    # Exact row objects, including old labels/masks/metadata/tags;
                    # no synthetic loss weights or target edits are inserted.
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
        if len(ids[split]) != config['expected_rows'][split]: raise ValueError('Registered full union count differs')
        duplicates[split] = duplicate_count; source_tags[split] = dict(tag_count)
        totals[split]['rows_whose_partition_is_the_bound_source_file'] = file_split_rows
        outputs[split] = dict(path=str(dest.relative_to(ROOT)), sha256=sha(dest), records=len(ids[split]), groups=len(groups[split]))
        qpath = out/f'{split}-original-single-support-quarantine-private.jsonl'
        with qpath.open('x') as stream:
            for row in quarantines[split]: stream.write(json.dumps(row, separators=(',', ':'))+'\n')
        outputs[split+'_quarantine'] = dict(path=str(qpath.relative_to(ROOT)), sha256=sha(qpath), records=len(quarantines[split]))
    if (ids['train'] & ids['validation'] or groups['train'] & groups['validation']
            or groups['train'] & reserved or inputs['train'] & inputs['validation'] or len(reserved) != 675):
        raise ValueError('Merged data leaks reserved or opposite-split identity/input/family')
    for split, pool in quarantines.items():
        if (any(r['_meta']['id'] in ids[split] for r in pool)
                or len(pool) != config['expected_structural_quarantine'][split]):
            raise ValueError('Structural zero-gradient rows entered the active union')
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Source or active fit changed')
    write(out/'preparation-binding-private.json', dict(files=files, active_fit_files=active))
    result = dict(status='progression_union_v2_prepared_not_independently_audited_or_fit_admitted',
        files=outputs, cohorts=totals, source_tags=source_tags, legacy_mc=mc_counts,
        same_split_duplicate_visible_inputs_retained_and_reported=duplicates,
        cross_split_ids_groups_and_public_inputs=0, reserved_groups_excluded=675,
        original_full_legacy_validation_reference=original_validation_reference,
        all_source_row_objects_and_target_masks_unchanged=True, copies_added=0,
        new_teacher_source_games=0, source_teacher_families_reused=256,
        prior_train_mc_conditional_world_evidence_hash_rechecked=True,
        validation_world_payloads_opened=False, new_worlds_or_model_inference=False,
        loss_mix_and_update_budget_registered=False, parent_admitted=False, student_updates=0,
        active_fit_dependencies_unchanged=len(active), raw_data_private=True,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'data-ready-private.json', result); print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'source_tags')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    prepare(p.parse_args().out.resolve())
