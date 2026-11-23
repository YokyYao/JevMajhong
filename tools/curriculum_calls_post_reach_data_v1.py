"""Versioned calls replay plus a distinct, audited post-reach stratum.

This prepares data, not a parent seal or a student fit. All entry-point work
requires a bounded scheduler GPU allocation, including the companion CPU work.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_curriculum_calls_repair import MASS as OLD_MASS
from tools.build_curriculum_stage1 import public_input
from tools.curriculum_course_order import course_order
from tools.curriculum_transfer import read, sha

POST = 'followup:teacher:post_reach_discard'
MASS = {**{k: .98*v for k, v in OLD_MASS.items()}, POST: .02}
READY = 'calls_post_reach13_data_prepared_not_parent_or_fit_admitted'
AUDITED = 'calls_post_reach13_replay_split_encoder_and_sampler_audit_passed'
CONFIG = {
    'calls_data': 'runs/curriculum-calls-repair-data-v1/data',
    'post_reach_data': 'runs/curriculum-post-reach-teacher-data-v2/audit-v1',
    'source_registration': 'runs/curriculum-riichi-class-control-v1/fit-plan/registered.json',
    'out': 'runs/curriculum-calls-post-reach-data-v1/data',
    'audit_out': 'runs/curriculum-calls-post-reach-data-v1/audit-v1',
    'old_relative_loss_mass': .98, 'post_reach_loss_mass': .02,
    'post_reach_stratum': POST, 'epochs': 2, 'order_seed': 2026112209,
    'accumulation_search_min': 192, 'accumulation_search_max': 512,
    'minimum_tail_fraction': .75, 'maximum_example_weight': 16,
    'automatic_parent_admission': False, 'automatic_fit': False,
}
PUBLIC_FILES = (
    'tools/curriculum_calls_post_reach_data_v1.py',
    'tests/core/test_curriculum_calls_post_reach_data_v1.py',
    'configs/training/curriculum-calls-post-reach-data-v1.json',
    'docs/experiments/plans/curriculum-calls-post-reach-data-20261123.md',
    'cluster/curriculum-calls-post-reach-data-v1.slurm',
)


def gpu_only():
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
            or socket.gethostname().startswith(('lo-', 'login'))):
        raise RuntimeError('Bounded scheduler GPU allocation required')


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def fingerprint(row):
    return hashlib.sha256(json.dumps(public_input(row), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def layout(counts):
    if set(counts) != set(MASS) or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError('All thirteen positive integer source counts required')
    total = sum(counts.values())
    weights = {k: MASS[k]*total/counts[k] for k in MASS}
    if max(weights.values()) > CONFIG['maximum_example_weight']:
        raise ValueError('Prospective per-example weight cap exceeded')
    for size in range(CONFIG['accumulation_search_min'], CONFIG['accumulation_search_max']+1):
        updates = math.ceil(total/size)
        tail = total-(updates-1)*size
        if tail >= .75*size and min(counts.values()) >= updates:
            return dict(accumulation=size, updates_per_epoch=updates,
                        example_weights=weights, final_update_records=tail,
                        final_step=2*updates, record_exposures=2*total)
    raise ValueError('Cannot cover all thirteen sources without copies or a tiny tail')


def retag_post(row, split):
    q = row['questions']['action']
    meta = row['_meta']
    if (meta['split'] != split or meta['curriculum_stratum'] != 'post_reach_discard'
            or meta.get('previous_actual_own_action') != 'reach'
            or q['src'] != 'curriculum1:teacher:post_reach_discard'
            or q.get('target') is not None or q['label'] not in q['criteria']
            or not q['criteria'] or {c['type'] for c in q['criteria'].values()} != {'dahai'}):
        raise ValueError('Unchanged actual teacher post-reach hard-target discard required')
    result = copy.deepcopy(row)
    result['questions']['action']['src'] = 'curriculum2:'+POST
    return result


def check_splits(pools, old_pools, reserved):
    ids = {}; groups = {}; inputs = {}; post_inputs = set()
    for split, pool in pools.items():
        ids[split] = {r['_meta']['id'] for r in pool}
        groups[split] = {r['_meta']['group_id'] for r in pool}
        inputs[split] = {fingerprint(r) for r in pool}
        if len(ids[split]) != len(pool):
            raise ValueError('Duplicate decision IDs in merged replay')
        old_inputs = {fingerprint(r) for r in old_pools[split]}
        for row in pool[len(old_pools[split]):]:
            key = fingerprint(row)
            if key in old_inputs or key in post_inputs:
                raise ValueError('New post-reach input duplicates existing or new replay')
            post_inputs.add(key)
    if (ids['train'] & ids['validation'] or groups['train'] & groups['validation']
            or groups['train'] & reserved or inputs['train'] & inputs['validation']):
        raise ValueError('Merged replay leaks held-out identities, families or public inputs')
    return dict(groups={k: len(v) for k, v in groups.items()},
                cross_split_id_overlap=0, cross_split_group_overlap=0,
                cross_split_observed_input_overlap=0, new_duplicate_inputs=0)


def sources(config):
    if read(config) != CONFIG:
        raise ValueError('Prospective thirteen-source data recipe changed')
    files = {str((ROOT/p).resolve()): sha(ROOT/p) for p in PUBLIC_FILES}
    files[str(Path(config).resolve())] = sha(config)
    manifests = [ROOT/CONFIG['calls_data']/'data-ready-private.json',
                 ROOT/CONFIG['post_reach_data']/'summary-private.json']
    old, post = map(read, manifests)
    if (old['status'] != 'calls_repair12_source_data_prepared_not_parent_or_fit_admitted'
            or post['status'] != 'post_reach_teacher_native_normalized_encoder_replay_and_guard_audit_passed'
            or post['checked_complete_source_games'] != 256
            or post['checked_native_decisions'] != 387):
        raise ValueError('Complete unchanged twelve-source and audited post-reach sources required')
    for p in [*manifests, ROOT/CONFIG['source_registration'],
              ROOT/CONFIG['calls_data']/'preparation-binding-private.json',
              ROOT/CONFIG['post_reach_data']/'audit-binding-private.json']:
        files[str(p.resolve())] = sha(p)
        if p.name.endswith('binding-private.json'):
            for name, digest in read(p)['files'].items():
                if name in files and files[name] != digest:
                    raise ValueError('Conflicting source binding')
                files[name] = digest
    pools = []
    for manifest in (old, post):
        current = {}
        for split, spec in manifest['files'].items():
            path = (ROOT/spec['path']).resolve()
            if sha(path) != spec['sha256']:
                raise ValueError('Audited decision file changed')
            files[str(path)] = spec['sha256']
            current[split] = [json.loads(line) for line in path.open()]
            if len(current[split]) != spec['records']:
                raise ValueError('Audited row count differs')
        pools.append(current)
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('Immutable source/audit bindings changed')
    return old, post, *pools, files


def orders(counts):
    categories = [key for key, n in counts.items() for _ in range(n)]
    receipts = []
    rng = random.Random(CONFIG['order_seed'])
    for epoch in range(2):
        size = layout(counts)['accumulation']
        order = course_order(categories, rng, size)
        batches = [order[i:i+size] for i in range(0, len(order), size)]
        if (sorted(order) != list(range(len(categories)))
                or any({categories[i] for i in batch} != set(MASS) for batch in batches)):
            raise ValueError('Full multistratum permutation failed')
        receipts.append(dict(epoch=epoch+1, records=len(order), updates=len(batches),
                             minimum_update_records=min(map(len, batches)), copies_added=0,
                             all_updates_have_all_thirteen_strata=True,
                             order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest()))
    return receipts


def prepare(config):
    gpu_only()
    old, post, old_pools, post_pools, files = sources(config)
    out = ROOT/CONFIG['out']
    out.mkdir(parents=True, exist_ok=False)
    pools = {split: old_pools[split]+[retag_post(r, split) for r in post_pools[split]]
             for split in ('train', 'validation')}
    reserved_file = ROOT/CONFIG['calls_data']/'reserved-public-groups.json'
    files[str(reserved_file)] = sha(reserved_file)
    split_proof = check_splits(pools, old_pools, set(read(reserved_file)))
    counts = {}; outputs = {}
    for split, pool in pools.items():
        counts[split] = dict(Counter(r['questions']['action']['src'].removeprefix('curriculum2:') for r in pool))
        if (set(counts[split]) != set(MASS)
                or counts[split][POST] != {'train': 312, 'validation': 75}[split]
                or {k: v for k, v in counts[split].items() if k != POST} != old['categories'][split]):
            raise ValueError('Original replay or full new cohort not preserved')
        path = out/f'{split}.jsonl'
        with path.open('x') as stream:
            for row in pool:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
        outputs[split] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                              records=len(pool), groups=split_proof['groups'][split])
    proposed = layout(counts['train'])
    result = dict(status=READY, files=outputs, categories=counts, loss_mass=MASS,
                  example_weights=proposed['example_weights'], count_only_accumulation=proposed['accumulation'],
                  proposed_updates_per_epoch=proposed['updates_per_epoch'], proposed_epochs=2,
                  proposed_final_step=proposed['final_step'], proposed_record_exposures=proposed['record_exposures'],
                  original_twelve_source_rows_exact=True, all_basic_replay_rows_and_labels_retained=True,
                  post_reach_loss_tag_separate=True, public_inputs_and_labels_unchanged=True,
                  required_input_schema=None, parent_policy_aware_seal_required=True,
                  additional_independent_teacher_families=0, source_teacher_families_reused=256,
                  source_post_reach_opponent_riichi_coverage=[0], reserved_groups_excluded=len(read(reserved_file)),
                  copies_added=0, student_optimizer_updates=0, parent_model_admitted=False, raw_private=True,
                  **split_proof)
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('Source changed during preparation')
    write(out/'epoch-layout-private.json', dict(layout=proposed, prospective_orders=orders(counts['train'])))
    write(out/'data-ready-private.json', result)
    write(out/'preparation-binding-private.json', dict(files=files, node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(result), flush=True)


def audit(config):
    gpu_only()
    old, post, old_pools, post_pools, files = sources(config)
    source = ROOT/CONFIG['out']; out = ROOT/CONFIG['audit_out']
    out.mkdir(parents=True, exist_ok=False)
    ready = read(source/'data-ready-private.json')
    if ready['status'] != READY or ready['loss_mass'] != MASS:
        raise ValueError('Prepared thirteen-source contract differs')
    for p, h in read(source/'preparation-binding-private.json')['files'].items():
        if sha(p) != h:
            raise ValueError('Preparation binding changed')
        files[p] = h
    pools = {}; counts = {}
    for split, spec in ready['files'].items():
        path = ROOT/spec['path']
        if sha(path) != spec['sha256']:
            raise ValueError('Prepared output changed')
        files[str(path.resolve())] = spec['sha256']
        pool = [json.loads(line) for line in path.open()]
        n = len(old_pools[split])
        if pool[:n] != old_pools[split] or len(pool) != n+len(post_pools[split]):
            raise ValueError('Original replay is not exact or new cohort incomplete')
        for original, actual in zip(post_pools[split], pool[n:]):
            restored = copy.deepcopy(actual)
            if restored['questions']['action']['src'] != 'curriculum2:'+POST:
                raise ValueError('Distinct post-reach tag lost')
            restored['questions']['action']['src'] = original['questions']['action']['src']
            if restored != original or public_input(actual) != public_input(original):
                raise ValueError('Private retag changed a source decision')
        pools[split] = pool
        counts[split] = dict(Counter(r['questions']['action']['src'].removeprefix('curriculum2:') for r in pool))
    reserved = ROOT/CONFIG['calls_data']/'reserved-public-groups.json'
    split_proof = check_splits(pools, old_pools, set(read(reserved)))
    proposed = layout(counts['train'])
    if (counts != ready['categories'] or proposed['example_weights'] != ready['example_weights']
            or ready['count_only_accumulation'] != proposed['accumulation']
            or read(source/'epoch-layout-private.json') != dict(layout=proposed, prospective_orders=orders(counts['train']))):
        raise ValueError('Count-only cadence, weights or exact two-epoch permutations differ')
    reg = read(ROOT/CONFIG['source_registration'])
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from types import SimpleNamespace
    from jevany.data import materialize
    from jevany.model import DecisionModel, load_preprocessor
    tokenizer = load_preprocessor(reg['training_config']['base_load_path'], backbone_adapter='text')
    proxy = SimpleNamespace(decision_mode='pointer', option_isolation=False)
    packed = {}; checked = 0
    for split, pool in post_pools.items():
        maximum = 0
        for original in pool:
            actual = retag_post(original, split)
            a = materialize(original); b = materialize(actual)
            stripped = copy.deepcopy(b)
            stripped['questions'][0]['src'] = a['questions'][0]['src']
            if stripped != a:
                raise ValueError('Materialized change extends beyond private source tag')
            ea = DecisionModel.encode(proxy, tokenizer, a, max_state=4096, max_branch=8192, strict=True)
            eb = DecisionModel.encode(proxy, tokenizer, b, max_state=4096, max_branch=8192, strict=True)
            if ea != eb or ea['state_truncated'] or len(ea['ids']) > 8192:
                raise ValueError('Post-reach retag changed or truncated actual packed input')
            checked += 1; maximum = max(maximum, len(ea['ids']))
        packed[split] = maximum
    for name in ('data-ready-private.json', 'epoch-layout-private.json', 'preparation-binding-private.json'):
        path = source/name; files[str(path.resolve())] = sha(path)
    files[str(reserved.resolve())] = sha(reserved)
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('Bound input changed during independent data audit')
    result = dict(status=AUDITED, files=ready['files'], categories=counts, loss_mass=MASS,
                  layout=proposed, packed_post_reach_rows_checked=checked, max_post_reach_packed=packed,
                  inherited_native_post_reach_audit_windows=387, new_native_replays=0,
                  original_replay_rows_checked={k: len(v) for k, v in old_pools.items()},
                  original_inputs_labels_metadata_splits_and_order_exact=True,
                  post_reach_only_private_source_tag_changed=True, complete_epoch_orders_checked=2,
                  reserved_groups_excluded=len(read(reserved)), additional_independent_families=0,
                  source_post_reach_opponent_riichi_coverage=[0], **split_proof,
                  student_optimizer_updates=0, parent_model_admitted=False,
                  later_course_fit_admitted=False, raw_private=True)
    write(out/'summary-private.json', result)
    write(out/'audit-binding-private.json', dict(files=files, node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'audit'), required=True)
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else audit)(args.config.resolve())
