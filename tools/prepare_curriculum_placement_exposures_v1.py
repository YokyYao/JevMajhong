"""Materialize actual placement exposures/orders after count-only proposal.

This prepares a corpus and a count-derived recipe, not a student parent seal
or fit registration. All entry-point computation requires scheduled GPU
resources, including the companion CPU work.
"""
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.curriculum_placement_mix_v1 import layout
from tools.curriculum_placement_partition_v1 import membership, require_partition, expand_train
from tools.curriculum_placement_exposures_v1 import REPEAT_FIELD, choose_accumulation, epoch_order


def proposed_schedule(total):
    accumulation = choose_accumulation(total)
    updates = math.ceil(total/accumulation)
    final = 2*updates
    interval = 100*math.ceil(final/400)
    points = [0, interval, 2*interval, 3*interval, final]
    if points != sorted(set(points)) or points[-2] >= final:
        raise ValueError('Five distinct count-derived curve points required')
    return dict(epochs=2, accumulation=accumulation, batch=1,
        updates_per_epoch=updates, proposed_final_step=final,
        expected_record_exposures=2*total, final_update_records=total-(updates-1)*accumulation,
        checkpoint_every_steps=100, eval_every_steps=interval,
        validation_and_game_steps=points, registered_training_budget=False,
        regular_evaluations_follow_complete_periodic_save=True,
        final_evaluation_requires_complete_final_save=True)


def prepare():
    gpu_only()
    config_path = ROOT/'configs/training/curriculum-placement-exposures-v1.json'
    c = read(config_path)
    if (c['status'] != 'prospective_exposure_preparation_not_fit_registration'
            or c['epochs'] != 2 or c['order_seed'] != 2026112306
            or c['student_fit'] or c['automatic_release']):
        raise ValueError('Fixed prospective exposure preparation recipe required')
    parent_path = ROOT/c['data_ready']; parent = read(parent_path)
    proposal_path = ROOT/c['mix_proposal']; proposal = read(proposal_path)
    if (parent['status'] != 'placement_all_sources_including_targeted_win_special_prepared_not_fit'
            or proposal['status'] != 'count_only_placement_mix_proposed_not_parent_or_fit_admitted'
            or parent['optimizer_updates'] != 0 or proposal['optimizer_updates'] != 0
            or proposal['student_fit_admitted']):
        raise ValueError('Completed coherent data3 and actual-count proposal required')
    files = {}
    for path in (parent_path, proposal_path):
        binding = path.parent/'completion-binding-private.json'
        for name, expected in read(binding)['files'].items():
            if name in files and files[name] != expected:
                raise ValueError('Conflicting immutable preparation ancestry')
            files[name] = expected
        files[str(path.resolve())] = sha(path); files[str(binding.resolve())] = sha(binding)
    source_path = ROOT/c['source_registration']; source = read(source_path)
    sys.path.insert(0, source['training_config']['upstream_source'])
    from jevany.resume import dataset_fingerprint
    for path in (Path(__file__), config_path, source_path,
                 ROOT/'tools/curriculum_placement_partition_v1.py',
                 ROOT/'tools/curriculum_placement_exposures_v1.py',
                 ROOT/'tools/curriculum_placement_metrics_v1.py',
                 ROOT/'tests/core/test_curriculum_placement_partition_v1.py',
                 ROOT/'tests/core/test_curriculum_placement_exposure_preparation_v1.py',
                 ROOT/'cluster/curriculum-placement-exposures-v1.slurm',
                 ROOT/'docs/experiments/plans/curriculum-placement-exposures-20261123.md'):
        name = str(path.resolve()); expected = sha(path)
        if name in files and files[name] != expected:
            raise ValueError('Prepared runtime source differs from data/proposal binding')
        files[name] = expected
    if any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('A completed prerequisite changed')
    pools = {}
    for split in ('train', 'validation'):
        spec = parent['files'][split]; path = ROOT/spec['path']
        if sha(path) != spec['sha256']:
            raise ValueError('Original placement data changed')
        pools[split] = [json.loads(line) for line in path.open()]
        if len(pools[split]) != spec['records']:
            raise ValueError('Complete original current partition required')
        files[str(path.resolve())] = spec['sha256']
    manifest = membership(pools)
    counts = dict(Counter(r['questions']['action']['src'] for r in pools['train']))
    families = {}
    for row in pools['train']:
        families.setdefault(row['questions']['action']['src'], set()).add(row['_meta']['group_id'])
    plan = layout(counts, {s: len(v) for s, v in families.items()})
    if any(proposal.get(key) != value for key, value in plan.items()):
        raise ValueError('Actual exposure layout differs from the completed count proposal')
    expanded = expand_train(pools['train'], plan, manifest)
    originals = {r['_meta']['id']: r for r in pools['train']}
    if not all(a is b for a, b in zip(pools['train'], expanded)):
        raise ValueError('Original TRAIN object prefix changed')
    for replica in expanded[len(pools['train']):]:
        trace = replica['_meta'][REPEAT_FIELD]; origin = originals[trace['original_id']]
        restored = dict(replica, _meta={k: v for k, v in replica['_meta'].items() if k != REPEAT_FIELD})
        restored['_meta']['id'] = trace['original_id']
        if restored != origin:
            raise ValueError('An exposure changed original input/supervision/source metadata')
    schedule = proposed_schedule(len(expanded))
    rng = random.Random(c['order_seed'])
    categories = [r['questions']['action']['src'] for r in expanded]
    receipts = []
    for epoch in range(1, 3):
        _, receipt = epoch_order(categories, rng, schedule['accumulation'])
        if receipt['optimizer_updates'] != schedule['updates_per_epoch']:
            raise ValueError('Prospective actual order and count-derived budget disagree')
        receipts.append(dict(epoch=epoch, **receipt))
    for split in pools:
        require_partition(pools[split], manifest, split)
    if any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('Immutable preparation inputs changed during expansion')
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    path = out/'train-explicit-exposures-private.jsonl'
    with path.open('x') as stream:
        for row in expanded:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    write(out/'current-membership-private.json', manifest)
    write(out/'expected-orders-private.json', dict(seed=c['order_seed'], epochs=receipts))
    outputs = dict(train=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), records=len(expanded)),
        validation=parent['files']['validation'])
    result = dict(status='placement_expanded_corpus_and_two_epoch_recipe_prepared_not_parent_or_fit_registered',
        files=outputs, unique_train_records=len(pools['train']), unique_validation_records=len(pools['validation']),
        training_exposures_per_pass=len(expanded), extra_exposures=len(expanded)-len(pools['train']),
        source_exposures=plan['virtual_exposures_by_source'],
        source_unique_records=plan['unique_train_records_by_source'],
        source_distinct_train_families=plan['distinct_train_families_by_source'],
        global_families={s: len(manifest['partitions'][s]['groups']) for s in pools},
        scalar_example_weights=plan['scalar_example_weights'], source_loss_mass=plan['proposed_loss_mass'],
        proposed_schedule=schedule, order_seed=c['order_seed'],
        expected_order_sha256=[r['order_sha256'] for r in receipts],
        actual_expanded_dataset_fingerprint=dataset_fingerprint(expanded),
        current_membership_file=str((out/'current-membership-private.json').relative_to(ROOT)),
        originals_preserved_object_exact=True, replicas_change_only_ID_and_explicit_exposure_trace=True,
        inherited_split_tags_preserved=True, validation_file_reused_unchanged=True,
        repeated_exposures_add_independent_samples=False, id_overlap=0, group_overlap=0,
        qualified_parent_not_selected=True, student_fit_registered=False, optimizer_updates=0,
        final_confirmation_grid_opened=False, raw_private=True,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'data-ready-private.json', result)
    write(out/'completion-binding-private.json', dict(files=files))
    public = {k: v for k, v in result.items() if k not in
        ('files', 'actual_expanded_dataset_fingerprint', 'current_membership_file')}
    write(out/'summary-public.json', public); print(json.dumps(public), flush=True)


if __name__ == '__main__':
    prepare()
