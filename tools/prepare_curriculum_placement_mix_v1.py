"""Prepare a count-only virtual-exposure proposal after all labels exist."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_placement_mix_v1 import layout, masses
from tools.curriculum_transfer import read, sha

READY = 'placement_all_sources_including_targeted_win_special_prepared_not_fit'


def prepare():
    gpu_only()
    parent_path = ROOT/'runs/curriculum-placement-data-v3/data-v1/data-ready-private.json'
    binding_path = parent_path.parent/'completion-binding-private.json'
    parent = read(parent_path)
    if parent['status'] != READY or parent['optimizer_updates'] != 0 or parent['source_group_overlap'] != 0:
        raise ValueError('Actual completed coherent data3 required; no candidate is admitted here')
    files = dict(read(binding_path)['files'])
    for path in (parent_path, binding_path, Path(__file__),
                 ROOT/'tools/curriculum_placement_mix_v1.py',
                 ROOT/'tools/curriculum_yaku_objective_v1.py',
                 ROOT/'tools/curriculum_progression_objective_v1.py',
                 ROOT/'tools/curriculum_calls_post_reach_data_v1.py',
                 ROOT/'tools/build_curriculum_calls_repair.py',
                 ROOT/'tools/curriculum_basic_repair.py',
                 ROOT/'tools/curriculum_transfer.py',
                 ROOT/'tests/core/test_curriculum_placement_mix_v1.py',
                 ROOT/'docs/experiments/plans/curriculum-placement-mix-20261123.md',
                 ROOT/'cluster/curriculum-placement-mix-v1.slurm'):
        digest = sha(path)
        if str(path.resolve()) in files and files[str(path.resolve())] != digest:
            raise ValueError('Conflicting immutable mix prerequisites')
        files[str(path.resolve())] = digest
    ids = set(); all_groups = {}; catalogs = {}; families = {}
    expected_sources = set(masses())
    for split in ('train', 'validation'):
        spec = parent['files'][split]; path = ROOT/spec['path']
        if sha(path) != spec['sha256']:
            raise ValueError('Completed data3 records changed')
        files[str(path.resolve())] = spec['sha256']
        counts = Counter(); source_families = defaultdict(set)
        for line in path.open():
            row = json.loads(line); meta = row['_meta']; source = row['questions']['action']['src']
            if (set(row['questions']) != {'action'} or source not in expected_sources
                    or not isinstance(meta['id'], str) or meta['id'] in ids
                    or not isinstance(meta['group_id'], str) or not meta['group_id']):
                raise ValueError('Unique original IDs and one known question source required')
            ids.add(meta['id']); counts[source] += 1; source_families[source].add(meta['group_id'])
        family_counts = {s: len(g) for s, g in source_families.items()}
        all_groups[split] = set().union(*source_families.values())
        if (set(counts) != expected_sources or sum(counts.values()) != spec['records']
                or dict(counts) != parent['source_records'][split]
                or family_counts != parent['source_distinct_families'][split]):
            raise ValueError('Actual full source/family catalogs differ from completed data3')
        catalogs[split] = dict(counts); families[split] = family_counts
    if all_groups['train'] & all_groups['validation']:
        raise ValueError('Source family crossed the original split')
    plan = layout(catalogs['train'], families['train'])
    if any(sha(path) != digest for path, digest in files.items()):
        raise ValueError('An immutable prerequisite changed')
    out = ROOT/'runs/curriculum-placement-mix-v1/proposal-v1'
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='count_only_placement_mix_proposed_not_parent_or_fit_admitted', **plan,
        unique_validation_records=sum(catalogs['validation'].values()),
        unique_validation_records_by_source=catalogs['validation'],
        distinct_validation_families_by_source=families['validation'],
        global_distinct_families={s: len(g) for s, g in all_groups.items()},
        validation_exposures_per_pass=sum(catalogs['validation'].values()),
        source_family_overlap=0, source_rows_filtered_by_values_or_confidence=False,
        existing_data_files_copied_or_edited=False, optimizer_updates=0,
        final_confirmation_grid_opened=False, no_fourth_escape_veto=True,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'proposal-private.json', result)
    write(out/'completion-binding-private.json', dict(files=files))
    write(out/'summary-public.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    prepare()

