"""Freeze complete augmented replay + existing six yaku sources, with no fit."""
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha
from tools.prepare_curriculum_yaku_data_v2 import fingerprint
from tools.curriculum_yaku_parent_loss_v1 import MASS, NEW_SOURCES
from tools.curriculum_yaku_ri_coverage_v1 import build, RI
from tools.curriculum_course_order import course_order

CONFIG = ROOT/'configs/training/curriculum-yaku-augmented-data-v1.json'
PUBLIC = ('tools/prepare_curriculum_yaku_augmented_data_v1.py',
    'tools/curriculum_yaku_ri_coverage_v1.py',
    'tests/core/test_curriculum_yaku_ri_coverage_v1.py',
    'configs/training/curriculum-yaku-augmented-data-v1.json',
    'docs/experiments/plans/curriculum-yaku-augmented-data-20261125.md')


def main():
    allocation = resource(); started = time.monotonic(); config = read(CONFIG)
    out = ROOT/config['out']
    if (out.exists() or config['method'] != 'exact_augmented19_replay_plus_existing_normalized_six_yaku_routes_v1'
            or config['student_fit_automatic'] or config['parent_admitted']
            or config['final_confirmation_opened'] or config['epochs'] != 2
            or config['source_mass'] != 'unchanged_yaku25_parent19_times0.8_plus_six_routes_each0.2_div6'
            or config['accumulation'] != 73 or config['max_repeat'] != 8
            or config['max_example_weight'] != 16 or len(config['exposure_seeds']) != 2):
        raise ValueError('Only a distinct data-only full augmented25-source recipe is eligible')
    files = {}
    for key in ('augmented19_binding', 'original_yaku_audit_binding'):
        path = ROOT/config[key]; binding = read(path)
        for name, expected in binding['files'].items():
            if name in files and files[name] != expected:
                raise ValueError('Conflicting actual original source bindings')
            files[name] = expected
        files[str(path)] = sha(path)
    if any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('Actual original augmented/yaku source evidence changed')
    parent_root = (ROOT/config['augmented19_binding']).parent
    parent = read(parent_root/'summary-private.json')
    ready_path = ROOT/config['original_yaku_ready']; ready = read(ready_path)
    if (parent['status'] != 'audited_augmented19_union_and_full_coverage_replay_frozen_not_fit_registered'
            or ready['status'] != 'exact_replay_and_normalized_targeted_yaku_corpus_prepared_not_fit_admitted'
            or not ready['original_replay_objects_labels_and_partitions_unchanged']
            or not ready['actual_new_labels_preserved_and_encoded']
            or ready['cross_split_family_and_public_input_overlap'] != 0):
        raise ValueError('Complete original audited data-only sources required')
    out.mkdir(parents=True)
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/core',
        '-p', 'test_curriculum_yaku_ri_coverage_v1.py', '-v'], cwd=ROOT,
        capture_output=True, text=True, timeout=60)
    (out/'tests.log').write_text(result.stdout+result.stderr)
    write(out/'tests-exit-private.json', dict(returncode=result.returncode, child_reaped=True))
    if result.returncode:
        raise ValueError('Actual25-source coverage/mass/rejection tests failed')
    parent_cohorts = read(parent_root/'cohort-ids-private.json')
    pools = {}; cohorts = {}; output_files = {}; source_counts = {}
    for split in ('train', 'validation'):
        parent_file = parent_root/f'{split}-union-private.jsonl'
        spec = ready['files'][split]; old_file = ROOT/spec['path']
        if sha(parent_file) != files[str(parent_file)] or sha(old_file) != spec['sha256']:
            raise ValueError('Exact original corpora required')
        old_lines = old_file.read_bytes().splitlines(keepends=True)
        parent_lines = parent_file.read_bytes().splitlines(keepends=True)
        n = config['expected_old19'][split]
        if (len(parent_lines) != config['expected_parent'][split]
                or len(old_lines) != config['expected_original_yaku'][split]
                or old_lines[:n] != parent_lines[:n]):
            raise ValueError('Full old19 prefix must match in both actual source corpora')
        route_lines = old_lines[n:]
        if len(route_lines) != config['expected_routes'][split]:
            raise ValueError('Every original normalized route row must survive')
        rows = [json.loads(line) for line in parent_lines]
        parent_ids = {row['_meta']['id'] for row in rows}
        parent_groups = {row['_meta']['group_id'] for row in rows}
        parent_inputs = {fingerprint(row) for row in rows}
        routes = [json.loads(line) for line in route_lines]
        if (any(row['questions']['action']['src'] not in NEW_SOURCES for row in routes)
                or parent_ids & {row['_meta']['id'] for row in routes}
                or parent_groups & {row['_meta']['group_id'] for row in routes}
                or parent_inputs & {fingerprint(row) for row in routes}):
            raise ValueError('Added route identities/groups/inputs must be distinct from the retained parent')
        rows.extend(routes)
        ids = [row['_meta']['id'] for row in rows]
        if (len(rows) != config['expected_union'][split] or len(set(ids)) != len(rows)
                or {row['questions']['action']['src'] for row in rows} != set(MASS)):
            raise ValueError('Complete unique actual25-source corpus required')
        destination = out/f'{split}-union-private.jsonl'
        with destination.open('xb') as handle:
            for line in (*parent_lines, *route_lines):
                if not line.endswith(b'\n'):
                    raise ValueError('Preserve every original complete JSONL line byte')
                handle.write(line)
        old_ids = parent_cohorts[split]['original']; basic_ids = parent_cohorts[split]['supplement']
        if (set(old_ids) | set(basic_ids) != parent_ids or set(old_ids) & set(basic_ids)
                or len(old_ids) != n):
            raise ValueError('Actual original/new basic cohorts must partition the complete parent')
        cohorts[split] = dict(original19=old_ids, basic_supplement=basic_ids,
            yaku_routes=[row['_meta']['id'] for row in routes])
        pools[split] = rows
        source_counts[split] = dict(Counter(row['questions']['action']['src'] for row in rows))
        output_files[split] = dict(path=str(destination.relative_to(ROOT)), sha256=sha(destination),
            records=len(rows), groups=len({row['_meta']['group_id'] for row in rows}))
        files[str(destination)] = sha(destination)
        print(dict(stage='actual_augmented_yaku_union', split=split, records=len(rows),
            original_line_bytes_preserved=True), flush=True)
    for kind in ('ids', 'groups', 'inputs'):
        def values(rows):
            return {row['_meta']['id'] if kind == 'ids' else row['_meta']['group_id']
                if kind == 'groups' else fingerprint(row) for row in rows}
        if values(pools['train']) & values(pools['validation']):
            raise ValueError('Cross-split complete union overlap: '+kind)
    # Also prevent fresh basic and yaku families from crossing their opposite
    # retained partitions; the complete union checks above cover those edges.
    categories = [row['questions']['action']['src'] for row in pools['train']]
    rng = random.Random(config['baseline_seed']); plans = []
    for epoch, seed in enumerate(config['exposure_seeds'], 1):
        baseline = course_order(categories, rng, 511, min_tail_fraction=0.)
        plan = build(categories, baseline, seed, batch=config['accumulation'],
            max_repeat=config['max_repeat'], max_weight=config['max_example_weight'])
        plan['summary']['epoch'] = epoch; plans.append(plan)
        path = out/f'epoch-{epoch:02d}-private.json'; write(path, plan); files[str(path)] = sha(path)
    if sum(source_counts['train'][source] for source in RI) != 4244:
        raise ValueError('All original and supplemental RI TRAIN opportunities required')
    cohort_path = out/'cohort-ids-private.json'; write(cohort_path, cohorts); files[str(cohort_path)] = sha(cohort_path)
    for path in [*[ROOT/name for name in PUBLIC], ready_path,
            parent_root/'summary-private.json', parent_root/'cohort-ids-private.json',
            ROOT/'tools/curriculum_ri_coverage_replay_v2.py',
            ROOT/'tools/curriculum_yaku_parent_loss_v1.py', ROOT/'tools/prepare_curriculum_yaku_data_v2.py']:
        files[str(path)] = sha(path)
    if any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('Actual source inputs changed during preparation')
    summary = dict(status='complete_augmented25_yaku_corpus_and_RI_coverage_plans_frozen_not_fit_admitted',
        files=output_files, source_counts=source_counts, source_masses=MASS,
        original_parent_and_route_line_bytes_preserved=True,
        all_existing_labels_masks_and_partitions_unchanged=True,
        complete_cross_split_ID_group_visible_input_overlap=0,
        separate_original_basic_supplement_and_route_cohorts=True,
        epochs=[plan['summary'] for plan in plans],
        optimizer_steps_if_later_registered=sum(plan['summary']['optimizer_updates'] for plan in plans),
        exposures_if_later_registered=sum(len(plan['indices']) for plan in plans),
        RI_TRAIN_records=4244, coverage_tests_passed=4,
        existing_source_native_encoder_audits_bound=True,
        new_combined_student_token_admission_performed=False,
        numeric_objective_parent_verifier_full_epoch_adapter_and_fit_still_pending=True,
        actual_parent_admitted=False, student_fit_admitted=False,
        current2472_or2596_inputs_changed=False, model_loaded=False,
        student_updates=0, new_games=0, final_confirmation_opened=False,
        seconds=time.monotonic()-started, allocation=allocation)
    write(out/'data-ready-private.json', summary)
    for path in [out/'data-ready-private.json', out/'tests.log', out/'tests-exit-private.json']:
        files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print({key:value for key,value in summary.items()
        if key not in ('allocation', 'files', 'source_counts', 'source_masses', 'epochs')}, flush=True)
    print(dict(epoch_summaries=summary['epochs']), flush=True)


if __name__ == '__main__':
    main()
