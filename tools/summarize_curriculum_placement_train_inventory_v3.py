"""Count the complete ended-writer TRAIN population, without admitting a fit.

Reuse completed base/source proofs. This publishes only per-source aggregate
counts and a prospective budget; full original VAL collection remains required.
All entry-point computation must run in a finite scheduler GPU allocation.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource, terminal_record, fields
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_placement_mix_v1 import layout, masses, NEW_SOURCES, REPLAY_MASS
from tools.prepare_curriculum_placement_exposures_v1 import proposed_schedule

OUT = ROOT/'runs/curriculum-placement-ended-train-inventory-v3'


def capability_group(source):
    if source.startswith('curriculum2:basic:'):
        return 'basic_discard_and_riichi'
    if source.startswith('curriculum2:followup:teacher:post_reach_discard'):
        return 'post_riichi_discard'
    if source.startswith('curriculum2:'):
        return 'calls_and_post_call_play'
    if source.startswith('curriculum3:mc:'):
        return 'paired_world_efficiency_and_defense'
    if source.startswith('curriculum3:teacher:'):
        return 'completion_and_riichi_defense'
    if source.startswith('curriculum3B:teacher:'):
        return 'yaku_routes_and_random_control'
    if source.startswith('curriculum4:rank_win:targeted:'):
        return 'placement_complete_win_riichi_and_kan_menus'
    if source.startswith('curriculum4:rank_win:control:'):
        return 'placement_win_accept_decline_controls'
    if source.startswith('curriculum4:rank_win:'):
        return 'placement_win_accept_decline'
    if source.startswith('curriculum4:rank_nonwin:'):
        return 'placement_discard_call_and_post_call'
    raise ValueError('Unclassified source: '+source)


def summarize():
    allocation = resource()
    if OUT.exists(): raise ValueError('Never overwrite an actual or partial TRAIN inventory')
    files = {str(Path(__file__).resolve()):sha(__file__)}

    def pinned(path, expected=None):
        path = Path(path).resolve(); actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Actual completed input changed: '+str(path))
        if str(path) in files and files[str(path)] != actual:
            raise ValueError('Conflicting actual input identity')
        files[str(path)] = actual
        return path

    base_root = ROOT/'runs/curriculum-placement-base-source-closure-v1'
    base_binding = read(pinned(base_root/'completion-binding-private.json'))['files']
    base_path = ROOT/'runs/curriculum-placement-data-v2/data-v1/data-ready-private.json'
    base = read(pinned(base_path, base_binding[str(base_path)]))
    counts_path = ROOT/'runs/curriculum-placement-base-pool-origin-proof-v3/actual-base-loss-source-counts-private.json'
    base_counts = read(pinned(counts_path, base_binding[str(counts_path)]))
    if (base['source_records'] != base_counts or len(base_counts['train']) != 48
            or sum(base_counts['train'].values()) != base['files']['train']['records']
            or sum(base_counts['validation'].values()) != base['files']['validation']['records']):
        raise ValueError('Actual complete base corpus counts and projection evidence required')
    collection_path = ROOT/'runs/curriculum-targeted-win-labels-v1/registered-private.json'
    collection = read(pinned(collection_path, base_binding[str(collection_path)]))
    selected = {spec['id']:spec for spec in collection['selected']}
    train_specs = {rid:spec for rid,spec in selected.items() if spec['split'] == 'train'}
    validation_specs = [spec for spec in selected.values() if spec['split'] == 'validation']
    if len(selected) != 108 or len(train_specs) != 72 or len(validation_specs) != 36:
        raise ValueError('The original fixed108 source population and splits are required')

    subset = ROOT/'runs/curriculum-completed-targeted-source-subset-v1'
    subset_binding = read(pinned(subset/'completion-binding-private.json'))['files']
    summary_path = subset/'summary-private.json'
    summary = read(pinned(summary_path,subset_binding[str(summary_path)]))
    if (summary['completed_shards'] != [0,1] or summary['actual_complete_case_files'] != 53
            or summary['unreviewed_numeric_contexts']):
        raise ValueError('Actual ended first54-view source observations required')
    folders = [ROOT/'runs/curriculum-targeted-win-resume-v3/shard-00/attempt-001',
        ROOT/'runs/curriculum-targeted-win-resume-v3/shard-01/attempt-004']
    rows = {}; quarantines = {}; target_counts = Counter(); target_families = defaultdict(set)

    def consume(folder, binding):
        for name in ('labels-private.jsonl','quarantine-private.jsonl'):
            path = (folder/name).resolve()
            pinned(path,binding[str(path)])
            with path.open() as stream:
                for line in stream:
                    if not line.endswith('\n'): raise ValueError('Only ended complete label files may be counted')
                    value = json.loads(line)
                    if name == 'labels-private.jsonl':
                        meta = value['_meta']; rid = meta['id']; spec = selected[rid]
                        if (meta['split'] != spec['split'] or meta['group_id'] != spec['group']
                                or meta['score_context'] != spec['context']
                                or value['questions']['action']['src'] != 'curriculum4:rank_win:targeted:'+spec['bucket']):
                            raise ValueError('Original source, family or split differs')
                        if spec['split'] != 'train': continue
                        if rid in rows: raise ValueError('Duplicate ended TRAIN label')
                        rows[rid] = value; source = value['questions']['action']['src']
                        target_counts[source] += 1; target_families[source].add(spec['group'])
                    else:
                        rid = value['source_id']; spec = selected[rid]
                        if (value['reason'] != 'public_prior_sampling_exhausted'
                                or not value['no_partial_target_admitted']
                                or value['group'] != spec['group'] or value['split'] != spec['split']):
                            raise ValueError('Only original bounded sampling quarantine is permitted')
                        if spec['split'] != 'train': continue
                        if rid in quarantines: raise ValueError('Duplicate ended quarantine')
                        quarantines[rid] = value

    for folder in folders:
        consume(folder,subset_binding)
    # The first18 source2 TRAIN views belong to an already ended original
    # writer. Its recovery admission has already reconstructed complete cases.
    # No currently appended label output is read or admitted here.
    admission_path = ROOT/'runs/curriculum-targeted-win-resume-v3/shard-02/attempt-001-admission-private.json'
    admission = read(pinned(admission_path))
    original_folder = ROOT/'runs/curriculum-targeted-win-labels-v1/shards/shard-02'
    original_entry = ROOT/'cluster/curriculum-targeted-win-labels-v1.slurm'
    original_archive_path = ROOT/'runs/curriculum-targeted-shard-terminals-v3/original-shard02-job322032-terminal-private.json'
    original_archive = read(pinned(original_archive_path))
    terminal = terminal_record(original_archive['raw'],'322032',original_entry)
    observed = admission['terminals']
    stable_fields = ('JobId','JobState','ExitCode','StartTime','EndTime','RunTime',
        'TimeLimit','NodeList','Command','ArrayJobId','ArrayTaskId')
    archived_fields = fields(original_archive['raw'])
    admitted_fields = fields(observed[0]['raw']) if len(observed) == 1 else {}
    complete_cases = [name for name in admission['files']
        if Path(name).parent == original_folder/'cases-private' and Path(name).suffix == '.json']
    if (terminal['state'] != 'FAILED' or terminal['exit_code'] != '124:0'
            or admission['status'] != 'actual_terminal_complete_cases_preserved_missing_cases_only'
            or not admission['same_fixed_population_worlds_teacher_and_targets']
            or admission['completed_views'] != len(complete_cases) or len(complete_cases) < 18
            or admission['remaining_views']+admission['completed_views'] != 27
            or len(observed) != 1
            or any(admitted_fields.get(key) != archived_fields.get(key) for key in stable_fields)
            or admission['priors'] != [str(original_folder)]):
        raise ValueError('Genuine original ended writer and exact complete-case recovery admission required')
    consume(original_folder,admission['files'])
    if (set(rows) & set(quarantines) or set(rows) | set(quarantines) != set(train_specs)
            or len(rows) != 71 or len(quarantines) != 1):
        raise ValueError('All72 fixed TRAIN views must be complete71 labels plus1 original quarantine')

    counts = dict(base_counts['train']); counts.update(target_counts)
    families = dict(base['source_distinct_families']['train'])
    families.update({source:len(values) for source,values in target_families.items()})
    plan = layout(counts,families)
    schedule = proposed_schedule(plan['virtual_exposures_per_pass'])
    validation_planned = Counter('curriculum4:rank_win:targeted:'+spec['bucket'] for spec in validation_specs)
    if len(target_counts) != 9 or len(validation_planned) != 9 or set(counts) != set(masses()):
        raise ValueError('All original57 source categories must be retained')
    OUT.mkdir(parents=True,exist_ok=False)
    columns = ['source','capability_group','unique_train_records','distinct_train_families',
        'existing_validation_records','targeted_validation_planned_records',
        'proposed_loss_mass','virtual_repeats_per_original','virtual_exposures_per_pass','scalar_example_weight']
    csv_path = OUT/'public-source-inventory.csv'
    with csv_path.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
        for source in sorted(counts):
            writer.writerow(dict(source=source,capability_group=capability_group(source),
                unique_train_records=counts[source],distinct_train_families=families[source],
                existing_validation_records=base_counts['validation'].get(source,0),
                targeted_validation_planned_records=validation_planned.get(source,0),
                proposed_loss_mass=plan['proposed_loss_mass'][source],
                virtual_repeats_per_original=plan['virtual_repeats_per_original_by_source'][source],
                virtual_exposures_per_pass=plan['virtual_exposures_by_source'][source],
                scalar_example_weight=plan['scalar_example_weights'][source]))
    result = dict(status='ended_complete_TRAIN_counts_and_prospective_budget_not_fit_registration',
        unique_train_records=plan['unique_train_records'],base_train_records=base['files']['train']['records'],
        complete_targeted_train_labels=len(rows),targeted_train_quarantines=len(quarantines),
        original_targeted_train_views=len(train_specs),loss_sources=len(counts),
        source2_actual_ended_complete_views=admission['completed_views'],
        existing_validation_records=base['files']['validation']['records'],
        targeted_validation_planned_records=len(validation_specs),
        full_expected_validation_records=base['files']['validation']['records']+len(validation_specs),
        validation_actual_final_count_still_requires_complete_merge=True,
        replay_25_sources_loss_mass=.8,new_32_terminal_sources_loss_mass=.2,
        retained25_means_number_of_sources_not25_percent=True,
        virtual_exposures_per_pass=plan['virtual_exposures_per_pass'],
        extra_virtual_exposures=plan['extra_virtual_exposures'],proposed_schedule=schedule,
        train_count_proposal_uses_only_ended_writer_labels=True,
        currently_appended_outputs_not_read=True,validation_target_values_not_used_for_proposal=True,
        all_labels_and_sources_retained_no_positive_value_filter=True,
        full108_merge_still_required=True,official_mix_and_fit_not_registered=True,
        new_games=0,model_loaded=False,optimizer_updates=0,allocation=allocation)
    write(OUT/'summary-private.json',result)
    write(OUT/'prospective-plan-private.json',plan)
    public = {key:value for key,value in result.items() if key != 'allocation'}
    write(OUT/'summary-public.json',public)
    for path in (ROOT/'tools/curriculum_placement_mix_v1.py',
                 ROOT/'tools/prepare_curriculum_placement_exposures_v1.py',
                 ROOT/'tools/curriculum_placement_exposures_v1.py',*OUT.glob('*')):
        pinned(path)
    write(OUT/'completion-binding-private.json',dict(files=files,actual_fit_admitted=False))
    print(json.dumps(public),flush=True)


if __name__ == '__main__': summarize()
