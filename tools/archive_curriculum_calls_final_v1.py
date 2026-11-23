"""Verify actual final-curve export and stage only minimal public aggregates."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.export_curriculum_calls_curves_v1 import GAME_FIELDS, TRAIN_FIELDS


def archive(export_root, terminal, exit_path, out):
    gpu_only()
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    scheduler_exit(read(exit_path), terminal.read_text(), export_root/'export.slurm')
    public = export_root/'snapshot-v1/public-ready'; private = export_root/'snapshot-v1/snapshots-private'
    proof = read(public/'provenance.json')
    if (proof['actual_train_through'] != 388 or proof['actual_validation_steps'] != [0, 194, 388]
            or proof['actual_matched16_steps'] != [0, 194, 388] or proof['actual_final64_steps'] != [388]
            or proof['independent_full_final_audit_checked'] is not True
            or proof['checked_cached_validation_menus'] != 37437 or proof['checked_actual_native_ledgers'] != 96):
        raise ValueError('Actual independently audited complete final388 export required')
    files = dict(read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files'])
    for path in (Path(__file__), export_root/'archive.slurm', terminal, exit_path, public/'provenance.json'):
        files[str(path.resolve())] = sha(path)
    for name, digest in proof['output_files'].items():
        path = public/name
        if sha(path) != digest: raise ValueError('Actual exported artifact changed')
        files[str(path)] = digest
    for name, digest in proof['captured_journal_sha256'].items():
        path = private/name
        if sha(path) != digest: raise ValueError('Actual captured raw journal changed')
        files[str(path)] = digest
    for name, n in (('train-curve.csv', 388), ('validation-curve.csv', 3), ('operation-validation.csv', 30),
                    ('evaluation-matched16.csv', 3), ('evaluation-final64.csv', 1)):
        with (public/name).open() as stream:
            reader = csv.DictReader(stream); records = list(reader)
            if len(records) != n: raise ValueError('Actual minimal public row count differs: '+name)
            if name == 'train-curve.csv' and (tuple(reader.fieldnames) != TRAIN_FIELDS or int(records[-1]['records_seen']) != 98044):
                raise ValueError('Private training fields or wrong final exposure')
    parent = ROOT/'runs/curriculum-calls-classroute-v1'; regpath = parent/'fit-plan/registered.json'; reg = read(regpath)
    evidence = read(parent/'final-evidence-private.json'); fit = ROOT/reg['curriculum']['out']
    if evidence['status'] != 'stage2_calls_classroute_final_evidence_audit_passed': raise ValueError('Actual final audit missing')
    validation = [json.loads(line) for line in (fit/'metrics/validation_metrics.jsonl').open()][-1]
    if validation['local_step'] != 388: raise ValueError('Final actual validation required')
    comparisons = []
    for policy, label in (('calls13-parent-final608', 'reference-parent'), ('calls13-final388', 'step-000388'), ('frozen-v8', 'reference-v8')):
        folder = fit/'evaluations'/label; summary = read(folder/'summary.json')
        if summary['matches'] != 64 or summary['seed_groups'] != 8: raise ValueError('Matching full64 references required')
        comparisons.append(dict(policy=policy, **{k: summary[k] for k in GAME_FIELDS}))
        for name in ('protocol.json', 'summary.json', 'matches-private.jsonl'): files[str(folder/name)] = sha(folder/name)
    dest = ROOT/'docs/experiments/results/curriculum-calls388-verified-20261123'; dest.mkdir(parents=True, exist_ok=False)
    names = ('train-curve.csv', 'validation-curve.csv', 'operation-validation.csv', 'evaluation-matched16.csv',
             'evaluation-final64.csv', 'calls13-curves.svg', 'provenance.json')
    for name in names: shutil.copyfile(public/name, dest/name)
    with (dest/'matched-reference64.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('policy', *GAME_FIELDS)); writer.writeheader(); writer.writerows(comparisons)
    diagnosis = dict(status='actual_final388_curves_and_matched64_reference_aggregates_verified',
        actual_updates=388, actual_exposures=98044, original_route_parameter_identity=read(fit/'evaluations/step-000388/summary.json')['checkpoint_sha256'],
        original_route_policy_identity=read(fit/'evaluations/step-000388/summary.json')['policy_identity'],
        actual_full64_comparisons=comparisons, paired_eight_seed_cluster_comparisons=evidence['comparisons'],
        original_registered_stage2_grade=evidence['stage2_advancement'],
        current_selected_basic_source_quality=validation['basic_replay_source_quality'],
        current_raw_basic_source_quality=validation['raw_basic_source_quality'],
        source_specific_retention=evidence['source_specific_basic_retention'],
        five_call_balanced_accuracy_gain=evidence['five_call_type_balanced_gain'],
        final_hard_operation_counts=validation['operation_type_counts'],
        actual_final12k_reload_probability_error=evidence['saved_final_max_probability_error'],
        independent_audit_complete_games_checked=evidence['checked_complete_games'],
        actual_curve_points=[0, 194, 388], curve_native_grid16_with_full64_separate=True,
        supervised_reward=None, actual_game_reward='terminal_net_points_div8000', visual_PNG_review_completed=True,
        no_new_inference_games_or_updates=True, raw_records_predictions_gradients_and_weights_private=True,
        final_reserved_grid_unused=True, independent_v8_parity_not_proven=True, goal_achieved=False)
    write(dest/'diagnosis-summary.json', diagnosis)
    for path in (regpath, parent/'final-evidence-private.json', fit/'metrics/validation_metrics.jsonl'):
        files[str(path)] = sha(path)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Immutable evidence changed during staging')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'completion-binding-private.json', dict(files=files,
        staged_public_files={str(p.relative_to(ROOT)): sha(p) for p in dest.iterdir()}))
    write(out/'complete-private.json', dict(status=diagnosis['status'], public_whitelist=list(names)+['matched-reference64.csv', 'diagnosis-summary.json'],
        automatic_publication=False, visual_review_completed=True, source_recipe_unchanged=True))
    print(json.dumps(dict(status=diagnosis['status'], comparisons=comparisons)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('export-root', 'terminal', 'exit', 'out'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); archive(a.export_root.resolve(), a.terminal.resolve(), a.exit.resolve(), a.out.resolve())
