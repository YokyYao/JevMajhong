"""Export observed route-label and retained-operation diagnostics, without inference.

Read an already completed curve16 report. Teacher agreement is not yaku mastery
or a measure of game strength. Execute only in a finite scheduler GPU allocation.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha

ROUTES = dict(kokushi='Kokushi musou', chuuren='Chuuren poutou',
    suuankou='Suuankou', daisangen='Daisangen', chiitoitsu='Chiitoitsu',
    unselected_random_initial_hand_control='Random initial-hand control')
OPERATIONS = ('reach', 'chi', 'pon', 'ankan', 'daiminkan', 'kakan')


def export(comparison, out):
    allocation = resource(); comparison = Path(comparison).resolve(); out = Path(out).resolve()
    if out.exists() or not out.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Use a new private runs/ output; never overwrite evidence')
    binding_path = comparison/'completion-binding-private.json'; bound = read(binding_path)['files']
    files = {str(binding_path): sha(binding_path), str(Path(__file__)): sha(__file__)}
    def checked(path):
        path = Path(path).resolve(); actual = sha(path)
        if bound.get(str(path)) != actual: raise ValueError('Completed diagnostic input changed')
        files[str(path)] = actual
        return read(path)
    done = checked(comparison/'summary-private.json')
    if (done['status'] != 'actual_same16_development_points_and_full_validation_summarized'
            or not done['same_grid_and_execution_policy'] or done['new_games'] != 0
            or done['independent_strength_proved']):
        raise ValueError('An actual completed same-grid development report is required')
    steps = done['actual_steps']; journals = checked(comparison/'selected-journal-points-private.json')
    validation = {step:journals['validation'][str(step)] for step in steps}
    registrations = [Path(name) for name in bound if name.endswith('/fit-plan/registered.json')]
    if len(registrations) != 1: raise ValueError('The actual completed comparison registration is required')
    reg = checked(registrations[0]); ready = checked(ROOT/reg['curriculum']['data_ready'])
    rows = []
    for route, label in ROUTES.items():
        source = 'curriculum3B:teacher:'+route
        n = ready['coverage']['validation']['sources'][source]
        for step in steps:
            row = validation[step]; ce = row['stratum_ce'][source]
            agreement = row['new_source_teacher_agreement'][source]
            if (row['stratum_counts'][source] != n or not math.isfinite(ce) or ce < 0
                    or not 0 <= agreement <= 1):
                raise ValueError('Stable real route denominators and finite observations required')
            rows.append(dict(route=route, label=label, step=step,
                train_menus=ready['coverage']['train']['sources'][source], validation_menus=n,
                teacher_target_ce=ce, teacher_action_agreement=agreement))
    if sum(ready['coverage']['validation']['sources']['curriculum3B:teacher:'+r] for r in ROUTES) != 828:
        raise ValueError('All actual828 route/control validation menus are required')
    operations = []
    for op in OPERATIONS:
        support = None
        for step in steps:
            counts = validation[step]['original_calls_operation_type_counts'][op]
            values = {key:counts.get(key, 0) for key in ('tp', 'fn', 'tn', 'fp')}
            if any(type(n) is not int or n < 0 for n in values.values()):
                raise ValueError('Actual nonnegative operation confusion counts required')
            yes = values['tp']+values['fn']; no = values['tn']+values['fp']
            if not yes or not no or (support is not None and support != (yes, no)):
                raise ValueError('Every operation requires fixed positive/negative label denominators')
            support = yes, no
            operations.append(dict(operation=op, step=step, positive_menus=yes, negative_menus=no,
                accept_recall=values['tp']/yes, decline_recall=values['tn']/no, **values))
    out.mkdir(parents=True)
    for name, values in (('route-label-diagnostics.csv', rows), ('retained-operation-diagnostics.csv', operations)):
        with (out/name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    lines = ['# Observed 3B route and retained-operation diagnostics', '',
        'These observations reuse completed full validation. No additional model inference, games or updates were run.', '',
        '## Route teacher labels', '',
        'The 828 route/control validation menus come from 45 source families; repeated decisions are not independent games. '
        'CE measures probability assigned to the teacher action, and agreement measures the selected action matching that label. '
        'Neither proves that the model can complete the named yaku in real play.', '',
        '| Route | Train menus | Validation menus | '+
        ' | '.join('CE '+str(s) for s in steps)+' | Agreement '+str(steps[0])+' | Agreement '+str(steps[-1])+' |',
        '| --- | ---: | ---: | '+' | '.join('---:' for _ in steps)+' | ---: | ---: |']
    for route, label in ROUTES.items():
        points = [r for r in rows if r['route'] == route]
        lines.append('| '+label+' | '+str(points[0]['train_menus'])+' | '+str(points[0]['validation_menus'])+
            ' | '+' | '.join(f"{r['teacher_target_ce']:.5f}" for r in points)+
            f" | {points[0]['teacher_action_agreement']:.2%} | {points[-1]['teacher_action_agreement']:.2%} |")
    lines += ['', '## Retained operation labels', '',
        'The positive and negative menus are the unchanged original-call validation subset, excluding the new route menus. '
        'Accept recall means selecting the operation when the cached teacher/legacy label selects it; decline recall means '
        'declining when that label does. These are imitation diagnostics, not expert judgments about profitable calls.', '',
        '| Operation | Positive menus | Negative menus | Accept '+str(steps[0])+' | Accept '+str(steps[-1])+
        ' | Decline '+str(steps[0])+' | Decline '+str(steps[-1])+' |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for op in OPERATIONS:
        points = [r for r in operations if r['operation'] == op]; a, b = points[0], points[-1]
        lines.append(f"| {op} | {a['positive_menus']} | {a['negative_menus']} | {a['accept_recall']:.2%} | "
            f"{b['accept_recall']:.2%} | {a['decline_recall']:.2%} | {b['decline_recall']:.2%} |")
    lines += ['', 'The course and original 2500-update budget remain unchanged. This report does not impose an intermediate '
        'grade veto, select a checkpoint, open reserved confirmation seeds or replace production weights. Raw examples stay private.', '']
    (out/'route-diagnostics.md').write_text('\n'.join(lines))
    write(out/'summary-private.json', dict(status='completed_validation_route_and_operation_diagnostics_exported',
        actual_steps=steps, route_validation_menus=828, route_source_families=45,
        new_games=0, model_loaded=False, student_updates=0, independent_strength_proved=False,
        rare_yaku_mastery_proved=False, recipe_unchanged=True, allocation=allocation))
    for path in out.iterdir(): files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps(dict(status='completed_validation_route_and_operation_diagnostics_exported',
        steps=steps, routes=rows, retained_operations=operations)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); export(args.comparison, args.out)
