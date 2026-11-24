"""Audit all currently prepared placement TRAIN objectives, without a model.

Nine targeted terminal pools are still collecting. This is explicitly not
full57-source data admission, source weighting, a qualified parent or a fit.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource


def main():
    allocation = resource(); started = time.monotonic()
    from tools import curriculum_placement_parent_loss_v1 as new
    from tools import curriculum_yaku_parent_loss_v1 as replay
    from tools import curriculum_placement_objective_v1 as terminal
    reg_path = ROOT/'runs/curriculum-progression-parent-retention-v1/fit-plan/registered.json'
    reg = read(reg_path); sys.path[:0] = [reg['source'], reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from jevany.data import materialize
    ready_path = ROOT/'runs/curriculum-placement-data-v2/data-v1/data-ready-private.json'; ready = read(ready_path)
    if (ready['status'] != 'placement_replay_partial_win_full_win_controls_and_nonwin_prepared_not_fit'
            or ready['files']['train']['records'] != 91647 or ready['files']['validation']['records'] != 24607):
        raise ValueError('Actual coherent replay plus currently completed terminal controls required')
    test_log = ROOT/'logs/curriculum-placement-parent-loss-tests-v1.log'
    if 'Ran 5 tests' not in test_log.read_text() or not test_log.read_text().endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Five original-value/gradient/annotation/mask/observer tests must pass first')
    parent_proof = ROOT/'runs/curriculum-yaku-parent-loss-v1/preflight-v1'
    earlier = read(parent_proof/'summary-private.json')
    files = dict(read(parent_proof/'completion-binding-private.json')['files'])
    if earlier['method'] != replay.METHOD or earlier['actual_TRAIN_records'] != 91247:
        raise ValueError('Actual retained3B replay contract required')
    public = ('tools/curriculum_placement_parent_loss_v1.py',
        'tools/preflight_curriculum_placement_parent_loss_v1.py',
        'tests/core/test_curriculum_placement_parent_loss_v1.py',
        'docs/experiments/plans/curriculum-placement-parent-loss-20261124.md')
    paths = [ROOT/ready['files'][s]['path'] for s in ('train', 'validation')]
    for split, path in zip(('train', 'validation'), paths):
        if sha(path) != ready['files'][split]['sha256']: raise ValueError('Actual prepared data changed')
    for p in (reg_path, ready_path, test_log, parent_proof/'summary-private.json',
              parent_proof/'completion-binding-private.json', *paths, *[ROOT/p for p in public],
              ROOT/'tools/curriculum_placement_objective_v1.py', ROOT/'tools/curriculum_placement_mix_v1.py'):
        files[str(p.resolve())] = sha(p)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('A prior bound source changed')
    out = ROOT/'runs/curriculum-placement-parent-loss-v1/objective-preflight-v1'; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files, allocation=allocation,
        synthetic_logits=True, model_loaded=False, student_updates=0, missing_pools_not_imputed=True))
    counts = Counter(); representatives = {}; maximum = 0.
    try:
        with paths[0].open() as stream, torch.no_grad():
            for line in stream:
                row = json.loads(line); source = row['questions']['action']['src']; counts[source] += 1
                original_request = copy.deepcopy(row); original = materialize(row)
                encoded = new.annotate(copy.deepcopy(original), row); new.materialization_unchanged(original, encoded)
                if row != original_request: raise ValueError('Original request mutated')
                module = replay if source in replay.MASS else terminal
                expected = module.annotate(copy.deepcopy(original), row)['questions'][0]; q = encoded['questions'][0]
                z = torch.linspace(-1.1, 1.2, len(q['keys']), dtype=torch.float64)
                before = module.parts(z, expected, 'cpu'); after = new.parts(z, q, 'cpu')
                for key in ('objective', 'standard_target_ce'):
                    difference = float(abs(before[key]-after[key])); maximum = max(maximum, difference)
                    if difference: raise ValueError('Actual source objective/CE changed: '+source)
                if source in replay.MASS:
                    if any(not torch.equal(before[k], after[k]) for k in replay.COMPONENTS):
                        raise ValueError('Actual inherited replay component changed')
                representatives.setdefault(source, row)
        if dict(counts) != ready['source_records']['train'] or sum(counts.values()) != 91647:
            raise ValueError('Full actual prepared TRAIN inventory required')
        if sum(counts[s] for s in replay.MASS) != 91247 or sum(counts[s] for s in new.NEW_SOURCES) != 400:
            raise ValueError('Exact3B replay and all currently prepared terminal rows required')
        missing = sorted(set(new.masses())-set(counts))
        if len(counts) != 48 or len(missing) != 9 or any(':targeted:' not in s for s in missing):
            raise ValueError('Explicitly retain the nine still-collecting targeted terminal pools')
        # Unit scalars prove loss/gradient equality; they are not a production mix.
        weights = dict.fromkeys(new.masses(), 1.); observer = []; losses = []; gradient_error = 0.
        for source, row in representatives.items():
            original = materialize(row); q = new.annotate(copy.deepcopy(original), row)['questions'][0]
            module = replay if source in replay.MASS else terminal
            expected_q = module.annotate(copy.deepcopy(original), row)['questions'][0]
            x = torch.linspace(-.8, .7, len(q['keys']), dtype=torch.float64, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            a = module.parts(x, expected_q, 'cpu')['objective']
            b = new.weighted_loss(y, q, 'cpu', weights, observer)
            a.backward(); b.backward(); error = float((x.grad-y.grad).abs().max())
            gradient_error = max(gradient_error, error)
            if error or not torch.equal(a.detach(), b.detach()): raise ValueError('Actual representative gradient changed')
            losses.append(float(b.detach()))
        observed = new.summarize(observer, weights)
        if abs(observed['loss']-sum(losses)/48) > 1e-10: raise ValueError('Actual sparse observer accounting differs')
        if any(sha(p) != h for p, h in files.items()): raise ValueError('A frozen input changed')
        summary = dict(status='available_placement_TRAIN_objective_continuity_passed_not_full57_admission',
            method=new.METHOD, actual_prepared_TRAIN_records=91647, exact_retained_replay_records=91247,
            currently_prepared_terminal_records=400, actual_sources_checked=48, required_final_sources=57,
            missing_targeted_source_pools=missing, maximum_all_records_objective_or_CE_difference=maximum,
            representative_gradient_sources=48, maximum_representative_gradient_difference=gradient_error,
            every_actual_materialization_labels_masks_sources_unchanged=True,
            new_terminal_objective_unchanged=True, actual57_source_weights_not_registered=True,
            unit_scalars_only_for_software_proof=True, model_loaded=False, model_forwards=0,
            new_games=0, student_updates=0, separately_executed_unit_tests=5,
            full57_data_validation_metrics_trainer_auditor_and_parent_fit_still_required=True,
            final_grid_unused=True, seconds=time.monotonic()-started)
        write(out/'summary-private.json', summary)
        for p in (out/'registered-private.json', out/'summary-private.json'): files[str(p.resolve())] = sha(p)
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(error=repr(error), checked_counts=dict(counts)))
        raise


if __name__ == '__main__': main()
