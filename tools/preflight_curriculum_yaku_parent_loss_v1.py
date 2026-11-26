"""Actual complete TRAIN objective-continuity proof, not 3B admission or fit."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha


def main():
    allocation = resource(); started = time.monotonic()
    from tools import curriculum_yaku_parent_loss_v1 as new
    from tools import curriculum_yaku_objective_v1 as previous
    from tools import curriculum_progression_parent_loss_v1 as parent
    reg_path = ROOT/'runs/curriculum-progression-parent-retention-v1/fit-plan/registered.json'
    reg = read(reg_path)
    sys.path[:0] = [reg['source'], reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from jevany.data import materialize
    ready_path = ROOT/'runs/curriculum-yaku-scenarios-v1/data-v2/data-ready-private.json'
    ready = read(ready_path)
    if (ready['status'] != 'exact_replay_and_normalized_targeted_yaku_corpus_prepared_not_fit_admitted'
            or ready['files']['train']['records'] != 91247 or ready['files']['validation']['records'] != 24516
            or not ready['original_replay_objects_labels_and_partitions_unchanged']):
        raise ValueError('Actual prepared full3B data required')
    paths = [ROOT/ready['files'][s]['path'] for s in ('train', 'validation')]
    if any(sha(p) != ready['files'][s]['sha256'] for s, p in zip(('train', 'validation'), paths)):
        raise ValueError('Prepared actual source files changed')
    test_log = ROOT/'logs/curriculum-yaku-parent-loss-tests-v1.log'
    log = test_log.read_text()
    if 'Ran 4 tests' not in log or not log.endswith('\nOK\n'):
        raise ValueError('The separately executed four objective/gradient contract tests must pass first')
    out = ROOT/'runs/curriculum-yaku-parent-loss-v1/preflight-v1'
    out.mkdir(parents=True, exist_ok=False)
    public = ('tools/curriculum_yaku_parent_loss_v1.py',
        'tools/preflight_curriculum_yaku_parent_loss_v1.py',
        'tests/core/test_curriculum_yaku_parent_loss_v1.py',
        'docs/experiments/plans/curriculum-yaku-parent-loss-continuity-20261124.md')
    dependencies = ('tools/curriculum_yaku_objective_v1.py',
        'tools/curriculum_progression_parent_loss_v1.py', 'tools/curriculum_progression_objective_v1.py',
        'tools/curriculum_calls_classroute_v1.py', 'tools/curriculum_calls_riichi_retention_v1.py',
        'tools/curriculum_riichi_class_loss.py', 'tools/curriculum_losses.py',
        'tools/curriculum_course_order.py', 'tests/core/test_curriculum_progression_parent_loss_v1.py')
    files = {str(p.resolve()): sha(p) for p in
        [*[ROOT/p for p in public+dependencies], reg_path, ready_path, test_log, *paths]}
    write(out/'registered-private.json', dict(files=files, allocation=allocation,
        method=new.METHOD, only_synthetic_logits=True, student_model_loaded=False,
        student_updates=0, future_parent_grade_not_assumed=True))
    counts = Counter(); representatives = {}; maximum = 0.
    try:
        with paths[0].open() as stream, torch.no_grad():
            for line in stream:
                row = json.loads(line); source = row['questions']['action']['src']; counts[source] += 1
                original_request = copy.deepcopy(row); material = materialize(row)
                annotated = new.annotate(copy.deepcopy(material), row)
                new.materialization_unchanged(material, annotated)
                if row != original_request: raise ValueError('Actual request mutated')
                q = annotated['questions'][0]
                old = previous if source in new.NEW_SOURCES else parent
                reference = old.annotate(copy.deepcopy(material), row)['questions'][0]
                z = torch.linspace(-1.1, 1.2, len(q['keys']), dtype=torch.float64)
                expected = old.parts(z, reference, 'cpu'); actual = new.parts(z, q, 'cpu')
                for field in ('objective', 'standard_target_ce'):
                    difference = float(abs(expected[field]-actual[field])); maximum = max(maximum, difference)
                    if difference: raise ValueError('Actual source objective/CE changed: '+source)
                if (source in new.NEW_SOURCES and not torch.equal(actual['new_route_objective'], actual['objective'])):
                    raise ValueError('New-route objective diagnostic differs')
                if source not in new.NEW_SOURCES and float(actual['new_route_objective']) != 0:
                    raise ValueError('Replay acquired a new-route loss')
                representatives.setdefault(source, row)
        if (sum(counts.values()) != 91247 or dict(counts) != ready['coverage']['train']['sources']
                or sum(counts[s] for s in parent.MASS) != 87595
                or sum(counts[s] for s in new.NEW_SOURCES) != 3652):
            raise ValueError('Full exact19-replay/six-route TRAIN inventory required')
        scalar_weights = new.weights(dict(counts)); observer = []; weighted_losses = []; gradient_error = 0.
        for source, row in representatives.items():
            material = materialize(row); q = new.annotate(copy.deepcopy(material), row)['questions'][0]
            old = previous if source in new.NEW_SOURCES else parent
            expected_q = old.annotate(copy.deepcopy(material), row)['questions'][0]
            x = torch.linspace(-.8, .7, len(q['keys']), dtype=torch.float64, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            expected = old.parts(x, expected_q, 'cpu')['objective']*scalar_weights[source]
            actual = new.weighted_loss(y, q, 'cpu', scalar_weights, observer)
            expected.backward(); actual.backward()
            difference = float((x.grad-y.grad).abs().max()); gradient_error = max(gradient_error, difference)
            if difference or not torch.equal(expected.detach(), actual.detach()):
                raise ValueError('Actual representative weighted gradient/objective changed: '+source)
            weighted_losses.append(float(actual.detach()))
        observed = new.summarize(observer, scalar_weights)
        if (len(observer) != 25 or abs(observed['loss']-sum(weighted_losses)/25) > 1e-10
                or abs(observed['loss']-observed['ce']-observed['signed_objective_adjustment']) > 1e-10):
            raise ValueError('Actual weighted observer journal accounting differs')
        if any(sha(p) != h for p, h in files.items()): raise ValueError('A preflight-bound input changed')
        summary = dict(status='actual91247_TRAIN_yaku_parent_loss_continuity_passed_not_fit_admitted',
            method=new.METHOD, actual_TRAIN_records=91247, inherited_replay_records=87595,
            new_route_records=3652, validation_records_not_inferred=24516, source_counts=dict(counts),
            maximum_actual_objective_or_CE_difference=maximum,
            representative_weighted_gradient_sources=25, maximum_representative_gradient_difference=gradient_error,
            every_actual_materialization_labels_masks_and_source_unchanged=True,
            source_masses_weights_and_original_epoch_sampler_unchanged=True,
            replay_mass=.8, new_route_mass=.2, separately_executed_unit_tests=4,
            synthetic_logits_not_model_predictions=True, student_model_loaded=False,
            student_updates=0, new_games=0, actual_parent_qualification_still_required=True,
            validation_trainer_auditor_registration_and_GPU_fit_not_implemented_here=True,
            seconds=time.monotonic()-started)
        write(out/'summary-private.json', summary)
        for p in (out/'registered-private.json', out/'summary-private.json'): files[str(p)] = sha(p)
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(error=repr(error), source_counts=dict(counts)))
        raise


if __name__ == '__main__': main()
