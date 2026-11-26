"""Full cached validation and exact source-loss preflight, without model updates."""
import ast
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_progression_parent_control_v1 import CONTROL, PREFLIGHT, PUBLIC
from tools.curriculum_progression_parent_loss_v1 import METHOD, MASS


def main():
    allocation = resource(); started = time.monotonic()
    if PREFLIGHT.exists(): raise ValueError('Do not overwrite preflight evidence')
    regpath = CONTROL/'fit-plan/registered.json'; reg = read(regpath)
    sys.path[:0] = [reg['source'], reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from tools.curriculum_progression_parent_fit_v1 import profile
    from tools.curriculum_progression_parent_loss_v1 import weighted_loss, summarize, annotate, parts
    from jevmahjong.training.base_lora import target_cross_entropy
    from jevany.data import materialize
    from tools import curriculum_calls_classroute_v1 as parent
    from tools import curriculum_progression_objective_v1 as old_objective
    from tools.curriculum_progression_paired_v1 import verify_retry
    verify_retry(reg)
    proofroot = CONTROL/'parent-loss-preflight-v1'
    previous = read(proofroot/'summary-private.json')
    previous_binding = read(proofroot/'completion-binding-private.json')
    if (previous['actual_TRAIN_rows'] != 87595 or previous['maximum_objective_difference'] != 0
            or previous['maximum_representative_gradient_difference'] != 0):
        raise ValueError('All actual TRAIN loss and source gradient evidence required')
    for p, h in previous_binding['files'].items():
        if sha(p) != h: raise ValueError('Original loss preflight changed: '+p)
    files = [ROOT/p for p in PUBLIC]
    for p in files:
        if p.suffix == '.py': ast.parse(p.read_text(), filename=str(p))
    subprocess.run(['bash', '-n', str(ROOT/'cluster/curriculum-progression-parent-retention-v1.slurm')], check=True)
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    rows = [json.loads(line) for line in Path(args['data']).open()]
    counts = Counter(r['questions']['action']['src'] for r in rows)
    if len(rows) != 87595 or dict(counts) != previous['source_counts']:
        raise ValueError('Exact source-complete TRAIN cohort required')
    representatives = {}
    for r in rows: representatives.setdefault(r['questions']['action']['src'], r)
    pending = []; weighted_sum = 0.; weights = reg['course_loss_weights']
    for src, row in representatives.items():
        q = annotate(materialize(row), row)['questions'][0]
        z = torch.linspace(-1.2, 1.3, len(q['keys']), dtype=torch.float64, requires_grad=True)
        loss = weighted_loss(z, q, 'cpu', weights, pending)
        reference = (parent.parts(z, q, 'cpu') if src.startswith('curriculum2:')
                     else old_objective.parts(z, q, 'cpu'))['objective']*weights[src]
        if not torch.equal(loss, reference): raise ValueError('Actual weighted objective differs')
        a, = torch.autograd.grad(loss, z, retain_graph=True)
        b, = torch.autograd.grad(reference, z)
        if not torch.equal(a, b): raise ValueError('Actual weighted gradient differs')
        weighted_sum += float(loss.detach())
    observed = summarize(pending, weights)
    if (len(pending) != 19 or abs(observed['loss']-weighted_sum/19) > 1e-10
            or observed['loss_method'] != METHOD):
        raise ValueError('Actual optimizer component aggregation differs')
    validation = [json.loads(line) for line in Path(args['eval_data']).open()]
    predictions_path = ROOT/reg['curriculum']['out']/'training-run/training_eval/step-000000/development/rows.json'
    actual = profile(validation, read(predictions_path), {}, target_cross_entropy)
    old = read(ROOT/reg['curriculum']['out']/'validation-curve.json')[0]
    changed = {'weighted_class_objective', 'extra_RI_class_loss', 'signed_objective_adjustment',
               'source_loss_components', 'source_objective', 'loss_method', 'step', 'upstream_hard_label_nll'}
    for k, v in old.items():
        if k not in changed and actual.get(k) != v:
            raise ValueError('Classification/policy/CE validation diagnostic changed: '+k)
    if sum(actual['stratum_counts'].values()) != 23688 or actual['loss_method'] != METHOD:
        raise ValueError('Complete retained-parent validation objective required')
    PREFLIGHT.mkdir()
    test_run = subprocess.run([sys.executable, '-m', 'unittest',
        'tests.core.test_curriculum_progression_parent_loss_v1',
        'tests.core.test_curriculum_progression_parent_control_v1'],
        capture_output=True, text=True, timeout=180)
    (PREFLIGHT/'tests-private.log').write_text(test_run.stdout+test_run.stderr)
    if test_run.returncode: raise RuntimeError('Meaningful objective/recipe tests failed; see private log')
    write(PREFLIGHT/'validation-zero-objective-private.json', actual)
    summary = dict(status='full_validation_parent_retention_launch_preflight_passed',
        method=METHOD, train_records=len(rows), validation_records=len(validation),
        source_count=len(counts), actual_weighted_loss_and_gradients_exact=True,
        classification_and_policy_diagnostics_unchanged=True,
        zero_weighted_objective=actual['weighted_class_objective'],
        zero_weighted_target_ce=actual['weighted_target_ce'],
        zero_signed_adjustment=actual['signed_objective_adjustment'],
        unit_test_returncode=test_run.returncode, student_updates=0, student_model_loaded=False,
        new_games=0, resource_allocation=allocation, seconds=time.monotonic()-started)
    write(PREFLIGHT/'summary-private.json', summary)
    files += [regpath, Path(args['data']), Path(args['eval_data']), predictions_path,
              proofroot/'summary-private.json', proofroot/'completion-binding-private.json',
              *PREFLIGHT.glob('*-private.json'), PREFLIGHT/'tests-private.log']
    write(PREFLIGHT/'completion-binding-private.json', dict(files={str(p.resolve()): sha(p) for p in files}))
    print(json.dumps({k: v for k, v in summary.items() if k != 'resource_allocation'}), flush=True)


if __name__ == '__main__': main()
