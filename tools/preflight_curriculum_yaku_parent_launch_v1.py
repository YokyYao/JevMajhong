"""Exercise actual retained3B logging/audit code without a model or admission.

Synthetic logits below are software inputs, never training or performance data.
All computation must run in an existing scheduler GPU allocation.
"""
import ast
from collections import Counter
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha
from tools.curriculum_yaku_parent_launch_v1 import CONFIG, PUBLIC, PREFLIGHT, require_same_recipe
from tools.curriculum_yaku_parent_loss_v1 import (
    MASS, METHOD, COMPONENTS, NEW_SOURCES, annotate, weighted_loss, summarize,
)


def compiled_node(node, filename):
    return compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), filename, 'exec')


def main():
    allocation = resource(); started = time.monotonic(); c = read(CONFIG)
    original_path = ROOT/'configs/training/curriculum-yaku-fit-v2.json'
    require_same_recipe(c, read(original_path))
    reg_path = ROOT/'runs/curriculum-progression-parent-retention-v1/fit-plan/registered.json'
    parent = read(reg_path)
    sys.path[:0] = [parent['source'], parent['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from jevany.data import materialize
    from jevmahjong.training.metric_journal import MetricJournal, LocalTracker

    files = {str((ROOT/p).resolve()): sha(ROOT/p) for p in PUBLIC}
    files.update({str(p.resolve()): sha(p) for p in (CONFIG, original_path, reg_path,
        Path(parent['source'])/'jevmahjong/training/metric_journal.py')})
    proofs = {}
    for name in ('objective_preflight', 'validation_adapter_preflight', 'epoch_recipe'):
        path = ROOT/c[name]; proof = read(path); binding = path.parent/'completion-binding-private.json'
        bound = read(binding)['files']
        if any(sha(p) != h for p, h in bound.items()): raise ValueError('Prior proof input changed: '+name)
        files.update(bound)
        files[str(path.resolve())] = sha(path); files[str(binding.resolve())] = sha(binding)
        proofs[name] = proof
    train_proof = proofs['objective_preflight']; val_proof = proofs['validation_adapter_preflight']
    recipe = proofs['epoch_recipe']
    if (train_proof['method'] != METHOD or train_proof['actual_TRAIN_records'] != 91247
            or train_proof['maximum_actual_objective_or_CE_difference'] != 0
            or train_proof['maximum_representative_gradient_difference'] != 0
            or val_proof['method'] != METHOD or val_proof['validation_menus'] != 24516
            or not val_proof['canonical_CE_and_every_classification_metric_unchanged']
            or not val_proof['retained_nine_component_profile_and_signed_accounting_verified']
            or recipe['proposed_loss_mass'] != MASS or recipe['actual_optimizer_steps'] != 0):
        raise ValueError('Complete retained TRAIN/VAL contracts and unchanged numerical recipe required')
    for name in PUBLIC:
        if name.endswith('.py'): ast.parse((ROOT/name).read_text(), filename=name)
    subprocess.run(['bash', '-n', str(ROOT/'cluster/curriculum-yaku-parent-v1.slurm')], check=True, timeout=10)
    test_log = ROOT/'logs/curriculum-yaku-parent-launch-tests-v1.log'
    log = test_log.read_text()
    if 'Ran 8 tests' not in log or not log.endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Actual eight guard/runtime/I/O unit tests must pass first')
    files[str(test_log)] = sha(test_log)

    ready_path = ROOT/c['data_ready']; ready = read(ready_path)
    train_path = ROOT/ready['files']['train']['path']
    if sha(train_path) != ready['files']['train']['sha256']: raise ValueError('Actual TRAIN data changed')
    representatives = {}
    with train_path.open() as stream:
        for line in stream:
            row = json.loads(line); representatives.setdefault(row['questions']['action']['src'], row)
            if len(representatives) == 25: break
    if set(representatives) != set(MASS): raise ValueError('All25 actual source representatives required')
    files[str(train_path.resolve())] = sha(train_path); files[str(ready_path.resolve())] = sha(ready_path)
    synthetic_profile = (ROOT/c['validation_adapter_preflight']).parent/'synthetic-profile-private.json'
    files[str(synthetic_profile.resolve())] = sha(synthetic_profile)
    PREFLIGHT.mkdir(exist_ok=False)
    write(PREFLIGHT/'registered-private.json', dict(files=files, allocation=allocation,
        only_synthetic_logits=True, student_model_loaded=False, student_updates=0, parent_not_admitted=True))
    try:
        weights = recipe['proposed_example_weights']; pending = []; losses = []
        for i in range(73):
            row = list(representatives.values())[i % 25]
            q = annotate(materialize(copy.deepcopy(row)), row)['questions'][0]
            z = torch.linspace(-.9, .8, len(q['keys']), dtype=torch.float64, requires_grad=True)
            losses.append(float(weighted_loss(z, q, 'cpu', weights, pending).detach()))
        driver = ROOT/'tools/train_curriculum_yaku_parent_v1.py'
        tree = ast.parse(driver.read_text())
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        tracker_node = next(n for n in train.body if isinstance(n, ast.ClassDef) and n.name == 'Tracker')
        fake_root = PREFLIGHT/'software-contract-only'; fake_root.mkdir()
        validation_point = dict(read(synthetic_profile), step=0)
        write(fake_root/'validation-curve.json', [validation_point])
        validation_calls = []
        scope = dict(LocalTracker=LocalTracker, summarize=summarize, pending=pending,
            reg={'course_loss_weights': weights}, c=c, out=fake_root, read=read,
            fence=SimpleNamespace(validation=validation_calls.append))
        exec(compiled_node(tracker_node, str(driver)), scope)
        tracker = scope['Tracker'](MetricJournal(fake_root/'metrics',
            'SOFTWARE-CONTRACT-NOT-A-MODEL', 'SYNTHETIC-LOGGING-CONTRACT', 0), 'software-only')
        tracker.log({'optimizer_step': 1, 'train/objective': sum(losses)/73,
            'train/lr': c['learning_rate'], 'train/grad_norm_pre_clip': 1., 'train/records_seen': 73})
        if pending: raise ValueError('Actual tracker failed to clear its observations')
        tracker.log({'optimizer_step': 0})
        if validation_calls != [0]: raise ValueError('Actual tracker validation fence differs')
        journal = fake_root/'metrics/train_metrics.jsonl'
        logged = [json.loads(line) for line in journal.open()]
        val_logged = [json.loads(line) for line in (fake_root/'metrics/validation_metrics.jsonl').open()]
        if len(logged) != 1 or len(val_logged) != 1 or logged[0]['loss_method'] != METHOD:
            raise ValueError('Actual tracker/journal schema or method differs')
        auditor = ROOT/'tools/audit_curriculum_yaku_parent_v1.py'
        audit_tree = ast.parse(auditor.read_text())
        audit = next(n for n in audit_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
        loss_loop = next(n for n in audit.body if isinstance(n, ast.For)
            and isinstance(n.iter, ast.Name) and n.iter.id == 'logs')
        code = compiled_node(loss_loop, str(auditor))
        def check(rows):
            context = dict(logs=rows, METHOD=METHOD, MASS=MASS, COMPONENTS=COMPONENTS,
                NEW_SOURCES=NEW_SOURCES, math=math, exposed=Counter(), cumulative=0,
                reg={'course_loss_weights': weights})
            exec(code, context)
            return context
        checked = check(logged)
        if checked['cumulative'] != 73 or len(checked['exposed']) != 25:
            raise ValueError('Actual audit loop source/exposure contract differs')
        rejected = []
        for field in ('loss', 'signed_objective_adjustment', 'new_route_objective', 'records_seen'):
            corrupted = copy.deepcopy(logged); corrupted[0][field] += 1.
            try: check(corrupted)
            except ValueError: rejected.append(field)
            else: raise ValueError('Actual auditor accepted corrupted '+field)
        if any(sha(p) != h for p, h in files.items()): raise ValueError('A bound source/proof changed')
        summary = dict(status='retained_yaku_entry_preflight_passed_not_parent_or_fit_admitted',
            method=METHOD, unit_tests_passed=8, original_numerical_recipe_unchanged=True,
            prior_full_TRAIN_contract_records=91247, prior_full_VAL_contract_menus=24516,
            actual_sources_represented_in_synthetic_logging=25,
            synthetic_microbatch_observations=73, actual_nested_tracker_and_MetricJournal_exercised=True,
            actual_auditor_loss_loop_exercised=True, corrupted_journal_fields_rejected=rejected,
            actual_tracker_validation_branch_exercised=True, student_model_loaded=False,
            model_forwards=0, student_updates=0, new_games=0, child_jobs_submitted=False,
            actual_parent_qualification_and_full_model_warmstart_still_required=True,
            software_contract_journal_not_model_performance=True, seconds=time.monotonic()-started)
        write(PREFLIGHT/'summary-private.json', summary)
        for p in (PREFLIGHT/'registered-private.json', PREFLIGHT/'summary-private.json',
                  journal, fake_root/'metrics/validation_metrics.jsonl', fake_root/'validation-curve.json'):
            files[str(p.resolve())] = sha(p)
        write(PREFLIGHT/'completion-binding-private.json', dict(files=files))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(PREFLIGHT/'error-private.json', dict(error=repr(error)))
        raise


if __name__ == '__main__': main()
