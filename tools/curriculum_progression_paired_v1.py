"""Conditional same-parent 3A retry with paired RI accumulation orders.

No active run is modified. Registration requires an actual completed original
fit/audit with a failed RI criterion. Training reuses the original driver,
warm-start proofs, objective, native grid and independent final auditor.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tomllib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_riichi_pair_order_v1 import pair_within_blocks
from tools.curriculum_progression_objective_v1 import epoch_order

METHOD = 'prospective_RI_pairs_within_original_mixing_blocks_v1'
RUN = ROOT / 'runs/curriculum-progression-riichi-paired-v1'
PUBLIC = ('tools/curriculum_progression_paired_v1.py',
          'tools/curriculum_riichi_pair_order_v1.py',
          'tests/core/test_curriculum_progression_paired_v1.py',
          'tests/core/test_curriculum_riichi_pair_order_v1.py',
          'cluster/curriculum-progression-paired-v1.slurm',
          'docs/experiments/plans/curriculum-riichi-paired-replay-20261124.md')


def require_failed_RI_control(done, evidence, execution, last, limits):
    if (done['status'] != 'progression2400_fit_and_real_curves_complete'
            or done['optimizer_steps'] != 2400 or done['actual_records_seen'] != 175190
            or evidence['status'] != 'progression2400_final_evidence_audit_passed'
            or evidence['optimizer_steps'] != 2400 or evidence['actual_records_seen'] != 175190
            or not evidence['all376_optimizer_states_match'] or evidence['saved_final_reloaded_menus'] != 23688
            or not evidence['source_and_data_hashes_unchanged']
            or evidence['stage3_advancement']['passed'] is not False
            or evidence['basic_retention']['passed'] is not False
            or execution['status'] != 'actual_progression2400_full_evidence_independently_audited'
            or execution['original_launch_binding_unchanged'] is not True):
        raise ValueError('Completed original fit and independent failed RI-retention grade required')
    recalls = {}
    for source in ('legacy', 'teacher'):
        values = last['riichi_by_source'][source]
        if (set(values) != {'tp', 'fn', 'tn', 'fp'}
                or any(type(n) is not int or n < 0 for n in values.values())
                or values['tp'] + values['fn'] == 0 or values['tn'] + values['fp'] == 0):
            raise ValueError('Actual positive and negative RI cohorts required')
        recalls[source] = dict(accept=values['tp'] / (values['tp'] + values['fn']),
                               decline=values['tn'] / (values['tn'] + values['fp']))
    gains = evidence['basic_retention']['riichi_balanced_gain_by_source']
    failed = (min(value for row in recalls.values() for value in row.values())
              < limits['min_both_RI_recalls_by_legacy_and_teacher_source']
              or min(gains.values()) < -limits['max_RI_balanced_accuracy_drop_from_step0'])
    if not failed:
        raise ValueError('No actual failed RI criterion; do not launch a speculative retry')
    return dict(original_advancement_passed=False, original_retention_passed=False,
                actual_RI_recalls=recalls, actual_RI_balanced_gains=gains,
                other_failed_conditions_not_waived=True)


def paired_epoch(categories, baseline_rng, epoch):
    original, receipt = epoch_order(categories, baseline_rng)
    original_sha = hashlib.sha256(json.dumps(original).encode()).hexdigest()
    paired, diagnostic = pair_within_blocks(original, categories, random.Random(2026112406 + epoch))
    receipt = dict(receipt)
    receipt['source_counts_by_update'] = [dict(Counter(categories[index] for index in paired[start:start+73]))
                                         for start in range(0, len(paired), 73)]
    receipt.update(epoch=epoch, order_sha256=hashlib.sha256(json.dumps(paired).encode()).hexdigest(),
                   baseline_order_sha256=original_sha, paired_order_seed=2026112406 + epoch,
                   paired_order_method=METHOD, pairing_diagnostic=diagnostic)
    return paired, receipt


class PairedEpochRandom(random.Random):
    def __init__(self, seed, categories, directory, expected):
        super().__init__(seed)
        self.categories = list(categories)
        self.directory = Path(directory)
        self.expected = expected
        self.epochs = 0

    def shuffle(self, order):
        if (sorted(order) != list(range(87595)) or self.epochs >= 2):
            raise ValueError('Exactly two complete original TRAIN permutations required')
        baseline = random.Random()
        baseline.setstate(self.getstate())
        selected, receipt = paired_epoch(self.categories, baseline, self.epochs + 1)
        expected = {key: value for key, value in receipt.items() if key != 'source_counts_by_update'}
        if expected != self.expected[self.epochs]:
            raise ValueError('Actual paired epoch differs from its immutable preflight')
        self.setstate(baseline.getstate())
        self.epochs += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f'epoch-{self.epochs:02d}.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        order[:] = selected


def random_proxy(original, seed, categories, directory, expected):
    created = []
    def factory(value=None):
        if type(value) is int and value == seed:
            if created:
                raise ValueError('Unexpected duplicate candidate epoch RNG')
            instance = PairedEpochRandom(value, categories, directory, expected)
            created.append(instance)
            return instance
        return original.Random(value)
    return SimpleNamespace(seed=original.seed, Random=factory), created


def require_matched_recipe(reg, control, arguments, original_arguments):
    c, old = reg['curriculum'], control['curriculum']
    expected_c = dict(old, run_id='jev-curriculum-progression-riichi-paired-v1', stage='stage3-riichi-paired-v1',
        out=str(RUN.relative_to(ROOT)/'stage3-fit'), plan_root=str(RUN.relative_to(ROOT)/'fit-plan'),
        paired_order_method=METHOD, retry_from_qualified_calls_not_control_final=True)
    expected_args = dict(original_arguments, out=str(RUN/'stage3-fit/training-run'))
    if c != expected_c or arguments != expected_args:
        raise ValueError('Only output identity and the explicit paired epoch ordering may change')
    for key in ('source', 'source_sha256', 'parent_model', 'course_loss_weights', 'course_loss_mass',
                'expected_record_exposures', 'train_records', 'validation_records', 'registered_final_step'):
        if reg[key] != control[key]:
            raise ValueError('Matched control numerical/data contract changed: ' + key)
    expected_config = dict(control['training_config'], model_id=c['run_id'], out=c['out'],
        trainer_config=str(RUN/'fit-plan/trainer.toml'),
        trainer_config_sha256=reg['training_config']['trainer_config_sha256'])
    if reg['training_config'] != expected_config:
        raise ValueError('Matched base/parent/objective configuration changed')


def verify_retry(reg):
    from tools.curriculum_transfer import read, sha
    from tools.curriculum_progression_runtime_v1 import require_registration
    require_registration(reg)
    control = read(reg['paired_control_registration'])
    require_registration(control)
    require_matched_recipe(reg, control,
        tomllib.loads(Path(reg['training_config']['trainer_config']).read_text()),
        tomllib.loads(Path(control['training_config']['trainer_config']).read_text()))
    if (reg['curriculum'].get('paired_order_method') != METHOD
            or reg['parent_model'] != control['parent_model']
            or reg['course_loss_weights'] != control['course_loss_weights']
            or reg['course_loss_mass'] != control['course_loss_mass']
            or reg['expected_record_exposures'] != control['expected_record_exposures']
            or reg['curriculum']['development_targets'] != control['curriculum']['development_targets']
            or reg['curriculum']['out'] == control['curriculum']['out']):
        raise ValueError('Same qualified calls parent, data objective, budget and original grade required')
    for path, expected in reg['inputs'].items():
        if sha(path) != expected:
            raise ValueError('Immutable retry/control input changed: ' + path)


def prepare(control_path):
    from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
    from tools.curriculum_transfer import read, sha
    from tools.curriculum_progression_runtime_v1 import require_registration
    from tools.curriculum_progression_parent_v1 import verify_parent
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    gpu_only()
    old = read(control_path)
    require_registration(old)
    control_run = control_path.parent.parent
    fit = ROOT / old['curriculum']['out']
    paths = [control_path, fit/'completion.json', control_run/'final-evidence-private.json',
             control_run/'audit-execution-private.json', control_run/'fit-terminal-scheduler-private.json',
             fit/'metrics/validation_metrics.jsonl', control_run/'launch-binding-private.json']
    done, evidence, execution, terminal = (read(path) for path in paths[1:5])
    last = [json.loads(line) for line in paths[5].open()][-1]
    trigger = require_failed_RI_control(done, evidence, execution, last, old['curriculum']['development_targets'])
    scheduler_exit(terminal['exit'], terminal['raw'], ROOT/'cluster/curriculum-progression-v4.slurm')
    if str(execution['actual_fit_job']) != str(terminal['exit']['job_id']):
        raise ValueError('Independent control audit and actual fit terminal differ')
    old_binding = read(control_run/'launch-binding-private.json')
    for path, expected in {**old['inputs'], **old_binding['files']}.items():
        if sha(path) != expected:
            raise ValueError('Original control source changed: ' + path)
    sys.path[:0] = [old['source'], old['training_config']['upstream_source']]
    verify_parent(old['training_config'])
    if RUN.exists():
        raise ValueError('Never overwrite a retry registration or fit')
    arguments = tomllib.loads(Path(old['training_config']['trainer_config']).read_text())
    categories = [json.loads(line)['questions']['action']['src'] for line in Path(arguments['data']).open()]
    baseline_rng = random.Random(old['curriculum']['seed'])
    expected_orders = []
    for epoch in (1, 2):
        _, receipt = paired_epoch(categories, baseline_rng, epoch)
        if receipt['baseline_order_sha256'] != old['expected_orders'][epoch-1]['order_sha256']:
            raise ValueError('Paired experiment no longer reconstructs the exact original control orders')
        expected_orders.append({key:value for key,value in receipt.items() if key != 'source_counts_by_update'})
    reg = copy.deepcopy(old)
    c, config = reg['curriculum'], reg['training_config']
    c.update(run_id='jev-curriculum-progression-riichi-paired-v1', stage='stage3-riichi-paired-v1',
             out=str(RUN.relative_to(ROOT)/'stage3-fit'), plan_root=str(RUN.relative_to(ROOT)/'fit-plan'),
             paired_order_method=METHOD, retry_from_qualified_calls_not_control_final=True)
    plan = RUN/'fit-plan'
    plan.mkdir(parents=True)
    arguments['out'] = str(RUN/'stage3-fit/training-run')
    trainer = plan/'trainer.toml'
    trainer.write_text(''.join(f'{key} = {json.dumps(value)}\n' for key,value in arguments.items()))
    config.update(model_id=c['run_id'], out=c['out'], trainer_config=str(trainer), trainer_config_sha256=sha(trainer))
    config_path = plan/'training-config.json'
    write(config_path, config)
    files = dict(old['inputs'])
    for path in paths + [trainer, config_path] + [ROOT/name for name in PUBLIC]:
        files[str(path.resolve())] = sha(path)
    reg.update(inputs=files, expected_orders=expected_orders, paired_control_registration=str(control_path),
               paired_control_failed_RI_grade=trigger, control2400_parameters_not_in_candidate_ancestry=True,
               original_budget_and_gradient_recipe_unchanged=False,
               same_parent_data_objective_budget_only_epoch_order_changed=True)
    verify_retry(reg)
    registered = plan/'registered.json'
    write(registered, reg)
    write(RUN/'launch-binding-private.json', dict(files={**files,str(registered):sha(registered)},
          registered_sha256=sha(registered), original_failed_control_grade_preserved=trigger,
          selected_initializer='original_qualified_calls388_not_failed3A_final', new_optimizer_updates=0))
    print(json.dumps(dict(status='conditional_paired_3A_retry_registered_not_started', candidate_updates=0)), flush=True)


def train(registered):
    from tools.curriculum_calls_post_reach_data_v1 import gpu_only
    from tools.curriculum_transfer import read
    gpu_only()
    verify_retry(read(registered))
    from tools import train_curriculum_progression_v3 as driver
    original = driver.random_proxy
    driver.random_proxy = random_proxy
    try:
        driver.train(registered)
    finally:
        driver.random_proxy = original


def audit(registered, fit_job):
    from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
    from tools.curriculum_transfer import read, sha
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    gpu_only()
    reg = read(registered)
    verify_retry(reg)
    run = registered.parent.parent
    exit_path = ROOT/'logs'/f'curriculum-progression-paired-exit-{fit_job}.json'
    receipt = read(exit_path)
    if receipt['phase'] != 'fit':
        raise ValueError('Actual completed retry FIT phase required')
    raw = subprocess.check_output(['scontrol','show','job','-o',str(fit_job)], text=True, timeout=20)
    scheduler_exit(receipt, raw, ROOT/'cluster/curriculum-progression-paired-v1.slurm')
    write(run/'fit-terminal-scheduler-private.json', dict(raw=raw, exit=receipt))
    binding = read(run/'launch-binding-private.json')
    if any(sha(path) != expected for path,expected in binding['files'].items()):
        raise ValueError('Retry launcher changed before audit')
    from tools.audit_curriculum_progression_v1 import audit as original_audit
    original_audit(registered, run/'final-evidence-private.json')
    verify_retry(reg)
    write(run/'audit-execution-private.json', dict(status='actual_progression2400_full_evidence_independently_audited',
          actual_fit_job=fit_job, original_launch_binding_unchanged=True, paired_order_method=METHOD,
          control_failed_grade_preserved=reg['paired_control_failed_RI_grade'], automatic_release=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare','fit','audit'), required=True)
    parser.add_argument('--control-registration', type=Path, default=ROOT/'runs/curriculum-progression-v4/fit-plan/registered.json')
    parser.add_argument('--registered', type=Path, default=RUN/'fit-plan/registered.json')
    parser.add_argument('--fit-job', type=int)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.control_registration.resolve())
    elif args.phase == 'fit':
        train(args.registered.resolve())
    else:
        if args.fit_job is None:
            parser.error('--fit-job is required for audit')
        audit(args.registered.resolve(), args.fit_job)
