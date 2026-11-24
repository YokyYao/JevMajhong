"""Synthetic interface contracts; no real parent, model or dataset is admitted."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tools import curriculum_placement_parent_runtime_v1 as runtime
from tools import curriculum_placement_retained_parent_v1 as parent
from tools import curriculum_placement_parent_loss_v2 as loss
from tools import curriculum_yaku_parent_loss_v1 as yaku
from tools.audit_curriculum_placement_parent_v1 import journal_evidence
from tools.train_curriculum_placement_parent_v1 import ROOT
from tools.curriculum_transfer import read
from jevmahjong.training.metric_journal import MetricJournal, LocalTracker
from tests.core.test_curriculum_placement_runtime_v1 import registration_fixture
from tests.core.test_curriculum_progression_parent_loss_v1 import fixture


def registration():
    reg = registration_fixture(); reg['curriculum']['loss_method'] = runtime.LOSS
    reg.update(training_config={'loss_method': runtime.LOSS},
        retained_parent_replay_objective_explicit=True,
        upstream_normalized_terminal_alignment_required=True, parent_loss_method=runtime.PARENT_LOSS)
    return reg


def parent_fixture():
    p = dict(status='yaku2500_registered_not_started', registered_final_step=2500,
        train_records=91247, validation_records=24516, course_loss_mass=yaku.MASS,
        training_config={'loss_method': yaku.METHOD}, retained_parent_replay_objective_explicit=True,
        curriculum=dict(epochs=2, accumulation=73, seed=2026112305, loss_method=yaku.METHOD))
    from tools.curriculum_complete_legal_class_policy_v1 import METHOD as ROUTE
    p['curriculum'].update(selection_method=ROUTE, expected_record_exposures=182494,
        validation_and_game_steps=[0, 625, 1250, 1875, 2500], optimizer_resume=False, new_optimizer=True)
    done = dict(status='yaku2500_fit_and_real_curves_complete', loss_method=yaku.METHOD,
        optimizer_steps=2500, actual_records_seen=182494)
    evidence = dict(status='yaku2500_final_evidence_audit_passed', loss_method=yaku.METHOD,
        optimizer_steps=2500, actual_records_seen=182494, stage3B_advancement={'passed': True},
        saved_final_reloaded_menus=24516, all376_optimizer_states_match=True,
        source_and_data_hashes_unchanged=True, final_reserved_grid_unused=True)
    execution = dict(status='actual_yaku2500_full_evidence_independently_audited', loss_method=yaku.METHOD,
        original_launch_binding_unchanged=True, actual_fit_job='123')
    cap = dict(status='actual_retained_yaku_terminals_preserved_not_stage4_admitted',
        strategy_grade_overridden=False, stage4_parent_sealed=False, student_updates=0,
        original_strategy_grade={'passed': True},
        scheduler_proofs={role: dict(successful=True, job_id='123' if role == 'fit' else role)
            for role in ('prepare', 'fit', 'audit')})
    return p, done, evidence, execution, cap


def actual_tracker_contract(directory):
    source = ROOT/'tools/train_curriculum_placement_parent_v1.py'; tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    tracker = next(n for n in function.body if isinstance(n, ast.ClassDef) and n.name == 'Tracker')
    reg = dict(registered_final_step=2, expected_record_exposures=114, course_loss_weights=dict.fromkeys(loss.masses(), 1.),
        curriculum=dict(stage_offset=0))
    c = dict(learning_rate=3e-6, head_learning_rate=1.5e-5); pending = []
    scope = dict(LocalTracker=LocalTracker, summarize=loss.summarize, pending=pending, reg=reg, c=c, read=read,
        out=directory, METHOD=loss.METHOD)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tracker], type_ignores=[])), str(source), 'exec'), scope)
    target = scope['Tracker'](MetricJournal(directory/'metrics', 'SOFTWARE-CONTRACT-NOT-MODEL', 'synthetic57', 0), 'synthetic')
    sources = list(loss.masses())
    for step in (1, 2):
        values = []
        for index, source_name in enumerate(sources):
            if source_name in yaku.MASS:
                if source_name.endswith(('riichi_yes', 'riichi_no')):
                    kinds = ['reach', 'dahai', 'dahai']; label = 0 if source_name.endswith('riichi_yes') else 1
                elif ':call' in source_name or ':kan' in source_name: kinds, label = ['none', 'pon', 'daiminkan'], 0
                else: kinds, label = ['dahai']*3, 1
                soft = [.4, .6, 0.] if ':mc:' in source_name else None
            elif index % 2:
                kinds, label, soft = ['reach', 'dahai', 'dahai'], 2, [.2, .3, .5+1e-15]
            else: kinds, label, soft = ['none', 'pon', 'chi'], 1, [.4, .6, 0.]
            request, encoded = fixture(source_name, kinds, label, soft)
            if soft is not None: encoded['questions'][0]['target'] = [v/sum(soft) for v in soft]
            q = loss.annotate(encoded, request)['questions'][0]
            z = torch.linspace(-.8, .9, 3, dtype=torch.float64, requires_grad=True)
            values.append(float(loss.weighted_loss(z, q, 'cpu', reg['course_loss_weights'], pending).detach()))
        target.log(dict(optimizer_step=step, **{'train/objective': sum(values)/57, 'train/lr': 3e-6,
            'train/grad_norm_pre_clip': 1., 'train/records_seen': step*57}))
        if pending: raise AssertionError('Actual tracker failed to clear observations')
    with (directory/'metrics/train_metrics.jsonl').open() as stream:
        logs = [json.loads(line) for line in stream]
    orders = [dict(source_exposure_counts_by_update=[dict.fromkeys(sources, 1)]) for _ in range(2)]
    reg['expected_orders'] = orders
    return reg, logs, orders


class PlacementParentInterfaceTests(unittest.TestCase):
    def test_new_runtime_preserves_the_original_numeric_contract_without_mutation(self):
        reg = registration(); original = copy.deepcopy(reg); runtime.require_registration(reg)
        self.assertEqual(reg, original)
        self.assertEqual(runtime.evaluation_groups(reg, 0), 8)
        self.assertEqual(runtime.evaluation_groups(reg, reg['registered_final_step']), 8)

    def test_old_loss_flags_and_weakened_budget_or_baseline_are_rejected(self):
        for key in ('retained_parent_replay_objective_explicit', 'upstream_normalized_terminal_alignment_required'):
            reg = registration(); reg[key] = False
            with self.subTest(key=key), self.assertRaises(ValueError): runtime.require_registration(reg)
        for key, value in (('loss_method', runtime.original.LOSS), ('epochs', 1), ('stage_zero_seed_groups', 2)):
            reg = registration(); reg['curriculum'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): runtime.require_registration(reg)

    def test_complete_synthetic_parent_contract_can_pass_but_is_not_an_actual_seal(self):
        parent.require_parent_evidence(*parent_fixture())

    def test_parent_loss_failed_grade_budget_and_phase_failures_are_rejected(self):
        for index, key, value in ((1, 'loss_method', 'old'), (1, 'optimizer_steps', 1250),
                (2, 'stage3B_advancement', {'passed': False}), (2, 'saved_final_reloaded_menus', 100),
                (3, 'loss_method', 'old')):
            values = parent_fixture(); values[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): parent.require_parent_evidence(*values)
        values = parent_fixture(); values[-1]['scheduler_proofs']['prepare']['successful'] = False
        with self.assertRaises(ValueError): parent.require_parent_evidence(*values)

    def test_actual_nested_tracker_and_auditor_accept_full57_component_journal(self):
        with tempfile.TemporaryDirectory() as path:
            reg, logs, orders = actual_tracker_contract(Path(path))
            evidence = journal_evidence(reg, logs, orders)
            self.assertEqual(evidence['actual_record_exposures'], 114)
            self.assertEqual(len(evidence['actual_source_exposures']), 57)
            self.assertEqual(evidence['actual_optimizer_observations'], 2)

    def test_actual_auditor_rejects_corrupted_signed_or_route_accounting(self):
        with tempfile.TemporaryDirectory() as path:
            reg, logs, orders = actual_tracker_contract(Path(path))
            for field in ('loss', 'ce', 'signed_objective_adjustment', 'new_route_objective', 'terminal_soft_objective', 'records_seen'):
                bad = copy.deepcopy(logs); bad[-1][field] += 1
                with self.subTest(field=field), self.assertRaises(ValueError): journal_evidence(reg, bad, orders)
            bad = copy.deepcopy(logs); bad[-1]['loss_method'] = 'old'
            with self.assertRaises(ValueError): journal_evidence(reg, bad, orders)

    def test_full57_journal_rejects_missing_step_or_changed_source_order(self):
        with tempfile.TemporaryDirectory() as path:
            reg, logs, orders = actual_tracker_contract(Path(path))
            with self.assertRaises(ValueError): journal_evidence(reg, logs[:-1], orders)
            bad = copy.deepcopy(orders); bad[-1]['source_exposure_counts_by_update'][0][next(iter(loss.masses()))] += 1
            with self.assertRaises(ValueError): journal_evidence(reg, logs, bad)


if __name__ == '__main__': unittest.main()
